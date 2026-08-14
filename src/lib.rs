//! Rust core for the PORA collision-risk metric hot loop.
//!
//! Reference semantics come from the Python packages
//! `pora-replication` (github.com/chenggma/pora-replication) and the
//! `pora_score` wiring in `risk-metric-bench`. Every function here is
//! differential-tested against that reference (tests/test_parity.py):
//! same grid conventions, same operation order per cell, same clamps.
//!
//! Three entry points, coarse to fine:
//!   * `pora_score` — fused end-to-end scalar (the fast path: no
//!     intermediate arrays cross the FFI boundary, grids live in two
//!     reused buffers)
//!   * `cv_gaussian_grid` — constant-velocity-Gaussian occupancy source
//!   * `pora_step` — AV-frame resample + risk for one timestep
//!
//! Parallelism: rayon over grid rows; each Python call releases the GIL
//! for the duration of the compute.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

/// (x, y, vx, vy, length, width)
type Foe = (f64, f64, f64, f64, f64, f64);

// ------------------------------------------------------------------ geometry

/// Stopping sight distance, SI form: v*r + v^2 / (2a).
#[inline]
fn ssd(speed_ms: f64, reaction_time_s: f64, decel_ms2: f64) -> f64 {
    speed_ms * reaction_time_s + speed_ms * speed_ms / (2.0 * decel_ms2)
}

/// Safety box Phi / guaranteed-collision subarea phi, as half-dimensions.
#[derive(Clone, Copy)]
struct BoxDims {
    hx_out: f64,
    hy_out: f64,
    hx_in: f64,
    hy_in: f64,
}

impl BoxDims {
    fn from_full(phi_length: f64, phi_width: f64, core_length: f64, core_width: f64) -> Self {
        BoxDims {
            hx_out: phi_length / 2.0,
            hy_out: phi_width / 2.0,
            hx_in: core_length / 2.0,
            hy_in: core_width / 2.0,
        }
    }

    fn derive(
        av_length: f64,
        av_width: f64,
        fleet_max_length: f64,
        fleet_min_width: f64,
        speed_ms: f64,
        reaction_time_s: f64,
        decel_ms2: f64,
    ) -> Self {
        let phi_length = av_length + fleet_max_length + ssd(speed_ms, reaction_time_s, decel_ms2);
        let phi_width = av_width + fleet_max_length;
        let core_length = (av_length + fleet_min_width).min(phi_length);
        let core_width = (av_width + fleet_min_width).min(phi_width);
        Self::from_full(phi_length, phi_width, core_length, core_width)
    }

    /// P(C|O): 1 inside phi, 0 outside Phi, linear nested-rectangle decay
    /// between (same interpolation as the Python reference).
    #[inline]
    fn pcgo(&self, x: f64, y: f64) -> f64 {
        if x.abs() > self.hx_out || y.abs() > self.hy_out {
            return 0.0;
        }
        let dx = self.hx_out - self.hx_in;
        let dy = self.hy_out - self.hy_in;
        let tx = if dx <= 0.0 {
            0.0
        } else {
            (x.abs() - self.hx_in) / dx
        };
        let ty = if dy <= 0.0 {
            0.0
        } else {
            (y.abs() - self.hy_in) / dy
        };
        1.0 - tx.max(ty).clamp(0.0, 1.0)
    }
}

// ----------------------------------------------------------------- sampling

/// Bilinear sample of a row-major (ny, nx) grid whose cell [j, i] center is
/// origin + (i*res, j*res). Points outside return 0 (unobserved = free),
/// matching OccupancyGrid.sample_bilinear.
#[inline]
#[allow(clippy::too_many_arguments)]
fn bilinear(data: &[f64], ny: i64, nx: i64, ox: f64, oy: f64, res: f64, x: f64, y: f64) -> f64 {
    let gx = (x - ox) / res;
    let gy = (y - oy) / res;
    let i0 = gx.floor() as i64;
    let j0 = gy.floor() as i64;
    let fx = gx - i0 as f64;
    let fy = gy - j0 as f64;
    let mut out = 0.0;
    let corners = [
        (i0, j0, (1.0 - fx) * (1.0 - fy)),
        (i0 + 1, j0, fx * (1.0 - fy)),
        (i0, j0 + 1, (1.0 - fx) * fy),
        (i0 + 1, j0 + 1, fx * fy),
    ];
    for (ii, jj, w) in corners {
        if ii >= 0 && ii < nx && jj >= 0 && jj < ny {
            out += data[(jj * nx + ii) as usize] * w;
        }
    }
    out
}

// ------------------------------------------------- occupancy source (CV+Gauss)

#[allow(clippy::too_many_arguments)]
fn cv_gaussian_impl(
    foes: &[Foe],
    lead_time: f64,
    ox: f64,
    oy: f64,
    ny: usize,
    nx: usize,
    res: f64,
    sigma0: f64,
    sigma_growth: f64,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), ny * nx);
    // Exactness-preserving tail cutoff: when p = density * footprint is
    // below ~5.5e-17, the reference's `free *= 1.0 - p` rounds to exactly
    // `free` in f64, so skipping the cell is bit-identical, not an
    // approximation. Solve density(d2) * footprint = P_CUT for d2.
    const P_CUT: f64 = 1e-20;

    // Pre-derive per-foe constants once (mean position, 2*sigma^2,
    // footprint, squared cutoff radius).
    let pre: Vec<(f64, f64, f64, f64, f64)> = foes
        .iter()
        .filter_map(|f| {
            let mx = f.0 + f.2 * lead_time;
            let my = f.1 + f.3 * lead_time;
            let sigma = sigma0 + sigma_growth * lead_time + (f.4 + f.5) / 8.0;
            let footprint = f.4 * f.5;
            let two_s2 = 2.0 * sigma * sigma;
            let peak = footprint / (PI * two_s2);
            if peak < P_CUT {
                return None; // whole foe is below the cutoff everywhere
            }
            let cutoff_d2 = two_s2 * (peak / P_CUT).ln();
            Some((mx, my, two_s2, footprint, cutoff_d2))
        })
        .collect();

    out.par_chunks_mut(nx).enumerate().for_each(|(j, row)| {
        let y = oy + res * j as f64;
        for (i, cell) in row.iter_mut().enumerate() {
            let x = ox + res * i as f64;
            let mut free = 1.0_f64;
            for &(mx, my, two_s2, footprint, cutoff_d2) in &pre {
                let d2 = (x - mx) * (x - mx) + (y - my) * (y - my);
                if d2 > cutoff_d2 {
                    continue;
                }
                let density = (-d2 / two_s2).exp() / (PI * two_s2);
                let p = (density * footprint).clamp(0.0, 1.0);
                free *= 1.0 - p;
            }
            *cell = 1.0 - free;
        }
    });
}

// ------------------------------------------------------- AV-frame risk step

struct StepOut {
    po: Vec<f64>,
    raw_max: f64,
    adj_max: f64,
}

/// Output window dims for a half-extent at a resolution (matches
/// to_av_frame: n = max(1, round(2*half/res) + 1)).
#[inline]
fn window_len(half: f64, res: f64) -> usize {
    std::cmp::max(1, (2.0 * half / res).round() as i64 + 1) as usize
}

#[allow(clippy::too_many_arguments)]
fn pora_step_impl(
    global: &[f64],
    gny: i64,
    gnx: i64,
    gox: f64,
    goy: f64,
    gres: f64,
    av_x: f64,
    av_y: f64,
    heading: f64,
    half_length: f64,
    half_width: f64,
    res: f64,
    bx: BoxDims,
    prev_po: Option<&[f64]>,
    beta: f64,
) -> StepOut {
    let nx = window_len(half_length, res);
    let ny = window_len(half_width, res);
    let (c, s) = (heading.cos(), heading.sin());
    let exp_beta = beta.exp();

    let mut po = vec![0.0_f64; ny * nx];
    let (raw_max, adj_max) = po
        .par_chunks_mut(nx)
        .enumerate()
        .map(|(j, row)| {
            let yl = -half_width + res * j as f64;
            let mut raw = f64::NEG_INFINITY;
            let mut adj = f64::NEG_INFINITY;
            for (i, cell) in row.iter_mut().enumerate() {
                let xl = -half_length + res * i as f64;
                let wx = av_x + c * xl - s * yl;
                let wy = av_y + s * xl + c * yl;
                let p_occ = bilinear(global, gny, gnx, gox, goy, gres, wx, wy);
                *cell = p_occ;
                let pc = bx.pcgo(xl, yl) * p_occ;
                raw = raw.max(pc);
                let a = match prev_po {
                    None => pc,
                    Some(prev) => {
                        let d_po = p_occ - prev[j * nx + i];
                        pc * (beta * d_po).exp() / exp_beta
                    }
                };
                adj = adj.max(a);
            }
            (raw, adj)
        })
        .reduce(
            || (f64::NEG_INFINITY, f64::NEG_INFINITY),
            |a, b| (a.0.max(b.0), a.1.max(b.1)),
        );

    StepOut {
        po,
        raw_max,
        adj_max,
    }
}

// ---------------------------------------------------------------- py bindings

/// Constant-velocity Gaussian occupancy grid (parity:
/// pora.occupancy_sources.constant_velocity_gaussian).
#[pyfunction]
#[pyo3(signature = (foes, lead_time, origin, shape, resolution, sigma0=0.5, sigma_growth=0.5))]
#[allow(clippy::too_many_arguments)]
fn cv_gaussian_grid<'py>(
    py: Python<'py>,
    foes: Vec<Foe>,
    lead_time: f64,
    origin: (f64, f64),
    shape: (usize, usize),
    resolution: f64,
    sigma0: f64,
    sigma_growth: f64,
) -> Bound<'py, PyArray2<f64>> {
    let (ny, nx) = shape;
    let mut out = vec![0.0_f64; ny * nx];
    py.detach(|| {
        cv_gaussian_impl(
            &foes,
            lead_time,
            origin.0,
            origin.1,
            ny,
            nx,
            resolution,
            sigma0,
            sigma_growth,
            &mut out,
        )
    });
    Array2::from_shape_vec((ny, nx), out)
        .unwrap()
        .into_pyarray(py)
}

/// One PORA timestep from a global grid: resample into the AV frame,
/// apply the safety box and (optionally) the Cox adjustment.
/// Returns (av_frame_occupancy, per_step_unadjusted_max, per_step_max).
/// Parity: pora.transform.to_av_frame + pora.risk.collision_probability
/// + pora.risk.cox_adjust.
#[pyfunction]
#[pyo3(signature = (global, origin, resolution, av_x, av_y, av_heading, half_length, half_width,
                    out_resolution, box_dims, prev_po=None, beta=1.0))]
#[allow(clippy::too_many_arguments)]
fn pora_step<'py>(
    py: Python<'py>,
    global: PyReadonlyArray2<'py, f64>,
    origin: (f64, f64),
    resolution: f64,
    av_x: f64,
    av_y: f64,
    av_heading: f64,
    half_length: f64,
    half_width: f64,
    out_resolution: f64,
    box_dims: (f64, f64, f64, f64),
    prev_po: Option<PyReadonlyArray2<'py, f64>>,
    beta: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, f64, f64)> {
    let g = global.as_array();
    let (gny, gnx) = (g.nrows() as i64, g.ncols() as i64);
    let gslice = g.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("global grid must be C-contiguous")
    })?;
    let prev = prev_po.as_ref().map(|p| p.as_array());
    let prev_slice = match prev.as_ref() {
        None => None,
        Some(p) => Some(p.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("prev_po must be C-contiguous")
        })?),
    };
    let bx = BoxDims::from_full(box_dims.0, box_dims.1, box_dims.2, box_dims.3);

    let out = py.detach(|| {
        pora_step_impl(
            gslice,
            gny,
            gnx,
            origin.0,
            origin.1,
            resolution,
            av_x,
            av_y,
            av_heading,
            half_length,
            half_width,
            out_resolution,
            bx,
            prev_slice,
            beta,
        )
    });
    let nx = window_len(half_length, out_resolution);
    let ny = window_len(half_width, out_resolution);
    Ok((
        Array2::from_shape_vec((ny, nx), out.po)
            .unwrap()
            .into_pyarray(py),
        out.raw_max,
        out.adj_max,
    ))
}

/// Fused end-to-end PORA scalar for a constant-velocity ego plan against
/// constant-velocity-Gaussian foes (parity: bench.metrics.pora_score in
/// risk-metric-bench). ego = (x, y, vx, vy, heading_rad, speed, length, width).
#[pyfunction]
#[pyo3(signature = (ego, foes, horizon_s=2.5, dt=0.5, beta=1.0, resolution=0.5, extent=40.0,
                    reaction_time_s=1.0, decel_ms2=9.81, sigma0=0.5, sigma_growth=0.5))]
#[allow(clippy::too_many_arguments)]
fn pora_score(
    py: Python<'_>,
    ego: (f64, f64, f64, f64, f64, f64, f64, f64),
    foes: Vec<Foe>,
    horizon_s: f64,
    dt: f64,
    beta: f64,
    resolution: f64,
    extent: f64,
    reaction_time_s: f64,
    decel_ms2: f64,
    sigma0: f64,
    sigma_growth: f64,
) -> f64 {
    if foes.is_empty() {
        return 0.0;
    }
    py.detach(|| {
        let (ex, ey, evx, evy, heading, speed, el, ew) = ego;
        let fleet_max_l = foes.iter().map(|f| f.4).fold(f64::NEG_INFINITY, f64::max);
        let fleet_min_w = foes.iter().map(|f| f.5).fold(f64::INFINITY, f64::min);

        let n = (2.0 * extent / resolution).round() as usize + 1;
        let (ox, oy) = (ex - extent, ey - extent);
        let steps = (horizon_s / dt).round() as usize + 1;

        let bx = BoxDims::derive(
            el,
            ew,
            fleet_max_l,
            fleet_min_w,
            speed,
            reaction_time_s,
            decel_ms2,
        );
        let half_len = bx.hx_out + 1.0; // phi_length/2 + 1, as in the bench
        let half_wid = bx.hy_out + 1.0;

        let mut global = vec![0.0_f64; n * n];
        let mut prev_po: Option<Vec<f64>> = None;
        let mut scalar = f64::NEG_INFINITY;
        for k in 0..steps {
            let t = k as f64 * dt;
            cv_gaussian_impl(
                &foes,
                t,
                ox,
                oy,
                n,
                n,
                resolution,
                sigma0,
                sigma_growth,
                &mut global,
            );
            let out = pora_step_impl(
                &global,
                n as i64,
                n as i64,
                ox,
                oy,
                resolution,
                ex + evx * t,
                ey + evy * t,
                heading,
                half_len,
                half_wid,
                resolution,
                bx,
                prev_po.as_deref(),
                beta,
            );
            scalar = scalar.max(out.adj_max);
            prev_po = Some(out.po);
        }
        scalar
    })
}

#[pymodule]
fn pora_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cv_gaussian_grid, m)?)?;
    m.add_function(wrap_pyfunction!(pora_step, m)?)?;
    m.add_function(wrap_pyfunction!(pora_score, m)?)?;
    Ok(())
}
