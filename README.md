# pora-rs

[![ci](https://github.com/chenggma/pora-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/chenggma/pora-rs/actions/workflows/ci.yml)

Rust rewrite of the hot loop of the PORA collision-risk metric, with PyO3
bindings — **differential-tested against the Python reference
implementation** ([pora-replication](https://github.com/chenggma/pora-replication))
on randomized inputs at 1e-12 relative tolerance.

**7.0–16.2x faster** than the numpy-vectorized reference on the
end-to-end scoring path (see [docs/benchmarks.md](docs/benchmarks.md);
Apple M-series, parity asserted on every benchmark call).

> PORA is from Wang, Yeo, Paiva, Utke, Delle Monache,
> [arXiv:2501.16480](https://arxiv.org/abs/2501.16480). This repo, like
> `pora-replication`, is unofficial and independent; the metric semantics
> replicated here are those of that Python package, gaps and all.

## C++ comparison implementation

`cpp/` holds a single-threaded C++17 implementation of the same hot loop,
bound with pybind11 and held to the same bar: the differential suite
asserts it against both the Python reference and the Rust extension at
1e-12 on randomized scenarios.

```bash
pip install ./cpp && python -m pytest tests/
```

On the benchmark cases it lands at **3.8–6.2x** over numpy, against
7.0–16.2x for the Rust extension. The gap is parallelism, not language:
the Rust path fans grid rows out with rayon, the C++ is single-threaded
by design so the comparison isolates per-core codegen. See
[docs/benchmarks.md](docs/benchmarks.md) for the three-way table.

## Why

`risk-metric-bench` labels Monte-Carlo SUMO scenarios with surrogate
safety metrics. PORA under a constant-velocity Gaussian occupancy source
costs ~11 ms per (ego, timestep) in numpy — the dominant term of a
60-seed benchmark run, and far too slow for larger sweeps. The pipeline
is: build a global occupancy grid per horizon step (Gaussian per foe per
cell), resample it into the AV frame (bilinear inverse mapping), apply
the safety-box collision field and the Cox adjustment, take maxima.

## What made it fast

1. **Fusion.** The Rust scoring path materializes no intermediate arrays:
   resample, safety-box weighting, Cox adjustment and the max-reduction
   happen per cell in one pass. The numpy reference allocates ~6 grid
   temporaries per step; here two reused buffers cross the whole horizon.
2. **Parallelism.** rayon over grid rows; the GIL is released for the
   duration of every call.
3. **An exactness-preserving tail cutoff.** When a foe's Gaussian tail
   contribution `p` satisfies `p < 1e-20`, the reference's
   `free *= 1.0 - p` rounds to exactly `free` in f64 — so skipping such
   cells is **bit-identical, not an approximation**. Each foe gets a
   precomputed cutoff radius; cost becomes proportional to occupied area
   instead of grid area. This is where the wide-grid case's 17.6x comes
   from — and it is why the differential tests still pass at 1e-12.

## Usage

```python
import pora_rs

ego = (x, y, vx, vy, heading_rad, speed, length, width)
foes = [(x, y, vx, vy, length, width), ...]

score = pora_rs.pora_score(ego, foes, horizon_s=2.5, dt=0.5,
                           beta=1.0, resolution=0.5, extent=40.0)
```

Drop-in for `bench.metrics.pora_score` in
[risk-metric-bench](https://github.com/chenggma/risk-metric-bench)
(same wiring, same defaults). Granular pieces for other predictors /
pipelines:

```python
grid = pora_rs.cv_gaussian_grid(foes, lead_time, origin, (ny, nx), resolution)
po, raw_max, adj_max = pora_rs.pora_step(
    grid, origin, resolution, av_x, av_y, heading,
    half_length, half_width, out_resolution,
    (phi_length, phi_width, core_length, core_width),
    prev_po=last_po, beta=1.0)
```

## Correctness story

`tests/test_parity.py` runs the Rust and Python implementations on the
same randomized scenarios and asserts agreement:

* occupancy source: 9 parameter combinations, dense grids;
* single step: headings, with/without Cox adjustment, window-rounding
  edge cases (the output grid must have *exactly* the reference's shape);
* end-to-end scalar: 8 random scenes, plus behavioural checks (empty
  scene scores 0; traffic that never enters the safety box scores
  exactly 0; head-on approach registers risk).

The benchmark script re-asserts parity on every timed call, so the
published speedups cannot drift away from correctness.

Tolerance rationale: both sides perform the same operations in the same
per-cell order; residual differences are libm `exp` ulps. 1e-12 relative
is orders of magnitude below any decision the metric could inform.

## Build

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release
pip install -e ".[test]" && pytest          # parity suite
python scripts/bench.py                      # benchmark table
```

Requires a Rust toolchain (rustup or `brew install rust`).

## Related

* [pora-replication](https://github.com/chenggma/pora-replication) — the
  Python reference this crate is tested against
* [risk-metric-bench](https://github.com/chenggma/risk-metric-bench) —
  where the metric is consumed (AUROC vs ground-truth collisions)

## License

MIT
