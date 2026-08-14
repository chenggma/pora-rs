"""Differential tests: pora_rs must reproduce the Python reference
(pora-replication) to float tolerance on randomized inputs.

Tolerances: the two implementations perform the same operations in the
same per-cell order, but libm exp() and instruction scheduling may differ
by ~1 ulp; 1e-12 relative is far below any physical meaning of the metric
and far above accumulated ulp noise.
"""

import math
import random

import numpy as np
import pytest

import pora_rs
from pora.geometry import SafetyBox, VehicleDims
from pora.heatmap import OccupancyGrid
from pora.occupancy_sources import FoeState, constant_velocity_gaussian
from pora.risk import collision_probability, cox_adjust, pora_horizon
from pora.transform import to_av_frame

RTOL = 1e-12
ATOL = 1e-14

rng = random.Random(20260814)


def random_foes(n):
    return [
        FoeState(
            x=rng.uniform(-40, 40),
            y=rng.uniform(-40, 40),
            vx=rng.uniform(-15, 15),
            vy=rng.uniform(-15, 15),
            dims=VehicleDims(rng.uniform(3.5, 16.0), rng.uniform(1.6, 2.6)),
        )
        for _ in range(n)
    ]


def as_tuples(foes):
    return [(f.x, f.y, f.vx, f.vy, f.dims.length, f.dims.width) for f in foes]


# ------------------------------------------------------------ CV Gaussian

@pytest.mark.parametrize("n_foes", [1, 3, 12])
@pytest.mark.parametrize("lead_time", [0.0, 1.5, 4.0])
def test_cv_gaussian_parity(n_foes, lead_time):
    foes = random_foes(n_foes)
    origin = (-30.0, -25.0)
    shape = (101, 121)
    ref = constant_velocity_gaussian(
        foes, lead_time, origin, shape, resolution=0.5)
    got = pora_rs.cv_gaussian_grid(
        as_tuples(foes), lead_time, origin, shape, 0.5)
    assert got.shape == ref.data.shape
    np.testing.assert_allclose(got, ref.data, rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------- one step

def make_box(speed):
    return SafetyBox(
        av=VehicleDims(4.8, 1.9),
        fleet_max_length=12.0,
        fleet_min_width=1.8,
        speed_ms=speed,
    )


@pytest.mark.parametrize("heading", [0.0, 0.7, -2.4])
@pytest.mark.parametrize("prev", [False, True])
def test_step_parity(heading, prev):
    foes = random_foes(5)
    origin = (-40.0, -40.0)
    grid = constant_velocity_gaussian(foes, 1.0, origin, (161, 161), 0.5)
    box = make_box(11.0)
    half_l = box.phi_length / 2.0 + 1.0
    half_w = box.phi_width / 2.0 + 1.0
    av = (3.2, -7.5)

    local = to_av_frame(grid, av[0], av[1], heading, half_l, half_w, 0.5)
    pc = collision_probability(local, box)
    if prev:
        prev_grid = to_av_frame(
            constant_velocity_gaussian(foes, 0.5, origin, (161, 161), 0.5),
            av[0], av[1], heading, half_l, half_w, 0.5)
        ref_adj = cox_adjust(pc, local.data, prev_grid.data, beta=1.0)
        prev_arg = prev_grid.data
    else:
        ref_adj = pc
        prev_arg = None

    po, raw_max, adj_max = pora_rs.pora_step(
        grid.data, origin, 0.5, av[0], av[1], heading, half_l, half_w, 0.5,
        (box.phi_length, box.phi_width, box.core_length, box.core_width),
        prev_po=prev_arg, beta=1.0)

    np.testing.assert_allclose(po, local.data, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(raw_max, pc.max(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(adj_max, ref_adj.max(), rtol=RTOL, atol=ATOL)


def test_step_window_dims_match_reference():
    """Window sizing must reproduce to_av_frame's rounding exactly."""
    foes = random_foes(2)
    grid = constant_velocity_gaussian(foes, 0.0, (-20.0, -20.0), (81, 81), 0.5)
    for half_l, half_w, res in [(7.3, 3.1, 0.5), (10.0, 4.999, 0.25),
                                (0.1, 0.1, 0.5), (16.84, 5.45, 0.5)]:
        local = to_av_frame(grid, 0.0, 0.0, 0.3, half_l, half_w, res)
        po, _, _ = pora_rs.pora_step(
            grid.data, (-20.0, -20.0), 0.5, 0.0, 0.0, 0.3, half_l, half_w,
            res, (10.0, 5.0, 6.0, 3.0))
        assert po.shape == local.data.shape


# ------------------------------------------------------- fused end-to-end

def python_pora_score(ego, foes, horizon_s=2.5, dt=0.5, beta=1.0,
                      resolution=0.5, extent=40.0):
    """Reference wiring, identical to bench.metrics.pora_score in
    risk-metric-bench (inlined here so this repo only depends on
    pora-replication)."""
    if not foes:
        return 0.0
    foe_states = [FoeState(f[0], f[1], f[2], f[3], VehicleDims(f[4], f[5]))
                  for f in foes]
    fleet_max_l = max(f[4] for f in foes)
    fleet_min_w = min(f[5] for f in foes)
    ex, ey, evx, evy, heading, speed, el, ew = ego

    n = int(round(2.0 * extent / resolution)) + 1
    origin = (ex - extent, ey - extent)
    steps = int(round(horizon_s / dt)) + 1
    grids, boxes = [], []
    for k in range(steps):
        t = k * dt
        g = constant_velocity_gaussian(
            foe_states, t, origin, (n, n), resolution)
        box = SafetyBox(av=VehicleDims(el, ew), fleet_max_length=fleet_max_l,
                        fleet_min_width=fleet_min_w, speed_ms=speed)
        grids.append(to_av_frame(
            g, ex + evx * t, ey + evy * t, heading,
            box.phi_length / 2.0 + 1.0, box.phi_width / 2.0 + 1.0,
            resolution))
        boxes.append(box)
    return pora_horizon(grids, boxes, beta=beta).scalar


@pytest.mark.parametrize("seed", range(8))
def test_pora_score_parity(seed):
    r = random.Random(seed)
    ego = (r.uniform(-5, 5), r.uniform(-5, 5),
           r.uniform(-14, 14), r.uniform(-14, 14),
           r.uniform(-math.pi, math.pi), r.uniform(0, 20), 4.8, 1.9)
    foes = [(r.uniform(-30, 30), r.uniform(-30, 30),
             r.uniform(-15, 15), r.uniform(-15, 15),
             r.uniform(3.5, 14.0), r.uniform(1.6, 2.6))
            for _ in range(r.randint(1, 10))]
    ref = python_pora_score(ego, foes)
    got = pora_rs.pora_score(ego, foes)
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_pora_score_no_foes():
    assert pora_rs.pora_score((0, 0, 0, 0, 0.0, 10.0, 4.8, 1.9), []) == 0.0


def test_pora_score_nontrivial():
    """Head-on foe must produce materially more risk than a distant one."""
    ego = (0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 4.8, 1.9)
    close = pora_rs.pora_score(ego, [(25.0, 0.0, -10.0, 0.0, 4.5, 1.8)])
    far = pora_rs.pora_score(ego, [(0.0, 35.0, 0.0, 10.0, 4.5, 1.8)])
    assert close > 0.05, "head-on approach must register risk"
    assert far == 0.0, "traffic that never enters the safety box scores 0"
