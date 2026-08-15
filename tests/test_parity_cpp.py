"""Differential tests for the C++ comparison implementation.

pora_cpp must reproduce both the Python reference and the Rust extension
to the same tolerance the Rust extension is held to. Skipped when the
module has not been built (pip install ./cpp).
"""

import random

import numpy as np
import pytest

pora_cpp = pytest.importorskip("pora_cpp")
import pora_rs  # noqa: E402

from test_parity import python_pora_score  # noqa: E402

RTOL = 1e-12
ATOL = 1e-14

rng = random.Random(20260815)


def random_scenario(n_foes):
    ego = (rng.uniform(-5, 5), rng.uniform(-5, 5),
           rng.uniform(-14, 14), rng.uniform(-14, 14),
           rng.uniform(-3.14, 3.14), rng.uniform(3, 20), 4.8, 1.9)
    foes = [(rng.uniform(-30, 30), rng.uniform(-30, 30),
             rng.uniform(-15, 15), rng.uniform(-15, 15),
             rng.uniform(3.5, 16.0), rng.uniform(1.6, 2.6))
            for _ in range(n_foes)]
    return ego, foes


@pytest.mark.parametrize("n_foes", [1, 2, 8, 32])
@pytest.mark.parametrize("trial", range(3))
def test_cpp_matches_python_reference(n_foes, trial):
    ego, foes = random_scenario(n_foes)
    got = pora_cpp.pora_score(ego, foes)
    want = python_pora_score(ego, foes)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_foes", [1, 8, 32])
@pytest.mark.parametrize("trial", range(3))
def test_cpp_matches_rust(n_foes, trial):
    ego, foes = random_scenario(n_foes)
    got = pora_cpp.pora_score(ego, foes)
    want = pora_rs.pora_score(ego, foes)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(horizon_s=1.0, dt=0.1),
        dict(beta=2.0),
        dict(resolution=0.25, extent=20.0),
        dict(sigma0=0.2, sigma_growth=1.0),
    ],
)
def test_cpp_matches_python_nondefault_params(kwargs):
    ego, foes = random_scenario(6)
    got = pora_cpp.pora_score(ego, foes, **kwargs)
    want = python_pora_score(ego, foes, **{
        k: v for k, v in kwargs.items()
        if k in ("horizon_s", "dt", "beta", "resolution", "extent")
    })
    if set(kwargs) <= {"horizon_s", "dt", "beta", "resolution", "extent"}:
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    else:
        # sigma params are not exposed by the reference wiring; hold the
        # C++ to the Rust extension instead.
        want_rs = pora_rs.pora_score(ego, foes, **kwargs)
        np.testing.assert_allclose(got, want_rs, rtol=RTOL, atol=ATOL)


def test_no_foes_is_zero():
    ego, _ = random_scenario(1)
    assert pora_cpp.pora_score(ego, []) == 0.0
