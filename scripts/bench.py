"""Benchmark: pora_rs.pora_score vs the Python reference pipeline.

Method: median-of-repeats wall clock for the same randomized scenario
set at each size; parity of the returned scalars is asserted on every
call so the benchmark cannot silently drift from correctness. Run on an
otherwise idle machine.

    python scripts/bench.py --out docs/benchmarks.md
"""

from __future__ import annotations

import argparse
import platform
import random
import statistics
import time

import os
import sys

import numpy as np

import pora_rs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_parity import python_pora_score  # noqa: E402

CASES = [
    # (label, n_foes, extent_m, horizon_s, dt_s)
    ("small: 2 foes, 40 m, K=6", 2, 40.0, 2.5, 0.5),
    ("bench-like: 8 foes, 40 m, K=6", 8, 40.0, 2.5, 0.5),
    ("dense: 32 foes, 40 m, K=6", 32, 40.0, 2.5, 0.5),
    ("wide: 8 foes, 80 m, K=6", 8, 80.0, 2.5, 0.5),
    ("long horizon: 8 foes, 40 m, K=26", 8, 40.0, 2.5, 0.1),
]


def scenario(r: random.Random, n_foes: int):
    ego = (r.uniform(-5, 5), r.uniform(-5, 5),
           r.uniform(-14, 14), r.uniform(-14, 14),
           r.uniform(-3.14, 3.14), r.uniform(3, 20), 4.8, 1.9)
    foes = [(r.uniform(-30, 30), r.uniform(-30, 30),
             r.uniform(-15, 15), r.uniform(-15, 15),
             r.uniform(3.5, 14.0), r.uniform(1.6, 2.6))
            for _ in range(n_foes)]
    return ego, foes


def time_fn(fn, scenarios, repeats):
    """Median over `repeats` passes of summed wall time over `scenarios`."""
    passes = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = [fn(ego, foes) for ego, foes in scenarios]
        passes.append(time.perf_counter() - t0)
    return statistics.median(passes), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = [
        "# Benchmarks: pora_rs vs Python reference\n",
        f"- machine: {platform.machine()}, {platform.processor() or 'apple silicon'}, "
        f"Python {platform.python_version()}",
        f"- method: median of {args.repeats} passes x {args.scenarios} random "
        "scenarios per case; scalars asserted equal (rtol 1e-12) every call",
        "- python reference: pora-replication (numpy-vectorized), same "
        "wiring as risk-metric-bench `pora_score`\n",
        "| case | python | rust | speedup | per call (rust) |",
        "|---|---|---|---|---|",
    ]
    for label, n_foes, extent, horizon, dt in CASES:
        r = random.Random(hash(label) & 0xFFFF)
        scenarios = [scenario(r, n_foes) for _ in range(args.scenarios)]

        def py_fn(ego, foes, extent=extent, horizon=horizon, dt=dt):
            return python_pora_score(ego, foes, horizon_s=horizon, dt=dt,
                                     extent=extent)

        def rs_fn(ego, foes, extent=extent, horizon=horizon, dt=dt):
            return pora_rs.pora_score(ego, foes, horizon_s=horizon, dt=dt,
                                      extent=extent)

        t_py, out_py = time_fn(py_fn, scenarios, args.repeats)
        t_rs, out_rs = time_fn(rs_fn, scenarios, args.repeats)
        np.testing.assert_allclose(out_rs, out_py, rtol=1e-12, atol=1e-14)

        per_call_ms = 1000 * t_rs / len(scenarios)
        row = (f"| {label} | {t_py / len(scenarios) * 1000:.1f} ms "
               f"| {per_call_ms:.2f} ms | **{t_py / t_rs:.1f}x** "
               f"| {per_call_ms:.2f} ms |")
        lines.append(row)
        print(row)

    report = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
