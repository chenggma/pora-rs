# Benchmarks: pora_rs vs Python reference

- machine: arm64, arm, Python 3.14.7
- method: median of 5 passes x 5 random scenarios per case; scalars asserted equal (rtol 1e-12) every call
- python reference: pora-replication (numpy-vectorized), same wiring as risk-metric-bench `pora_score`

| case | python | rust | speedup | per call (rust) |
|---|---|---|---|---|
| small: 2 foes, 40 m, K=6 | 4.2 ms | 0.64 ms | **6.5x** | 0.64 ms |
| bench-like: 8 foes, 40 m, K=6 | 11.5 ms | 1.14 ms | **10.0x** | 1.14 ms |
| dense: 32 foes, 40 m, K=6 | 38.5 ms | 3.86 ms | **10.0x** | 3.86 ms |
| wide: 8 foes, 80 m, K=6 | 39.2 ms | 2.23 ms | **17.6x** | 2.23 ms |
| long horizon: 8 foes, 40 m, K=26 | 52.8 ms | 5.84 ms | **9.0x** | 5.84 ms |
