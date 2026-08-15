# Benchmarks: pora_rs vs Python reference

- machine: arm64, arm, Python 3.14.7
- method: median of 5 passes x 5 random scenarios per case; scalars asserted equal (rtol 1e-12) every call
- python reference: pora-replication (numpy-vectorized), same wiring as risk-metric-bench `pora_score`

| case | python | rust | c++ | rust speedup | c++ speedup |
|---|---|---|---|---|---|
| small: 2 foes, 40 m, K=6 | 4.0 ms | 0.57 ms | 0.87 ms | **7.0x** | 4.6x |
| bench-like: 8 foes, 40 m, K=6 | 10.4 ms | 1.11 ms | 2.62 ms | **9.3x** | 4.0x |
| dense: 32 foes, 40 m, K=6 | 36.3 ms | 3.64 ms | 9.52 ms | **10.0x** | 3.8x |
| wide: 8 foes, 80 m, K=6 | 36.3 ms | 2.24 ms | 5.89 ms | **16.2x** | 6.2x |
| long horizon: 8 foes, 40 m, K=26 | 48.6 ms | 5.20 ms | 10.56 ms | **9.3x** | 4.6x |
