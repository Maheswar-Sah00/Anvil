# P-04 identity baseline (adapters.myteam:Engine)

Harness: K=16, N=64, noise_levels=[0.5, 0.7, 0.8], n_per_level=250, n_aniso=16.
Baseline comparator: DummyAgent (Pi=I). Engine also returns Pi=1; delta should be 0.

| seed | test1_accuracy | test2_spread_reduction | test1_delta_vs_baseline |
|------|----------------|-------------------------|-------------------------|
| 0 | 0.7613 | 1.00 | +0.0000 |
| 1 | 0.8013 | 1.00 | +0.0000 |
| 2 | 0.8613 | 1.00 | +0.0000 |
| 3 | 0.8453 | 1.00 | +0.0000 |
| 42 | 0.8733 | 1.00 | +0.0000 |
