# P-04 stress results

Adapter: `adapters.myteam:Engine`
Mode: fast (reduced queries)
Harness: noise_levels=[0.5, 0.7, 0.8], n_aniso_sample=20, beta=8.0

Penalty gate: `delta < 0` OR `spread_reduction <= 1.0` (vs DummyAgent Pi=I).
Test 2 spread: median of per-pattern spreads over 20 sampled attractors.

| config | seed | K | N | test1_acc | test1_delta | test2_spread | penalty | sec |
|--------|------|---|---|-----------|-------------|--------------|---------|-----|
| vanilla-0 | 0 | 16 | 64 | 0.7000 | -0.0067 | 0.54 | yes | 12.9 |
| vanilla-1 | 1 | 16 | 64 | 0.8200 | +0.0000 | 0.55 | yes | 55.3 |
| vanilla-2 | 2 | 16 | 64 | 0.8800 | +0.0067 | 0.55 | yes | 54.0 |
| vanilla-3 | 3 | 16 | 64 | 0.8533 | +0.0000 | 0.53 | yes | 42.6 |
| vanilla-4 | 4 | 16 | 64 | 0.8400 | +0.0000 | 0.53 | yes | 67.9 |
| vanilla-5 | 5 | 16 | 64 | 0.8267 | +0.0067 | 0.55 | yes | 36.9 |
| vanilla-6 | 6 | 16 | 64 | 0.7133 | -0.0600 | 0.55 | yes | 11.6 |
| vanilla-7 | 7 | 16 | 64 | 0.7533 | -0.0067 | 0.53 | yes | 12.2 |
| vanilla-8 | 8 | 16 | 64 | 0.8667 | -0.0067 | 0.53 | yes | 12.5 |
| vanilla-9 | 9 | 16 | 64 | 0.8067 | -0.0067 | 0.55 | yes | 12.8 |
| high-K | 0 | 400 | 64 | 0.0000 | +0.0000 | 1.22 | no | 7.5 |
| high-N | 0 | 200 | 128 | 0.0000 | +0.0000 | 1.05 | no | 8.0 |
| pca-mnist | 0 | 200 | 64 | 0.0000 | +0.0000 | 1.05 | no | 9.7 |
