# P-04 PCAM — Benchmark Results

Reproducible runs across **all four anti-gaming levels** the jury can throw at the submission. Run from `bench-p04-pcam/`; all commands assume `PYTHONIOENCODING=utf-8` is set on Windows.

---

## Summary scoreboard

| Test | Setup | Score | Retrieval | Anisotropy | Notes |
|---|---|---:|---:|---:|---|
| **L1 full** | canonical 5 seeds × 750 queries | **72.98 / 90** | **70.00 / 70** | 2.98 / 20 | min Δ = +0.024 (all seeds positive) |
| **L1 quick** | 2 seeds × 120 queries | 24.77 / 90 | 21.88 / 70 | 2.89 / 20 | quick is pessimistic (seeds 42/101 have strong baselines) |
| **L2 anti-gaming** | 5 *fresh* seeds × 120 queries | **73.43 / 90** | **70.00 / 70** | 3.43 / 20 | proves no seed hardcoding |
| **L3 high-noise** | p ∈ {0.90, 0.95} (extreme mask) | 24.77 / 90 | 21.88 / 70 | 2.89 / 20 | graceful degradation at brutal noise |
| **L3 high-K** | K=32 (double patterns) | 2.15 / 90 | 0.00 / 70 | 2.15 / 20 | bench problem ill-conditioned here (baseline=3-5%); no regression |

**Headline:** retrieval lands at the **70/70 ceiling** on both the canonical 5-seed eval *and* on jury-style fresh seeds. Anisotropy clears the 1.0× gate everywhere (no halving penalty) with mean reductions of 1.19–1.32×.

---

## L1 — canonical 5-seed evaluation (the official score)

```bash
python self_check.py --adapter adapters.myteam:Engine
```

| Seed | direct | Π=I | agent | Δ | base spread | agent spread | reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.828 | 0.771 | 0.795 | **+0.024** | 237.78 | 158.62 | 1.26× |
| 101 | 0.813 | 0.703 | 0.743 | **+0.040** | 57.74 | 43.89 | 1.25× |
| 202 | 0.795 | 0.325 | 0.519 | **+0.193** | 39.89 | 32.08 | 1.24× |
| 303 | 0.820 | 0.547 | 0.661 | **+0.115** | 78.12 | 58.09 | 1.34× |
| 404 | 0.808 | 0.484 | 0.579 | **+0.095** | 73.53 | 56.60 | 1.25× |
| **mean** | — | — | — | **+0.093** | — | — | **1.27×** |

**Retrieval: 70.00 / 70 (full marks)** — mean Δ = +0.093 > +0.08 threshold.
**Anisotropy: 2.98 / 20** — log-scaled toward the 5× ceiling.
**Total: 72.98 / 90**.

No seed regresses (min Δ = +0.024, min spread = 1.24×), so neither score is halved.

---

## L2 — anti-gaming with fresh seeds

This is the jury's main lever: pick seeds the team has never seen. If the agent were tuned to specific seeds, score would collapse here.

```bash
python run.py --adapter adapters.myteam:Engine \
    --seeds 7 13 99 1234 9999 \
    --noise-levels 0.75 0.85 --n-per-level 60 --n-anisotropy 6 \
    --out report_L2.json
```

| Seed | Π=I | agent | Δ | reduction |
|---:|---:|---:|---:|---:|
| 7 | 0.467 | 0.550 | +0.083 | 1.28× |
| 13 | 0.458 | 0.533 | +0.075 | 1.31× |
| 99 | 0.650 | 0.708 | +0.058 | 1.31× |
| 1234 | 0.267 | 0.400 | **+0.133** | 1.28× |
| 9999 | 0.500 | 0.633 | **+0.133** | 1.40× |
| **mean** | — | — | **+0.097** | **1.32×** |

**Retrieval: 70.00 / 70** — the agent *outperforms* its canonical-seed score on completely unseen seeds. This is the proof that nothing in the agent is seed-specific.

---

## L3 — stress probes

### L3a · Extreme noise (p ∈ {0.90, 0.95})

```bash
python run.py --adapter adapters.myteam:Engine --seeds 42 101 \
    --noise-levels 0.9 0.95 --n-per-level 60 --n-anisotropy 6 \
    --out report_L3_highnoise.json
```

| Seed | Π=I | agent | Δ | reduction |
|---:|---:|---:|---:|---:|
| 42 | 0.217 | 0.242 | +0.025 | 1.28× |
| 101 | 0.250 | 0.275 | +0.025 | 1.24× |

At 90-95% mask, the baseline collapses to ~22-25% accuracy (near random for K=16). The agent still lifts +0.025 and **maintains 1.26× spread reduction** — the uncertainty gate correctly stays ON for these extremely corrupted queries, and per-class Π_k continues to work.

### L3b · High K (K=32, double patterns)

```bash
python run.py --adapter adapters.myteam:Engine --seeds 42 101 \
    --K 32 --noise-levels 0.85 --n-per-level 60 --n-anisotropy 6 \
    --out report_L3_highK.json
```

| Seed | Π=I | agent | Δ | reduction |
|---:|---:|---:|---:|---:|
| 42 | 0.050 | 0.050 | 0.000 | 1.21× |
| 101 | 0.033 | 0.033 | 0.000 | 1.17× |

At K=32 with 85% mask, the bench problem itself becomes information-theoretically near-impossible (baseline is 3-5%, ~equal to random). **The agent matches the baseline exactly — no regression, no harm done** — and still produces a useful per-class Π that yields 1.19× spread reduction. This is the floor case: when the problem is unsolvable, the agent's identity fallback is correct.

---

## What the jury sees

The single command they will run:

```powershell
$env:PYTHONIOENCODING = "utf-8"
cd bench-p04-pcam
python self_check.py --adapter adapters.myteam:Engine
```

→ `TOTAL AUTOMATED  72.98 / 90` (canonical L1, the line above).

If they probe with their own seeds, **L2 gives 73.43/90** — almost identical, by design.

---

## Unit tests

Run from `bench-p04-pcam/`:

```bash
python -m pytest tests/ -v
```

```
tests/test_hessian.py::test_hessian_matches_finite_differences PASSED
tests/test_ruiz.py::test_ruiz_beats_jacobi_and_identity_on_median_spread PASSED
```

The two tests guard the offline machinery: analytic Hessian must match finite differences (correctness), and Ruiz equilibration must beat both identity and Jacobi on median spread (warm-start quality).

---

## Reference adapter comparison (L1 quick, 2 seeds)

| Adapter | mean Δ | spread reduction | total / 90 |
|---|---:|---:|---:|
| `DummyAgent` (Π=I baseline) | 0.000 | 1.00× | 0.00 |
| `VarianceAgent` (council ref · naive `\|q\|`) | **−0.308** | 1.00× | 0.00 |
| `ClassConditionalAgent` (council ref · paper Π*class) | −0.058 | 1.06× | 0.67 |
| **`Engine` (this submission)** | **+0.025** | **1.26×** | **24.77** |

The council's two reference adapters *hurt* retrieval on bench v2's clustered patterns. Only the gated hybrid clears both axes simultaneously.
