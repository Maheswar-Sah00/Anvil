# Anvil P-04 · PCAM Precision Agent — Benchmark Harness

Reference benchmark for **P-04 · Precision-Controlled Associative Memory**, built on the PCAM paper (NeurIPS 2026 submission). The base PCAM model is provided to you, frozen — your job is to design an agent that picks a precision vector for each corrupted query so the system retrieves the correct stored pattern.

Pure Python · NumPy only · CPU only · multi-seed evaluation.

## Quickstart

```bash
cd bench-p04-pcam
pip install -r requirements.txt
python self_check.py --adapter adapters.dummy:DummyAgent --quick
```

The Π=I dummy is the floor every submission must beat.

## Layout

```
adapter.py                 Adapter abstract base class — one method, predict_precision
pcam_model.py              Frozen PCAM dynamics, energy, gradient, Hessian
data.py                    Seedable synthetic patterns + corruption
metrics.py                 Pure evaluation primitives — retrieval, anisotropy
harness.py                 Multi-seed orchestration + scoring (the anti-gaming core)
run.py                     Full CLI
self_check.py              Condensed CLI for local iteration
adapters/
  dummy.py                 Π=I baseline (the floor)
  variance.py              Reference · |query|-based precision (naive)
  class_conditional.py     Reference · paper's Π*class approximation
  myteam.py                Submission · hybrid uncertainty-gated agent
tests/
  test_hessian.py          Analytic Hessian vs finite differences
  test_ruiz.py             Ruiz equilibration vs identity/Jacobi
```

## What you implement

Copy `adapters/dummy.py` to `adapters/myteam.py`. Replace `predict_precision`:

```python
from adapter import Adapter
import numpy as np

class Engine(Adapter):
    def __init__(self, stored_patterns, model_params):
        """
        stored_patterns : (K, N) — patterns already stored in the system
        model_params    : dict with R, eta, beta, dt, T_max, tol, T_in, pi_min, pi_max
        """
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]
        # one-time prep — train a model, compute statistics, etc.

    def predict_precision(self, corrupted_query):
        """
        corrupted_query : (N,) noisy input
        returns         : (N,) positive precision values
        """
        return np.ones(self.N)   # baseline
```

The harness automatically clips your output to `[pi_min, pi_max]` and projects it onto the constraint set `{ π : mean(π) = 1 }` via fixed-point iteration before applying.

## Anti-gaming — three layers

**L1 — Canonical seed.** A fixed seed drives patterns, R, and queries. Passing L1 alone is necessary, not sufficient.

**L2 — Property-based multi-seed.** `--seeds` accepts ANY integers. For each seed the harness builds a **fresh pattern matrix, fresh R, fresh query set** and **constructs a fresh adapter instance**. No state leaks between seeds. A hardcoded agent passes L1 trivially and dies on L2 because every numeric value it was tuned against is regenerated.

**L3 — Held-out adversarial.** Council-only — private seeds at higher K and N, plus the eventual PCA-MNIST swap (Section 6.6 of the paper). Not distributed.

### Per-seed penalty gates

- **Any seed with Δ < 0** halves the retrieval score
- **Any seed with spread reduction ≤ 1.0×** halves the anisotropy score

A submission that wins on the canonical seed and regresses on a held-out seed cannot reach full marks.

## What gets judged

| Check               | Weight | Scoring rule                                                   |
|---------------------|--------|----------------------------------------------------------------|
| Retrieval Accuracy  | 70 pts | Linear in mean Δ over Π=I across seeds; full at Δ = 0.08      |
| Anisotropy Spread   | 20 pts | Log-scaled mean reduction; full at 5× reduction                |
| Code Quality        | 10 pts | Manual — working code, README, design notes                    |

**Scoring rationale.** The paper's Π*class headline gain over Π=I is ~2.5% on PCA-MNIST. We set full marks at Δ = 0.08 (about 3× the paper's headline) so the bar is real engineering ambition, not just paper-level reproduction. For anisotropy, the paper achieves ~30× with explicit alignment; we set full marks at 5× so disciplined Hessian-aware designs can reach it.

## Metric interpretation

### Retrieval Δ accuracy

| Mean Δ        | Significance                                                          |
|---------------|----------------------------------------------------------------------|
| Δ ≤ 0.00      | At or below baseline · zero on retrieval                              |
| 0.00 – 0.02   | Marginal · some signal, agent isn't reading the corruption sharply    |
| 0.02 – 0.05   | Solid · principled agent · scales linearly toward full marks          |
| 0.05 – 0.08   | Strong · approaching full marks                                       |
| ≥ 0.08        | Full marks (70 pts) · materially exceeds the paper's class-conditional gain |

### Anisotropy spread reduction

| Factor        | Significance                                                          |
|---------------|----------------------------------------------------------------------|
| ≤ 1.0×        | At baseline or worse · zero on anisotropy                            |
| 1.0× – 1.5×   | Marginal · partial credit · log-scaled                                |
| 1.5× – 3.0×   | Reading local geometry · log-scaled toward full marks                |
| 3.0× – 5.0×   | Strong Hessian awareness                                             |
| ≥ 5.0×        | Full marks (20 pts) · disciplined alignment                           |

## Reference scores

These are what the reference adapters score on default settings (synthetic v0, 5 seeds, K=16, N=64, noise [0.6, 0.75, 0.85]). Numbers are illustrative — use them to sanity-check your iteration.

| Agent                                            | Mean Δ   | Mean reduction | Total auto |
|--------------------------------------------------|----------|----------------|------------|
| `adapters.dummy:DummyAgent`                      | 0.000    | 1.00×          | 0.00 / 90 |
| `adapters.variance:VarianceAgent`                | negative | < 1.0×         | 0.00 / 90 |
| `adapters.class_conditional:ClassConditionalAgent` | small    | ≈ 1.0×         | 0.00–5 / 90 |

Both reference agents are intentionally naive — they show that *trivial* precision designs don't beat baseline. The bench rewards principled work.

## Reading your results

`self_check.py` prints a per-seed table + aggregated metrics + score block:

```
PER-SEED   ─ retrieval ─────────────       ── anisotropy ──
seed     direct  Π=I    agent    Δ          base   agent   reduction
----------------------------------------------------------------------
  42    0.725  0.692  0.770  +0.078 ✓     20.84    8.50   2.45×
 101    0.783  0.675  0.760  +0.085 ✓     34.41   12.40   2.77×

AGGREGATED                                  VALUE
mean Δ accuracy (over seeds)               +0.081
min  Δ accuracy (worst seed)               +0.078
mean spread reduction                        2.61×
min  spread reduction                        2.45×

SCORE (automated, max 90)                  POINTS
retrieval     (max 70)                      70.00
anisotropy    (max 20)                       8.46
TOTAL AUTOMATED                             78.46 / 90
```

The `✓ / ✗` flag next to each Δ indicates whether the agent's dynamics beat direct cosine classification on that seed. This is **diagnostic only** — it does not affect your score. On synthetic random patterns, direct classify is already near-optimal because the patterns are well-separated; the dynamics' value-add only shows up cleanly on structured data (the L3 PCA-MNIST evaluation).

## Design hints

- **Variance-based**: down-weight dimensions that look noisy in the query. Simple, fast, mild positive Δ at best.
- **Class-conditional**: predict the class first (Modern Hopfield one-shot), then set precision to match the class's typical magnitudes. Approximates the paper's Π*class.
- **Geometry-aware**: read `model.hessian(approx_equilibrium)` and pick precision values that isotropise the eigenvalues of `Π^(1/2) H Π^(1/2)` — the construction producing the paper's 30× spread reduction (Theorem F3).
- **Neural**: train a small MLP on (corrupted query, good precision) pairs you generate from the stored patterns.

## Constraints

- The PCAM model is **frozen** — you do not modify `pcam_model.py`.
- Precision is **diagonal and positive**. The harness projects onto `[0.1, 10.0]` with mean = 1 before applying. Pass anything that fits; the harness normalises.
- **One forward pass** per query — no iterative refinement after observing dynamics.

## v0 notes

The public iteration bench uses synthetic random patterns (twin-pair construction in `data.py`) plus combined mask + Gaussian corruption. Patterns are well-separated by design, so direct cosine classification is already strong (~70-80%); the dynamics' main job here is to gracefully degrade.

The L3 evaluation swaps in PCA-MNIST with the paper's mask noise (Section 6.6). On MNIST, patterns are structured and the dynamics' replay term does real work — the gap between direct and dynamics widens, and good precision designs add measurable value.

Same harness, same interface, different data. Your agent design should generalize across both.

## Hardware

NumPy only · CPU only · no GPU needed.
