# Anvil P-04 · PCAM Precision Agent — Benchmark Harness

Reference benchmark for **P-04 · Precision-Controlled Associative Memory**.

Built on the PCAM paper (NeurIPS 2026 submission). The base PCAM model is provided to you, frozen — your job is to design an agent that picks a precision vector for each corrupted query so the system retrieves the correct stored pattern.

Pure Python · NumPy only · CPU only · multi-seed evaluation.

## Quickstart

```bash
cd bench-p04-pcam
pip install -r requirements.txt
python self_check.py --adapter adapters.dummy:DummyAgent --quick
```

The dummy (Π=I) is the floor every submission must beat.

## Layout

```
adapter.py        Adapter abstract base class — one method, predict_precision
pcam_model.py     Frozen PCAM dynamics: energy, gradient, Hessian, integrator
data.py           Synthetic pattern + corrupted-query generation (seedable)
checks.py         Per-seed retrieval accuracy + spread-reduction primitives
harness.py        Multi-seed orchestration + scoring (the anti-gaming core)
run.py            Full CLI
self_check.py     Condensed CLI for local iteration
adapters/
  dummy.py        Π=I baseline (no precision modulation)
```

## What you implement

Copy `adapters/dummy.py` to `adapters/myteam.py` and replace `predict_precision`:

```python
from adapter import Adapter
import numpy as np

class Engine(Adapter):
    def __init__(self, stored_patterns, model_params):
        """
        stored_patterns: (K, N) — patterns already stored
        model_params:    dict with R, eta, beta, dt, T_max, tol, pi_min, pi_max
        """
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]
        # one-time prep here — train a model, compute statistics, etc.

    def predict_precision(self, corrupted_query):
        """
        corrupted_query: (N,) noisy input
        returns:         (N,) positive precision values
        """
        return np.ones(self.N)
```

## Anti-gaming — three layers

Same defence model as P-01 and P-02. You see the first two; you do not see the third.

**L1 — Canonical seed.** A fixed seed (42) drives the patterns, graph, and queries. Passing L1 means your agent handles one known instance.

**L2 — Property-based multi-seed.** `--seeds` accepts ANY integers. For each seed, the harness builds a **fresh pattern matrix, fresh structured operator R, fresh query set** and **constructs a fresh adapter instance**. State and tuning cannot leak between seeds. A hardcoded agent passes L1 trivially and fails L2 immediately because every numeric value the agent was tuned against is regenerated.

```bash
python run.py --adapter adapters.myteam:Engine \
  --seeds 7 13 31 97 211 503 1009 --K 16 --N 64
```

**L3 — Held-out adversarial.** The council holds private seeds at higher K and N, plus the eventual PCA-MNIST swap (Section 6.6 of the paper). Used only at final evaluation. Not distributed.

### Score penalties

- **Any seed with Δ < 0** (agent regresses below Π=I on that seed) halves the retrieval score.
- **Any seed with spread reduction ≤ 1.0×** halves the anisotropy score.

So an agent that "wins on average" by being great on some seeds and terrible on others scores far below an agent that's consistently good.

## What gets judged

| Check                | Weight | How it scores                                                              |
|----------------------|--------|----------------------------------------------------------------------------|
| Retrieval Accuracy   | 70%    | Linear in mean Δ over Π=I (across seeds). Full at Δ ≥ 0.05. Min-seed gate. |
| Anisotropy Spread    | 20%    | Log-scaled mean spread reduction. Full at 10×. Min-seed gate.              |
| Code Quality         | 10%    | Manual — working code, README, design notes.                               |

## Metric interpretation

**Retrieval Δ accuracy** — how much better your agent is than Π=I, averaged across seeds:

| Δ range       | Meaning                                                                       |
|---------------|-------------------------------------------------------------------------------|
| Δ ≤ 0.00      | At or below baseline — zero on retrieval. Precision is not helping.           |
| 0.00 – 0.02   | Marginal — some signal, but the agent isn't reading corruption sharply.       |
| 0.02 – 0.05   | Solid — agent is principled. Scales linearly toward full marks.               |
| ≥ 0.05        | Full marks (70 pts). Reproducing the paper's class-conditional gain.          |

**Anisotropy spread reduction** — how much more uniform the convergence rates are under your precision:

| Factor        | Meaning                                                                       |
|---------------|-------------------------------------------------------------------------------|
| ≤ 1.0×        | Anti-aligned or identical to baseline — zero on this axis.                    |
| 1.0× – 2.0×   | Some isotropisation, not yet principled.                                      |
| 2.0× – 10.0×  | Reading local geometry. Log-scaled toward full marks at 10×.                  |
| ≥ 10.0×       | Full marks (20 pts). Approaching the paper's aligned construction (~30×).     |

**Common patterns** —

| Δ          | Spread     | What it signifies                                                       |
|------------|-----------|-------------------------------------------------------------------------|
| ≈ 0        | ≈ 1×       | Agent is effectively π = 1. No modulation in effect.                    |
| ≈ 0        | High       | Geometry is being shaped but it isn't helping retrieval — possibly pulling toward the wrong attractor. |
| Positive   | ≈ 1×       | Heuristic that helps retrieval but isn't grounded in the Hessian.       |
| Positive   | High       | You've cracked it — precision is both useful and principled.            |
| Negative   | Any        | Precision is hurting retrieval — likely a clip / normalisation bug.     |

## Design hints

- **Variance-based**: down-weight dimensions that look like noise in the query.
- **Class-conditional**: predict the class first (nearest stored pattern by cosine similarity), then set precision to match that class's typical signature.
- **Geometry-aware**: read `model.hessian(approx_attractor)` and pick precision values that isotropise the eigenvalues of `Π^(1/2) H Π^(1/2)` — this is the construction producing the paper's ~30× spread reduction (Theorem F3).
- **Neural**: train a small MLP to map corrupted queries to good precision vectors.

## Constraints

- The PCAM model is **frozen** — you do not modify `pcam_model.py`.
- Precision is **diagonal and positive**. The harness clips to `[0.1, 10.0]` and mean-normalises to 1 before applying.
- **One forward pass** per query: no iterative refinement after observing the dynamics.

## v0 notes

This is the public iteration bench. Synthetic patterns (twin-pair construction in `data.py`) plus combined mask + Gaussian corruption. The L3 evaluation swaps in PCA-MNIST with mask noise per Section 6.6 of the paper. Same harness, same interface, different data.

## Hardware

NumPy only. CPU only. No GPU required.
