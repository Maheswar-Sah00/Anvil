# Anvil P-04 · Precision-Controlled Associative Memory

Team submission for the MetaCognition track (P-04). Adapter: `bench-p04-pcam/adapters/myteam.py` (`Engine`). Other Anvil benches live under `bench-p01-crdt/`, `bench-p02-context/`; see `index.html` for the full dashboard.

## Problem statement

The harness stores `K` patterns and, for each corrupted query `q`, asks your agent for a length-`N` precision vector `pi`. Frozen PCAM dynamics then integrate from `a0 = q` with element-wise scaled gradients (`update = -pi * grad E`) until convergence; retrieval succeeds if the final state classifies to the true pattern. The bench scores you against `DummyAgent` (always `pi = 1`) on Test 1 (retrieval accuracy and per-seed delta) and Test 2 (anisotropy spread reduction at equilibria). Official scoring halves retrieval or anisotropy points if any seed has `delta < 0` or spread reduction `<= 1.0` versus the identity baseline (see `docs/BENCH_NOTES.md`).

## Architecture

Offline, for each stored class `k` we compute the analytic Hessian at `X[k]`, derive a Ruiz-equilibrated precision template `pi_hess[k]` (kept only if it lowers eigenvalue spread; otherwise a Jacobi-style `1/|diag(H)|` fallback), and a class-conditional variance template `pi_class[k]`. These mix by geometric mean: `pi_combined[k] ∝ pi_hess[k]^hw · pi_class[k]^(1-hw)` with `hessian_weight = hw` (default 0.5). At query time, softmax weights `w` over classes from similarities `X @ q` produce a soft posterior `pi = exp(w @ log pi_combined)`, then confidence gating `pi ← pi^(w_max^confidence_exp)` (constructor default `confidence_exp = 1.0`; sweep-best 0.5 in `docs/SWEEP_RESULTS.md`) shrinks toward identity when the posterior is flat; values are clipped to `[0.1, 10]` and mean-normalized before dynamics. This hybrid design couples Hessian-aware preconditioning (Test 2) with class structure and conservative query-time modulation (Test 1).

## Results

From `docs/STRESS_RESULTS.md` (adapter `adapters.myteam:Engine`, fast mode).

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

Hyperparameter sweep rank 1 (fast eval, seeds 0–4): `cls_beta_mult=1.0`, `confidence_exp=0.5`, `hessian_weight=0.5` — mean delta `+0.0027`, min delta `+0.0000` (`docs/SWEEP_RESULTS.md`).

## How to run

```bash
# deps (bench is NumPy-only; notebook/plots need matplotlib, scipy, sklearn)
cd bench-p04-pcam
pip install -r requirements.txt
pip install matplotlib scipy scikit-learn jupyter

# official harness self-check (quick, then full)
python self_check.py --adapter adapters.myteam:Engine --quick
python self_check.py --adapter adapters.myteam:Engine

# unit tests (repo root)
cd ..
python -m pytest tests/test_hessian.py tests/test_ruiz.py

# reproduce reported stress / sweep tables
python scripts/run_stress.py --adapter adapters.myteam:Engine --fast
python scripts/run_sweep.py --fast

# hackathon Q&A figures
cd notebooks
jupyter nbconvert --execute hackathon_qa_plots.ipynb --to notebook --inplace
```

Optional: `cd bench-p04-pcam && python capture_baseline.py` writes `docs/BASELINE.md`.

## File map

| Path | Role |
|------|------|
| `bench-p04-pcam/adapters/myteam.py` | Submission `Engine` |
| `bench-p04-pcam/` | Frozen PCAM harness (`self_check.py`, `harness.py`, `pcam_model.py`) |
| `docs/BENCH_NOTES.md` | Harness Q&A reference |
| `docs/STRESS_RESULTS.md`, `docs/SWEEP_RESULTS.md` | Eval artifacts |
| `docs/writeup.md` | Technical writeup |
| `scripts/run_stress.py`, `scripts/run_sweep.py` | Table generators |
| `tests/test_hessian.py`, `tests/test_ruiz.py` | Analytic Hessian + Ruiz tests |
| `notebooks/hackathon_qa_plots.ipynb`, `notebooks/figures/` | Plots for Q&A |
