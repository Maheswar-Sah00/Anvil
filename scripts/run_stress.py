"""
P-04 stress harness: vanilla seeds, high-K, high-N, and PCA-MNIST patterns.

Run from repo root:

    python scripts/run_stress.py
    python scripts/run_stress.py --adapter adapters.myteam:Engine --fast

Writes docs/STRESS_RESULTS.md and prints the markdown table.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench-p04-pcam"
sys.path.insert(0, str(BENCH))

from adapters.dummy import DummyAgent  # noqa: E402
from checks import per_pattern_spread, retrieval_accuracy  # noqa: E402
from data import make_patterns, make_test_queries  # noqa: E402
from harness import pack_params  # noqa: E402
from pcam_model import PCAMModel, build_default_R  # noqa: E402
from self_check import agent_factory_from_spec  # noqa: E402

DEFAULT_BETA = 8.0
DEFAULT_NOISE = [0.5, 0.7, 0.8]
N_ANISO_SAMPLE = 20


@dataclass
class StressRow:
    config: str
    seed: int
    K: int
    N: int
    test1_accuracy: float
    test1_delta: float
    test2_spread_reduction: float
    penalty_fired: bool
    duration_s: float


def _median_spread_reduction(model: PCAMModel,
                             agent,
                             dummy,
                             pattern_indices: list[int],
                             seed: int = 0) -> tuple[float, float, float]:
    """Median per-pattern spread for agent/baseline; return factor, med_base, med_agent."""
    rng = np.random.default_rng(seed)
    base_spreads: list[float] = []
    agent_spreads: list[float] = []

    for idx in pattern_indices:
        pattern = model.X[idx]
        probe = pattern + rng.standard_normal(model.N) * 0.05
        n = np.linalg.norm(probe)
        if n > 1e-12:
            probe = probe / n
        pi_a = agent.predict_precision(probe)
        pi_b = dummy.predict_precision(probe)
        s_a = per_pattern_spread(model, pi_a, pattern)
        s_b = per_pattern_spread(model, pi_b, pattern)
        if s_a is not None and s_b is not None:
            agent_spreads.append(s_a)
            base_spreads.append(s_b)

    if not agent_spreads:
        return 0.0, float("inf"), float("inf")

    med_b = float(np.median(base_spreads))
    med_a = float(np.median(agent_spreads))
    factor = med_b / med_a if med_a > 0 and np.isfinite(med_a) else 0.0
    return factor, med_b, med_a


def run_one_config(agent_factory: Callable[[np.ndarray, dict[str, Any]], Any],
                   config: str,
                   seed: int,
                   K: int,
                   N: int,
                   noise_levels: list[float],
                   n_per_level: int,
                   n_aniso: int = N_ANISO_SAMPLE,
                   stored_patterns: np.ndarray | None = None) -> StressRow:
    """End-to-end run mirroring harness.run_one_seed with optional custom X."""
    if stored_patterns is not None:
        X = np.asarray(stored_patterns, dtype=np.float64)
        if X.shape != (K, N):
            raise ValueError(f"stored_patterns shape {X.shape} != ({K}, {N})")
    else:
        X = make_patterns(K=K, N=N, seed=seed)

    R = build_default_R(N=N, seed=seed)
    model = PCAMModel(X, R, beta=DEFAULT_BETA)
    params = pack_params(model)

    agent = agent_factory(X, params)
    dummy = DummyAgent(X, params)
    queries, truths, _ = make_test_queries(X, noise_levels, n_per_level, seed=seed)

    t0 = time.monotonic()
    base_acc = retrieval_accuracy(model, dummy, queries, truths)
    agent_acc = retrieval_accuracy(model, agent, queries, truths)
    delta = float(agent_acc - base_acc)

    rng = np.random.default_rng(seed)
    indices = rng.choice(K, size=min(n_aniso, K), replace=False).tolist()
    spread_red, _, _ = _median_spread_reduction(model, agent, dummy, indices, seed=seed)

    penalty = (delta < 0) or (spread_red <= 1.0)
    dur = time.monotonic() - t0

    return StressRow(
        config=config,
        seed=seed,
        K=K,
        N=N,
        test1_accuracy=float(agent_acc),
        test1_delta=delta,
        test2_spread_reduction=spread_red,
        penalty_fired=penalty,
        duration_s=round(dur, 1),
    )


def load_mnist_patterns(n_digits: int = 200,
                        n_components: int = 64,
                        seed: int = 0) -> np.ndarray:
    """200 random MNIST digits projected to n_components via PCA (unit-norm rows)."""
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "PCA-MNIST stress case requires scikit-learn: pip install scikit-learn"
        ) from e

    cache = REPO / ".cache" / "openml"
    cache.mkdir(parents=True, exist_ok=True)
    data = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto",
        data_home=str(cache),
    )
    X_all = np.asarray(data.data, dtype=np.float64) / 255.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_all.shape[0], size=n_digits, replace=False)
    X_sub = X_all[idx]
    X_proj = PCA(n_components=n_components, random_state=seed).fit_transform(X_sub)
    norms = np.linalg.norm(X_proj, axis=1, keepdims=True)
    return X_proj / np.maximum(norms, 1e-12)


def build_configs(fast: bool) -> list[dict[str, Any]]:
    n_per = 50 if fast else 250
    n_per_heavy = 30 if fast else 100

    configs: list[dict[str, Any]] = []

    for s in range(10):
        configs.append({
            "config": f"vanilla-{s}",
            "seed": s,
            "K": 16,
            "N": 64,
            "n_per_level": n_per,
        })

    configs.append({
        "config": "high-K",
        "seed": 0,
        "K": 400,
        "N": 64,
        "n_per_level": n_per_heavy,
    })

    configs.append({
        "config": "high-N",
        "seed": 0,
        "K": 200,
        "N": 128,
        "n_per_level": n_per_heavy,
    })

    configs.append({
        "config": "pca-mnist",
        "seed": 0,
        "K": 200,
        "N": 64,
        "n_per_level": n_per_heavy,
        "stored_patterns": None,  # filled at run time
    })

    return configs


def format_markdown(rows: list[StressRow], adapter: str, fast: bool) -> str:
    lines = [
        "# P-04 stress results",
        "",
        f"Adapter: `{adapter}`",
        f"Mode: {'fast (reduced queries)' if fast else 'full'}",
        f"Harness: noise_levels={DEFAULT_NOISE}, n_aniso_sample={N_ANISO_SAMPLE}, "
        f"beta={DEFAULT_BETA}",
        "",
        "Penalty gate: `delta < 0` OR `spread_reduction <= 1.0` (vs DummyAgent Pi=I).",
        "Test 2 spread: median of per-pattern spreads over 20 sampled attractors.",
        "",
        "| config | seed | K | N | test1_acc | test1_delta | test2_spread | "
        "penalty | sec |",
        "|--------|------|---|---|-----------|-------------|--------------|"
        "---------|-----|",
    ]
    for r in rows:
        pen = "yes" if r.penalty_fired else "no"
        lines.append(
            f"| {r.config} | {r.seed} | {r.K} | {r.N} | "
            f"{r.test1_accuracy:.4f} | {r.test1_delta:+.4f} | "
            f"{r.test2_spread_reduction:.2f} | {pen} | {r.duration_s:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P-04 stress runner")
    ap.add_argument("--adapter", default="adapters.myteam:Engine")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Fewer queries per noise level (faster; high-K/N still slow at init).",
    )
    ap.add_argument("--skip-mnist", action="store_true", help="Skip PCA-MNIST case.")
    ap.add_argument(
        "--only",
        default=None,
        help="Run a single config label (e.g. pca-mnist, vanilla-0, high-K).",
    )
    args = ap.parse_args(argv)

    factory = agent_factory_from_spec(args.adapter)
    configs = build_configs(args.fast)
    if args.only:
        configs = [c for c in configs if c["config"] == args.only]
        if not configs:
            raise SystemExit(f"Unknown config label: {args.only!r}")
    rows: list[StressRow] = []

    for i, cfg in enumerate(configs):
        label = cfg["config"]
        print(f"[{i + 1}/{len(configs)}] {label} ...", flush=True)

        stored = cfg.get("stored_patterns")
        if label == "pca-mnist":
            if args.skip_mnist:
                print("  skipped (--skip-mnist)")
                continue
            try:
                stored = load_mnist_patterns(
                    n_digits=cfg["K"], n_components=cfg["N"], seed=cfg["seed"]
                )
            except ImportError as e:
                print(f"  TODO: {e}")
                continue

        row = run_one_config(
            factory,
            config=label,
            seed=int(cfg["seed"]),
            K=int(cfg["K"]),
            N=int(cfg["N"]),
            noise_levels=DEFAULT_NOISE,
            n_per_level=int(cfg["n_per_level"]),
            stored_patterns=stored,
        )
        rows.append(row)
        print(
            f"  acc={row.test1_accuracy:.3f} delta={row.test1_delta:+.3f} "
            f"spread={row.test2_spread_reduction:.2f}x penalty={row.penalty_fired}",
            flush=True,
        )

    table = format_markdown(rows, args.adapter, args.fast)
    out = REPO / "docs" / "STRESS_RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table, encoding="utf-8")
    print()
    print(table)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
