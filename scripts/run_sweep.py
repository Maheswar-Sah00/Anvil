"""
Coarse hyperparameter sweep for adapters.myteam.Engine.

Grid: cls_beta_mult x confidence_exp x hessian_weight (36 configs)
Evaluated on seeds 0..4 with harness.run_one_seed.

Run from repo root:

    python scripts/run_sweep.py --fast
    python scripts/run_sweep.py --full

Writes docs/SWEEP_RESULTS.md (~30-90 min with --fast).
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench-p04-pcam"
sys.path.insert(0, str(BENCH))

from adapters.myteam import Engine  # noqa: E402
from harness import run_one_seed  # noqa: E402

CLS_BETA_MULTS = [0.5, 1.0, 2.0]
CONFIDENCE_EXPS = [0.5, 1.0, 2.0, 4.0]
HESSIAN_WEIGHTS = [0.3, 0.5, 0.7]
SEEDS = [0, 1, 2, 3, 4]
K, N = 16, 64
NOISE_LEVELS = [0.5, 0.7, 0.8]


@dataclass
class SweepRow:
    cls_beta_mult: float
    confidence_exp: float
    hessian_weight: float
    mean_delta: float
    min_delta: float
    mean_spread: float
    duration_s: float


def make_factory(cls_beta_mult: float,
                 confidence_exp: float,
                 hessian_weight: float) -> Callable[[np.ndarray, dict[str, Any]], Any]:
    def factory(X: np.ndarray, params: dict[str, Any]) -> Engine:
        return Engine(
            X,
            params,
            cls_beta_mult=cls_beta_mult,
            confidence_exp=confidence_exp,
            hessian_weight=hessian_weight,
        )
    return factory


def evaluate_config(cls_beta_mult: float,
                    confidence_exp: float,
                    hessian_weight: float,
                    n_per_level: int) -> SweepRow:
    factory = make_factory(cls_beta_mult, confidence_exp, hessian_weight)
    deltas: list[float] = []
    spreads: list[float] = []
    t0 = time.monotonic()

    for seed in SEEDS:
        report = run_one_seed(
            factory,
            seed=seed,
            K=K,
            N=N,
            noise_levels=NOISE_LEVELS,
            n_per_level=n_per_level,
            n_aniso=16,
        )
        deltas.append(report.delta)
        spreads.append(report.spread_reduction)

    return SweepRow(
        cls_beta_mult=cls_beta_mult,
        confidence_exp=confidence_exp,
        hessian_weight=hessian_weight,
        mean_delta=float(np.mean(deltas)),
        min_delta=float(np.min(deltas)),
        mean_spread=float(np.mean(spreads)),
        duration_s=round(time.monotonic() - t0, 1),
    )


def rank_rows(rows: list[SweepRow]) -> list[SweepRow]:
    return sorted(rows, key=lambda r: (r.min_delta, r.mean_delta), reverse=True)


def format_markdown(rows: list[SweepRow], ranked: list[SweepRow], fast: bool) -> str:
    mode = "fast (50 queries/level)" if fast else "full (250 queries/level)"
    lines = [
        "# P-04 hyperparameter sweep (Engine)",
        "",
        f"Grid: cls_beta_mult {CLS_BETA_MULTS} x confidence_exp {CONFIDENCE_EXPS} "
        f"x hessian_weight {HESSIAN_WEIGHTS} ({len(rows)} configs)",
        f"Seeds: {SEEDS}, K={K}, N={N}, noise_levels={NOISE_LEVELS}, mode={mode}",
        "",
        "Ranking: **min delta** (worst seed) descending, tiebreak **mean delta**.",
        "",
        "## Top 5 (by min delta, tiebreak mean delta)",
        "",
        "| rank | cls_beta_mult | confidence_exp | hessian_weight | mean_delta | "
        "min_delta | mean_spread | sec |",
        "|------|---------------|----------------|----------------|------------|"
        "-----------|-------------|-----|",
    ]
    for i, r in enumerate(ranked[:5], start=1):
        lines.append(
            f"| {i} | {r.cls_beta_mult} | {r.confidence_exp} | {r.hessian_weight} | "
            f"{r.mean_delta:+.4f} | {r.min_delta:+.4f} | {r.mean_spread:.2f} | "
            f"{r.duration_s:.1f} |"
        )

    lines.extend([
        "",
        "## All configurations",
        "",
        "| cls_beta_mult | confidence_exp | hessian_weight | mean_delta | min_delta | "
        "mean_spread | sec |",
        "|---------------|----------------|----------------|------------|-----------|"
        "-------------|-----|",
    ])
    for r in rows:
        lines.append(
            f"| {r.cls_beta_mult} | {r.confidence_exp} | {r.hessian_weight} | "
            f"{r.mean_delta:+.4f} | {r.min_delta:+.4f} | {r.mean_spread:.2f} | "
            f"{r.duration_s:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P-04 Engine hyperparameter sweep")
    ap.add_argument(
        "--full",
        action="store_true",
        help="250 queries per noise level (default: 50, fast).",
    )
    args = ap.parse_args(argv)
    fast = not args.full
    n_per_level = 50 if fast else 250

    grid = list(itertools.product(CLS_BETA_MULTS, CONFIDENCE_EXPS, HESSIAN_WEIGHTS))
    rows: list[SweepRow] = []
    n_total = len(grid)

    for i, (cbm, ce, hw) in enumerate(grid, start=1):
        print(
            f"[{i}/{n_total}] cls_beta_mult={cbm} confidence_exp={ce} "
            f"hessian_weight={hw} ...",
            flush=True,
        )
        row = evaluate_config(cbm, ce, hw, n_per_level)
        rows.append(row)
        print(
            f"  mean_delta={row.mean_delta:+.4f} min_delta={row.min_delta:+.4f} "
            f"mean_spread={row.mean_spread:.2f}x ({row.duration_s}s)",
            flush=True,
        )

    ranked = rank_rows(rows)
    md = format_markdown(rows, ranked, fast)

    print()
    print("## Top 5")
    for i, r in enumerate(ranked[:5], start=1):
        print(
            f"  {i}. cbm={r.cls_beta_mult} ce={r.confidence_exp} hw={r.hessian_weight} "
            f"min_delta={r.min_delta:+.4f} mean_delta={r.mean_delta:+.4f}"
        )

    out = REPO / "docs" / "SWEEP_RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
