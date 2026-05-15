"""
P-04 self-check.

    python self_check.py --adapter adapters.dummy:DummyAgent --quick
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from typing import Any, Callable

import numpy as np

from harness import run_multi


def agent_factory_from_spec(spec: str) -> Callable[[np.ndarray, dict[str, Any]], Any]:
    module_name, class_name = spec.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    def factory(X: np.ndarray, params: dict[str, Any]):
        return cls(X, params)
    return factory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P-04 self-check")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="Smaller K and fewer seeds — fast iteration.")
    args = ap.parse_args(argv)

    if args.quick:
        seeds = [42, 101]
        K, N = 16, 64
        noise_levels = [0.7, 0.8]
        n_per_level = 50
        n_aniso = 5
    else:
        seeds = [42, 101, 202, 303, 404]
        K, N = 16, 64
        noise_levels = [0.5, 0.7, 0.8]
        n_per_level = 250
        n_aniso = 16

    factory = agent_factory_from_spec(args.adapter)
    t0 = time.monotonic()
    report = run_multi(
        agent_factory=factory,
        seeds=seeds,
        K=K, N=N,
        noise_levels=noise_levels,
        n_per_level=n_per_level,
        n_aniso=n_aniso,
    )
    total_ms = (time.monotonic() - t0) * 1000.0

    agg = report["aggregated"]
    sc = report["score"]

    print()
    print("ANVIL · P-04 · PCAM Precision Agent — Self-Check")
    print("=" * 62)
    print(f"  total wall time          {total_ms:>10.1f} ms")
    print(f"  seeds                    {agg['n_seeds']:>10d}")
    print(f"  stored patterns (K)      {K:>10d}")
    print(f"  state dim (N)            {N:>10d}")
    print(f"  noise levels             {noise_levels}")
    print()
    print("  PER-SEED  ─ retrieval ─       ── anisotropy ──")
    print("  seed      Π=I      agent  Δ     base    agent  ratio")
    print("  " + "-" * 58)
    for r in report["per_seed"]:
        print(f"  {r['seed']:>4}     {r['baseline_accuracy']:.3f}    "
              f"{r['agent_accuracy']:.3f}  {r['delta']:+.3f}  "
              f"{r['spread_baseline']:>6.2f}  {r['spread_agent']:>6.2f}  "
              f"{r['spread_reduction']:>5.2f}×")
    print()
    print("  AGGREGATED                       VALUE")
    print("  " + "-" * 58)
    print(f"  mean Δ accuracy (over seeds)    {agg['mean_delta']:+.3f}")
    print(f"  min  Δ accuracy (worst seed)    {agg['min_delta']:+.3f}")
    print(f"  mean spread reduction           {agg['mean_spread']:>6.2f}×")
    print(f"  min  spread reduction           {agg['min_spread']:>6.2f}×")
    print()
    print("  SCORE (automated, max 90)         POINTS")
    print("  " + "-" * 58)
    print(f"  retrieval     (max 70)            {sc['retrieval_pts']:>6.2f}")
    print(f"  anisotropy    (max 20)            {sc['anisotropy_pts']:>6.2f}")
    print(f"  code quality  (max 10)            (manual)")
    print(f"  TOTAL AUTOMATED                   {sc['total_automated']:>6.2f}  / 90")
    print()

    delta = agg["mean_delta"]
    spread = agg["mean_spread"]
    min_d = agg["min_delta"]

    if delta <= 0:
        print("  ⚠  Mean Δ ≤ 0 — your agent does not beat Π=I on average. "
              "Zero on retrieval.")
    elif delta < 0.02:
        print("  ▸  Mean Δ is small. The agent is helping a bit but not "
              "principled. Aim for Δ ≥ 0.05.")
    else:
        print("  ✓  Solid retrieval gain on average.")

    if min_d < 0:
        print("  ⚠  Min Δ < 0 — your agent regresses on at least one seed. "
              "Retrieval score halved.")
    if spread < 2.0:
        print("  ▸  Spread reduction near baseline. Try Hessian-aware design "
              "(see README hints).")
    elif spread < 10.0:
        print("  ✓  Anisotropy improving — log-scaled toward 10×.")
    else:
        print("  ✓✓ Excellent anisotropy reduction.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
