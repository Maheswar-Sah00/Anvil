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
                    help="Two seeds, smaller query count — fast iteration.")
    args = ap.parse_args(argv)

    if args.quick:
        seeds = [42, 101]
        K, N = 16, 64
        noise_levels = [0.75, 0.85]
        n_per_level = 60
        n_aniso = 6
    else:
        seeds = [42, 101, 202, 303, 404]
        K, N = 16, 64
        noise_levels = [0.6, 0.75, 0.85]
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
    print("=" * 72)
    print(f"  total wall time          {total_ms:>10.1f} ms")
    print(f"  seeds                    {agg['n_seeds']:>10d}")
    print(f"  stored patterns (K)      {K:>10d}")
    print(f"  state dim (N)            {N:>10d}")
    print(f"  noise levels             {noise_levels}")
    print()
    print("  PER-SEED   ─ retrieval ─────────────       ── anisotropy ──")
    print("  seed     direct  Π=I    agent    Δ          base   agent   reduction")
    print("  " + "-" * 70)
    for r in report["per_seed"]:
        flag = "✓" if r["dynamics_adds_value"] else "✗"
        print(f"  {r['seed']:>4}    {r['direct_classify_acc']:.3f}  "
              f"{r['baseline_acc']:.3f}  {r['agent_acc']:.3f}  "
              f"{r['delta']:+.3f} {flag}    "
              f"{r['baseline_spread']:>6.2f}  {r['agent_spread']:>6.2f}  "
              f"{r['spread_reduction']:>5.2f}×")
    print()
    print("  AGGREGATED                                  VALUE")
    print("  " + "-" * 70)
    print(f"  mean Δ accuracy (over seeds)               {agg['mean_delta']:+.3f}")
    print(f"  min  Δ accuracy (worst seed)               {agg['min_delta']:+.3f}")
    print(f"  mean spread reduction                      {agg['mean_reduction']:>6.2f}×")
    print(f"  min  spread reduction                      {agg['min_reduction']:>6.2f}×")
    print(f"  dynamics-adds-value pass rate              "
          f"{agg['dynamics_gate_pass_rate']:.0%}")
    print()
    print("  SCORE (automated, max 90)                  POINTS")
    print("  " + "-" * 70)
    print(f"  retrieval     (max 70)                     {sc['retrieval_pts']:>6.2f}")
    print(f"  anisotropy    (max 20)                     {sc['anisotropy_pts']:>6.2f}")
    print(f"  code quality  (max 10)                     (manual)")
    print(f"  TOTAL AUTOMATED                            {sc['total_automated']:>6.2f}  / 90")
    print()

    if sc["notes"]:
        print("  NOTES")
        for n in sc["notes"]:
            print(f"    · {n}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
