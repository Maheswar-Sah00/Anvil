"""One-shot runner for the full P-04 PCAM benchmark suite.

Runs every anti-gaming level the jury might exercise:
    L1 full        canonical 5-seed evaluation       (the official score)
    L2 fresh seeds 5 jury-style unseen seeds         (anti-gaming probe)
    L3 high-noise  brutal mask fractions             (stress)
    L3 high-K      double the stored patterns        (stress)

Prints one formatted scoreboard at the end. JSON reports are saved into
bench-p04-pcam/ for later inspection.

Usage:
    python scripts/run_benchmarks.py [--quick]

Options:
    --quick   smaller config for fast iteration (~3 min total).
              omit for the canonical settings (~10-12 min total).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

# Make the bench importable from the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "bench-p04-pcam")
ADAPTERS = os.path.join(BENCH, "adapters")
for p in (BENCH, ADAPTERS):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness import run_multi                                       # noqa: E402
from adapters.myteam import Engine                                  # noqa: E402

# Force UTF-8 stdout on Windows so the box-drawing characters render.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _factory(X, params):
    return Engine(X, params)


def build_suite(quick: bool) -> list[dict[str, Any]]:
    """Each entry is one benchmark configuration."""
    if quick:
        n_per_level, n_aniso = 60, 6
        l1_n_per_level, l1_seeds = 60, [42, 101]
    else:
        n_per_level, n_aniso = 120, 6
        l1_n_per_level, l1_seeds = 250, [42, 101, 202, 303, 404]

    return [
        {
            "name":          "L1 canonical",
            "level":         "L1",
            "description":   ("canonical 5-seed eval" if not quick
                              else "canonical 2-seed quick eval"),
            "out":           "report_L1.json",
            "kwargs": dict(
                seeds=l1_seeds, K=16, N=64,
                noise_levels=[0.6, 0.75, 0.85] if not quick else [0.75, 0.85],
                n_per_level=l1_n_per_level, n_aniso=16 if not quick else 6,
            ),
        },
        {
            "name":          "L2 fresh seeds",
            "level":         "L2",
            "description":   "5 jury-style unseen seeds (anti-gaming probe)",
            "out":           "report_L2.json",
            "kwargs": dict(
                seeds=[7, 13, 99, 1234, 9999], K=16, N=64,
                noise_levels=[0.75, 0.85],
                n_per_level=n_per_level, n_aniso=n_aniso,
            ),
        },
        {
            "name":          "L3 high-noise",
            "level":         "L3",
            "description":   "extreme mask fractions  (p ∈ {0.90, 0.95})",
            "out":           "report_L3_highnoise.json",
            "kwargs": dict(
                seeds=[42, 101], K=16, N=64,
                noise_levels=[0.90, 0.95],
                n_per_level=n_per_level, n_aniso=n_aniso,
            ),
        },
        {
            "name":          "L3 high-K",
            "level":         "L3",
            "description":   "double stored patterns  (K=32)",
            "out":           "report_L3_highK.json",
            "kwargs": dict(
                seeds=[42, 101], K=32, N=64,
                noise_levels=[0.85],
                n_per_level=n_per_level, n_aniso=n_aniso,
            ),
        },
    ]


def run_one(entry: dict[str, Any]) -> dict[str, Any]:
    """Run a single config and write its JSON report."""
    t0 = time.monotonic()
    print(f"\n→ {entry['name']:<18}  {entry['description']}", flush=True)
    report = run_multi(agent_factory=_factory, **entry["kwargs"])
    dur = time.monotonic() - t0
    out_path = os.path.join(BENCH, entry["out"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"   wall {dur:6.1f}s   →  {entry['out']}", flush=True)
    return {
        "name":        entry["name"],
        "level":       entry["level"],
        "description": entry["description"],
        "duration_s":  dur,
        "score":       report["score"]["total_automated"],
        "retrieval":   report["score"]["retrieval_pts"],
        "anisotropy":  report["score"]["anisotropy_pts"],
        "mean_delta":  report["aggregated"]["mean_delta"],
        "min_delta":   report["aggregated"]["min_delta"],
        "mean_red":    report["aggregated"]["mean_reduction"],
        "min_red":     report["aggregated"]["min_reduction"],
        "n_seeds":     report["aggregated"]["n_seeds"],
        "seeds":       report["aggregated"]["seeds"],
    }


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

W_TABLE = 88


def _hline(char: str = "─") -> str:
    return char * W_TABLE


def _grade(score: float) -> str:
    """Visual marker for a score band (out of 90)."""
    if score >= 70:   return "★★★★★"
    if score >= 50:   return "★★★★ "
    if score >= 25:   return "★★★  "
    if score >= 5:    return "★★   "
    return "★    "


def print_scoreboard(rows: list[dict[str, Any]]) -> None:
    print()
    print("╔" + "═" * (W_TABLE - 2) + "╗")
    print("║" + "  P-04 PCAM  ·  full benchmark scoreboard".ljust(W_TABLE - 2) + "║")
    print("║" + f"  Engine  (adapters.myteam:Engine)".ljust(W_TABLE - 2) + "║")
    print("╚" + "═" * (W_TABLE - 2) + "╝")
    print()
    print(_hline())
    print(f"  {'Benchmark':<18}  {'Score':>10}    "
          f"{'Retrieval':>11}  {'Anisotropy':>11}    {'mean Δ':>8}    {'spread':>7}")
    print(_hline())
    for r in rows:
        print(
            f"  {r['name']:<18}  "
            f"{r['score']:>5.2f} / 90    "
            f"{r['retrieval']:>5.2f} / 70  "
            f"{r['anisotropy']:>5.2f} / 20    "
            f"{r['mean_delta']:+.3f}    "
            f"{r['mean_red']:>5.2f}×    "
            f"{_grade(r['score'])}"
        )
    print(_hline())

    # Sub-table: per-seed deltas where available.
    print()
    print("  per-seed sanity check  (no halving = min Δ ≥ 0 and min spread ≥ 1×)")
    print("  " + "─" * (W_TABLE - 4))
    for r in rows:
        regress_delta = "REGRESSION" if r["min_delta"] < 0 else "ok"
        regress_red = "REGRESSION" if r["min_red"] <= 1.0 else "ok"
        print(
            f"  {r['name']:<18}  "
            f"seeds={r['n_seeds']:>2}   "
            f"min Δ = {r['min_delta']:+.3f} [{regress_delta:>10}]   "
            f"min spread = {r['min_red']:.2f}× [{regress_red:>10}]"
        )
    print()

    # Headline summary
    l1 = next((r for r in rows if r["name"] == "L1 canonical"), None)
    l2 = next((r for r in rows if r["name"] == "L2 fresh seeds"), None)
    if l1 and l2:
        delta_vs_canonical = l2["score"] - l1["score"]
        sign = "+" if delta_vs_canonical >= 0 else ""
        print("  HEADLINE")
        print("  " + "─" * (W_TABLE - 4))
        print(f"  L1 (canonical / what the jury runs) :   {l1['score']:>6.2f} / 90")
        print(f"  L2 (fresh seeds / anti-gaming probe):   {l2['score']:>6.2f} / 90   "
              f"(delta vs L1: {sign}{delta_vs_canonical:.2f})")
        if l2["score"] >= l1["score"] - 5.0:
            print("  → L2 close to L1: NO seed-specific behaviour detected.")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the full P-04 PCAM benchmark suite.")
    ap.add_argument("--quick", action="store_true",
                    help="Smaller config for fast iteration (~3 min total).")
    args = ap.parse_args(argv)

    print("=" * W_TABLE)
    mode = "quick" if args.quick else "canonical"
    print(f"  Running benchmark suite  ·  mode = {mode}")
    print("=" * W_TABLE)

    suite = build_suite(args.quick)
    rows: list[dict[str, Any]] = []
    t_total = time.monotonic()
    for entry in suite:
        rows.append(run_one(entry))

    print()
    print(f"all runs complete in {time.monotonic() - t_total:.1f}s total")

    print_scoreboard(rows)

    # Persist a single consolidated summary too
    summary_path = os.path.join(BENCH, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "results": rows}, f, indent=2, default=str)
    print(f"  consolidated summary saved →  {os.path.relpath(summary_path, ROOT)}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
