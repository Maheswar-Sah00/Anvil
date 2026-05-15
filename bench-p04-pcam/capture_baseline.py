"""
Capture P-04 identity baseline metrics per seed.

Run from bench-p04-pcam/:

    python capture_baseline.py

Writes docs/BASELINE.md at the repo root and prints the same markdown table.
"""
from __future__ import annotations

from pathlib import Path

from harness import run_one_seed
from self_check import agent_factory_from_spec

SEEDS = [0, 1, 2, 3, 42]
ADAPTER_SPEC = "adapters.myteam:Engine"
K = 16
N = 64
NOISE_LEVELS = [0.5, 0.7, 0.8]
N_PER_LEVEL = 250
N_ANISO = 16


def collect_rows() -> list[dict[str, float | int]]:
    factory = agent_factory_from_spec(ADAPTER_SPEC)
    rows: list[dict[str, float | int]] = []
    for seed in SEEDS:
        report = run_one_seed(
            factory,
            seed=seed,
            K=K,
            N=N,
            noise_levels=NOISE_LEVELS,
            n_per_level=N_PER_LEVEL,
            n_aniso=N_ANISO,
        )
        rows.append({
            "seed": report.seed,
            "test1_accuracy": report.agent_accuracy,
            "test2_spread_reduction": report.spread_reduction,
            "test1_delta_vs_baseline": report.delta,
        })
    return rows


def format_markdown_table(rows: list[dict[str, float | int]]) -> str:
    header = (
        "# P-04 identity baseline (adapters.myteam:Engine)\n"
        "\n"
        f"Harness: K={K}, N={N}, noise_levels={NOISE_LEVELS}, "
        f"n_per_level={N_PER_LEVEL}, n_aniso={N_ANISO}.\n"
        "Baseline comparator: DummyAgent (Pi=I). Engine also returns Pi=1; "
        "delta should be 0.\n"
        "\n"
        "| seed | test1_accuracy | test2_spread_reduction | "
        "test1_delta_vs_baseline |\n"
        "|------|----------------|-------------------------|"
        "-------------------------|\n"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            f"| {r['seed']} "
            f"| {r['test1_accuracy']:.4f} "
            f"| {r['test2_spread_reduction']:.2f} "
            f"| {r['test1_delta_vs_baseline']:+.4f} |"
        )
    return header + "\n".join(body_lines) + "\n"


def main() -> int:
    rows = collect_rows()
    table = format_markdown_table(rows)

    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "docs" / "BASELINE.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")

    print(table)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
