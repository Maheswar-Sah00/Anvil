"""Render the benchmark scoreboard across L1/L2/L3 + reference adapters.

Produces docs/figures/benchmarks.png.

Usage:  python docs/figures/make_benchmarks.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "benchmarks.png")


def make(out_path: str = OUT_PATH) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=140,
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # ─── (1) Anti-gaming scoreboard (L1, L2, L3) ───────────────────────
    setups = [
        "L1 quick\n(2 seeds × 120)",
        "L1 full\n(5 seeds × 750)",
        "L2 fresh seeds\n(5 unseen)",
        "L3 high-noise\n(p=0.90, 0.95)",
        "L3 high-K\n(K=32, p=0.85)",
    ]
    retrieval = [21.88, 70.00, 70.00, 21.88, 0.00]
    anisotropy = [2.89, 2.98, 3.43, 2.89, 2.15]
    colors = ["#94a3b8", "#16a34a", "#16a34a", "#f59e0b", "#dc2626"]

    x = np.arange(len(setups))
    width = 0.35

    ax1.bar(x - width/2, retrieval, width, color=colors, edgecolor="#0f172a",
            label="Retrieval (max 70)", lw=1.0)
    ax1.bar(x + width/2, anisotropy, width, color=colors, edgecolor="#0f172a",
            alpha=0.55, label="Anisotropy (max 20)", lw=1.0,
            hatch="//")

    # Annotate totals on top
    for i, (r, a) in enumerate(zip(retrieval, anisotropy)):
        total = r + a
        ax1.text(i, max(r, a) + 4, f"{total:.2f}/90",
                 ha="center", fontsize=10, fontweight="bold", color="#0f172a",
                 family="monospace")

    ax1.axhline(70, ls="--", color="#16a34a", alpha=0.45, lw=1.2)
    ax1.text(4.4, 71, "70 (retrieval ceiling)", fontsize=8,
             color="#16a34a", family="monospace", ha="right")

    ax1.set_xticks(x)
    ax1.set_xticklabels(setups, fontsize=9)
    ax1.set_ylim(0, 92)
    ax1.set_ylabel("points", fontsize=10)
    ax1.set_title("Engine — anti-gaming scoreboard\n"
                  "L1 official + L2 (fresh seeds) + L3 stress",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax1.grid(True, axis="y", alpha=0.25, ls=":")

    # ─── (2) Reference adapter comparison (L1 quick, 2 seeds) ──────────
    adapters = [
        "DummyAgent\n(Π=I baseline)",
        "VarianceAgent\n(council ref)",
        "ClassConditional\n(council ref)",
        "Engine\n(this submission)",
    ]
    totals = [0.00, 0.00, 0.67, 24.77]
    deltas = [0.000, -0.308, -0.058, 0.025]
    bar_colors = ["#94a3b8", "#dc2626", "#f59e0b", "#16a34a"]

    bars = ax2.barh(adapters, totals, color=bar_colors, edgecolor="#0f172a", lw=1.2)
    for b, t, d in zip(bars, totals, deltas):
        sign = "+" if d >= 0 else ""
        ax2.text(t + 0.6, b.get_y() + b.get_height()/2,
                 f"{t:.2f}/90    Δ = {sign}{d:.3f}",
                 va="center", fontsize=9.5, fontweight="bold",
                 family="monospace", color="#0f172a")

    ax2.set_xlim(0, 35)
    ax2.set_xlabel("total / 90", fontsize=10)
    ax2.set_title("Adapter comparison  ·  L1 quick (2 seeds)\n"
                  "the council's references HURT retrieval — Engine is the only positive",
                  fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(True, axis="x", alpha=0.25, ls=":")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = make()
    print(f"wrote {path}")
