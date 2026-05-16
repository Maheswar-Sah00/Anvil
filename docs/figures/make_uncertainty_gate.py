"""Render the uncertainty gate function — the architectural unlock.

Produces docs/figures/uncertainty_gate.png.

Usage:  python docs/figures/make_uncertainty_gate.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "uncertainty_gate.png")


def make(out_path: str = OUT_PATH) -> str:
    threshold = 0.92
    scale = 60.0

    w_max = np.linspace(0.0, 1.0, 400)
    u_gate = 1.0 / (1.0 + np.exp(scale * (w_max - threshold)))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)

    ax.plot(w_max, u_gate, lw=2.6, color="#dc2626",
            label="u_gate = σ(60·(0.92 − w_max))")

    # Corrupted region
    ax.axvspan(0.10, 0.90, ymin=0, ymax=1, color="#fecaca", alpha=0.35,
               label="corrupted query  (gate ≈ 1)")
    # Probe region
    ax.axvspan(0.94, 0.99, ymin=0, ymax=1, color="#bbf7d0", alpha=0.45,
               label="Test-2 probe  (gate ≈ 0)")

    ax.axvline(threshold, ls="--", color="#0f172a", alpha=0.5, lw=1.2)
    ax.text(threshold + 0.005, 0.97, "threshold = 0.92", rotation=90,
            va="top", fontsize=9, family="monospace", color="#0f172a")

    # Annotate the regimes
    ax.annotate("variance term FULLY ACTIVE\n(boost retrieval on masked dims)",
                xy=(0.45, 1.00), xytext=(0.45, 1.18),
                ha="center", fontsize=10, color="#7f1d1d",
                arrowprops=dict(arrowstyle="-|>", color="#7f1d1d"))
    ax.annotate("variance term OFF\n(preserve per-class spread reduction)",
                xy=(0.965, 0.02), xytext=(0.65, 0.30),
                ha="center", fontsize=10, color="#14532d",
                arrowprops=dict(arrowstyle="-|>", color="#14532d"))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.35)
    ax.set_xlabel("w_max  (classification confidence)", fontsize=11)
    ax.set_ylabel("u_gate", fontsize=11)
    ax.set_title("Uncertainty Gate — the architectural unlock\n"
                 "cleanly separates the Test-2 probe regime from corrupted queries",
                 fontsize=12, fontweight="bold", color="#0f172a")

    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = make()
    print(f"wrote {path}")
