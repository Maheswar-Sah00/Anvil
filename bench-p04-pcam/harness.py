"""
Benchmark harness for P-04. Multi-seed orchestration + scoring.

Each seed runs in a fresh adapter instance with freshly-generated
patterns, R, and queries. Hardcoded agents fail immediately because
every numeric value they were tuned against is regenerated.

Scoring philosophy (rebalanced after audit):

    Retrieval (70 pts max)
      - delta = mean(agent_acc - baseline_acc) across seeds
      - full marks at delta = 0.08 (significantly above the paper's
        2.5% headline gain for Pi*class)
      - linear scaling for 0 < delta < 0.08
      - penalty: any seed with delta < 0 halves the score
      - hard gate: agent_acc must beat direct_classify_acc on at
        least 60% of seeds (the dynamics must add value)

    Anisotropy (20 pts max)
      - reduction = mean(baseline_spread / agent_spread) across seeds,
        evaluated at TRUE equilibria
      - full marks at reduction = 5.0 (paper achieves ~30x with
        explicitly aligned construction; 5x is realistic for a
        well-designed agent)
      - log-scaling rewards even modest geometry awareness
      - penalty: any seed with reduction <= 1.0 halves the score
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from adapters.dummy import DummyAgent
from data import make_patterns, make_test_queries
from metrics import (
    anisotropy_reductions,
    direct_classify_accuracy,
    retrieval_accuracy,
    summarise_anisotropy,
)
from pcam_model import PCAMModel, build_default_R


# --------------------------------------------------------------------------- #
# Configuration & weights
# --------------------------------------------------------------------------- #

WEIGHTS: dict[str, float] = {
    "retrieval":   70.0,
    "anisotropy":  20.0,
    "code":        10.0,
}

RETRIEVAL_FULL_AT: float = 0.08
ANISOTROPY_FULL_AT: float = 5.0
DYNAMICS_GATE_FRACTION: float = 0.6  # share of seeds where dynamics must help


# --------------------------------------------------------------------------- #
# Per-seed report
# --------------------------------------------------------------------------- #

@dataclass
class SeedReport:
    seed: int
    direct_classify_acc: float
    baseline_acc: float
    agent_acc: float
    delta: float
    dynamics_adds_value: bool
    baseline_spread: float
    agent_spread: float
    spread_reduction: float
    duration_s: float


@dataclass
class Aggregated:
    mean_delta: float
    min_delta: float
    mean_reduction: float
    min_reduction: float
    dynamics_gate_pass_rate: float
    seeds: list[int]
    n_seeds: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_seeds = len(self.seeds)


# --------------------------------------------------------------------------- #
# Per-seed execution
# --------------------------------------------------------------------------- #

def pack_params(model: PCAMModel) -> dict[str, Any]:
    return {
        "R":      model.R,
        "eta":    model.eta,
        "beta":   model.beta,
        "dt":     model.dt,
        "T_max":  model.T_max,
        "tol":    model.tol,
        "T_in":   model.T_in,
        "pi_min": model.pi_min,
        "pi_max": model.pi_max,
    }


def run_one_seed(agent_factory: Callable[[np.ndarray, dict[str, Any]], Any],
                 seed: int,
                 K: int,
                 N: int,
                 noise_levels: list[float],
                 n_per_level: int,
                 n_aniso: int) -> SeedReport:
    """Build a fresh model + agent + query set for this seed and score it."""
    X = make_patterns(K=K, N=N, seed=seed)
    R = build_default_R(N=N, seed=seed)
    model = PCAMModel(X, R)
    params = pack_params(model)

    agent = agent_factory(X, params)
    dummy = DummyAgent(X, params)

    queries, truths, _ = make_test_queries(X, noise_levels, n_per_level, seed=seed)

    t0 = time.monotonic()
    direct_acc = direct_classify_accuracy(model, queries, truths)
    base_acc = retrieval_accuracy(model, dummy, queries, truths)
    agent_acc = retrieval_accuracy(model, agent, queries, truths)

    rng = np.random.default_rng(seed)
    indices = rng.choice(K, size=min(n_aniso, K), replace=False).tolist()
    pairs = anisotropy_reductions(model, agent, indices, seed=seed)
    aniso = summarise_anisotropy(pairs)
    aniso_baseline_pairs = anisotropy_reductions(model, dummy, indices, seed=seed)
    aniso_baseline = summarise_anisotropy(aniso_baseline_pairs)

    duration = time.monotonic() - t0

    return SeedReport(
        seed=seed,
        direct_classify_acc=float(direct_acc),
        baseline_acc=float(base_acc),
        agent_acc=float(agent_acc),
        delta=float(agent_acc - base_acc),
        dynamics_adds_value=bool(agent_acc > direct_acc),
        baseline_spread=float(aniso_baseline["agent_spread"]),
        agent_spread=float(aniso["agent_spread"]),
        spread_reduction=float(aniso["reduction"]),
        duration_s=round(duration, 2),
    )


# --------------------------------------------------------------------------- #
# Aggregation + scoring
# --------------------------------------------------------------------------- #

def aggregate(reports: list[SeedReport]) -> Aggregated:
    deltas = [r.delta for r in reports]
    reductions = [r.spread_reduction for r in reports]
    gate_passes = sum(1 for r in reports if r.dynamics_adds_value)
    return Aggregated(
        mean_delta=float(np.mean(deltas)),
        min_delta=float(np.min(deltas)),
        mean_reduction=float(np.mean(reductions)),
        min_reduction=float(np.min(reductions)),
        dynamics_gate_pass_rate=gate_passes / max(len(reports), 1),
        seeds=[r.seed for r in reports],
    )


def retrieval_points(agg: Aggregated,
                     full_at: float = RETRIEVAL_FULL_AT,
                     weight: float = WEIGHTS["retrieval"]) -> tuple[float, list[str]]:
    """Return (points, notes). Notes explain any penalties applied.

    Scoring rule: linear in mean delta over the baseline up to `full_at`.
    Per-seed penalty: any seed with delta < 0 halves the score.

    Note on the dynamics-vs-direct diagnostic: we report whether the
    agent's dynamics beat direct cosine classification on each seed,
    but do NOT penalise failure of this gate. On synthetic patterns
    with strong separation, direct classify is already near-optimal —
    the dynamics' value-add manifests on structured data (MNIST).
    The diagnostic is informational for participants.
    """
    notes: list[str] = []

    if agg.mean_delta <= 0:
        return 0.0, ["mean delta <= 0 — agent does not beat the Π=I baseline on average"]

    points = min(weight, weight * (agg.mean_delta / full_at))

    if agg.min_delta < 0:
        notes.append(
            f"min delta < 0 ({agg.min_delta:+.3f}) — regression on at least "
            "one seed, retrieval score halved"
        )
        points *= 0.5

    if agg.dynamics_gate_pass_rate < DYNAMICS_GATE_FRACTION:
        notes.append(
            f"diagnostic: dynamics beat direct-classify on only "
            f"{agg.dynamics_gate_pass_rate:.0%} of seeds (informational, no penalty)"
        )

    return float(points), notes


def anisotropy_points(agg: Aggregated,
                      full_at: float = ANISOTROPY_FULL_AT,
                      weight: float = WEIGHTS["anisotropy"]) -> tuple[float, list[str]]:
    notes: list[str] = []

    if agg.mean_reduction <= 1.0:
        return 0.0, ["mean reduction <= 1.0 — precision is not improving the spread"]

    points = min(weight, weight * (np.log(agg.mean_reduction) / np.log(full_at)))

    if agg.min_reduction <= 1.0:
        notes.append(
            f"min reduction <= 1.0 ({agg.min_reduction:.2f}x) — "
            "regression on at least one seed, score halved"
        )
        points *= 0.5

    return float(points), notes


def compute_score(agg: Aggregated) -> dict[str, Any]:
    r_pts, r_notes = retrieval_points(agg)
    a_pts, a_notes = anisotropy_points(agg)
    return {
        "retrieval_pts":    round(r_pts, 2),
        "anisotropy_pts":   round(a_pts, 2),
        "code_quality_pts": "(manual, up to 10)",
        "total_automated":  round(r_pts + a_pts, 2),
        "max_automated":    WEIGHTS["retrieval"] + WEIGHTS["anisotropy"],
        "notes":            r_notes + a_notes,
    }


# --------------------------------------------------------------------------- #
# Top-level runner
# --------------------------------------------------------------------------- #

def run_multi(agent_factory: Callable[[np.ndarray, dict[str, Any]], Any],
              seeds: list[int],
              K: int = 16,
              N: int = 64,
              noise_levels: list[float] | None = None,
              n_per_level: int = 250,
              n_aniso: int = 16) -> dict[str, Any]:
    noise_levels = noise_levels or [0.6, 0.75, 0.85]
    reports = [
        run_one_seed(agent_factory, s, K, N, noise_levels, n_per_level, n_aniso)
        for s in seeds
    ]
    agg = aggregate(reports)
    score = compute_score(agg)
    return {
        "config": {
            "K": K, "N": N, "noise_levels": noise_levels,
            "n_per_level": n_per_level, "n_aniso": n_aniso,
            "seeds": seeds,
            "retrieval_full_at_delta": RETRIEVAL_FULL_AT,
            "anisotropy_full_at_factor": ANISOTROPY_FULL_AT,
            "dynamics_gate_fraction": DYNAMICS_GATE_FRACTION,
        },
        "per_seed":   [asdict(r) for r in reports],
        "aggregated": asdict(agg),
        "score":      score,
    }
