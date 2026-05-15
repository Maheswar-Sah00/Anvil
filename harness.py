"""Benchmark harness for P-04. Multi-seed orchestration + scoring.

Each seed runs in a fresh adapter instance with a freshly-generated
pattern set + Erdos-Renyi graph + query set. Agents that hardcode for
a specific seed fail immediately on the rest — this is the anti-gaming
defence shared with P-02.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from adapters.dummy import DummyAgent
from checks import retrieval_accuracy, spread_reduction
from data import make_patterns, make_test_queries
from pcam_model import PCAMModel, build_default_R


# Indicative axis weights. Council may rebalance before the event.
WEIGHTS: dict[str, float] = {
    "retrieval":   70.0,
    "anisotropy":  20.0,
    "code":        10.0,
}


@dataclass
class SeedReport:
    seed: int
    agent_accuracy: float
    baseline_accuracy: float
    delta: float
    spread_baseline: float
    spread_agent: float
    spread_reduction: float
    duration_s: float


@dataclass
class Aggregated:
    mean_delta: float
    min_delta: float
    mean_spread: float
    min_spread: float
    seeds: list[int]
    n_seeds: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_seeds = len(self.seeds)


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
    """Build a fresh model, agent, and query set for this seed."""
    X = make_patterns(K=K, N=N, seed=seed)
    R = build_default_R(N=N, seed=seed)
    model = PCAMModel(X, R)
    params = pack_params(model)

    agent = agent_factory(X, params)
    dummy = DummyAgent(X, params)

    queries, truths, _ = make_test_queries(X, noise_levels, n_per_level, seed=seed)

    t0 = time.monotonic()
    base_acc = retrieval_accuracy(model, dummy, queries, truths)
    agent_acc = retrieval_accuracy(model, agent, queries, truths)

    rng = np.random.default_rng(seed)
    indices = rng.choice(K, size=min(n_aniso, K), replace=False).tolist()
    spread = spread_reduction(model, agent, dummy, indices, seed=seed)
    dur = time.monotonic() - t0

    return SeedReport(
        seed=seed,
        agent_accuracy=float(agent_acc),
        baseline_accuracy=float(base_acc),
        delta=float(agent_acc - base_acc),
        spread_baseline=float(spread["baseline_spread"]),
        spread_agent=float(spread["agent_spread"]),
        spread_reduction=float(spread["reduction_factor"]),
        duration_s=round(dur, 2),
    )


def aggregate(seed_reports: list[SeedReport]) -> Aggregated:
    deltas = [r.delta for r in seed_reports]
    spreads = [r.spread_reduction for r in seed_reports]
    return Aggregated(
        mean_delta=float(np.mean(deltas)),
        min_delta=float(np.min(deltas)),
        mean_spread=float(np.mean(spreads)),
        min_spread=float(np.min(spreads)),
        seeds=[r.seed for r in seed_reports],
    )


def retrieval_points(mean_delta: float,
                     min_delta: float,
                     full_at: float = 0.05,
                     weight: float = WEIGHTS["retrieval"]) -> float:
    """Score retrieval. Penalises agents that regress on any seed."""
    if mean_delta <= 0:
        return 0.0
    base = min(weight, weight * (mean_delta / full_at))
    # Per-seed sanity gate: any regression below baseline halves the points.
    if min_delta < 0:
        base *= 0.5
    return float(base)


def anisotropy_points(mean_spread: float,
                      min_spread: float,
                      full_at: float = 10.0,
                      weight: float = WEIGHTS["anisotropy"]) -> float:
    if mean_spread <= 1.0:
        return 0.0
    base = min(weight, weight * (np.log(mean_spread) / np.log(full_at)))
    # If any seed produced spread <= 1 (no improvement), penalise.
    if min_spread <= 1.0:
        base *= 0.5
    return float(base)


def compute_score(agg: Aggregated) -> dict[str, Any]:
    r = retrieval_points(agg.mean_delta, agg.min_delta)
    a = anisotropy_points(agg.mean_spread, agg.min_spread)
    return {
        "retrieval_pts":    round(r, 2),
        "anisotropy_pts":   round(a, 2),
        "code_quality_pts": "(manual, up to 10)",
        "total_automated":  round(r + a, 2),
        "max_automated":    WEIGHTS["retrieval"] + WEIGHTS["anisotropy"],
    }


def run_multi(agent_factory: Callable[[np.ndarray, dict[str, Any]], Any],
              seeds: list[int],
              K: int = 16,
              N: int = 64,
              noise_levels: list[float] | None = None,
              n_per_level: int = 250,
              n_aniso: int = 16) -> dict[str, Any]:
    noise_levels = noise_levels or [0.5, 0.7, 0.8]
    per_seed = [
        run_one_seed(agent_factory, s, K, N, noise_levels, n_per_level, n_aniso)
        for s in seeds
    ]
    agg = aggregate(per_seed)
    score = compute_score(agg)
    return {
        "config": {
            "K": K, "N": N, "noise_levels": noise_levels,
            "n_per_level": n_per_level, "n_aniso": n_aniso,
            "seeds": seeds,
        },
        "per_seed":   [asdict(r) for r in per_seed],
        "aggregated": asdict(agg),
        "score":      score,
    }
