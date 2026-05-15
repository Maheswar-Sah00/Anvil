"""
The two checks for P-04: retrieval accuracy and anisotropy spread.
"""
from __future__ import annotations

import numpy as np

from pcam_model import PCAMModel


def retrieval_accuracy(model: PCAMModel,
                       agent,
                       queries: np.ndarray,
                       truths: np.ndarray) -> float:
    correct = 0
    for q, t in zip(queries, truths):
        pi = agent.predict_precision(q)
        a_star = model.run(q, pi, u_const=q)
        if model.classify(a_star) == int(t):
            correct += 1
    return correct / len(queries)


def per_pattern_spread(model: PCAMModel, pi: np.ndarray, pattern: np.ndarray) -> float | None:
    """Spread mu_max / mu_min of the symmetrised contraction operator at the
    given (approximate) equilibrium, under precision pi."""
    pi = model.clip_and_normalise(pi)
    H = model.hessian(pattern)
    H = 0.5 * (H + H.T)
    eig_H = np.linalg.eigvalsh(H)
    if eig_H.min() <= 0:
        return None  # not in a stable basin under PCAM assumptions
    pi_sqrt = np.sqrt(pi)
    S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    if len(eigs) < 2:
        return None
    return float(eigs.max() / eigs.min())


def anisotropy_spread(model: PCAMModel,
                      agent,
                      pattern_indices: list[int],
                      probe_sigma: float = 0.05,
                      seed: int = 0) -> float:
    """For each sampled stored pattern, query the agent with a lightly-perturbed
    version (so it sees a realistic input) and measure the spread of the
    resulting precision-weighted contraction operator at the attractor."""
    rng = np.random.default_rng(seed)
    spreads: list[float] = []
    for idx in pattern_indices:
        pattern = model.X[idx]
        probe = pattern + rng.standard_normal(model.N) * probe_sigma
        n = np.linalg.norm(probe)
        if n > 1e-12:
            probe = probe / n
        pi = agent.predict_precision(probe)
        s = per_pattern_spread(model, pi, pattern)
        if s is not None:
            spreads.append(s)
    return float(np.mean(spreads)) if spreads else float("inf")


def spread_reduction(model: PCAMModel,
                     agent,
                     baseline,
                     pattern_indices: list[int],
                     seed: int = 0) -> dict[str, float]:
    base = anisotropy_spread(model, baseline, pattern_indices, seed=seed)
    yours = anisotropy_spread(model, agent, pattern_indices, seed=seed)
    factor = base / yours if yours > 0 and np.isfinite(yours) else 0.0
    return {
        "baseline_spread": round(base, 4),
        "agent_spread":    round(yours, 4),
        "reduction_factor": round(factor, 4),
    }
