"""
Evaluation primitives for P-04. Pure functions — no orchestration.

Two metrics, both grounded in the PCAM paper:

1. Retrieval accuracy (paper Section 6.6): fraction of corrupted
   queries the dynamics resolve to the correct stored pattern.

2. Anisotropy spread reduction (paper Theorem F3): how much more
   uniform the local convergence rates become under the agent's
   precision, evaluated at the TRUE equilibrium of each stored
   pattern (not the pattern itself).

A third diagnostic — direct_classify_accuracy — is reported alongside
to show how much of the work the dynamics are actually doing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pcam_model import PCAMModel


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def retrieval_accuracy(model: PCAMModel,
                       agent,
                       queries: np.ndarray,
                       truths: np.ndarray) -> float:
    """Run the agent's precision through the PCAM dynamics, classify the
    final state, return fraction correct.
    """
    correct = 0
    for q, t in zip(queries, truths):
        pi = agent.predict_precision(q)
        a_star = model.run(q, pi, u_const=q)
        if model.classify(a_star) == int(t):
            correct += 1
    return correct / len(queries)


def direct_classify_accuracy(model: PCAMModel,
                             queries: np.ndarray,
                             truths: np.ndarray) -> float:
    """Diagnostic: classify the corrupted query directly, no dynamics.

    Compare against retrieval_accuracy to see how much value the
    dynamics add. If direct >= dynamics, your agent is regressing.
    """
    correct = 0
    for q, t in zip(queries, truths):
        if model.classify(q) == int(t):
            correct += 1
    return correct / len(queries)


# --------------------------------------------------------------------------- #
# Anisotropy — evaluated at TRUE equilibria
# --------------------------------------------------------------------------- #

def _symmetrised_spread(pi: np.ndarray, H: np.ndarray) -> Optional[float]:
    """Spread of the eigenvalues of Pi^(1/2) H Pi^(1/2) — equivalent to
    the spread of Pi H (which is the linearised contraction operator),
    but symmetrised so the eigenvalues are guaranteed real.

    Returns None if H is not positive definite (i.e. the point isn't
    a stable equilibrium under PCAM assumptions).
    """
    eig_H = np.linalg.eigvalsh(0.5 * (H + H.T))
    if eig_H.min() <= 0:
        return None
    pi_sqrt = np.sqrt(np.clip(pi, 1e-12, None))
    S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    if len(eigs) < 2:
        return None
    return float(eigs.max() / eigs.min())


def anisotropy_reductions(model: PCAMModel,
                          agent,
                          pattern_indices: list[int],
                          probe_sigma: float = 0.05,
                          seed: int = 0) -> list[tuple[float, float]]:
    """For each sampled stored pattern:

    1. Find the TRUE equilibrium a* by running dynamics from x_i with
       pi = I and no input (paper Lemma E3 — equilibria sit near
       eta * R^-1 * x_i, not at x_i).
    2. Compute H(a*) at the equilibrium.
    3. Probe the agent with a lightly-noisy version of x_i so it
       produces a realistic precision vector.
    4. Measure the spread of Pi^(1/2) H Pi^(1/2) at the true
       equilibrium for both pi = I (baseline) and pi = agent's choice.

    Returns a list of (baseline_spread, agent_spread) pairs. The mean
    reduction factor is computed by the harness.
    """
    rng = np.random.default_rng(seed)
    pi_I = np.ones(model.N)
    results: list[tuple[float, float]] = []

    for idx in pattern_indices:
        pattern = model.X[idx]
        a_star = model.find_equilibrium(pattern)

        # Probe so the agent's predict_precision sees a realistic input.
        probe = pattern + rng.standard_normal(model.N) * probe_sigma
        probe = probe / max(np.linalg.norm(probe), 1e-12)
        pi_agent_raw = agent.predict_precision(probe)
        pi_agent = model.clip_and_normalise(pi_agent_raw)

        H = model.hessian(a_star)
        s_base = _symmetrised_spread(pi_I, H)
        s_agent = _symmetrised_spread(pi_agent, H)
        if s_base is None or s_agent is None:
            continue
        results.append((s_base, s_agent))

    return results


def summarise_anisotropy(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """Aggregate per-pattern spreads into a single reduction factor."""
    if not pairs:
        return {
            "baseline_spread": float("nan"),
            "agent_spread":    float("nan"),
            "reduction":       0.0,
            "n":               0,
        }
    base = np.array([p[0] for p in pairs])
    agent = np.array([p[1] for p in pairs])
    reductions = base / np.maximum(agent, 1e-12)
    return {
        "baseline_spread": float(np.mean(base)),
        "agent_spread":    float(np.mean(agent)),
        "reduction":       float(np.mean(reductions)),
        "reduction_min":   float(np.min(reductions)),
        "n":               len(pairs),
    }
