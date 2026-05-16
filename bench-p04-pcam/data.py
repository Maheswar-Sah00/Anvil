"""
Clustered synthetic patterns and query corruption for the P-04 bench.

Random unit-norm patterns in N-dim are near-orthogonal — too well-
separated for cosine classification to fail, which defeats the purpose
of a precision-controlled-dynamics benchmark.

This module generates **clustered patterns** instead: K patterns spread
across C clusters, where within-cluster patterns share a common
direction (cosine ≈ intra_sim). This mirrors MNIST's class structure
and creates real ambiguity under corruption — cosine classification of
the corrupted query is no longer trivial, and the precision-controlled
dynamics must do actual recovery work.

Corruption is per-dimension mask plus a low-magnitude Gaussian, tuned
so direct cosine classification on the corrupted query is well below
the achievable retrieval ceiling.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Clustered pattern generation
# --------------------------------------------------------------------------- #

def make_patterns(K: int = 16,
                  N: int = 64,
                  seed: int = 42,
                  n_clusters: int = 4,
                  intra_sim: float = 0.5) -> np.ndarray:
    """K unit-norm patterns drawn from `n_clusters` tight clusters.

    Within a cluster, patterns share a common direction at cosine
    `intra_sim` with the cluster center. Across clusters, centers are
    approximately orthogonal. Within-cluster pairwise cosine is
    approximately `intra_sim ** 2`.

    This mimics MNIST: patterns in the same class are similar, patterns
    in different classes are distinct. Random unit-norm patterns don't
    produce this — they are all approximately orthogonal regardless of
    "class" — which makes cosine classification trivial and defeats
    the bench's purpose.

    K patterns are assigned round-robin to clusters.
    """
    rng = np.random.default_rng(seed)
    n_clusters = max(2, min(n_clusters, K))
    intra_sim = float(np.clip(intra_sim, 0.0, 0.99))
    perp_weight = float(np.sqrt(max(1.0 - intra_sim ** 2, 1e-9)))

    # Cluster centers — random unit-norm, approximately orthogonal in N dim.
    centers = rng.standard_normal((n_clusters, N))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    X = np.empty((K, N), dtype=np.float64)
    for k in range(K):
        c = k % n_clusters
        center = centers[c]
        perp = rng.standard_normal(N)
        perp = perp - (perp @ center) * center
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 1e-12:
            perp = perp / perp_norm
        x = intra_sim * center + perp_weight * perp
        x_norm = np.linalg.norm(x)
        X[k] = x / x_norm if x_norm > 1e-12 else center
    return X


# --------------------------------------------------------------------------- #
# Corruption
# --------------------------------------------------------------------------- #

def corrupt(query: np.ndarray,
            p: float,
            rng: np.random.Generator,
            sigma: float = 0.4) -> np.ndarray:
    """Mask `p` fraction of dimensions to zero, add Gaussian noise of
    scaled magnitude `sigma / sqrt(N)`, then re-normalise to unit L2.

    With clustered patterns, the combined corruption regularly pushes
    queries past the basin boundary of the true pattern into a
    same-cluster neighbour — making cosine classification on the
    corrupted query insufficient and forcing the dynamics to recover.
    """
    N = query.shape[0]
    mask = rng.random(N) < p
    out = query.copy()
    out[mask] = 0.0
    out = out + rng.standard_normal(N) * (sigma / np.sqrt(N))
    norm = np.linalg.norm(out)
    if norm > 1e-12:
        out = out / norm
    return out


def make_test_queries(X: np.ndarray,
                      noise_levels: list[float],
                      n_per_level: int,
                      seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a (queries, truths, levels) triple covering all noise levels."""
    rng = np.random.default_rng(seed)
    K = X.shape[0]
    queries: list[np.ndarray] = []
    truths: list[int] = []
    levels: list[float] = []
    for p in noise_levels:
        for _ in range(n_per_level):
            idx = int(rng.integers(K))
            q = corrupt(X[idx], p, rng)
            queries.append(q)
            truths.append(idx)
            levels.append(float(p))
    return np.array(queries), np.array(truths), np.array(levels)
