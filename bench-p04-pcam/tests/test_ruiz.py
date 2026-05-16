"""Verify Ruiz equilibration reduces eigenvalue spread vs identity and Jacobi."""
from __future__ import annotations

import numpy as np

from myteam import _ruiz_equilibrate, _spread


def _random_spd(n: int, cond: float, rng: np.random.Generator) -> np.ndarray:
    """SPD matrix with eigenvalues in [1, cond] and skewed row/column scaling.

    Diagonal similarity S H S makes Jacobi's one-shot diag scaling suboptimal
    while symmetric Ruiz can rebalance row max-norms over many iterations.
    """
    eigs = np.geomspace(1.0, cond, n)
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    core = (Q * eigs) @ Q.T
    s = np.exp(rng.uniform(-1.5, 1.5, n))
    return (s[:, None] * core) * s[None, :]


def _jacobi_scaled(H: np.ndarray) -> np.ndarray:
    d = np.diag(H)
    s = np.sqrt(1.0 / np.maximum(np.abs(d), 1e-12))
    return (s[:, None] * H) * s[None, :]


def _ruiz_scaled(H: np.ndarray, pi: np.ndarray) -> np.ndarray:
    s = np.sqrt(pi)
    return (s[:, None] * H) * s[None, :]


def test_ruiz_beats_jacobi_and_identity_on_median_spread(capsys):
    rng = np.random.default_rng(42)
    n = 64
    n_mats = 20

    spread_identity: list[float] = []
    spread_jacobi: list[float] = []
    spread_ruiz: list[float] = []

    for i in range(n_mats):
        cond = float(rng.uniform(10.0, 1000.0))
        H = _random_spd(n, cond, rng)

        pi_ruiz = _ruiz_equilibrate(H)
        si = _spread(H)
        sj = _spread(_jacobi_scaled(H))
        sr = _spread(_ruiz_scaled(H, pi_ruiz))

        spread_identity.append(si)
        spread_jacobi.append(sj)
        spread_ruiz.append(sr)

        print(
            f"matrix {i:2d}  cond~{cond:8.1f}  "
            f"identity={si:10.4f}  jacobi={sj:10.4f}  ruiz={sr:10.4f}"
        )

    med_i = float(np.median(spread_identity))
    med_j = float(np.median(spread_jacobi))
    med_r = float(np.median(spread_ruiz))
    print(
        f"\nmedian  identity={med_i:.4f}  jacobi={med_j:.4f}  ruiz={med_r:.4f}"
    )

    assert med_r < med_j < med_i, (
        f"expected median spread_ruiz < spread_jacobi < spread_identity, "
        f"got {med_r:.4f} < {med_j:.4f} < {med_i:.4f}"
    )
