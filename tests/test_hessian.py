"""Verify _hessian_at against finite differences of the bench energy."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, "..", "bench-p04-pcam"))
ADAPTERS = os.path.join(BENCH, "adapters")
for p in (BENCH, ADAPTERS):
    if p not in sys.path:
        sys.path.insert(0, p)

from pcam_model import PCAMModel, build_default_R  # noqa: E402
from myteam import _hessian_at  # noqa: E402


def _energy(model: PCAMModel, a: np.ndarray) -> float:
    """Canonical PCAM energy from pcam_model.py docstring lines 8-11.

    E(a) = 1/2 a^T R a - (eta/beta) log sum_i exp(beta x_i^T a).
    """
    a = np.asarray(a, dtype=np.float64)
    quad = 0.5 * a @ (model.R @ a)
    z = model.beta * (model.X @ a)
    zmax = z.max()
    lse = zmax + np.log(np.exp(z - zmax).sum())
    return float(quad - (model.eta / model.beta) * lse)


def _fd_hessian(model: PCAMModel, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    N = x.shape[0]
    H = np.zeros((N, N), dtype=np.float64)
    inv = 1.0 / (4.0 * h * h)
    for i in range(N):
        for j in range(i, N):
            xpp = x.copy(); xpp[i] += h; xpp[j] += h
            xpm = x.copy(); xpm[i] += h; xpm[j] -= h
            xmp = x.copy(); xmp[i] -= h; xmp[j] += h
            xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
            v = (_energy(model, xpp) - _energy(model, xpm)
                 - _energy(model, xmp) + _energy(model, xmm)) * inv
            H[i, j] = v
            H[j, i] = v
    return H


def test_hessian_matches_finite_differences():
    cases = [(5, 8), (8, 12), (16, 24), (10, 16), (6, 20)]
    for case_idx, (K, N) in enumerate(cases):
        rng = np.random.default_rng(1000 + case_idx)
        Xraw = rng.standard_normal((K, N))
        X = Xraw / np.linalg.norm(Xraw, axis=1, keepdims=True)
        R = build_default_R(N=N, seed=case_idx)
        model = PCAMModel(X, R)
        x = rng.standard_normal(N) * 0.3

        params = {"R": model.R, "eta": model.eta, "beta": model.beta}
        H_analytic = _hessian_at(X, params, x)
        H_fd = _fd_hessian(model, x, h=1e-4)

        err = float(np.max(np.abs(H_analytic - H_fd)))
        assert err < 1e-4, (
            f"case {case_idx} (K={K}, N={N}): max-abs error {err:.3e} >= 1e-4"
        )
