"""
Team submission for P-04: Hessian + class precision templates with soft
classification and confidence gating in predict_precision.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter
from pcam_model import build_default_R


def _fill_model_params(X: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """Merge partial model_params with bench defaults for ad-hoc / smoke tests."""
    _, N = X.shape
    out = dict(params)
    if "R" not in out:
        out["R"] = build_default_R(N=N, seed=0)
    out.setdefault("eta", 0.5)
    out.setdefault("beta", 8.0)
    return out


def _hessian_at(X: np.ndarray,
                params: dict[str, Any],
                x: np.ndarray) -> np.ndarray:
    """Analytic Hessian of the bench PCAM energy (see pcam_model.py / BENCH_NOTES).

    E(a) = 1/2 a^T R a - (eta/beta) log sum_i exp(beta x_i^T a)
    H(x) = R - eta * beta * X^T (diag(s) - s s^T) X,  s = softmax(beta X x).
    Returned matrix is symmetrized: 0.5 * (H + H.T).
    """
    X = np.asarray(X, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    R = np.asarray(params["R"], dtype=np.float64)
    eta = float(params["eta"])
    beta = float(params["beta"])

    z = beta * (X @ x)
    z = z - z.max()
    e = np.exp(z)
    s = e / e.sum()

    D = np.diag(s) - np.outer(s, s)
    H = R - eta * beta * (X.T @ (D @ X))
    H = 0.5 * (H + H.T)
    return H


def _spread(M: np.ndarray, eig_floor: float = 1e-10) -> float:
    """Eigenvalue spread max/min of the symmetric part of M."""
    M = 0.5 * (np.asarray(M, dtype=np.float64) + np.asarray(M, dtype=np.float64).T)
    eigs = np.linalg.eigvalsh(M)
    eigs = np.maximum(eigs, eig_floor)
    return float(eigs.max() / eigs.min())


def _ruiz_equilibrate(H: np.ndarray,
                      n_iters: int = 30,
                      eps: float = 1e-8) -> np.ndarray:
    """Symmetric Ruiz equilibration; returns pi with diag(sqrt(pi)) H diag(sqrt(pi)) ~ equilibrated.

    Regularises H with 1e-3 * I, then for each iteration sets r_i = sqrt(max_j |H_ij|, eps),
    scales H <- D H D with D = diag(1/r), and accumulates diagonal scale d. Returns pi = d^2.
    """
    H = 0.5 * (np.asarray(H, dtype=np.float64) + np.asarray(H, dtype=np.float64).T)
    N = H.shape[0]
    H = H + 1e-3 * np.eye(N)

    d = np.ones(N, dtype=np.float64)
    for _ in range(n_iters):
        row_max = np.max(np.abs(H), axis=1)
        r = np.sqrt(np.maximum(row_max, eps))
        inv_r = 1.0 / r
        H = (inv_r[:, None] * H) * inv_r[None, :]
        d *= inv_r

    return d * d


def _apply_pi(H: np.ndarray, pi: np.ndarray) -> np.ndarray:
    s = np.sqrt(pi)
    return (s[:, None] * H) * s[None, :]


class Engine(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any],
                 **kwargs: Any) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.model_params = _fill_model_params(self.X, model_params)
        self.beta = float(self.model_params["beta"])
        self.cls_beta_mult = float(kwargs.pop("cls_beta_mult", 1.0))
        self.confidence_exp = float(kwargs.pop("confidence_exp", 1.0))
        self.hessian_weight = float(kwargs.pop("hessian_weight", 0.5))
        self._precompute_hessian_templates()
        self._precompute_class_templates()
        self._precompute_combined()

    def _precompute_hessian_templates(self) -> None:
        self.pi_hess = np.zeros((self.K, self.N), dtype=np.float64)
        self.H_cache = np.zeros((self.K, self.N, self.N), dtype=np.float64)

        for k in range(self.K):
            H_k = _hessian_at(self.X, self.model_params, self.X[k])
            H_k = H_k + 1e-3 * np.eye(self.N)
            self.H_cache[k] = H_k

            pi_ruiz = _ruiz_equilibrate(H_k)
            spread_id = _spread(H_k)
            spread_ruiz = _spread(_apply_pi(H_k, pi_ruiz))

            if np.all(pi_ruiz > 0) and spread_ruiz < spread_id:
                pi_k = pi_ruiz
            else:
                pi_k = 1.0 / np.maximum(np.abs(np.diag(H_k)), 1e-12)

            mean = max(float(pi_k.mean()), 1e-12)
            self.pi_hess[k] = pi_k / mean

    def _precompute_class_templates(self) -> None:
        x_mean = self.X.mean(axis=0)
        diff_sq = (self.X - x_mean) ** 2 + 1e-3
        diff_sq = 0.9 * diff_sq + 0.1 * diff_sq.mean(axis=1, keepdims=True)
        self.pi_class = diff_sq / diff_sq.mean(axis=1, keepdims=True)

    def _precompute_combined(self) -> None:
        log_h = np.log(np.maximum(self.pi_hess, 1e-3))
        log_c = np.log(np.maximum(self.pi_class, 1e-3))
        hw = self.hessian_weight
        log_combined = hw * log_h + (1.0 - hw) * log_c
        self.pi_combined = np.exp(log_combined)
        self.pi_combined /= self.pi_combined.mean(axis=1, keepdims=True)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=np.float64).reshape(self.N)

        sims = self.X @ q
        sims = sims - sims.max()
        cls_beta = self.beta * self.cls_beta_mult
        w = np.exp(cls_beta * sims)
        w_sum = w.sum()
        if w_sum < 1e-12:
            return np.ones(self.N, dtype=np.float64)

        w = w / w_sum
        log_pi_mix = w @ np.log(np.maximum(self.pi_combined, 1e-3))
        pi = np.exp(log_pi_mix)

        c = float(w.max()) ** self.confidence_exp
        pi = pi ** c

        pi = pi / pi.mean()
        pi = np.clip(pi, 0.1, 10.0)
        pi = pi / pi.mean()
        return pi.astype(np.float64)
