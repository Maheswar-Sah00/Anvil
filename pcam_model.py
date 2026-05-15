"""
Frozen PCAM dynamics for P-04.

Implements the precision-controlled associative memory system from
the PCAM paper (NeurIPS 2026 submission). Participants do not modify
this file — the base model is frozen by problem constraint.

Energy:   E(a) = 1/2 a^T R a  -  (eta/beta) log sum_i exp(beta x_i^T a)
Gradient: grad E(a) = R a  -  eta X^T softmax(beta X a)
Dynamics: a_{t+1} = a_t + dt * ( -pi * grad E(a_t)  +  J * u(t) )
Hessian:  H(a) = R  -  eta * beta * X^T (diag(s) - s s^T) X     where s = softmax(beta X a)
"""
from __future__ import annotations

import numpy as np


def build_default_R(N: int = 64,
                    gamma: float = 0.2,
                    delta: float = 0.1,
                    alpha: float = 0.5,
                    edge_p: float = 0.1,
                    seed: int = 0) -> np.ndarray:
    """R = A + gamma * L + delta * 11^T, with L the normalised Laplacian of an
    Erdos-Renyi graph. Matches the paper's default operator (Section 6.1)."""
    rng = np.random.default_rng(seed)
    A = alpha * np.eye(N)
    adj_upper = (rng.random((N, N)) < edge_p).astype(float)
    adj_upper = np.triu(adj_upper, 1)
    adj = adj_upper + adj_upper.T
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    norm_adj = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    L = np.eye(N) - norm_adj
    R = A + gamma * L + delta * np.ones((N, N))
    R = 0.5 * (R + R.T)
    return R


class PCAMModel:
    """Frozen precision-controlled associative memory."""

    def __init__(self,
                 X: np.ndarray,
                 R: np.ndarray | None = None,
                 eta: float = 0.5,
                 beta: float = 8.0,
                 dt: float = 0.01,
                 T_max: int = 3000,
                 tol: float = 1e-6,
                 T_in: int = 100,
                 pi_min: float = 0.1,
                 pi_max: float = 10.0) -> None:
        self.X = X.astype(np.float64)
        self.K, self.N = self.X.shape
        self.R = (R if R is not None else build_default_R(self.N)).astype(np.float64)
        self.eta = float(eta)
        self.beta = float(beta)
        self.dt = float(dt)
        self.T_max = int(T_max)
        self.tol = float(tol)
        self.T_in = int(T_in)
        self.pi_min = float(pi_min)
        self.pi_max = float(pi_max)

    def _softmax(self, a: np.ndarray) -> np.ndarray:
        z = self.beta * (self.X @ a)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def gradient(self, a: np.ndarray) -> np.ndarray:
        s = self._softmax(a)
        return self.R @ a - self.eta * (self.X.T @ s)

    def hessian(self, a: np.ndarray) -> np.ndarray:
        s = self._softmax(a)
        D = np.diag(s) - np.outer(s, s)
        return self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))

    def clip_and_normalise(self, pi: np.ndarray) -> np.ndarray:
        pi = np.asarray(pi, dtype=np.float64).reshape(self.N)
        pi = np.clip(pi, self.pi_min, self.pi_max)
        mean = pi.mean()
        if mean > 0:
            pi = pi / mean
        return pi

    def run(self,
            a0: np.ndarray,
            pi: np.ndarray,
            u_const: np.ndarray | None = None) -> np.ndarray:
        """Integrate the dynamics with the given diagonal precision.

        a0       : initial state (N,)
        pi       : diagonal precision (N,), clipped & mean-normalised internally
        u_const  : constant external input applied during the input window
        """
        pi = self.clip_and_normalise(pi)
        a = np.asarray(a0, dtype=np.float64).copy()
        for t in range(self.T_max):
            g = self.gradient(a)
            update = -pi * g
            if u_const is not None and t < self.T_in:
                update = update + u_const
            a_new = a + self.dt * update
            if np.linalg.norm(a_new - a) < self.tol:
                a = a_new
                break
            a = a_new
        return a

    def classify(self, a: np.ndarray) -> int:
        """Return the index of the stored pattern that best aligns with a.

        PCAM equilibria sit at approximately eta * R^-1 * x_i (Lemma E3),
        so their magnitude differs from the stored patterns. We classify
        by direction, matching the paper's convention.
        """
        a_norm = np.linalg.norm(a)
        if a_norm < 1e-12:
            return 0
        cosines = self.X @ (a / a_norm)  # X rows are already unit-norm
        return int(np.argmax(cosines))
