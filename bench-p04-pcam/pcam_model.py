"""
Frozen PCAM dynamics for P-04.

Implements the precision-controlled associative memory system from the
PCAM paper (NeurIPS 2026 submission). Participants do not modify this
file — the base model is frozen by problem constraint.

Reference equations (paper Section 3):

    E(a)       = (1/2) a^T R a  -  (eta / beta) log sum_i exp(beta x_i^T a)
    grad E(a)  = R a  -  eta X^T softmax(beta X a)
    H(a)       = R  -  eta beta X^T (diag(s) - s s^T) X    where s = softmax(beta X a)
    dynamics   = a_{t+1} = a_t + dt * ( -pi(t) odot grad E(a_t)  +  J u(t) )

The precision operator pi is the participant's inference-time control.
It is constrained to be diagonal, positive, and to satisfy
    pi_min <= pi_i <= pi_max  AND  mean(pi_i) = 1
which is enforced by `clip_and_normalise` below.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Structured operator R = A + gamma * L + delta * 11^T
# --------------------------------------------------------------------------- #

def build_default_R(N: int = 64,
                    gamma: float = 0.2,
                    delta: float = 0.1,
                    alpha: float = 0.5,
                    edge_p: float = 0.1,
                    seed: int = 0) -> np.ndarray:
    """Build the paper's default structured operator (Section 6.1).

    R = alpha * I  +  gamma * L  +  delta * 11^T

    where L is the symmetrically normalised Laplacian of an Erdos-Renyi
    graph with edge probability `edge_p`.
    """
    rng = np.random.default_rng(seed)
    A = alpha * np.eye(N)

    upper = (rng.random((N, N)) < edge_p).astype(np.float64)
    upper = np.triu(upper, 1)
    adj = upper + upper.T

    deg = adj.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    normalised_adj = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    L = np.eye(N) - normalised_adj

    R = A + gamma * L + delta * np.ones((N, N))
    return 0.5 * (R + R.T)


# --------------------------------------------------------------------------- #
# PCAM model
# --------------------------------------------------------------------------- #

class PCAMModel:
    """Frozen precision-controlled associative memory."""

    def __init__(self,
                 X: np.ndarray,
                 R: Optional[np.ndarray] = None,
                 eta: float = 0.5,
                 beta: float = 8.0,
                 dt: float = 0.01,
                 T_max: int = 3000,
                 tol: float = 1e-6,
                 T_in: int = 100,
                 pi_min: float = 0.1,
                 pi_max: float = 10.0) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.R = np.asarray(R if R is not None else build_default_R(self.N),
                            dtype=np.float64)
        self.eta = float(eta)
        self.beta = float(beta)
        self.dt = float(dt)
        self.T_max = int(T_max)
        self.tol = float(tol)
        self.T_in = int(T_in)
        self.pi_min = float(pi_min)
        self.pi_max = float(pi_max)

    # ------------------------------- energy ------------------------------ #

    def _softmax(self, a: np.ndarray) -> np.ndarray:
        z = self.beta * (self.X @ a)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def energy(self, a: np.ndarray) -> float:
        """E(a) per Section 3.1 of the paper."""
        quad = 0.5 * a @ (self.R @ a)
        z = self.beta * (self.X @ a)
        z_max = z.max()
        log_sum = z_max + np.log(np.exp(z - z_max).sum())
        return float(quad - (self.eta / self.beta) * log_sum)

    def gradient(self, a: np.ndarray) -> np.ndarray:
        s = self._softmax(a)
        return self.R @ a - self.eta * (self.X.T @ s)

    def hessian(self, a: np.ndarray) -> np.ndarray:
        s = self._softmax(a)
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        return 0.5 * (H + H.T)

    # ----------------------------- projection ---------------------------- #

    def clip_and_normalise(self, pi: np.ndarray) -> np.ndarray:
        """Project pi onto { pi : pi_min <= pi <= pi_max AND mean(pi) = 1 }.

        The constraint set is non-empty since pi_min <= 1 <= pi_max. We do
        an iterative clip+rescale that converges in 1-3 iterations in
        practice. Final clip guarantees the bounds are respected even if
        the iteration stops on a marginal case.
        """
        pi = np.asarray(pi, dtype=np.float64).reshape(self.N)
        if not np.all(np.isfinite(pi)):
            return np.ones(self.N)

        for _ in range(20):
            pi = np.clip(pi, self.pi_min, self.pi_max)
            m = pi.mean()
            if m <= 1e-12:
                return np.ones(self.N)
            pi = pi / m
            within_bounds = (pi.min() >= self.pi_min - 1e-9
                             and pi.max() <= self.pi_max + 1e-9)
            mean_ok = abs(pi.mean() - 1.0) < 1e-8
            if within_bounds and mean_ok:
                break

        return np.clip(pi, self.pi_min, self.pi_max)

    # ----------------------------- dynamics ------------------------------ #

    def run(self,
            a0: np.ndarray,
            pi: np.ndarray,
            u_const: Optional[np.ndarray] = None,
            T_max: Optional[int] = None) -> np.ndarray:
        """Integrate the precision-modulated dynamics with explicit Euler.

        a0       : initial state, shape (N,)
        pi       : diagonal precision (clipped + normalised internally)
        u_const  : constant external input during the input window
        T_max    : optional override of self.T_max
        """
        pi = self.clip_and_normalise(pi)
        a = np.asarray(a0, dtype=np.float64).copy()
        T = self.T_max if T_max is None else int(T_max)
        T_in = self.T_in

        for t in range(T):
            g = self.gradient(a)
            update = -pi * g
            if u_const is not None and t < T_in:
                update = update + u_const
            a_new = a + self.dt * update
            if np.linalg.norm(a_new - a) < self.tol:
                a = a_new
                break
            a = a_new
        return a

    def find_equilibrium(self, x0: np.ndarray) -> np.ndarray:
        """Run dynamics from x0 with pi = I and no external input.

        Useful for locating the true attractor associated with a stored
        pattern. Equilibria sit at approximately eta * R^-1 * x_i for
        large beta (paper Lemma E3), not at x_i itself.
        """
        return self.run(x0, np.ones(self.N), u_const=None)

    # ----------------------------- classify ------------------------------ #

    def classify(self, a: np.ndarray) -> int:
        """Nearest-pattern classification of a converged state.

        We use direction (cosine similarity) because the equilibria sit
        at scaled translates of the stored patterns; their magnitude is
        not equal to ||x_i||. The paper's argmin-L2 is equivalent to
        argmax-inner-product when both sides are unit-normalised.
        """
        n = np.linalg.norm(a)
        if n < 1e-12:
            return 0
        cosines = self.X @ (a / n)
        return int(np.argmax(cosines))
