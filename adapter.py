"""
Adapter interface for P-04 (PCAM Precision Agent) submissions.

Every submission provides a concrete Adapter that wraps its agent.
The harness calls predict_precision once per corrupted query and the
returned vector is fed straight into the PCAM dynamics.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Adapter(ABC):
    @abstractmethod
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        """
        stored_patterns : ndarray (K, N) — the K patterns already in the system
        model_params    : dict with frozen system parameters
                          (R, eta, beta, dt, T_max, tol, pi_min, pi_max)
        """

    @abstractmethod
    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        """
        corrupted_query : ndarray (N,) — the noisy input
        returns         : ndarray (N,) of positive values, your precision weights.
                          Will be clipped to [pi_min, pi_max] and mean-normalised
                          by the harness before being applied.
        """
