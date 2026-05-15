"""
Identity-precision baseline (pi = 1 everywhere).

This is the floor every submission must beat on retrieval accuracy.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter


class DummyAgent(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        return np.ones(self.N)
