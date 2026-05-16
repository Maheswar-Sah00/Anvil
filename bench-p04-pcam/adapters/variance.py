"""
Reference agent · variance-based precision.

A simple heuristic: trust dimensions where the corrupted query has
large absolute value; down-weight dimensions that look masked (close
to zero) or noisy. Useful as a sanity-check reference — easy to beat
with a better design, hard to do worse than.

This is a SIMPLE EXAMPLE for participants to study. It is not the
intended submission template.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter


class VarianceAgent(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        # Higher precision where the query has more signal magnitude.
        # Small floor so masked dims still participate slightly.
        return np.abs(corrupted_query) + 0.1
