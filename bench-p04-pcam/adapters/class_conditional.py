"""
Reference agent · class-conditional precision.

Approximates the paper's Pi*class design (Section 6.6). Runs a fast
one-shot Modern-Hopfield-style prediction on the corrupted query to
guess the class, then sets precision proportional to the predicted
class's typical magnitudes.

This is a SIMPLE REFERENCE for participants to study. The paper's
Pi*class achieves ~2.5% accuracy gain over Pi=I on MNIST; expect
similar order-of-magnitude gains on the synthetic bench.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter


class ClassConditionalAgent(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]
        self.beta = float(model_params.get("beta", 8.0))

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        # One-shot softmax retrieval to guess the class
        z = self.beta * (self.X @ corrupted_query)
        z = z - z.max()
        sm = np.exp(z)
        sm = sm / sm.sum()
        # Precision proportional to the predicted target pattern's magnitudes
        target = self.X.T @ sm  # (N,) — softmax-weighted pattern
        return np.abs(target) + 0.1
