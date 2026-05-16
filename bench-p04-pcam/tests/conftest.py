"""Pytest fixtures + path setup for the bench tests.

Adds the bench root and `adapters/` to sys.path so tests can
`from pcam_model import ...` and `from myteam import ...` directly.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ADAPTERS = os.path.join(BENCH, "adapters")
for p in (BENCH, ADAPTERS):
    if p not in sys.path:
        sys.path.insert(0, p)
