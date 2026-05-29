"""Baseline strategies for benchmark comparison."""
from .independent import IndependentBaseline
from .union_fps import UnionFpsBaseline

__all__ = ["IndependentBaseline", "UnionFpsBaseline"]
