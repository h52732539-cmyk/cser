"""Benchmark runner + reporter."""
from .runner import BenchmarkRunner
from .reporter import generate_tables

__all__ = ["BenchmarkRunner", "generate_tables"]
