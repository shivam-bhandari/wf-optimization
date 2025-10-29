"""
Benchmarking module for workflow optimization algorithms

This module provides comprehensive benchmarking infrastructure including:
- BenchmarkRunner: Multi-algorithm, multi-workflow benchmark orchestration
- BenchmarkConfig: Configuration for benchmark execution
- Benchmark: Base class for custom benchmark implementations (legacy/extensibility)
- BenchmarkResult: Result structure for individual benchmark runs
"""

from src.benchmarking.runner import BenchmarkConfig, BenchmarkRunner
from src.benchmarking.benchmark import Benchmark, BenchmarkResult

__all__ = ["BenchmarkConfig", "BenchmarkRunner", "Benchmark", "BenchmarkResult"]
