"""
Base Benchmark class for workflow optimization

This module provides foundational classes for creating custom benchmarks.
For most use cases, use BenchmarkRunner instead which provides a complete
implementation for running multiple algorithms across multiple workflows.
"""

from typing import Dict, Any, List
import networkx as nx
from pydantic import BaseModel


class BenchmarkResult(BaseModel):
    """
    Result structure for a single benchmark run.

    Attributes:
        algorithm_name (str): Name of the algorithm that was run
        execution_time (float): Time taken to execute in seconds
        solution_quality (Dict[str, float]): Quality metrics (e.g., cost, time)
        metadata (Dict[str, Any]): Additional metadata about the run
    """

    algorithm_name: str
    execution_time: float
    solution_quality: Dict[str, float]
    metadata: Dict[str, Any]


class Benchmark:
    """
    Base class for workflow optimization benchmarks.

    This is an abstract base class that can be subclassed to create custom
    benchmark implementations with specialized evaluation logic.

    For most use cases, consider using BenchmarkRunner which provides a
    complete, ready-to-use implementation for running multiple algorithms
    across multiple workflows with comprehensive statistics and error handling.

    Attributes:
        name (str): Name of the benchmark
        workflow_graph (nx.DiGraph): The workflow graph to benchmark against

    Example:
        ```python
        class CustomBenchmark(Benchmark):
            def run(self, algorithm):
                # Custom benchmark logic
                start = time.time()
                solution = algorithm.solve(self.workflow_graph)
                end = time.time()

                return BenchmarkResult(
                    algorithm_name=algorithm.name,
                    execution_time=end - start,
                    solution_quality={'cost': solution['total_cost']},
                    metadata={'nodes': len(solution['path'])}
                )

            def evaluate(self, solution):
                # Custom evaluation logic
                return {'cost': solution['total_cost']}
        ```
    """

    def __init__(self, name: str, workflow_graph: nx.DiGraph):
        """
        Initialize the benchmark.

        Args:
            name (str): Name identifier for this benchmark
            workflow_graph (nx.DiGraph): The workflow graph to benchmark against
        """
        self.name = name
        self.workflow_graph = workflow_graph

    def run(self, algorithm) -> BenchmarkResult:
        """
        Run the benchmark with the given algorithm.

        This method should be implemented by subclasses to define specific
        benchmark behavior.

        Args:
            algorithm: An optimization algorithm instance with a solve() method

        Returns:
            BenchmarkResult: The benchmark results

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Subclasses must implement run method. "
            "For a ready-to-use implementation, consider using BenchmarkRunner."
        )

    def evaluate(self, solution: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a solution and return quality metrics.

        This method should be implemented by subclasses to define specific
        evaluation logic.

        Args:
            solution: A solution dictionary from an algorithm

        Returns:
            Dict[str, float]: Dictionary of metric names to values

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Subclasses must implement evaluate method. "
            "For a ready-to-use implementation, consider using BenchmarkRunner."
        )
