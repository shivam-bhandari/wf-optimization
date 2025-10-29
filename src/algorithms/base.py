"""
Base class for optimization algorithms
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import networkx as nx


class OptimizationAlgorithm(ABC):
    """
    Abstract base class for workflow optimization algorithms
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs

    @abstractmethod
    def solve(self, workflow_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Solve the optimization problem

        Args:
            workflow_graph: The workflow to optimize

        Returns:
            Solution dictionary containing optimization results
        """
        raise NotImplementedError("Subclasses must implement solve method")

    @abstractmethod
    def validate_solution(
        self, solution: Dict[str, Any], workflow_graph: nx.DiGraph
    ) -> bool:
        """
        Validate that a solution is feasible

        Args:
            solution: The solution to validate
            workflow_graph: The workflow graph

        Returns:
            True if solution is valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate_solution method")
