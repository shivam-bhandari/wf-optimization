"""
Tests for the OptimizationAlgorithm base class and interface.

This module tests the abstract base class behavior and interface contract.
For tests of concrete algorithm implementations, see test_phase1_algorithms.py.
"""

import pytest
import networkx as nx
from src.algorithms.base import OptimizationAlgorithm


def test_base_algorithm_abstract():
    """Test that OptimizationAlgorithm cannot be instantiated directly."""
    
    # Can't instantiate abstract class
    with pytest.raises(TypeError):
        OptimizationAlgorithm(name="test")


def test_algorithm_interface_contract():
    """
    Test that a concrete implementation must implement required abstract methods.
    
    This test verifies the interface contract - any concrete algorithm must
    implement solve() and validate_solution() methods.
    """
    
    # Test that missing solve() raises TypeError
    class MissingSolve(OptimizationAlgorithm):
        def validate_solution(self, solution, workflow_graph):
            return True
    
    with pytest.raises(TypeError):
        MissingSolve(name="incomplete")
    
    # Test that missing validate_solution() raises TypeError
    class MissingValidate(OptimizationAlgorithm):
        def solve(self, workflow_graph):
            return {}
    
    with pytest.raises(TypeError):
        MissingValidate(name="incomplete")
    
    # Test that complete implementation works
    class CompleteAlgorithm(OptimizationAlgorithm):
        def solve(self, workflow_graph):
            return {'path': [], 'total_cost': 0.0}
        
        def validate_solution(self, solution, workflow_graph):
            return True
    
    algorithm = CompleteAlgorithm(name="test_alg")
    assert algorithm.name == "test_alg"
    assert isinstance(algorithm.solve(nx.DiGraph()), dict)
