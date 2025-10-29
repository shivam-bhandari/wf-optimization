"""
Tests for workflow generation
"""

import pytest
import networkx as nx
from src.datasets.generator import WorkflowGenerator


def test_workflow_generator_init():
    """Test WorkflowGenerator initialization"""
    generator = WorkflowGenerator(random_seed=42)
    assert generator is not None


def test_generate_linear_workflow():
    """Test linear workflow generation"""
    generator = WorkflowGenerator(random_seed=42)
    workflow = generator.generate_linear_workflow(num_nodes=5)
    
    assert isinstance(workflow, nx.DiGraph)
    assert len(workflow.nodes()) == 5
    assert len(workflow.edges()) == 4
    assert nx.is_directed_acyclic_graph(workflow)


def test_generate_dag():
    """Test DAG generation"""
    generator = WorkflowGenerator(random_seed=42)
    workflow = generator.generate_dag(num_nodes=10, edge_probability=0.3)
    
    assert isinstance(workflow, nx.DiGraph)
    assert len(workflow.nodes()) == 10
    assert nx.is_directed_acyclic_graph(workflow)


def test_generate_fork_join_workflow():
    """Test fork-join workflow generation"""
    generator = WorkflowGenerator(random_seed=42)
    workflow = generator.generate_fork_join_workflow(num_parallel_tasks=3)
    
    assert isinstance(workflow, nx.DiGraph)
    # Should have start node, parallel tasks, and join node
    assert len(workflow.nodes()) == 5
    assert nx.is_directed_acyclic_graph(workflow)
