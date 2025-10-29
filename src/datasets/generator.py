"""
Workflow generation utilities
"""

import networkx as nx
import random
from typing import Dict, List, Optional


class WorkflowGenerator:
    """
    Generate synthetic workflow graphs for benchmarking
    """

    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            random.seed(random_seed)

    def generate_dag(
        self,
        num_nodes: int,
        edge_probability: float = 0.3,
        node_weights: Optional[Dict] = None,
    ) -> nx.DiGraph:
        """
        Generate a directed acyclic graph (DAG)

        Args:
            num_nodes: Number of nodes in the workflow
            edge_probability: Probability of edge creation
            node_weights: Optional dictionary of node attributes

        Returns:
            A directed acyclic graph
        """
        # Create a random DAG
        graph = nx.DiGraph()

        for i in range(num_nodes):
            graph.add_node(i, weight=node_weights.get(i, 1) if node_weights else 1)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_probability:
                    graph.add_edge(i, j, weight=random.randint(1, 10))

        # Ensure it's a DAG by removing cycles if any
        if not nx.is_directed_acyclic_graph(graph):
            graph = nx.DiGraph(nx.topological_sort(graph.edges()))

        return graph

    def generate_linear_workflow(self, num_nodes: int) -> nx.DiGraph:
        """
        Generate a linear workflow (chain)

        Args:
            num_nodes: Number of nodes in the workflow

        Returns:
            A linear directed graph
        """
        graph = nx.DiGraph()

        for i in range(num_nodes):
            graph.add_node(i, weight=random.randint(1, 10))

        for i in range(num_nodes - 1):
            graph.add_edge(i, i + 1, weight=random.randint(1, 10))

        return graph

    def generate_fork_join_workflow(self, num_parallel_tasks: int) -> nx.DiGraph:
        """
        Generate a fork-join workflow

        Args:
            num_parallel_tasks: Number of parallel tasks

        Returns:
            A fork-join directed graph
        """
        graph = nx.DiGraph()

        # Start node
        graph.add_node(0, weight=1)

        # Parallel tasks
        for i in range(1, num_parallel_tasks + 1):
            graph.add_node(i, weight=random.randint(1, 10))
            graph.add_edge(0, i, weight=1)

        # Join node
        join_node = num_parallel_tasks + 1
        graph.add_node(join_node, weight=1)

        for i in range(1, num_parallel_tasks + 1):
            graph.add_edge(i, join_node, weight=1)

        return graph
