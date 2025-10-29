"""
Dijkstra's shortest path algorithm implementation using NetworkX.

This module implements Dijkstra's algorithm for finding shortest paths in graphs with
non-negative edge weights. It leverages NetworkX's optimized built-in implementation
while providing additional metrics, validation, and error handling.

When to use:
- Dijkstra: General graphs with non-negative weights, O((V+E)log V) with binary heap
- DAG-DP: Only for DAGs, faster O(V+E) linear time using topological sort
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from src.algorithms.base import OptimizationAlgorithm

# Configure logger
logger = logging.getLogger(__name__)


class DijkstraOptimizer(OptimizationAlgorithm):
    """
    Dijkstra's algorithm implementation for finding shortest paths in weighted graphs.

    This optimizer uses NetworkX's efficient implementation of Dijkstra's algorithm,
    which employs a binary heap for priority queue operations. It is suitable for
    general directed or undirected graphs with non-negative edge weights.

    Algorithm Comparison:
    ---------------------
    Dijkstra's Algorithm:
        - Time Complexity: O((V + E) log V) with binary heap
        - Space Complexity: O(V)
        - Requirements: Non-negative edge weights
        - Use case: General graphs (cyclic or acyclic)
        - Priority queue maintains nodes by distance

    DAG Dynamic Programming:
        - Time Complexity: O(V + E) - faster for DAGs
        - Space Complexity: O(V)
        - Requirements: Must be a DAG (no cycles)
        - Use case: Directed acyclic graphs only
        - Processes nodes in topological order

    Recommendation: If your graph is a DAG, use DAG-DP for better performance.

    Attributes:
        name (str): Algorithm name identifier
        config (dict): Configuration parameters
        weight_attr (str): Edge attribute to use as weight
    """

    def __init__(self, name: str = "dijkstra_optimizer", **kwargs):
        """
        Initialize the Dijkstra optimizer.

        Args:
            name (str): Name identifier for the algorithm
            **kwargs: Additional configuration parameters:
                - source: Source node ID (required for solve())
                - target: Target node ID (required for solve())
                - weight_attr (str): Edge attribute to use as weight.
                                    Defaults to 'weight'. Common values: 'cost', 'time', 'weight'
        """
        super().__init__(name, **kwargs)
        self.weight_attr = kwargs.get("weight_attr", "weight")

    def solve(self, workflow_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Find the shortest path using Dijkstra's algorithm.

        This method performs the following steps:
        1. Validates input parameters (source, target exist)
        2. Checks all edge weights are non-negative
        3. Logs a warning if the graph is a DAG (recommending DAG-DP)
        4. Uses NetworkX's single_source_dijkstra() for optimal path finding
        5. Calculates comprehensive metrics including resource utilization
        6. Tracks execution time

        Args:
            workflow_graph (nx.DiGraph): The workflow graph to optimize.
                                         Can be directed or undirected.

        Returns:
            Dict[str, Any]: Solution dictionary containing:
                - path (List[Any]): Ordered list of node IDs from source to target
                - total_cost (float): Total cost/weight of the path
                - total_time (float): Total time metric (if 'time' attribute exists)
                - nodes_explored (int): Number of nodes in the shortest path
                - execution_time_seconds (float): Algorithm execution time
                - resource_utilization (float): Graph coverage metric (nodes in path / total nodes)
                - algorithm (str): Algorithm name
                - weight_attr (str): Edge attribute used for weights

        Raises:
            ValueError: If source or target not specified in config
            ValueError: If source or target nodes don't exist in graph
            ValueError: If any edge has negative weight
            nx.NetworkXNoPath: If no path exists from source to target
            nx.NodeNotFound: If source node doesn't exist

        Time Complexity: O((V + E) log V) using binary heap
        Space Complexity: O(V) for distance and predecessor storage
        """
        start_time = time.perf_counter()

        # Extract source and target from config
        source = self.config.get("source")
        target = self.config.get("target")

        if source is None or target is None:
            raise ValueError(
                "Both 'source' and 'target' must be specified in configuration"
            )

        # Validate source and target nodes exist
        if source not in workflow_graph.nodes():
            raise ValueError(f"Source node '{source}' not found in graph")

        if target not in workflow_graph.nodes():
            raise ValueError(f"Target node '{target}' not found in graph")

        # Validate all edge weights are non-negative
        self._validate_non_negative_weights(workflow_graph)

        # Check if graph is a DAG and log recommendation
        if nx.is_directed_acyclic_graph(workflow_graph):
            logger.warning(
                "Graph is a Directed Acyclic Graph (DAG). "
                "Consider using DAGDynamicProgramming algorithm for better performance "
                f"(O(V+E) vs O((V+E)log V)). Current graph: {workflow_graph.number_of_nodes()} nodes, "
                f"{workflow_graph.number_of_edges()} edges."
            )

        # Handle special case: source equals target
        if source == target:
            end_time = time.perf_counter()
            return {
                "path": [source],
                "total_cost": 0.0,
                "total_time": 0.0,
                "nodes_explored": 1,
                "execution_time_seconds": end_time - start_time,
                "resource_utilization": 1.0 / max(workflow_graph.number_of_nodes(), 1),
                "algorithm": self.name,
                "weight_attr": self.weight_attr,
            }

        # Use NetworkX's Dijkstra implementation
        try:
            # single_source_dijkstra returns (distance, path)
            distance, path = nx.single_source_dijkstra(
                workflow_graph, source=source, target=target, weight=self.weight_attr
            )
        except nx.NetworkXNoPath:
            raise ValueError(
                f"No path exists from source '{source}' to target '{target}'"
            )
        except nx.NodeNotFound as e:
            raise ValueError(f"Node not found in graph: {e}")

        # Calculate comprehensive metrics
        metrics = self._calculate_path_metrics(workflow_graph, path)

        # Calculate execution time
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Build solution dictionary
        solution = {
            "path": path,
            "total_cost": distance,
            "total_time": metrics["total_time"],
            "nodes_explored": len(path),
            "execution_time_seconds": execution_time,
            "resource_utilization": metrics["resource_utilization"],
            "algorithm": self.name,
            "weight_attr": self.weight_attr,
        }

        return solution

    def _validate_non_negative_weights(self, graph: nx.DiGraph) -> None:
        """
        Validate that all edge weights are non-negative.

        Dijkstra's algorithm requires non-negative edge weights to guarantee correctness.
        This method checks all edges and raises an error if any negative weight is found.

        Args:
            graph (nx.DiGraph): The graph to validate

        Raises:
            ValueError: If any edge has a negative weight

        Time Complexity: O(E) where E is the number of edges
        """
        for u, v, data in graph.edges(data=True):
            # Try to get weight from specified attribute, with fallbacks
            weight = data.get(self.weight_attr)

            # Fallback to common weight attributes if primary not found
            if weight is None:
                for attr in ["weight", "cost", "time"]:
                    if attr in data:
                        weight = data[attr]
                        break

            # Default to 1.0 if no weight found
            if weight is None:
                weight = 1.0

            # Check for negative weight
            if weight < 0:
                raise ValueError(
                    f"Edge ({u}, {v}) has negative weight {weight}. "
                    f"Dijkstra's algorithm requires non-negative edge weights. "
                    f"Consider using Bellman-Ford algorithm for graphs with negative weights."
                )

    def _calculate_path_metrics(
        self, graph: nx.DiGraph, path: List[Any]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for a given path.

        Computes various metrics that provide insights into the path characteristics
        and resource utilization:
        - Total time: Sum of 'time' attribute along path edges
        - Resource utilization: Ratio of path nodes to total graph nodes

        Args:
            graph (nx.DiGraph): The workflow graph
            path (List[Any]): Ordered list of nodes in the path

        Returns:
            Dict[str, float]: Dictionary containing:
                - total_time (float): Sum of 'time' weights along the path
                - resource_utilization (float): Fraction of graph nodes in path

        Time Complexity: O(P) where P is the length of the path
        """
        metrics = {"total_time": 0.0, "resource_utilization": 0.0}

        # Calculate total time if 'time' attribute exists
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i + 1])
            if edge_data and "time" in edge_data:
                metrics["total_time"] += edge_data["time"]

        # Calculate resource utilization (percentage of nodes used)
        total_nodes = graph.number_of_nodes()
        if total_nodes > 0:
            metrics["resource_utilization"] = len(path) / total_nodes

        return metrics

    def validate_solution(
        self, solution: Dict[str, Any], workflow_graph: nx.DiGraph
    ) -> bool:
        """
        Validate that a solution is feasible and correct.

        Performs comprehensive validation including:
        1. Presence of all required keys in solution
        2. Path is non-empty and contains valid nodes
        3. Consecutive nodes in path are connected by edges
        4. Path starts at source and ends at target (if specified)
        5. Total cost matches the sum of edge weights along the path
        6. All metrics are within valid ranges

        Args:
            solution (Dict[str, Any]): The solution dictionary to validate
            workflow_graph (nx.DiGraph): The workflow graph

        Returns:
            bool: True if solution is valid and feasible, False otherwise

        Time Complexity: O(P) where P is the length of the path
        """
        # Check for required keys
        required_keys = {
            "path",
            "total_cost",
            "total_time",
            "nodes_explored",
            "execution_time_seconds",
            "resource_utilization",
        }
        if not all(key in solution for key in required_keys):
            logger.error(
                f"Solution missing required keys. Expected: {required_keys}, Got: {solution.keys()}"
            )
            return False

        path = solution["path"]

        # Empty path is invalid (except for source == target, but that should have [source])
        if not path:
            logger.error("Path is empty")
            return False

        # Check all nodes in path exist in graph
        graph_nodes = set(workflow_graph.nodes())
        if not all(node in graph_nodes for node in path):
            invalid_nodes = [node for node in path if node not in graph_nodes]
            logger.error(f"Path contains invalid nodes: {invalid_nodes}")
            return False

        # Check path connectivity: verify edges exist between consecutive nodes
        for i in range(len(path) - 1):
            if not workflow_graph.has_edge(path[i], path[i + 1]):
                logger.error(
                    f"No edge exists between consecutive nodes: {path[i]} -> {path[i + 1]}"
                )
                return False

        # Verify path endpoints match source and target if specified
        source = self.config.get("source")
        target = self.config.get("target")

        if source is not None and path[0] != source:
            logger.error(
                f"Path does not start at source. Expected: {source}, Got: {path[0]}"
            )
            return False

        if target is not None and path[-1] != target:
            logger.error(
                f"Path does not end at target. Expected: {target}, Got: {path[-1]}"
            )
            return False

        # Validate total_cost matches actual path weight
        calculated_cost = 0.0
        for i in range(len(path) - 1):
            edge_data = workflow_graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                weight = edge_data.get(self.weight_attr, edge_data.get("weight", 1.0))
                calculated_cost += weight

        # Allow small floating point tolerance
        if abs(calculated_cost - solution["total_cost"]) > 1e-9:
            logger.error(
                f"Total cost mismatch. Calculated: {calculated_cost}, "
                f"Reported: {solution['total_cost']}"
            )
            return False

        # Validate nodes_explored matches path length
        if solution["nodes_explored"] != len(path):
            logger.error(
                f"Nodes explored mismatch. Path length: {len(path)}, "
                f"Reported: {solution['nodes_explored']}"
            )
            return False

        # Validate metrics are within valid ranges
        if solution["total_cost"] < 0:
            logger.error(f"Total cost is negative: {solution['total_cost']}")
            return False

        if solution["execution_time_seconds"] < 0:
            logger.error(
                f"Execution time is negative: {solution['execution_time_seconds']}"
            )
            return False

        if not (0 <= solution["resource_utilization"] <= 1.0):
            logger.error(
                f"Resource utilization out of range [0,1]: {solution['resource_utilization']}"
            )
            return False

        return True
