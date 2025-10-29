"""
DAG-specific shortest path algorithm using dynamic programming.

This module implements an efficient shortest path algorithm for Directed Acyclic Graphs (DAGs)
using dynamic programming and topological sorting. The algorithm achieves O(V+E) time complexity
by processing nodes in topological order, ensuring each node is processed only once.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
import networkx as nx
from src.algorithms.base import OptimizationAlgorithm


class DAGDynamicProgramming(OptimizationAlgorithm):
    """
    Dynamic Programming algorithm for finding shortest paths in Directed Acyclic Graphs (DAGs).

    This algorithm leverages the topological ordering property of DAGs to compute shortest paths
    efficiently in linear time O(V+E), where V is the number of vertices and E is the number of edges.

    Time Complexity: O(V + E)
        - Topological sort: O(V + E)
        - Distance initialization: O(V)
        - Edge relaxation: O(E)
        - Path reconstruction: O(V)

    Space Complexity: O(V)
        - Distance dictionary: O(V)
        - Predecessor dictionary: O(V)
        - Topological order list: O(V)

    Attributes:
        name (str): Algorithm name identifier
        config (dict): Configuration parameters including source, target, and weight_attr
    """

    def __init__(self, name: str = "dag_dynamic_programming", **kwargs):
        """
        Initialize the DAG Dynamic Programming algorithm.

        Args:
            name (str): Name identifier for the algorithm
            **kwargs: Additional configuration parameters:
                - source: Source node ID (required for solve())
                - target: Target node ID (required for solve())
                - weight_attr (str): Edge attribute to use as weight ('cost', 'time', 'weight')
                                    Defaults to 'weight'
        """
        super().__init__(name, **kwargs)
        self.weight_attr = kwargs.get("weight_attr", "weight")

    def solve(self, workflow_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Find the shortest path in a DAG using dynamic programming.

        This method implements the classic DAG shortest path algorithm:
        1. Validates that the graph is acyclic
        2. Performs topological sort
        3. Initializes distances (source=0, others=infinity)
        4. Processes nodes in topological order, relaxing all outgoing edges
        5. Reconstructs the optimal path using predecessor tracking

        Args:
            workflow_graph (nx.DiGraph): Directed graph to find shortest path in.
                                         Must be acyclic (DAG).

        Returns:
            Dict[str, Any]: Solution dictionary containing:
                - path (List[Any]): Ordered list of node IDs from source to target
                - total_cost (float): Total cost of the path
                - total_time (float): Execution time in seconds
                - nodes_explored (int): Number of nodes processed
                - algorithm (str): Name of the algorithm used
                - weight_attr (str): Edge attribute used for weights

        Raises:
            ValueError: If graph contains cycles (not a DAG)
            ValueError: If source or target nodes don't exist in graph
            ValueError: If no path exists from source to target
            ValueError: If source or target not specified in config

        Time Complexity: O(V + E) where V = vertices, E = edges
        Space Complexity: O(V) for distance and predecessor dictionaries
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

        # Validate that the graph is a DAG
        if not nx.is_directed_acyclic_graph(workflow_graph):
            raise ValueError(
                "Graph contains cycles; DAG shortest path requires an acyclic graph"
            )

        # Perform topological sort
        try:
            topo_order = list(nx.topological_sort(workflow_graph))
        except nx.NetworkXError as e:
            raise ValueError(f"Failed to perform topological sort: {e}")

        # Initialize distances dictionary: all nodes to infinity except source
        distances: Dict[Any, float] = {
            node: float("inf") for node in workflow_graph.nodes()
        }
        distances[source] = 0.0

        # Initialize predecessor dictionary for path reconstruction
        predecessors: Dict[Any, Optional[Any]] = {
            node: None for node in workflow_graph.nodes()
        }

        # Track nodes explored
        nodes_explored = 0

        # Process nodes in topological order
        for node in topo_order:
            # Skip nodes that are unreachable
            if distances[node] == float("inf"):
                continue

            nodes_explored += 1

            # Relax all outgoing edges from this node
            for successor in workflow_graph.successors(node):
                self._relax_edge(
                    workflow_graph, node, successor, distances, predecessors
                )

        # Check if target is reachable
        if distances[target] == float("inf"):
            raise ValueError(
                f"No path exists from source '{source}' to target '{target}'"
            )

        # Reconstruct path from source to target
        path = self._reconstruct_path(predecessors, source, target)

        # Calculate total cost
        total_cost = distances[target]

        # Calculate execution time
        end_time = time.perf_counter()
        total_time = end_time - start_time

        return {
            "path": path,
            "total_cost": total_cost,
            "total_time": total_time,
            "nodes_explored": nodes_explored,
            "algorithm": self.name,
            "weight_attr": self.weight_attr,
        }

    def _relax_edge(
        self,
        graph: nx.DiGraph,
        u: Any,
        v: Any,
        distances: Dict[Any, float],
        predecessors: Dict[Any, Optional[Any]],
    ) -> None:
        """
        Perform edge relaxation for the edge (u, v).

        Edge relaxation checks if the path to v through u is shorter than the
        currently known shortest path to v. If so, it updates the distance and
        predecessor information.

        Args:
            graph (nx.DiGraph): The directed graph
            u (Any): Source node of the edge
            v (Any): Target node of the edge
            distances (Dict[Any, float]): Current shortest distances from source
            predecessors (Dict[Any, Optional[Any]]): Predecessor map for path reconstruction

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        # Get edge weight, default to 1.0 if not present
        edge_data = graph.get_edge_data(u, v)

        # Try to get weight from specified attribute, fallback to alternatives
        weight = None
        if edge_data:
            # Try primary weight attribute
            weight = edge_data.get(self.weight_attr)

            # Fallback to other common weight attributes if primary not found
            if weight is None:
                for attr in ["weight", "cost", "time"]:
                    if attr in edge_data:
                        weight = edge_data[attr]
                        break

        # Default weight if none found
        if weight is None:
            weight = 1.0

        # Perform relaxation: if path through u is shorter, update
        new_distance = distances[u] + weight
        if new_distance < distances[v]:
            distances[v] = new_distance
            predecessors[v] = u

    def _reconstruct_path(
        self, predecessors: Dict[Any, Optional[Any]], source: Any, target: Any
    ) -> List[Any]:
        """
        Reconstruct the shortest path from source to target using predecessor information.

        Args:
            predecessors (Dict[Any, Optional[Any]]): Map of each node to its predecessor
            source (Any): Source node ID
            target (Any): Target node ID

        Returns:
            List[Any]: Ordered list of node IDs representing the path from source to target

        Raises:
            ValueError: If path reconstruction fails (shouldn't happen after reachability check)

        Time Complexity: O(V) in worst case (path includes all vertices)
        Space Complexity: O(V) for the path list
        """
        path = []
        current = target

        # Trace back from target to source using predecessors
        while current is not None:
            path.append(current)
            if current == source:
                break
            current = predecessors[current]

        # Verify we reached the source
        if current != source:
            raise ValueError(
                f"Failed to reconstruct path from '{source}' to '{target}'"
            )

        # Reverse to get path from source to target
        path.reverse()

        return path

    def validate_solution(
        self, solution: Dict[str, Any], workflow_graph: nx.DiGraph
    ) -> bool:
        """
        Validate that a solution is feasible and correct.

        Checks:
        1. Solution contains all required keys
        2. Path is a valid sequence of nodes in the graph
        3. All consecutive nodes in path are connected by edges
        4. Path starts with source and ends with target (if specified in config)

        Args:
            solution (Dict[str, Any]): Solution dictionary to validate
            workflow_graph (nx.DiGraph): The workflow graph

        Returns:
            bool: True if solution is valid and feasible, False otherwise
        """
        # Check for required keys
        required_keys = {"path", "total_cost", "total_time", "nodes_explored"}
        if not all(key in solution for key in required_keys):
            return False

        path = solution["path"]

        # Empty path is invalid
        if not path:
            return False

        # Check all nodes in path exist in graph
        graph_nodes = set(workflow_graph.nodes())
        if not all(node in graph_nodes for node in path):
            return False

        # Check path connectivity: verify edges exist between consecutive nodes
        for i in range(len(path) - 1):
            if not workflow_graph.has_edge(path[i], path[i + 1]):
                return False

        # If source and target are specified, verify path endpoints
        source = self.config.get("source")
        target = self.config.get("target")

        if source is not None and path[0] != source:
            return False

        if target is not None and path[-1] != target:
            return False

        # Validate total_cost is non-negative
        if solution["total_cost"] < 0:
            return False

        # Validate nodes_explored is positive
        if solution["nodes_explored"] < 1:
            return False

        return True
