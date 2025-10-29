"""
Bellman-Ford shortest path algorithm implementation.

This module implements the Bellman-Ford algorithm for finding shortest paths in graphs
that may contain negative edge weights. Unlike Dijkstra's algorithm, Bellman-Ford can
handle negative weights and can detect negative cycles, though it has a slower time
complexity of O(V*E).

Algorithm Comparison:
---------------------
Bellman-Ford:
    - Time Complexity: O(V*E)
    - Handles negative edge weights: ✓
    - Detects negative cycles: ✓
    - Use case: Graphs with negative weights, cycle detection needed
    
Dijkstra:
    - Time Complexity: O((V+E)log V)
    - Handles negative edge weights: ✗
    - Detects negative cycles: ✗
    - Use case: Non-negative weights, faster performance needed

A* with zero heuristic:
    - Time Complexity: O((V+E)log V)
    - Handles negative edge weights: ✗
    - Detects negative cycles: ✗
    - Use case: Same as Dijkstra with optional heuristics

DAG-DP:
    - Time Complexity: O(V+E)
    - Handles negative edge weights: ✓
    - Detects negative cycles: N/A (requires DAG)
    - Use case: Fastest for DAGs, handles negative weights
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from src.algorithms.base import OptimizationAlgorithm

# Configure logger
logger = logging.getLogger(__name__)


class NegativeCycleError(Exception):
    """
    Exception raised when a negative cycle is detected in the graph.

    A negative cycle is a cycle whose total weight is negative. When such a cycle
    exists and is reachable from the source, there is no shortest path because you
    can keep traversing the cycle to make the path cost arbitrarily small.
    """

    pass


class BellmanFordOptimizer(OptimizationAlgorithm):
    """
    Bellman-Ford algorithm for shortest path finding with negative weight support.

    The Bellman-Ford algorithm computes shortest paths from a single source vertex
    to all other vertices in a weighted directed graph. Unlike Dijkstra's algorithm,
    it can handle graphs with negative edge weights and can detect negative cycles.

    Algorithm Description:
    ----------------------
    1. Initialize: Set distance to source as 0, all others as infinity
    2. Relax Edges: For V-1 iterations, relax all edges
       - For each edge (u,v) with weight w: if dist[u] + w < dist[v], update dist[v]
    3. Detect Negative Cycles: Run one more iteration
       - If any distance can still be reduced, a negative cycle exists
    4. Reconstruct Path: Use predecessor information to build the path

    Time Complexity: O(V * E)
        - V-1 iterations: O(V)
        - Each iteration processes all edges: O(E)
        - Total: O(V * E)

    Space Complexity: O(V)
        - Distance dictionary: O(V)
        - Predecessor dictionary: O(V)

    Advantages over Dijkstra:
    - Can handle negative edge weights
    - Can detect negative cycles
    - Simpler implementation

    Disadvantages compared to Dijkstra:
    - Slower: O(V*E) vs O((V+E)log V)
    - Less efficient for non-negative weights

    Attributes:
        name (str): Algorithm name identifier
        config (dict): Configuration parameters
        weight_attr (str): Edge attribute to use as weight
    """

    def __init__(self, name: str = "bellman_ford_optimizer", **kwargs):
        """
        Initialize the Bellman-Ford optimizer.

        Args:
            name (str): Name identifier for the algorithm
            **kwargs: Additional configuration parameters:
                - source: Source node ID (required for solve())
                - target: Target node ID (required for solve())
                - weight_attr (str): Edge attribute to use as weight. Defaults to 'weight'
        """
        super().__init__(name, **kwargs)
        self.weight_attr = kwargs.get("weight_attr", "weight")

    def solve(self, workflow_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Find the shortest path using the Bellman-Ford algorithm.

        This method implements the classic Bellman-Ford algorithm:
        1. Initializes distances (source=0, others=infinity)
        2. Performs V-1 iterations of edge relaxation
        3. Detects negative cycles with a final iteration
        4. Reconstructs the path from source to target

        The algorithm relaxes all edges V-1 times, which is sufficient to find
        shortest paths in graphs without negative cycles. The final iteration
        checks if any distance can still be reduced, indicating a negative cycle.

        Args:
            workflow_graph (nx.DiGraph): The workflow graph to search.
                                         Can contain negative edge weights.

        Returns:
            Dict[str, Any]: Solution dictionary containing:
                - path (List[Any]): Ordered list of node IDs from source to target
                - total_cost (float): Total cost/weight of the path
                - total_time (float): Total time metric (if 'time' attribute exists)
                - nodes_explored (int): Number of nodes with finite distance
                - execution_time_seconds (float): Algorithm execution time
                - resource_utilization (float): Ratio of path nodes to total nodes
                - algorithm (str): Algorithm name
                - weight_attr (str): Edge attribute used for weights
                - iterations (int): Number of relaxation iterations performed
                - edges_relaxed (int): Total number of edges relaxed
                - convergence_info (List[int]): Edges relaxed per iteration
                - negative_cycle_detected (bool): Whether a negative cycle was found

        Raises:
            ValueError: If source or target not specified in config
            ValueError: If source or target nodes don't exist in graph
            NegativeCycleError: If a negative cycle is detected in the graph
            ValueError: If no path exists from source to target

        Time Complexity: O(V * E) where V = vertices, E = edges
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

        # Get graph properties
        num_vertices = workflow_graph.number_of_nodes()
        num_edges = workflow_graph.number_of_edges()

        logger.info(
            f"Starting Bellman-Ford algorithm: {num_vertices} vertices, "
            f"{num_edges} edges, source={source}, target={target}"
        )

        # Step 1: Initialize distances and predecessors
        distances: Dict[Any, float] = {
            node: float("inf") for node in workflow_graph.nodes()
        }
        distances[source] = 0.0
        predecessors: Dict[Any, Optional[Any]] = {
            node: None for node in workflow_graph.nodes()
        }

        # Track convergence: edges relaxed per iteration
        convergence_info: List[int] = []
        total_edges_relaxed = 0

        # Step 2: Relax edges V-1 times
        logger.info(f"Performing {num_vertices - 1} iterations of edge relaxation")

        for iteration in range(num_vertices - 1):
            edges_relaxed_this_iteration = 0

            # Relax all edges
            for u, v, edge_data in workflow_graph.edges(data=True):
                # Get edge weight
                weight = edge_data.get(self.weight_attr, edge_data.get("weight", 1.0))

                # Relax edge if possible
                if (
                    distances[u] != float("inf")
                    and distances[u] + weight < distances[v]
                ):
                    old_distance = distances[v]
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
                    edges_relaxed_this_iteration += 1

                    logger.debug(
                        f"Iteration {iteration + 1}: Relaxed edge ({u}, {v}), "
                        f"distance {old_distance} -> {distances[v]}"
                    )

            convergence_info.append(edges_relaxed_this_iteration)
            total_edges_relaxed += edges_relaxed_this_iteration

            logger.info(
                f"Iteration {iteration + 1}/{num_vertices - 1}: "
                f"Relaxed {edges_relaxed_this_iteration} edges"
            )

            # Early termination: if no edges were relaxed, we've converged
            if edges_relaxed_this_iteration == 0:
                logger.info(f"Converged early at iteration {iteration + 1}")
                # Fill remaining iterations with 0
                for _ in range(iteration + 1, num_vertices - 1):
                    convergence_info.append(0)
                break

        # Step 3: Check for negative cycles
        logger.info("Checking for negative cycles")
        has_negative_cycle = self._detect_negative_cycle(workflow_graph, distances)

        if has_negative_cycle:
            logger.error("Negative cycle detected in graph")
            raise NegativeCycleError(
                "Graph contains a negative cycle reachable from source. "
                "No shortest path exists because you can keep traversing the "
                "cycle to make the path cost arbitrarily small."
            )

        logger.info("No negative cycles detected")

        # Check if target is reachable
        if distances[target] == float("inf"):
            raise ValueError(
                f"No path exists from source '{source}' to target '{target}'"
            )

        # Step 4: Reconstruct path
        path = self._reconstruct_path(predecessors, source, target)

        # Calculate metrics
        total_cost = distances[target]
        total_time = self._calculate_total_time(workflow_graph, path)

        # Count nodes with finite distance (nodes explored)
        nodes_explored = sum(1 for dist in distances.values() if dist != float("inf"))

        # Calculate resource utilization
        resource_utilization = len(path) / num_vertices if num_vertices > 0 else 0.0

        # Calculate execution time
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        logger.info(
            f"Bellman-Ford completed: path_length={len(path)}, "
            f"cost={total_cost}, time={execution_time:.4f}s, "
            f"edges_relaxed={total_edges_relaxed}"
        )

        return {
            "path": path,
            "total_cost": total_cost,
            "total_time": total_time,
            "nodes_explored": nodes_explored,
            "execution_time_seconds": execution_time,
            "resource_utilization": resource_utilization,
            "algorithm": self.name,
            "weight_attr": self.weight_attr,
            "iterations": len(convergence_info),
            "edges_relaxed": total_edges_relaxed,
            "convergence_info": convergence_info,
            "negative_cycle_detected": False,  # If we got here, no cycle was found
        }

    def _detect_negative_cycle(
        self, graph: nx.DiGraph, distances: Dict[Any, float]
    ) -> bool:
        """
        Detect if a negative cycle exists in the graph.

        After V-1 iterations of edge relaxation, all shortest paths should be found
        (assuming no negative cycles). This method performs one more iteration:
        if any edge can still be relaxed, it means we can reduce a distance further,
        which indicates the presence of a negative cycle.

        Args:
            graph (nx.DiGraph): The graph to check
            distances (Dict[Any, float]): Current shortest distances

        Returns:
            bool: True if a negative cycle is detected, False otherwise

        Time Complexity: O(E) - checks all edges once
        """
        # Try to relax all edges one more time
        for u, v, edge_data in graph.edges(data=True):
            weight = edge_data.get(self.weight_attr, edge_data.get("weight", 1.0))

            # If we can still reduce a distance, there's a negative cycle
            if distances[u] != float("inf") and distances[u] + weight < distances[v]:
                logger.warning(
                    f"Negative cycle detected: edge ({u}, {v}) can still be relaxed "
                    f"({distances[v]} -> {distances[u] + weight})"
                )
                return True

        return False

    def _reconstruct_path(
        self, predecessors: Dict[Any, Optional[Any]], source: Any, target: Any
    ) -> List[Any]:
        """
        Reconstruct the shortest path from source to target using predecessors.

        Args:
            predecessors (Dict[Any, Optional[Any]]): Map of each node to its predecessor
            source (Any): Source node ID
            target (Any): Target node ID

        Returns:
            List[Any]: Ordered list of node IDs representing the path

        Raises:
            ValueError: If path reconstruction fails

        Time Complexity: O(V) in worst case (path includes all vertices)
        """
        path = []
        current = target

        # Trace back from target to source
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

    def _calculate_total_time(self, graph: nx.DiGraph, path: List[Any]) -> float:
        """
        Calculate total time along the path if 'time' attribute exists.

        Args:
            graph (nx.DiGraph): The graph
            path (List[Any]): The path

        Returns:
            float: Total time along the path, or 0.0 if no time attributes

        Time Complexity: O(P) where P is the length of the path
        """
        total_time = 0.0

        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i + 1])
            if edge_data and "time" in edge_data:
                total_time += edge_data["time"]

        return total_time

    def validate_solution(
        self, solution: Dict[str, Any], workflow_graph: nx.DiGraph
    ) -> bool:
        """
        Validate that a solution is feasible and correct.

        Performs comprehensive validation including:
        1. Presence of all required keys
        2. Path validity and connectivity
        3. Cost calculation accuracy (including negative weights)
        4. Bellman-Ford specific metadata
        5. Metrics within valid ranges

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
            "iterations",
            "edges_relaxed",
            "convergence_info",
            "negative_cycle_detected",
        }
        if not all(key in solution for key in required_keys):
            logger.error(
                f"Solution missing required keys. Expected: {required_keys}, "
                f"Got: {solution.keys()}"
            )
            return False

        path = solution["path"]

        # Empty path is invalid
        if not path:
            logger.error("Path is empty")
            return False

        # Check all nodes in path exist in graph
        graph_nodes = set(workflow_graph.nodes())
        if not all(node in graph_nodes for node in path):
            invalid_nodes = [node for node in path if node not in graph_nodes]
            logger.error(f"Path contains invalid nodes: {invalid_nodes}")
            return False

        # Check path connectivity
        for i in range(len(path) - 1):
            if not workflow_graph.has_edge(path[i], path[i + 1]):
                logger.error(
                    f"No edge exists between consecutive nodes: "
                    f"{path[i]} -> {path[i + 1]}"
                )
                return False

        # Verify path endpoints
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

        # Validate total_cost matches actual path weight (including negative weights)
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

        # Validate Bellman-Ford specific fields
        if solution["iterations"] < 0:
            logger.error(f"Iterations must be non-negative: {solution['iterations']}")
            return False

        if solution["edges_relaxed"] < 0:
            logger.error(
                f"Edges relaxed must be non-negative: {solution['edges_relaxed']}"
            )
            return False

        if not isinstance(solution["convergence_info"], list):
            logger.error("convergence_info must be a list")
            return False

        if not isinstance(solution["negative_cycle_detected"], bool):
            logger.error("negative_cycle_detected must be a boolean")
            return False

        # Validate metrics are within valid ranges
        if solution["execution_time_seconds"] < 0:
            logger.error(
                f"Execution time is negative: {solution['execution_time_seconds']}"
            )
            return False

        if not (0 <= solution["resource_utilization"] <= 1.0):
            logger.error(
                f"Resource utilization out of range [0,1]: "
                f"{solution['resource_utilization']}"
            )
            return False

        if solution["nodes_explored"] < 1:
            logger.error(f"Nodes explored must be >= 1: {solution['nodes_explored']}")
            return False

        return True
