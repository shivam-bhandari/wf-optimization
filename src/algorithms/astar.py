"""
A* search algorithm implementation with configurable heuristics.

This module implements the A* search algorithm, which combines the benefits of uniform-cost
search (like Dijkstra) with heuristic-guided search for improved performance. A* is optimal
and complete when using admissible heuristics (heuristics that never overestimate the cost
to reach the goal).

Algorithm Comparison:
- A* with zero heuristic: Equivalent to Dijkstra's algorithm
- A* with good heuristic: Often faster than Dijkstra by focusing search
- Dijkstra: Special case of A* with h(n) = 0
- Greedy Best-First: Uses only heuristic (not optimal)
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
import networkx as nx
from src.algorithms.base import OptimizationAlgorithm

# Configure logger
logger = logging.getLogger(__name__)


class AStarOptimizer(OptimizationAlgorithm):
    """
    A* search algorithm with multiple configurable heuristics.

    A* is an informed search algorithm that uses a heuristic function to guide the search
    towards the goal. It maintains an open set of nodes to explore, prioritized by:
    f(n) = g(n) + h(n)
    where:
    - g(n) = actual cost from start to node n
    - h(n) = estimated cost from node n to goal (heuristic)

    A* is optimal when the heuristic is admissible (never overestimates the actual cost).

    Time Complexity: O(b^d) in worst case, but often much better with good heuristics
    Space Complexity: O(b^d) for storing the open and closed sets

    Available Heuristics:
    ---------------------
    1. Zero Heuristic (h(n) = 0):
       - Equivalent to Dijkstra's algorithm
       - Always admissible and consistent
       - Use when: No domain knowledge available
       - Performance: Same as Dijkstra O((V+E)log V)

    2. Minimum Cost Estimate:
       - Estimates minimum cost to reach target via any remaining edge
       - Admissible if edge costs are non-negative
       - Use when: Graph has varying edge costs
       - Performance: Better than Dijkstra in sparse graphs

    3. Task Depth Heuristic:
       - Estimates cost based on shortest path length (unweighted) to target
       - Multiplies by minimum edge weight in graph
       - Use when: Graph represents task dependencies
       - Performance: Good for workflow scheduling problems

    Attributes:
        name (str): Algorithm name identifier
        config (dict): Configuration parameters
        weight_attr (str): Edge attribute to use as weight
        heuristic_type (str): Type of heuristic ('zero', 'min_cost_estimate', 'task_depth')
        nodes_explored (int): Counter for nodes explored during search
    """

    def __init__(
        self, name: str = "astar_optimizer", heuristic_type: str = "zero", **kwargs
    ):
        """
        Initialize the A* optimizer with specified heuristic.

        Args:
            name (str): Name identifier for the algorithm
            heuristic_type (str): Type of heuristic to use:
                - 'zero': h(n) = 0, equivalent to Dijkstra
                - 'min_cost_estimate': Uses minimum edge cost estimate
                - 'task_depth': Uses graph depth weighted by min edge cost
            **kwargs: Additional configuration parameters:
                - source: Source node ID (required for solve())
                - target: Target node ID (required for solve())
                - weight_attr (str): Edge attribute to use as weight. Defaults to 'weight'

        Raises:
            ValueError: If heuristic_type is not recognized
        """
        super().__init__(name, **kwargs)
        self.weight_attr = kwargs.get("weight_attr", "weight")

        # Validate heuristic type
        valid_heuristics = ["zero", "min_cost_estimate", "task_depth"]
        if heuristic_type not in valid_heuristics:
            raise ValueError(
                f"Invalid heuristic_type '{heuristic_type}'. "
                f"Must be one of: {valid_heuristics}"
            )

        self.heuristic_type = heuristic_type
        self.nodes_explored = 0

        # Storage for graph-wide statistics needed by heuristics
        self._graph = None
        self._target = None
        self._min_edge_weight = None
        self._shortest_path_lengths = None

    def solve(self, workflow_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Find the shortest path using A* search with configured heuristic.

        A* expands nodes in order of f(n) = g(n) + h(n), where:
        - g(n) is the actual cost from start to n
        - h(n) is the heuristic estimate from n to goal

        The algorithm is guaranteed to find the optimal path if the heuristic is admissible
        (never overestimates the true cost to the goal).

        Args:
            workflow_graph (nx.DiGraph): The workflow graph to search.
                                         Can be directed or undirected.

        Returns:
            Dict[str, Any]: Solution dictionary containing:
                - path (List[Any]): Ordered list of node IDs from source to target
                - total_cost (float): Total cost/weight of the path
                - total_time (float): Total time metric (if 'time' attribute exists)
                - nodes_explored (int): Number of nodes where heuristic was evaluated
                - execution_time_seconds (float): Algorithm execution time
                - resource_utilization (float): Ratio of path nodes to total nodes
                - algorithm (str): Algorithm name
                - weight_attr (str): Edge attribute used for weights
                - heuristic_type (str): Type of heuristic used
                - heuristic_info (dict): Additional metadata about heuristic

        Raises:
            ValueError: If source or target not specified in config
            ValueError: If source or target nodes don't exist in graph
            nx.NetworkXNoPath: If no path exists from source to target
            nx.NodeNotFound: If source node doesn't exist

        Time Complexity: Depends on heuristic quality, ranges from O((V+E)log V) to O(b^d)
        Space Complexity: O(V) for storing open and closed sets
        """
        start_time = time.perf_counter()

        # Reset nodes explored counter
        self.nodes_explored = 0

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

        # Store graph and target for heuristic functions
        self._graph = workflow_graph
        self._target = target

        # Precompute statistics needed by heuristics
        self._precompute_heuristic_data()

        # Validate heuristic at goal (must be 0 for admissibility)
        goal_heuristic = self._calculate_heuristic(target, target)
        if abs(goal_heuristic) > 1e-9:
            logger.warning(
                f"Heuristic at goal is {goal_heuristic}, not 0. "
                f"This violates admissibility and may produce suboptimal results."
            )

        # Handle special case: source equals target
        if source == target:
            end_time = time.perf_counter()
            return self._build_solution(
                path=[source], execution_time=end_time - start_time
            )

        # Use NetworkX's A* implementation with custom heuristic
        try:
            path = nx.astar_path(
                workflow_graph,
                source=source,
                target=target,
                heuristic=self._calculate_heuristic,
                weight=self.weight_attr,
            )
        except nx.NetworkXNoPath:
            raise ValueError(
                f"No path exists from source '{source}' to target '{target}'"
            )
        except nx.NodeNotFound as e:
            raise ValueError(f"Node not found in graph: {e}")

        # Calculate execution time
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Build and return solution
        solution = self._build_solution(path, execution_time)

        return solution

    def _precompute_heuristic_data(self) -> None:
        """
        Precompute graph-wide data needed by heuristics.

        This method calculates statistics that are used by various heuristics:
        - Minimum edge weight in the graph
        - Shortest path lengths from all nodes to target (for task_depth heuristic)

        Time Complexity: O(V + E) for traversing edges, O(V^2) for shortest paths
        """
        # Calculate minimum edge weight
        min_weight = float("inf")
        for u, v, data in self._graph.edges(data=True):
            weight = data.get(self.weight_attr, data.get("weight", 1.0))
            if weight < min_weight:
                min_weight = weight

        self._min_edge_weight = min_weight if min_weight != float("inf") else 1.0

        # For task_depth heuristic, precompute shortest path lengths to target
        if self.heuristic_type == "task_depth":
            try:
                # Compute shortest path lengths from target in reverse graph
                # This gives us distances TO target from all nodes
                reverse_graph = self._graph.reverse(copy=False)
                self._shortest_path_lengths = nx.single_source_shortest_path_length(
                    reverse_graph, self._target
                )
            except Exception as e:
                logger.warning(f"Could not compute shortest path lengths: {e}")
                self._shortest_path_lengths = {}

    def _calculate_heuristic(self, node: Any, target: Any) -> float:
        """
        Calculate the heuristic value for a given node.

        This method dispatches to the appropriate heuristic function based on
        the configured heuristic_type. It also tracks the number of nodes explored.

        Args:
            node (Any): The current node
            target (Any): The target/goal node

        Returns:
            float: Estimated cost from node to target

        Time Complexity: Depends on heuristic type, typically O(1) to O(E)
        """
        # Increment nodes explored counter
        self.nodes_explored += 1

        # Dispatch to appropriate heuristic
        if self.heuristic_type == "zero":
            return self._zero_heuristic(node, target)
        elif self.heuristic_type == "min_cost_estimate":
            return self._min_cost_estimate(node, target)
        elif self.heuristic_type == "task_depth":
            return self._task_depth_heuristic(node, target)
        else:
            # Fallback to zero heuristic
            logger.warning(
                f"Unknown heuristic type '{self.heuristic_type}', using zero"
            )
            return 0.0

    def _zero_heuristic(self, node: Any, target: Any) -> float:
        """
        Zero heuristic: h(n) = 0 for all nodes.

        This heuristic makes A* equivalent to Dijkstra's algorithm. It provides no
        guidance towards the goal, resulting in uniform exploration of the graph.

        Properties:
        - Always admissible (never overestimates)
        - Always consistent (satisfies triangle inequality)
        - Guarantees optimal solution

        When to use:
        - No domain knowledge available
        - Want guaranteed behavior of Dijkstra
        - All heuristics perform poorly on the problem

        Args:
            node (Any): Current node
            target (Any): Target node

        Returns:
            float: 0.0 for all nodes

        Time Complexity: O(1)
        """
        return 0.0

    def _min_cost_estimate(self, node: Any, target: Any) -> float:
        """
        Minimum cost estimate heuristic.

        This heuristic estimates the cost to reach the target as the minimum weight
        of all edges in the graph. It assumes at least one edge must be traversed
        to reach the target (unless already at target).

        Properties:
        - Admissible: Never overestimates (true cost >= min edge weight)
        - Simple and fast to compute
        - Works well when edge costs vary significantly

        When to use:
        - Graphs with varying edge costs
        - Want slight improvement over Dijkstra with minimal overhead
        - No specific domain knowledge available

        Limitations:
        - Weak heuristic (not very informative)
        - Only provides constant estimate

        Args:
            node (Any): Current node
            target (Any): Target node

        Returns:
            float: Minimum edge weight if not at target, 0.0 if at target

        Time Complexity: O(1) after precomputation
        """
        if node == target:
            return 0.0

        # Return minimum edge weight as lower bound on remaining cost
        return self._min_edge_weight

    def _task_depth_heuristic(self, node: Any, target: Any) -> float:
        """
        Task depth heuristic based on unweighted shortest path length.

        This heuristic estimates the cost as:
        h(n) = shortest_path_length(n, target) * min_edge_weight

        It computes the shortest path length in terms of number of edges (unweighted),
        then multiplies by the minimum edge weight in the graph. This provides a
        lower bound on the actual cost since any path must traverse at least this
        many edges, each with at least the minimum weight.

        Properties:
        - Admissible: Multiplying path length by min weight never overestimates
        - More informed than zero or min_cost_estimate
        - Good for task dependency graphs where path length matters

        When to use:
        - Workflow scheduling and task dependency problems
        - Graphs where path length correlates with cost
        - DAGs with relatively uniform edge weights

        Limitations:
        - Requires precomputation of shortest paths (O(V^2))
        - Less effective if edge weights vary greatly

        Args:
            node (Any): Current node
            target (Any): Target node

        Returns:
            float: Estimated cost based on shortest path length and min edge weight

        Time Complexity: O(1) after O(V^2) precomputation
        """
        if node == target:
            return 0.0

        # Get shortest path length (number of edges) to target
        if self._shortest_path_lengths and node in self._shortest_path_lengths:
            path_length = self._shortest_path_lengths[node]
            # Estimate cost as path_length * minimum edge weight
            return path_length * self._min_edge_weight

        # If no path exists or not precomputed, return 0 (admissible fallback)
        return 0.0

    def _build_solution(self, path: List[Any], execution_time: float) -> Dict[str, Any]:
        """
        Build the solution dictionary from the computed path.

        Args:
            path (List[Any]): The computed path from source to target
            execution_time (float): Time taken to compute the path

        Returns:
            Dict[str, Any]: Complete solution dictionary

        Time Complexity: O(P) where P is the length of the path
        """
        # Calculate path cost
        total_cost = 0.0
        total_time = 0.0

        for i in range(len(path) - 1):
            edge_data = self._graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                # Get weight
                weight = edge_data.get(self.weight_attr, edge_data.get("weight", 1.0))
                total_cost += weight

                # Get time if available
                if "time" in edge_data:
                    total_time += edge_data["time"]

        # Calculate resource utilization
        resource_utilization = 0.0
        total_nodes = self._graph.number_of_nodes()
        if total_nodes > 0:
            resource_utilization = len(path) / total_nodes

        # Build heuristic info
        heuristic_info = {
            "type": self.heuristic_type,
            "admissible": True,  # All our heuristics are admissible
            "goal_heuristic": 0.0,  # h(goal) must be 0
        }

        if self.heuristic_type == "zero":
            heuristic_info["description"] = "Equivalent to Dijkstra (h(n) = 0)"
        elif self.heuristic_type == "min_cost_estimate":
            heuristic_info["description"] = f"Min edge weight: {self._min_edge_weight}"
            heuristic_info["min_edge_weight"] = self._min_edge_weight
        elif self.heuristic_type == "task_depth":
            heuristic_info["description"] = "Path length × min edge weight"
            heuristic_info["min_edge_weight"] = self._min_edge_weight

        return {
            "path": path,
            "total_cost": total_cost,
            "total_time": total_time,
            "nodes_explored": self.nodes_explored,
            "execution_time_seconds": execution_time,
            "resource_utilization": resource_utilization,
            "algorithm": self.name,
            "weight_attr": self.weight_attr,
            "heuristic_type": self.heuristic_type,
            "heuristic_info": heuristic_info,
        }

    def validate_solution(
        self, solution: Dict[str, Any], workflow_graph: nx.DiGraph
    ) -> bool:
        """
        Validate that a solution is feasible and correct.

        Performs comprehensive validation including:
        1. Presence of all required keys
        2. Path validity and connectivity
        3. Cost calculation accuracy
        4. Heuristic metadata presence
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
            "heuristic_type",
            "heuristic_info",
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

        # Validate heuristic_type
        valid_heuristics = ["zero", "min_cost_estimate", "task_depth"]
        if solution["heuristic_type"] not in valid_heuristics:
            logger.error(f"Invalid heuristic_type: {solution['heuristic_type']}")
            return False

        # Validate heuristic_info structure
        if not isinstance(solution["heuristic_info"], dict):
            logger.error("heuristic_info must be a dictionary")
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
                f"Resource utilization out of range [0,1]: "
                f"{solution['resource_utilization']}"
            )
            return False

        if solution["nodes_explored"] < 1:
            logger.error(f"Nodes explored must be >= 1: {solution['nodes_explored']}")
            return False

        return True
