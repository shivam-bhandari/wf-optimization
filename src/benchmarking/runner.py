"""
Comprehensive Benchmark Runner for Workflow Optimization Algorithms

This module provides a robust framework for benchmarking multiple optimization algorithms
across various workflows with comprehensive result collection, timeout handling, and
statistical analysis.

Key Features:
- Multi-algorithm, multi-workflow benchmarking with configurable trials
- Concurrent execution with timeout handling
- Comprehensive metrics collection and aggregation
- Automatic result persistence (CSV and JSON)
- Progress logging and error handling
- Statistical analysis (mean, std, min, max)

Example Usage:
    ```python
    from src.benchmarking.runner import BenchmarkConfig, BenchmarkRunner
    from src.algorithms.dijkstra import DijkstraOptimizer
    from src.datasets.healthcare import HealthcareWorkflowGenerator
    
    # Create workflows
    generator = HealthcareWorkflowGenerator(seed=42)
    workflows = [generator.generate_ehr_extraction() for _ in range(5)]
    
    # Create algorithms
    algorithms = [
        DijkstraOptimizer(source='start', target='end'),
        AStarOptimizer(source='start', target='end')
    ]
    
    # Configure and run benchmarks
    config = BenchmarkConfig(trials_per_combination=3, timeout_seconds=60.0)
    runner = BenchmarkRunner(algorithms, workflows, config)
    results_df = runner.run_benchmarks()
    ```
"""

import logging
import time
import json
import traceback
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from src.algorithms.base import OptimizationAlgorithm


# Configure module logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(message)s"
)


class BenchmarkConfig(BaseModel):
    """
    Configuration for benchmark execution.

    This dataclass defines all parameters controlling benchmark behavior including
    trial counts, timeouts, objectives to optimize, and result persistence settings.

    Attributes:
        trials_per_combination (int): Number of independent trials to run for each
            (workflow, algorithm, objective) combination. Higher values provide
            more robust statistical estimates but increase runtime. Default: 5.

        timeout_seconds (float): Maximum execution time in seconds for a single
            algorithm run. Prevents hanging on difficult problem instances.
            Default: 300.0 (5 minutes).

        random_seed (int): Random seed for reproducibility. Ensures consistent
            workflow generation and algorithm behavior across runs. Default: 42.

        objectives (List[str]): List of optimization objectives to benchmark.
            Common values: ['cost', 'time', 'weight']. Each objective will be
            run separately. Default: ['cost'].

        save_results (bool): Whether to automatically save results to disk.
            If True, saves CSV and JSON files to results_dir. Default: True.

        results_dir (Path): Directory path for saving benchmark results.
            Created automatically if it doesn't exist. Default: Path('results/').

        use_multiprocessing (bool): Whether to use ProcessPoolExecutor for
            parallel execution. If False, uses ThreadPoolExecutor which is
            more compatible with certain environments. Default: True.

    Example:
        ```python
        config = BenchmarkConfig(
            trials_per_combination=10,
            timeout_seconds=120.0,
            objectives=['cost', 'time'],
            save_results=True
        )
        ```
    """

    trials_per_combination: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=300.0, gt=0)
    random_seed: int = Field(default=42)
    objectives: List[str] = Field(default=["cost"])
    save_results: bool = Field(default=True)
    results_dir: Path = Field(default=Path("results/"))
    use_multiprocessing: bool = Field(default=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for workflow optimization algorithms.

    This class orchestrates the execution of multiple algorithms across multiple
    workflows, collecting detailed performance metrics, handling errors gracefully,
    and providing comprehensive statistical analysis of results.

    The runner supports:
    - Parallel execution with configurable timeouts
    - Multiple trials per combination for statistical robustness
    - Multiple optimization objectives
    - Automatic result aggregation and persistence
    - Comprehensive logging and progress tracking
    - Graceful error and timeout handling

    Workflow:
    1. For each workflow, algorithm, and objective:
       - Run N independent trials
       - Execute with timeout protection
       - Collect metrics (path, cost, time, nodes explored, etc.)
    2. Aggregate statistics (mean, std, min, max) per combination
    3. Save results to CSV and JSON with timestamps
    4. Return comprehensive DataFrame with all results

    Attributes:
        algorithms (List[OptimizationAlgorithm]): List of algorithm instances to benchmark
        workflows (List[Tuple[str, nx.DiGraph]]): List of (workflow_id, graph) tuples
        config (BenchmarkConfig): Configuration parameters
        results (List[Dict[str, Any]]): Collected results from all trials
    """

    def __init__(
        self,
        algorithms: List[OptimizationAlgorithm],
        workflows: List[Tuple[str, nx.DiGraph]],
        config: BenchmarkConfig,
    ):
        """
        Initialize the benchmark runner.

        Args:
            algorithms (List[OptimizationAlgorithm]): List of algorithm instances
                to benchmark. Each algorithm should be pre-configured with necessary
                parameters (e.g., source, target nodes).

            workflows (List[Tuple[str, nx.DiGraph]]): List of workflow tuples where
                each tuple contains (workflow_id, workflow_graph). The workflow_id
                should be a unique identifier for the workflow.

            config (BenchmarkConfig): Configuration parameters for benchmark execution.

        Raises:
            ValueError: If algorithms or workflows list is empty.

        Example:
            ```python
            algorithms = [
                DijkstraOptimizer(source='start', target='end'),
                AStarOptimizer(source='start', target='end')
            ]
            workflows = [
                ('ehr_extraction_1', graph1),
                ('ehr_extraction_2', graph2)
            ]
            config = BenchmarkConfig(trials_per_combination=5)
            runner = BenchmarkRunner(algorithms, workflows, config)
            ```
        """
        if not algorithms:
            raise ValueError("At least one algorithm must be provided")
        if not workflows:
            raise ValueError("At least one workflow must be provided")

        self.algorithms = algorithms
        self.workflows = workflows
        self.config = config
        self.results: List[Dict[str, Any]] = []

        # Ensure results directory exists
        if self.config.save_results:
            self.config.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized BenchmarkRunner with {len(algorithms)} algorithms, "
            f"{len(workflows)} workflows, {len(config.objectives)} objectives, "
            f"and {config.trials_per_combination} trials per combination"
        )
        logger.info(
            f"Total benchmark runs: {len(algorithms) * len(workflows) * len(config.objectives) * config.trials_per_combination}"
        )

    def run_benchmarks(self) -> pd.DataFrame:
        """
        Execute all benchmarks and return comprehensive results.

        This method orchestrates the complete benchmarking process:
        1. Iterates through all combinations of workflows, algorithms, and objectives
        2. Runs N trials for each combination
        3. Executes each trial with timeout protection
        4. Collects detailed metrics from each run
        5. Handles failures and timeouts gracefully
        6. Aggregates statistics across trials
        7. Saves results to disk (if configured)
        8. Returns comprehensive DataFrame

        The method provides detailed progress logging showing:
        - Current algorithm and workflow being processed
        - Progress counters (e.g., "Running algorithm 2/4 on workflow 3/10")
        - Timeout and error notifications
        - Aggregate statistics upon completion

        Returns:
            pd.DataFrame: Comprehensive results DataFrame with columns:
                - workflow_id (str): Workflow identifier
                - algorithm_name (str): Algorithm name
                - objective (str): Optimization objective
                - trial_number (int): Trial index (0-based)
                - path (List): Solution path (node IDs)
                - total_cost (float): Total cost/weight of solution
                - total_time_ms (float): Total time in milliseconds
                - execution_time_seconds (float): Algorithm execution time
                - nodes_explored (int): Number of nodes in solution path
                - success (bool): Whether the run completed successfully
                - error_message (str): Error description (if failed)
                - timestamp (str): ISO format timestamp of execution

        Raises:
            Exception: Critical errors are logged but execution continues for
                      remaining combinations.

        Example:
            ```python
            runner = BenchmarkRunner(algorithms, workflows, config)
            results_df = runner.run_benchmarks()

            # Analyze results
            print(results_df.groupby(['algorithm_name', 'workflow_id'])['total_cost'].mean())
            ```
        """
        logger.info("=" * 80)
        logger.info("Starting benchmark execution")
        logger.info("=" * 80)

        start_time = time.time()
        self.results = []

        total_workflows = len(self.workflows)
        total_algorithms = len(self.algorithms)
        total_objectives = len(self.config.objectives)

        # Iterate through all combinations
        for workflow_idx, (workflow_id, workflow_graph) in enumerate(self.workflows, 1):
            logger.info(f"\n{'='*80}")
            logger.info(
                f"Processing Workflow {workflow_idx}/{total_workflows}: {workflow_id}"
            )
            logger.info(
                f"  Nodes: {workflow_graph.number_of_nodes()}, Edges: {workflow_graph.number_of_edges()}"
            )
            logger.info(f"{'='*80}")

            for algo_idx, algorithm in enumerate(self.algorithms, 1):
                logger.info(
                    f"\n  Algorithm {algo_idx}/{total_algorithms}: {algorithm.name}"
                )

                for obj_idx, objective in enumerate(self.config.objectives, 1):
                    logger.info(
                        f"    Objective {obj_idx}/{total_objectives}: {objective}"
                    )

                    # Run N trials for this combination
                    for trial in range(self.config.trials_per_combination):
                        logger.info(
                            f"      Trial {trial + 1}/{self.config.trials_per_combination} "
                            f"[Progress: W{workflow_idx}/{total_workflows}, "
                            f"A{algo_idx}/{total_algorithms}, "
                            f"O{obj_idx}/{total_objectives}]"
                        )

                        # Execute with timeout
                        result = self._execute_with_timeout(
                            algorithm=algorithm,
                            workflow_graph=workflow_graph,
                            workflow_id=workflow_id,
                            objective=objective,
                            trial_number=trial,
                        )

                        self.results.append(result)

                        # Log trial result
                        if result["success"]:
                            logger.info(
                                f"        ✓ Success: cost={result['total_cost']:.2f}, "
                                f"time={result['execution_time_seconds']:.4f}s, "
                                f"nodes={result['nodes_explored']}"
                            )
                        else:
                            logger.warning(
                                f"        ✗ Failed: {result['error_message']}"
                            )

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Calculate aggregate statistics
        logger.info(f"\n{'='*80}")
        logger.info("Calculating aggregate statistics...")
        logger.info(f"{'='*80}")

        agg_df = self._aggregate_statistics(df)

        # Save results
        if self.config.save_results:
            self._save_results(df, agg_df)

        # Print summary
        elapsed_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info("Benchmark execution completed!")
        logger.info(f"{'='*80}")
        logger.info(f"Total runs: {len(self.results)}")
        logger.info(f"Successful runs: {df['success'].sum()}")
        logger.info(f"Failed runs: {(~df['success']).sum()}")
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to: results/{Path(self.config.results_dir).name}/")
        logger.info(f"{'='*80}\n")

        return df

    def export_web_results(self, results_df: pd.DataFrame, output_path: str) -> None:
        """
        Export benchmark results in a JSON format optimized for the web dashboard.
        Adds a 'workflows' section with full node/edge data for visualization (vis-network compatible),
        including a default demo workflow for immediate access/demo.
        """
        import numpy as np
        from datetime import datetime
        import json
        df = results_df.copy()
        # Avoid NaN issues for downstream
        df = df.replace({np.nan: None})

        # Metadata
        generated_at = datetime.now().isoformat(timespec="seconds")
        total_benchmarks = len(df)
        domains = sorted(df["workflow_id"].apply(lambda x: x.split("_")[0] if isinstance(x, str) and "_" in x else "other").unique())
        workflows = df["workflow_id"].nunique()
        algorithms_tested = sorted(df["algorithm_name"].unique())
        success_rate = round(df["success"].mean() * 100 if not df.empty else 0, 2)

        # Main algorithms stats
        algo_stats = []
        for algo in algorithms_tested:
            subset = df[df["algorithm_name"] == algo]
            display_name = (
                algo.replace("_optimizer", "").replace("dag_dynamic_programming", "DAG-DP").replace("bellman_ford_optimizer","Bellman-Ford").replace("astar", "A*").replace("dijkstra","Dijkstra").replace("_", " ").title().replace("A* ", "A*")
            )
            avg_execution_time = round(subset["execution_time_seconds"].mean() or 0, 4)
            avg_cost = round(subset["total_cost"].mean() or 0, 2)
            avg_nodes_explored = round(subset["nodes_explored"].mean() or 0, 2)
            success = round(subset["success"].mean() * 100 if len(subset) else 0, 2)
            total_runs = subset.shape[0]
            std_execution_time = round(subset["execution_time_seconds"].std() or 0, 5)
            d = dict(
                name=algo,
                display_name=display_name,
                avg_execution_time=avg_execution_time,
                avg_cost=avg_cost,
                avg_nodes_explored=avg_nodes_explored,
                success_rate=success,
                total_runs=total_runs,
                std_execution_time=std_execution_time
            )
            algo_stats.append(d)

        # Best algorithm (lowest avg_execution_time)
        best_algo = min(algo_stats, key=lambda x: (x["avg_execution_time"] if x["avg_execution_time"] is not None else float("inf")), default=None)
        all_optimal = all(a["avg_cost"] == algo_stats[0]["avg_cost"] for a in algo_stats) if algo_stats else True
        best_algorithm = best_algo["name"] if best_algo else None
        best_avg_time = best_algo["avg_execution_time"] if best_algo else None

        recommendation = f"Use {best_algo['display_name']} for production ({round(algo_stats[0]['avg_execution_time']/best_avg_time,1) if best_algo and best_avg_time else ''}× faster than alternatives)" if best_algo else "Review benchmark results for more information."

        # Group by workflow
        results_by_workflow = []
        workflows_group = df.groupby("workflow_id")
        for wf_id, wf_df in workflows_group:
            domain = wf_id.split("_")[0] if "_" in wf_id else "other"
            workflow_type = wf_id.split("_",1)[1] if "_" in wf_id else wf_id
            nodes = int(wf_df["nodes_explored"].mean() or 0) if "nodes_explored" in wf_df.columns else None
            results = []
            for algo in algorithms_tested:
                a_df = wf_df[wf_df["algorithm_name"] == algo]
                if not a_df.empty:
                    results.append(dict(
                        algorithm=algo,
                        avg_time=round(a_df["execution_time_seconds"].mean() or 0, 5),
                        avg_cost=round(a_df["total_cost"].mean() or 0, 2),
                        trials=int(a_df.shape[0])
                    ))
            results_by_workflow.append(dict(
                workflow_id=wf_id,
                domain=domain,
                workflow_type=workflow_type,
                nodes=nodes,
                results=results
            ))

        # Visualizations static metadata
        visualizations = [
            dict(filename="algorithm_comparison_time.png", title="Execution Time Comparison", description="Average execution time across all workflows"),
            dict(filename="algorithm_comparison_cost.png", title="Solution Cost Comparison", description="Total cost of solutions found by each algorithm"),
            dict(filename="scalability.png", title="Scalability Analysis", description="Performance across different workflow sizes"),
            dict(filename="cost_comparison.png", title="Cost by Domain", description="Cost comparison across Healthcare, Finance, and Legal domains"),
        ]

        # ---- WORKFLOWS SECTION ---- #
        def nx_to_vis(workflow_id, domain, g):
            """Convert DiGraph to vis-network format for nodes/edges and metadata."""
            # Node label/attr handling
            nodes = []
            for node, data in g.nodes(data=True):
                label = data.get('label', str(node).replace('_', ' ').title())
                task_type = data.get('task_type', 'task')
                exec_time_ms = data.get('execution_time_ms', data.get('exec_time_ms', 0))
                cost = data.get('cost_units', data.get('cost', 0))
                resource_req = data.get('resource_requirements', {})
                is_start = data.get('is_start', node == 'start')
                is_end = data.get('is_end', node == 'end')
                nodes.append({
                    'id': node,
                    'label': label,
                    'task_type': task_type,
                    'execution_time_ms': exec_time_ms,
                    'cost_units': cost,
                    'is_start': is_start,
                    'is_end': is_end,
                    'resource_requirements': resource_req
                })
            edges = []
            for u, v, data in g.edges(data=True):
                edges.append({
                    'from': u,
                    'to': v,
                    'transition_cost': data.get('transition_cost', data.get('weight', 0)),
                    'data_transfer_time_ms': data.get('data_transfer_time_ms', 0),
                    'label': data.get('label', ''),
                    'dashed': data.get('conditional', False)
                })
            # Compute metadata
            metadata = {
                'total_nodes': g.number_of_nodes(),
                'total_edges': g.number_of_edges(),
                'estimated_total_cost': float(sum([d.get('cost_units', d.get('cost', 0)) for _, d in g.nodes(data=True)])),
                'estimated_total_time_ms': float(sum([d.get('execution_time_ms', d.get('exec_time_ms', 0)) for _, d in g.nodes(data=True)])),
            }
            wf_type = None
            if '_' in workflow_id:
                wf_type = workflow_id.split('_', 1)[1]
            return {
                'workflow_id': workflow_id,
                'domain': domain,
                'type': wf_type or str(workflow_id),
                'nodes': nodes,
                'edges': edges,
                'metadata': metadata,
            }

        # Demo workflow (always included first)
        demo_nodes = [
            {'id': 'start', 'label': 'Start', 'task_type': 'start', 'execution_time_ms': 0, 'cost_units': 0, 'is_start': True, 'is_end': False, 'resource_requirements': {}},
            {'id': 'extract', 'label': 'Extract', 'task_type': 'extract', 'execution_time_ms': 500, 'cost_units': 0.12, 'is_start': False, 'is_end': False, 'resource_requirements': {'cpu_cores': 2}},
            {'id': 'validate', 'label': 'Validate', 'task_type': 'validate', 'execution_time_ms': 700, 'cost_units': 0.21, 'is_start': False, 'is_end': False, 'resource_requirements': {'cpu_cores': 2, 'memory_gb': 2}},
            {'id': 'map', 'label': 'Map', 'task_type': 'transform', 'execution_time_ms': 900, 'cost_units': 0.35, 'is_start': False, 'is_end': False, 'resource_requirements': {'cpu_cores': 1, 'memory_gb': 1}},
            {'id': 'upload', 'label': 'Upload', 'task_type': 'output', 'execution_time_ms': 600, 'cost_units': 0.16, 'is_start': False, 'is_end': False, 'resource_requirements': {'cpu_cores': 1}},
            {'id': 'end', 'label': 'End', 'task_type': 'end', 'execution_time_ms': 0, 'cost_units': 0, 'is_start': False, 'is_end': True, 'resource_requirements': {}},
        ]
        demo_edges = [
            {'from': 'start', 'to': 'extract', 'transition_cost': 0.05, 'data_transfer_time_ms': 50, 'label': '', 'dashed': False},
            {'from': 'extract', 'to': 'validate', 'transition_cost': 0.07, 'data_transfer_time_ms': 100, 'label': '', 'dashed': False},
            {'from': 'validate', 'to': 'map', 'transition_cost': 0.1, 'data_transfer_time_ms': 70, 'label': '', 'dashed': True},
            {'from': 'map', 'to': 'upload', 'transition_cost': 0.13, 'data_transfer_time_ms': 120, 'label': '', 'dashed': False},
            {'from': 'upload', 'to': 'end', 'transition_cost': 0.08, 'data_transfer_time_ms': 30, 'label': '', 'dashed': False},
        ]
        demo_metadata = {
            'total_nodes': 6,
            'total_edges': 5,
            'estimated_total_cost': sum(n['cost_units'] for n in demo_nodes),
            'estimated_total_time_ms': sum(n['execution_time_ms'] for n in demo_nodes),
        }
        workflows_out = [
            {
                'workflow_id': 'demo_etl_sample',
                'domain': 'demo',
                'type': 'extract_transform_load',
                'nodes': demo_nodes,
                'edges': demo_edges,
                'metadata': demo_metadata,
            }
        ]

        # Iterate through all actual workflows from self.workflows and add to workflows_out
        if hasattr(self, 'workflows'):
            for workflow_id, workflow_graph in self.workflows:
                # Try to guess domain from ID
                domain = str(workflow_id).split('_')[0] if '_' in str(workflow_id) else 'other'
                workflows_out.append(nx_to_vis(str(workflow_id), domain, workflow_graph))

        # Compose final dict
        web_result = {
            "metadata": {
                "generated_at": generated_at,
                "total_benchmarks": total_benchmarks,
                "success_rate": success_rate,
                "algorithms_tested": algorithms_tested,
                "workflows_tested": workflows,
                "domains": domains,
            },
            "summary": {
                "best_algorithm": best_algorithm,
                "best_avg_time": best_avg_time,
                "all_optimal": all_optimal,
                "recommendation": recommendation,
            },
            "algorithms": algo_stats,
            "results_by_workflow": results_by_workflow,
            "visualizations": visualizations,
            "workflows": workflows_out
        }
        # Save as pretty JSON
        with open(output_path, 'w') as f:
            json.dump(web_result, f, indent=2, default=str)
        logger.info(f"Exported web dashboard results to: {output_path}")

    def _execute_with_timeout(
        self,
        algorithm: OptimizationAlgorithm,
        workflow_graph: nx.DiGraph,
        workflow_id: str,
        objective: str,
        trial_number: int,
    ) -> Dict[str, Any]:
        """
        Execute a single algorithm run with timeout protection.

        This method wraps algorithm execution with comprehensive error handling
        and timeout protection using ProcessPoolExecutor. It ensures that:
        - Long-running algorithms are terminated after timeout
        - Algorithm exceptions are caught and logged
        - Consistent result format is maintained even for failures
        - Detailed error messages are captured

        The method updates the algorithm's weight attribute to match the objective,
        executes the solve method, and collects comprehensive metrics.

        Args:
            algorithm (OptimizationAlgorithm): Algorithm instance to execute
            workflow_graph (nx.DiGraph): Workflow graph to optimize
            workflow_id (str): Unique workflow identifier
            objective (str): Optimization objective (e.g., 'cost', 'time')
            trial_number (int): Trial index (0-based)

        Returns:
            Dict[str, Any]: Result dictionary containing:
                - workflow_id (str): Workflow identifier
                - algorithm_name (str): Algorithm name
                - objective (str): Optimization objective
                - trial_number (int): Trial index
                - path (List): Solution path (empty if failed)
                - total_cost (float): Total cost (NaN if failed)
                - total_time_ms (float): Total time in ms (NaN if failed)
                - execution_time_seconds (float): Execution time (NaN if failed)
                - nodes_explored (int): Nodes in path (0 if failed)
                - success (bool): Whether execution succeeded
                - error_message (str): Error description (empty if success)
                - timestamp (str): ISO format timestamp

        Note:
            Timeout errors and algorithm exceptions are logged as warnings,
            not errors, to allow benchmarking to continue for other combinations.
        """
        result = {
            "workflow_id": workflow_id,
            "algorithm_name": algorithm.name,
            "objective": objective,
            "trial_number": trial_number,
            "path": [],
            "total_cost": float("nan"),
            "total_time_ms": float("nan"),
            "execution_time_seconds": float("nan"),
            "nodes_explored": 0,
            "success": False,
            "error_message": "",
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Update algorithm weight attribute for this objective
            if hasattr(algorithm, "weight_attr"):
                algorithm.weight_attr = objective

            # Execute algorithm with timeout
            start_time = time.time()

            # Choose executor based on configuration
            executor_class = (
                ProcessPoolExecutor
                if self.config.use_multiprocessing
                else ThreadPoolExecutor
            )

            # Use executor for timeout
            try:
                with executor_class(max_workers=1) as executor:
                    future = executor.submit(
                        _run_algorithm_subprocess, algorithm, workflow_graph
                    )

                    try:
                        solution = future.result(timeout=self.config.timeout_seconds)
                    except FuturesTimeoutError:
                        logger.warning(
                            f"        ⏱ Timeout after {self.config.timeout_seconds}s "
                            f"for {algorithm.name} on {workflow_id}"
                        )
                        result[
                            "error_message"
                        ] = f"Timeout after {self.config.timeout_seconds}s"
                        return result
            except (OSError, PermissionError) as e:
                # Fallback to direct execution if multiprocessing fails
                logger.debug(
                    f"Multiprocessing failed ({e}), falling back to direct execution"
                )
                solution = _run_algorithm_subprocess(algorithm, workflow_graph)

            elapsed_time = time.time() - start_time

            # Extract metrics from solution
            result.update(
                {
                    "path": solution.get("path", []),
                    "total_cost": solution.get("total_cost", float("nan")),
                    "total_time_ms": solution.get("total_time", 0.0)
                    * 1000,  # Convert to ms
                    "execution_time_seconds": solution.get(
                        "execution_time_seconds", elapsed_time
                    ),
                    "nodes_explored": solution.get(
                        "nodes_explored", len(solution.get("path", []))
                    ),
                    "success": True,
                    "error_message": "",
                }
            )

        except Exception as e:
            logger.warning(
                f"        ✗ Error executing {algorithm.name} on {workflow_id}: {str(e)}"
            )
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            result["error_message"] = str(e)

        return result

    def _aggregate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregate statistics grouped by workflow and algorithm.

        This method computes comprehensive statistical summaries for each
        (workflow_id, algorithm_name, objective) combination, including:
        - Mean, standard deviation, minimum, and maximum for key metrics
        - Success rate (percentage of successful runs)
        - Sample size (number of trials)

        Statistics are calculated only for successful runs. Failed runs are
        excluded from metric calculations but counted in the success rate.

        Args:
            df (pd.DataFrame): Raw results DataFrame from all trials

        Returns:
            pd.DataFrame: Aggregated statistics DataFrame with columns:
                - workflow_id (str): Workflow identifier
                - algorithm_name (str): Algorithm name
                - objective (str): Optimization objective
                - num_trials (int): Total number of trials
                - num_successful (int): Number of successful trials
                - success_rate (float): Success rate (0.0 to 1.0)
                - mean_cost (float): Mean total cost
                - std_cost (float): Standard deviation of cost
                - min_cost (float): Minimum cost
                - max_cost (float): Maximum cost
                - mean_execution_time (float): Mean execution time in seconds
                - std_execution_time (float): Std dev of execution time
                - min_execution_time (float): Minimum execution time
                - max_execution_time (float): Maximum execution time
                - mean_nodes_explored (float): Mean number of nodes explored
                - std_nodes_explored (float): Std dev of nodes explored

        Example:
            ```python
            agg_df = runner._aggregate_statistics(results_df)
            print(agg_df[['algorithm_name', 'workflow_id', 'mean_cost', 'std_cost']])
            ```
        """
        if df.empty:
            logger.warning("No results to aggregate")
            return pd.DataFrame()

        # Filter successful runs for metric calculations
        successful_df = df[df["success"]].copy()

        if successful_df.empty:
            logger.warning("No successful runs to aggregate")
            return pd.DataFrame()

        # Group by workflow, algorithm, and objective
        groupby_cols = ["workflow_id", "algorithm_name", "objective"]

        # Calculate aggregate statistics
        agg_stats = (
            successful_df.groupby(groupby_cols)
            .agg(
                {
                    "total_cost": ["mean", "std", "min", "max"],
                    "execution_time_seconds": ["mean", "std", "min", "max"],
                    "nodes_explored": ["mean", "std", "min", "max"],
                    "success": "count",  # Count of successful runs
                }
            )
            .reset_index()
        )

        # Flatten column names
        agg_stats.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in agg_stats.columns.values
        ]

        # Rename columns for clarity
        agg_stats.rename(
            columns={
                "total_cost_mean": "mean_cost",
                "total_cost_std": "std_cost",
                "total_cost_min": "min_cost",
                "total_cost_max": "max_cost",
                "execution_time_seconds_mean": "mean_execution_time",
                "execution_time_seconds_std": "std_execution_time",
                "execution_time_seconds_min": "min_execution_time",
                "execution_time_seconds_max": "max_execution_time",
                "nodes_explored_mean": "mean_nodes_explored",
                "nodes_explored_std": "std_nodes_explored",
                "nodes_explored_min": "min_nodes_explored",
                "nodes_explored_max": "max_nodes_explored",
                "success_count": "num_successful",
            },
            inplace=True,
        )

        # Calculate total trials and success rate
        total_trials = df.groupby(groupby_cols).size().reset_index(name="num_trials")
        agg_stats = agg_stats.merge(total_trials, on=groupby_cols)
        agg_stats["success_rate"] = (
            agg_stats["num_successful"] / agg_stats["num_trials"]
        )

        # Log aggregate statistics
        logger.info("\nAggregate Statistics Summary:")
        logger.info("-" * 80)
        for _, row in agg_stats.iterrows():
            logger.info(
                f"{row['algorithm_name']} on {row['workflow_id']} ({row['objective']}): "
                f"cost={row['mean_cost']:.2f}±{row.get('std_cost', 0):.2f}, "
                f"time={row['mean_execution_time']:.4f}±{row.get('std_execution_time', 0):.4f}s, "
                f"success_rate={row['success_rate']:.1%}"
            )

        return agg_stats

    def _save_results(self, results_df: pd.DataFrame, agg_df: pd.DataFrame) -> None:
        """
        Save benchmark results to CSV and JSON files.

        This method persists both raw trial results and aggregated statistics
        to disk for later analysis. Files are saved with timestamps to prevent
        overwriting previous results.

        Files created:
        - benchmark_results_<timestamp>.csv: Raw trial results
        - benchmark_results_<timestamp>.json: Raw trial results (JSON format)
        - benchmark_aggregates_<timestamp>.csv: Aggregate statistics
        - benchmark_aggregates_<timestamp>.json: Aggregate statistics (JSON format)

        The JSON format converts path lists to strings for better readability
        and compatibility with JSON parsers.

        Args:
            results_df (pd.DataFrame): Raw results DataFrame
            agg_df (pd.DataFrame): Aggregated statistics DataFrame

        Returns:
            None

        Side Effects:
            Creates/writes files to self.config.results_dir

        Example File Names:
            - benchmark_results_20250128_143022.csv
            - benchmark_aggregates_20250128_143022.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        results_csv_path = (
            self.config.results_dir / f"benchmark_results_{timestamp}.csv"
        )
        results_json_path = (
            self.config.results_dir / f"benchmark_results_{timestamp}.json"
        )

        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Saved raw results to: {results_csv_path.name}")

        # Convert path lists to strings for JSON serialization
        results_json = results_df.copy()
        results_json["path"] = results_json["path"].apply(
            lambda x: str(x) if isinstance(x, list) else x
        )
        results_json.to_json(results_json_path, orient="records", indent=2)
        logger.info(f"Saved raw results to: {results_json_path.name}")

        # Save aggregate statistics
        if not agg_df.empty:
            agg_csv_path = (
                self.config.results_dir / f"benchmark_aggregates_{timestamp}.csv"
            )
            agg_json_path = (
                self.config.results_dir / f"benchmark_aggregates_{timestamp}.json"
            )

            agg_df.to_csv(agg_csv_path, index=False)
            logger.info(f"Saved aggregate statistics to: {agg_csv_path.name}")

            agg_df.to_json(agg_json_path, orient="records", indent=2)
            logger.info(f"Saved aggregate statistics to: {agg_json_path.name}")

        # Also export web results
        web_output_path = str(self.config.results_dir / "web_results.json")
        self.export_web_results(results_df, web_output_path)


def _run_algorithm_subprocess(
    algorithm: OptimizationAlgorithm, workflow_graph: nx.DiGraph
) -> Dict[str, Any]:
    """
    Helper function to run algorithm in a subprocess for timeout support.

    This function is executed in a separate process by ProcessPoolExecutor,
    allowing the main process to terminate it if it exceeds the timeout.

    Args:
        algorithm (OptimizationAlgorithm): Algorithm instance to execute
        workflow_graph (nx.DiGraph): Workflow graph to optimize

    Returns:
        Dict[str, Any]: Solution dictionary from algorithm.solve()

    Raises:
        Any exception raised by algorithm.solve()
    """
    return algorithm.solve(workflow_graph)
