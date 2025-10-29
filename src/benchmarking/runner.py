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
