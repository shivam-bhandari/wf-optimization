"""
Command-Line Interface for Workflow Optimization Benchmarking

This module provides a comprehensive CLI for running benchmarks, analyzing results,
and generating workflows using the Click framework.

Usage:
    python -m src.cli benchmark run --algorithms dijkstra,astar --trials 5
    python -m src.cli benchmark analyze --results-file results/benchmark_results.csv
    python -m src.cli benchmark workflows --domain healthcare --count 3
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd
import networkx as nx

from src.algorithms import (
    DAGDynamicProgramming,
    DijkstraOptimizer,
    AStarOptimizer,
    BellmanFordOptimizer,
)
from src.benchmarking.runner import BenchmarkConfig, BenchmarkRunner
from src.datasets.healthcare import HealthcareWorkflowGenerator
from src.datasets.finance import FinancialWorkflowGenerator
from src.datasets.legal import LegalWorkflowGenerator
from src.evaluation.visualizations import (
    plot_algorithm_comparison,
    plot_execution_time_scalability,
    plot_cost_comparison,
)
from src.reporting.report_generator import ReportGenerator


# Version constant
VERSION = "0.1.0"

# Algorithm mapping
ALGORITHM_MAP = {
    "dag_dp": ("DAG-DP", DAGDynamicProgramming),
    "dijkstra": ("Dijkstra", DijkstraOptimizer),
    "astar": ("A*", AStarOptimizer),
    "bellman_ford": ("Bellman-Ford", BellmanFordOptimizer),
}

# Domain mapping
DOMAIN_MAP = {
    "healthcare": HealthcareWorkflowGenerator,
    "finance": FinancialWorkflowGenerator,
    "legal": LegalWorkflowGenerator,
}

# Workflow type mapping per domain
WORKFLOW_TYPES = {
    "healthcare": [
        "medical_record_extraction",
        "insurance_claim_processing",
        "patient_intake",
    ],
    "finance": ["loan_approval", "fraud_detection", "risk_assessment"],
    "legal": ["contract_review", "compliance_check", "document_redlining"],
}


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(levelname)s - %(message)s"
    )


@click.group()
@click.option("--version", is_flag=True, help="Show version and exit.")
@click.option("--verbose", is_flag=True, help="Enable verbose output (DEBUG logging).")
@click.pass_context
def benchmark(ctx, version, verbose):
    """
    Workflow Optimization Benchmarking CLI

    A comprehensive command-line interface for benchmarking workflow optimization
    algorithms, analyzing results, and generating sample workflows across multiple
    domains (healthcare, finance, legal).

    \b
    Examples:
        # Run benchmarks with specific algorithms
        benchmark run --algorithms dijkstra,astar --trials 5

        # Analyze existing results
        benchmark analyze --results-file results/benchmark_results_20250128_143022.csv

        # Generate workflows
        benchmark workflows --domain healthcare --count 5
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    setup_logging(verbose)

    if version:
        click.echo(f"Benchmark CLI version {VERSION}")
        ctx.exit()


@benchmark.command()
@click.option(
    "--algorithms",
    type=str,
    help="Comma-separated list of algorithms (dijkstra,astar,dag_dp,bellman_ford). "
    "If not specified, runs all 4 algorithms.",
)
@click.option(
    "--domains",
    type=str,
    help="Comma-separated list of domains (healthcare,finance,legal). "
    "If not specified, uses all 3 domains.",
)
@click.option(
    "--trials",
    type=int,
    default=3,
    help="Number of trials per workflow-algorithm combination (default: 3).",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="Timeout in seconds for each algorithm run (default: 300).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/",
    help="Output directory for results (default: results/).",
)
@click.pass_context
def run(ctx, algorithms, domains, trials, timeout, output_dir):
    """
    Run a complete benchmark suite.

    This command executes a comprehensive benchmark across specified algorithms
    and domains, running multiple trials per combination and generating detailed
    performance metrics.

    \b
    The benchmark process:
    1. Generates 3 workflows per domain (one of each workflow type)
    2. Runs each algorithm on each workflow
    3. Executes multiple trials for statistical robustness
    4. Collects metrics (execution time, cost, path quality)
    5. Saves results to CSV and JSON files with timestamps
    6. Displays summary statistics

    \b
    Examples:
        # Run all algorithms on all domains with 5 trials
        benchmark run --trials 5

        # Run specific algorithms on healthcare domain
        benchmark run --algorithms dijkstra,astar --domains healthcare --trials 3

        # Run with custom timeout and output directory
        benchmark run --timeout 600 --output-dir my_results/ --trials 10

    \b
    Algorithm Options:
        - dag_dp: DAG Dynamic Programming (optimal for DAGs)
        - dijkstra: Dijkstra's algorithm (optimal for non-negative weights)
        - astar: A* search (optimal with admissible heuristic)
        - bellman_ford: Bellman-Ford algorithm (handles negative weights)

    \b
    Domain Options:
        - healthcare: Medical record extraction, insurance claims, patient intake
        - finance: Loan approval, fraud detection, risk assessment
        - legal: Contract review, compliance checking, document redlining
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo(click.style("\n" + "=" * 80, fg="cyan"))
    click.echo(
        click.style("  Workflow Optimization Benchmark Runner", fg="cyan", bold=True)
    )
    click.echo(click.style("=" * 80 + "\n", fg="cyan"))

    # Parse algorithm selection
    if algorithms:
        algo_keys = [a.strip().lower() for a in algorithms.split(",")]
        invalid_algos = [a for a in algo_keys if a not in ALGORITHM_MAP]
        if invalid_algos:
            click.echo(
                click.style(
                    f"✗ Error: Invalid algorithms: {', '.join(invalid_algos)}",
                    fg="red",
                    bold=True,
                )
            )
            click.echo(f"Valid options: {', '.join(ALGORITHM_MAP.keys())}")
            ctx.exit(1)
    else:
        algo_keys = list(ALGORITHM_MAP.keys())

    # Parse domain selection
    if domains:
        domain_keys = [d.strip().lower() for d in domains.split(",")]
        invalid_domains = [d for d in domain_keys if d not in DOMAIN_MAP]
        if invalid_domains:
            click.echo(
                click.style(
                    f"✗ Error: Invalid domains: {', '.join(invalid_domains)}",
                    fg="red",
                    bold=True,
                )
            )
            click.echo(f"Valid options: {', '.join(DOMAIN_MAP.keys())}")
            ctx.exit(1)
    else:
        domain_keys = list(DOMAIN_MAP.keys())

    # Display configuration
    click.echo(click.style("Configuration:", fg="yellow", bold=True))
    click.echo(f"  Algorithms: {click.style(', '.join(algo_keys), fg='cyan')}")
    click.echo(f"  Domains: {click.style(', '.join(domain_keys), fg='cyan')}")
    click.echo(f"  Trials per combination: {click.style(str(trials), fg='cyan')}")
    click.echo(f"  Timeout: {click.style(f'{timeout}s', fg='cyan')}")
    click.echo(f"  Output directory: {click.style(output_dir, fg='cyan')}")
    click.echo()

    try:
        # Generate workflows
        click.echo(click.style("Generating workflows...", fg="yellow"))
        workflows = []

        for domain_key in domain_keys:
            generator_class = DOMAIN_MAP[domain_key]
            generator = generator_class(seed=42)

            # Generate one workflow of each type for this domain
            for workflow_type in WORKFLOW_TYPES[domain_key]:
                if workflow_type == "medical_record_extraction":
                    graph = generator.generate_medical_record_extraction()
                elif workflow_type == "insurance_claim_processing":
                    graph = generator.generate_insurance_claim_processing()
                elif workflow_type == "patient_intake":
                    graph = generator.generate_patient_intake_workflow()
                elif workflow_type == "loan_approval":
                    graph = generator.generate_loan_approval()
                elif workflow_type == "fraud_detection":
                    graph = generator.generate_fraud_detection()
                elif workflow_type == "risk_assessment":
                    graph = generator.generate_risk_assessment()
                elif workflow_type == "contract_review":
                    graph = generator.generate_contract_review()
                elif workflow_type == "compliance_check":
                    graph = generator.generate_compliance_check()
                elif workflow_type == "document_redlining":
                    graph = generator.generate_document_redlining()

                workflow_id = f"{domain_key}_{workflow_type}"
                workflows.append((workflow_id, graph))
                click.echo(
                    f"  ✓ Generated: {click.style(workflow_id, fg='green')} "
                    f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
                )

        click.echo(
            click.style(
                f"\n✓ Generated {len(workflows)} workflows", fg="green", bold=True
            )
        )
        click.echo()

        # Create algorithm instances
        click.echo(click.style("Initializing algorithms...", fg="yellow"))
        algorithm_instances = []

        for algo_key in algo_keys:
            algo_name, algo_class = ALGORITHM_MAP[algo_key]
            algo_instance = algo_class(
                source="start", target="end", weight_attr="cost_units"
            )
            algorithm_instances.append(algo_instance)
            click.echo(f"  ✓ Initialized: {click.style(algo_name, fg='green')}")

        click.echo(
            click.style(
                f"\n✓ Initialized {len(algorithm_instances)} algorithms",
                fg="green",
                bold=True,
            )
        )
        click.echo()

        # Create benchmark configuration
        config = BenchmarkConfig(
            trials_per_combination=trials,
            timeout_seconds=float(timeout),
            objectives=["cost_units"],
            save_results=True,
            results_dir=Path(output_dir),
            use_multiprocessing=True,
        )

        # Run benchmarks
        click.echo(click.style("=" * 80, fg="cyan"))
        click.echo(
            click.style("Starting benchmark execution...", fg="yellow", bold=True)
        )
        click.echo(click.style("=" * 80 + "\n", fg="cyan"))

        runner = BenchmarkRunner(algorithm_instances, workflows, config)
        results_df = runner.run_benchmarks()

        # Display summary
        click.echo()
        click.echo(click.style("=" * 80, fg="cyan"))
        click.echo(click.style("Benchmark Summary", fg="cyan", bold=True))
        click.echo(click.style("=" * 80 + "\n", fg="cyan"))

        # Calculate summary statistics
        successful_df = results_df[results_df["success"]]

        if not successful_df.empty:
            summary = (
                successful_df.groupby("algorithm_name")
                .agg(
                    {
                        "execution_time_seconds": "mean",
                        "total_cost": "mean",
                        "success": "count",
                    }
                )
                .round(4)
            )

            click.echo(click.style("Algorithm Performance:", fg="yellow", bold=True))
            click.echo()

            # Header
            click.echo(
                f"{'Algorithm':<20} {'Avg Time (s)':<15} {'Avg Cost':<15} {'Runs':<10}"
            )
            click.echo("-" * 60)

            # Data rows
            for algo_name, row in summary.iterrows():
                time_str = f"{row['execution_time_seconds']:.4f}"
                cost_str = f"{row['total_cost']:.2f}"
                runs_str = f"{int(row['success'])}"

                click.echo(
                    f"{click.style(algo_name, fg='cyan'):<29} "
                    f"{click.style(time_str, fg='green'):<24} "
                    f"{click.style(cost_str, fg='green'):<24} "
                    f"{runs_str:<10}"
                )

            click.echo()

            # Overall statistics
            total_runs = len(results_df)
            successful_runs = results_df["success"].sum()
            failed_runs = total_runs - successful_runs
            success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0

            click.echo(click.style("Overall Statistics:", fg="yellow", bold=True))
            click.echo(f"  Total runs: {click.style(str(total_runs), fg='cyan')}")
            click.echo(f"  Successful: {click.style(str(successful_runs), fg='green')}")
            click.echo(
                f"  Failed: {click.style(str(failed_runs), fg='red' if failed_runs > 0 else 'green')}"
            )
            click.echo(
                f"  Success rate: {click.style(f'{success_rate:.1f}%', fg='green')}"
            )

            # Results location
            click.echo()
            click.echo(click.style("Results saved to:", fg="yellow", bold=True))
            click.echo(f"  {click.style(str(config.results_dir), fg='cyan')}")
        else:
            click.echo(
                click.style(
                    "⚠ Warning: No successful runs to summarize", fg="yellow", bold=True
                )
            )

        click.echo()
        click.echo(
            click.style("✓ Benchmark completed successfully!", fg="green", bold=True)
        )
        click.echo()

    except Exception as e:
        click.echo()
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo()
            click.echo(click.style("Traceback:", fg="red"))
            click.echo(traceback.format_exc())
        ctx.exit(1)


@benchmark.command()
@click.option(
    "--results-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to CSV results file to analyze.",
)
@click.pass_context
def analyze(ctx, results_file):
    """
    Analyze existing benchmark results.

    This command loads benchmark results from a CSV file and provides comprehensive
    statistical analysis including summary statistics, best-performing algorithms,
    and actionable recommendations.

    \b
    Analysis includes:
    - Summary statistics per algorithm (mean, std, min, max)
    - Best algorithm identification by metric
    - Success rates and failure analysis
    - Performance recommendations

    \b
    Examples:
        # Analyze results from a specific file
        benchmark analyze --results-file results/benchmark_results_20250128_143022.csv

        # Analyze with verbose output
        benchmark --verbose analyze --results-file results/benchmark_results.csv

    \b
    The analysis will show:
    - Execution time statistics
    - Cost statistics
    - Success rates
    - Best performing algorithm by each metric
    - Overall recommendation
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo(click.style("\n" + "=" * 80, fg="cyan"))
    click.echo(click.style("  Benchmark Results Analysis", fg="cyan", bold=True))
    click.echo(click.style("=" * 80 + "\n", fg="cyan"))

    try:
        # Load results
        click.echo(click.style("Loading results...", fg="yellow"))
        click.echo(f"  File: {click.style(results_file, fg='cyan')}")

        df = pd.read_csv(results_file)
        click.echo(click.style(f"✓ Loaded {len(df)} records", fg="green"))
        click.echo()

        # Filter successful runs
        successful_df = df[df["success"] == True]

        if successful_df.empty:
            click.echo(
                click.style(
                    "⚠ Warning: No successful runs found in the results file",
                    fg="yellow",
                    bold=True,
                )
            )
            ctx.exit(0)

        # Summary statistics
        click.echo(click.style("Summary Statistics:", fg="yellow", bold=True))
        click.echo()

        stats = (
            successful_df.groupby("algorithm_name")
            .agg(
                {
                    "execution_time_seconds": ["mean", "std", "min", "max"],
                    "total_cost": ["mean", "std", "min", "max"],
                    "nodes_explored": ["mean", "std", "min", "max"],
                }
            )
            .round(4)
        )

        # Execution time analysis
        click.echo(click.style("Execution Time (seconds):", fg="cyan", bold=True))
        click.echo(
            f"{'Algorithm':<20} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}"
        )
        click.echo("-" * 68)

        for algo_name in stats.index:
            mean = stats.loc[algo_name, ("execution_time_seconds", "mean")]
            std = stats.loc[algo_name, ("execution_time_seconds", "std")]
            min_val = stats.loc[algo_name, ("execution_time_seconds", "min")]
            max_val = stats.loc[algo_name, ("execution_time_seconds", "max")]

            click.echo(
                f"{algo_name:<20} "
                f"{mean:<12.4f} {std:<12.4f} {min_val:<12.4f} {max_val:<12.4f}"
            )

        click.echo()

        # Cost analysis
        click.echo(click.style("Total Cost:", fg="cyan", bold=True))
        click.echo(
            f"{'Algorithm':<20} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}"
        )
        click.echo("-" * 68)

        for algo_name in stats.index:
            mean = stats.loc[algo_name, ("total_cost", "mean")]
            std = stats.loc[algo_name, ("total_cost", "std")]
            min_val = stats.loc[algo_name, ("total_cost", "min")]
            max_val = stats.loc[algo_name, ("total_cost", "max")]

            click.echo(
                f"{algo_name:<20} "
                f"{mean:<12.2f} {std:<12.2f} {min_val:<12.2f} {max_val:<12.2f}"
            )

        click.echo()

        # Nodes explored analysis
        click.echo(click.style("Path Length (nodes):", fg="cyan", bold=True))
        click.echo(
            f"{'Algorithm':<20} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}"
        )
        click.echo("-" * 68)

        for algo_name in stats.index:
            mean = stats.loc[algo_name, ("nodes_explored", "mean")]
            std = stats.loc[algo_name, ("nodes_explored", "std")]
            min_val = stats.loc[algo_name, ("nodes_explored", "min")]
            max_val = stats.loc[algo_name, ("nodes_explored", "max")]

            click.echo(
                f"{algo_name:<20} "
                f"{mean:<12.1f} {std:<12.1f} {min_val:<12.0f} {max_val:<12.0f}"
            )

        click.echo()

        # Best algorithm identification
        click.echo(click.style("Best Algorithms by Metric:", fg="yellow", bold=True))

        mean_times = successful_df.groupby("algorithm_name")[
            "execution_time_seconds"
        ].mean()
        fastest = mean_times.idxmin()
        click.echo(
            f"  Fastest execution: {click.style(fastest, fg='green')} "
            f"({mean_times[fastest]:.4f}s)"
        )

        mean_costs = successful_df.groupby("algorithm_name")["total_cost"].mean()
        lowest_cost = mean_costs.idxmin()
        click.echo(
            f"  Lowest cost: {click.style(lowest_cost, fg='green')} "
            f"({mean_costs[lowest_cost]:.2f})"
        )

        success_rates = df.groupby("algorithm_name")["success"].mean()
        most_reliable = success_rates.idxmax()
        click.echo(
            f"  Most reliable: {click.style(most_reliable, fg='green')} "
            f"({success_rates[most_reliable]*100:.1f}% success rate)"
        )

        click.echo()

        # Overall recommendation
        click.echo(click.style("Recommendation:", fg="yellow", bold=True))

        if lowest_cost == fastest:
            click.echo(
                f"  {click.style(lowest_cost, fg='green', bold=True)} "
                f"is the clear winner - fastest and lowest cost!"
            )
        else:
            click.echo(
                f"  Use {click.style(fastest, fg='green')} for speed-critical applications"
            )
            click.echo(
                f"  Use {click.style(lowest_cost, fg='green')} for cost optimization"
            )

        click.echo()
        click.echo(click.style("✓ Analysis complete!", fg="green", bold=True))
        click.echo()

    except Exception as e:
        click.echo()
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo()
            click.echo(click.style("Traceback:", fg="red"))
            click.echo(traceback.format_exc())
        ctx.exit(1)


@benchmark.command()
@click.option(
    "--domain",
    type=click.Choice(["healthcare", "finance", "legal"], case_sensitive=False),
    required=True,
    help="Domain to generate workflows for.",
)
@click.option(
    "--count", type=int, default=1, help="Number of workflows to generate (default: 1)."
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="workflows/",
    help="Output directory for workflow files (default: workflows/).",
)
@click.pass_context
def workflows(ctx, domain, count, output_dir):
    """
    Generate sample workflows.

    This command generates workflow graphs for the specified domain and saves
    them as JSON files. Each workflow represents a realistic business process
    with tasks, dependencies, costs, and execution times.

    \b
    Generated workflows include:
    - Task nodes with execution times, costs, and resource requirements
    - Dependency edges with data transfer costs
    - Workflow metadata (ID, type, statistics)
    - Full graph structure in JSON format

    \b
    Examples:
        # Generate 1 healthcare workflow
        benchmark workflows --domain healthcare

        # Generate 5 finance workflows
        benchmark workflows --domain finance --count 5

        # Generate legal workflows in custom directory
        benchmark workflows --domain legal --count 3 --output-dir my_workflows/

    \b
    Healthcare Workflows:
        - Medical record extraction (document processing, entity extraction, ICD coding)
        - Insurance claim processing (verification, fraud detection, payment)
        - Patient intake (registration, insurance verification, scheduling)

    \b
    Finance Workflows:
        - Loan approval (credit checks, risk scoring, underwriting)
        - Fraud detection (pattern analysis, ML scoring, analyst review)
        - Risk assessment (portfolio analysis, VaR calculation, stress testing)

    \b
    Legal Workflows:
        - Contract review (clause extraction, risk detection, attorney review)
        - Compliance checking (regulation mapping, gap analysis, audit trails)
        - Document redlining (version comparison, risk assessment, attorney approval)

    \b
    Output Format:
        Workflows are saved as JSON files with the naming pattern:
        {domain}_{workflow_type}_{timestamp}.json
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo(click.style("\n" + "=" * 80, fg="cyan"))
    click.echo(click.style("  Workflow Generator", fg="cyan", bold=True))
    click.echo(click.style("=" * 80 + "\n", fg="cyan"))

    # Display configuration
    click.echo(click.style("Configuration:", fg="yellow", bold=True))
    click.echo(f"  Domain: {click.style(domain, fg='cyan')}")
    click.echo(f"  Count: {click.style(str(count), fg='cyan')}")
    click.echo(f"  Output directory: {click.style(output_dir, fg='cyan')}")
    click.echo()

    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize generator
        generator_class = DOMAIN_MAP[domain]

        click.echo(click.style("Generating workflows...", fg="yellow"))
        click.echo()

        generated_files = []

        for i in range(count):
            # Use different seed for each workflow to ensure variety
            generator = generator_class(seed=42 + i)

            # Cycle through workflow types for this domain
            workflow_types = WORKFLOW_TYPES[domain]
            workflow_type = workflow_types[i % len(workflow_types)]

            # Generate workflow
            if workflow_type == "medical_record_extraction":
                graph = generator.generate_medical_record_extraction()
            elif workflow_type == "insurance_claim_processing":
                graph = generator.generate_insurance_claim_processing()
            elif workflow_type == "patient_intake":
                graph = generator.generate_patient_intake_workflow()
            elif workflow_type == "loan_approval":
                graph = generator.generate_loan_approval()
            elif workflow_type == "fraud_detection":
                graph = generator.generate_fraud_detection()
            elif workflow_type == "risk_assessment":
                graph = generator.generate_risk_assessment()
            elif workflow_type == "contract_review":
                graph = generator.generate_contract_review()
            elif workflow_type == "compliance_check":
                graph = generator.generate_compliance_check()
            elif workflow_type == "document_redlining":
                graph = generator.generate_document_redlining()

            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                :17
            ]  # Include microseconds
            filename = f"{domain}_{workflow_type}_{timestamp}.json"
            filepath = output_path / filename

            # Convert graph to JSON-serializable format
            workflow_data = {
                "metadata": graph.graph,
                "nodes": {node: data for node, data in graph.nodes(data=True)},
                "edges": [
                    {"source": u, "target": v, "attributes": data}
                    for u, v, data in graph.edges(data=True)
                ],
            }

            # Save to file
            with open(filepath, "w") as f:
                json.dump(workflow_data, f, indent=2, default=str)

            generated_files.append(filepath)

            # Display progress
            click.echo(
                f"  [{i+1}/{count}] Generated: {click.style(workflow_type, fg='green')}"
            )
            click.echo(
                f"       Nodes: {graph.number_of_nodes()}, "
                f"Edges: {graph.number_of_edges()}"
            )
            click.echo(f"       File: {click.style(str(filepath), fg='cyan')}")
            click.echo()

        # Summary
        click.echo(click.style("Summary:", fg="yellow", bold=True))
        click.echo(
            f"  Generated: {click.style(str(len(generated_files)), fg='green')} workflows"
        )
        click.echo(f"  Location: {click.style(str(output_path.absolute()), fg='cyan')}")
        click.echo()
        click.echo(
            click.style("✓ Workflow generation complete!", fg="green", bold=True)
        )
        click.echo()

    except Exception as e:
        click.echo()
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo()
            click.echo(click.style("Traceback:", fg="red"))
            click.echo(traceback.format_exc())
        ctx.exit(1)


@benchmark.command()
@click.option(
    "--results-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to CSV results file to visualize.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/visualizations/",
    help="Output directory for visualization files (default: results/visualizations/).",
)
@click.pass_context
def visualize(ctx, results_file, output_dir):
    """
    Generate visualizations from benchmark results.

    This command creates comprehensive visualizations from benchmark results
    including algorithm comparisons, scalability analysis, and cost comparisons.
    All plots are saved as high-resolution PNG files.

    \b
    Generated visualizations:
    1. algorithm_comparison_time.png - Execution time by algorithm
    2. algorithm_comparison_cost.png - Solution cost by algorithm
    3. algorithm_comparison_nodes.png - Path length by algorithm
    4. scalability.png - How algorithms scale with workflow size
    5. cost_comparison.png - Cost comparison across workflows

    \b
    Examples:
        # Generate visualizations from benchmark results
        benchmark visualize --results-file results/benchmark_results_20251029_153133.csv

        # Specify custom output directory
        benchmark visualize --results-file results/latest.csv --output-dir plots/

    \b
    Output:
        All visualizations are saved as PNG files at 300 DPI resolution,
        suitable for presentations and publications.
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo(click.style("\n" + "=" * 80, fg="cyan"))
    click.echo(click.style("  Benchmark Visualization Generator", fg="cyan", bold=True))
    click.echo(click.style("=" * 80 + "\n", fg="cyan"))

    click.echo(click.style("Configuration:", fg="yellow", bold=True))
    click.echo(f"  Results file: {click.style(results_file, fg='cyan')}")
    click.echo(f"  Output directory: {click.style(output_dir, fg='cyan')}")
    click.echo()

    try:
        # Load results
        click.echo(click.style("Loading results...", fg="yellow"))
        df = pd.read_csv(results_file)
        click.echo(click.style(f"✓ Loaded {len(df)} records", fg="green"))

        # Show summary
        if "success" in df.columns:
            success_rate = df["success"].mean() * 100
            click.echo(
                f"  Success rate: {click.style(f'{success_rate:.1f}%', fg='green' if success_rate > 90 else 'yellow')}"
            )

        if "algorithm_name" in df.columns:
            algorithms = df["algorithm_name"].unique()
            click.echo(f"  Algorithms: {click.style(', '.join(algorithms), fg='cyan')}")

        click.echo()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        viz_count = 0

        # 1. Execution Time Comparison
        click.echo(
            click.style("[1/5] ", fg="cyan") + "Generating execution time comparison..."
        )
        try:
            plot_algorithm_comparison(
                df,
                metric="execution_time_seconds",
                save_path=str(output_path / "algorithm_comparison_time.png"),
            )
            click.echo(
                click.style("  ✓ Saved: algorithm_comparison_time.png", fg="green")
            )
            viz_count += 1
        except Exception as e:
            click.echo(click.style(f"  ✗ Error: {str(e)}", fg="red"))

        # 2. Cost Comparison
        click.echo(click.style("[2/5] ", fg="cyan") + "Generating cost comparison...")
        try:
            plot_algorithm_comparison(
                df,
                metric="total_cost",
                save_path=str(output_path / "algorithm_comparison_cost.png"),
            )
            click.echo(
                click.style("  ✓ Saved: algorithm_comparison_cost.png", fg="green")
            )
            viz_count += 1
        except Exception as e:
            click.echo(click.style(f"  ✗ Error: {str(e)}", fg="red"))

        # 3. Nodes Explored Comparison
        click.echo(
            click.style("[3/5] ", fg="cyan") + "Generating nodes explored comparison..."
        )
        try:
            plot_algorithm_comparison(
                df,
                metric="nodes_explored",
                save_path=str(output_path / "algorithm_comparison_nodes.png"),
            )
            click.echo(
                click.style("  ✓ Saved: algorithm_comparison_nodes.png", fg="green")
            )
            viz_count += 1
        except Exception as e:
            click.echo(click.style(f"  ✗ Error: {str(e)}", fg="red"))

        # 4. Scalability Analysis
        click.echo(
            click.style("[4/5] ", fg="cyan") + "Generating scalability analysis..."
        )
        try:
            plot_execution_time_scalability(
                df, save_path=str(output_path / "scalability.png")
            )
            click.echo(click.style("  ✓ Saved: scalability.png", fg="green"))
            viz_count += 1
        except Exception as e:
            click.echo(click.style(f"  ✗ Error: {str(e)}", fg="red"))

        # 5. Cost Comparison Across Workflows
        click.echo(
            click.style("[5/5] ", fg="cyan")
            + "Generating cost comparison across workflows..."
        )
        try:
            plot_cost_comparison(df, save_path=str(output_path / "cost_comparison.png"))
            click.echo(click.style("  ✓ Saved: cost_comparison.png", fg="green"))
            viz_count += 1
        except Exception as e:
            click.echo(click.style(f"  ✗ Error: {str(e)}", fg="red"))

        # Summary
        click.echo()
        click.echo(click.style("=" * 80, fg="cyan"))
        click.echo(click.style("Summary", fg="cyan", bold=True))
        click.echo(click.style("=" * 80, fg="cyan"))
        click.echo(
            f"  Visualizations created: {click.style(str(viz_count) + '/5', fg='green')}"
        )
        click.echo(
            f"  Output directory: {click.style(str(output_path.absolute()), fg='cyan')}"
        )
        click.echo()

        if viz_count == 5:
            click.echo(
                click.style(
                    "✓ All visualizations generated successfully!",
                    fg="green",
                    bold=True,
                )
            )
        elif viz_count > 0:
            click.echo(
                click.style(
                    f"⚠ {viz_count}/5 visualizations generated (some failed)",
                    fg="yellow",
                    bold=True,
                )
            )
        else:
            click.echo(
                click.style("✗ No visualizations were generated", fg="red", bold=True)
            )

        click.echo()

    except Exception as e:
        click.echo()
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo()
            click.echo(click.style("Traceback:", fg="red"))
            click.echo(traceback.format_exc())
        ctx.exit(1)


@benchmark.command()
@click.option(
    "--results-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to CSV results file to generate report from.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/",
    help="Output directory for report file (default: results/).",
)
@click.pass_context
def report(ctx, results_file, output_dir):
    """
    Generate a comprehensive markdown report from benchmark results.

    This command creates a detailed markdown report including executive summary,
    performance analysis, comparison tables, recommendations, and embedded
    visualizations. The report is suitable for documentation and presentations.

    \b
    Report sections include:
    - Executive summary with key findings
    - Test setup and configuration
    - Performance results table with best values highlighted
    - Detailed algorithm analysis
    - Production recommendations with tradeoffs
    - Embedded visualizations (if available)

    \b
    Examples:
        # Generate report from benchmark results
        benchmark report --results-file results/benchmark_results_20251029_153133.csv

        # Specify custom output directory
        benchmark report --results-file results/latest.csv --output-dir reports/

    \b
    Output:
        Creates a markdown file: benchmark_report_{timestamp}.md
        The report can be viewed in any markdown viewer or converted to PDF/HTML.
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo(click.style("\n" + "=" * 80, fg="cyan"))
    click.echo(click.style("  Benchmark Report Generator", fg="cyan", bold=True))
    click.echo(click.style("=" * 80 + "\n", fg="cyan"))

    click.echo(click.style("Configuration:", fg="yellow", bold=True))
    click.echo(f"  Results file: {click.style(results_file, fg='cyan')}")
    click.echo(f"  Output directory: {click.style(output_dir, fg='cyan')}")
    click.echo()

    try:
        # Load results
        click.echo(click.style("Loading results...", fg="yellow"))
        df = pd.read_csv(results_file)
        click.echo(click.style(f"✓ Loaded {len(df)} records", fg="green"))

        # Show summary
        if "success" in df.columns:
            success_rate = df["success"].mean() * 100
            click.echo(
                f"  Success rate: {click.style(f'{success_rate:.1f}%', fg='green' if success_rate > 90 else 'yellow')}"
            )

        if "algorithm_name" in df.columns:
            algorithms = df["algorithm_name"].unique()
            click.echo(f"  Algorithms: {click.style(', '.join(algorithms), fg='cyan')}")

        if "workflow_id" in df.columns:
            workflows = df["workflow_id"].nunique()
            click.echo(f"  Workflows: {click.style(str(workflows), fg='cyan')}")

        click.echo()

        # Generate report
        click.echo(
            click.style("Generating comprehensive markdown report...", fg="yellow")
        )
        generator = ReportGenerator(df, output_dir=output_dir)
        report_path = generator.generate_report()

        click.echo(click.style("✓ Report generated successfully!", fg="green"))
        click.echo()

        # Summary
        click.echo(click.style("=" * 80, fg="cyan"))
        click.echo(click.style("Summary", fg="cyan", bold=True))
        click.echo(click.style("=" * 80, fg="cyan"))
        click.echo(f"  Report saved to: {click.style(report_path, fg='green')}")
        click.echo()
        click.echo("The report includes:")
        click.echo(click.style("  • Executive summary with key findings", fg="cyan"))
        click.echo(click.style("  • Test setup and configuration", fg="cyan"))
        click.echo(click.style("  • Performance results table", fg="cyan"))
        click.echo(click.style("  • Detailed algorithm analysis", fg="cyan"))
        click.echo(click.style("  • Production recommendations", fg="cyan"))
        click.echo(click.style("  • Embedded visualizations (if available)", fg="cyan"))
        click.echo()
        click.echo(
            "You can view the report in any markdown viewer or convert it to PDF/HTML."
        )
        click.echo()

    except Exception as e:
        click.echo()
        click.echo(click.style(f"✗ Error: {str(e)}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo()
            click.echo(click.style("Traceback:", fg="red"))
            click.echo(traceback.format_exc())
        ctx.exit(1)


if __name__ == "__main__":
    benchmark()
