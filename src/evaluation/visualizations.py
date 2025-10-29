"""
Visualization Functions for Benchmark Results

This module provides visualization functions for analyzing and comparing
workflow optimization algorithm performance using matplotlib.

Functions:
- plot_algorithm_comparison: Compare algorithms across a specific metric
- plot_execution_time_scalability: Show how algorithms scale with workflow size
- plot_cost_comparison: Compare solution costs across algorithms and workflows
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Configure matplotlib style
def _setup_style():
    """Set up matplotlib style with fallback."""
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        try:
            plt.style.use("seaborn-darkgrid")
        except Exception:
            # Fallback to default if seaborn styles are not available
            plt.style.use("default")
            plt.rcParams["axes.grid"] = True
            plt.rcParams["grid.alpha"] = 0.3


def plot_algorithm_comparison(
    results_df: pd.DataFrame,
    metric: str = "execution_time_seconds",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a grouped bar chart comparing algorithms across a specific metric.

    This function aggregates benchmark results by algorithm and creates a
    bar chart showing the mean value of the specified metric for each algorithm.
    Useful for quick visual comparison of algorithm performance.

    Args:
        results_df (pd.DataFrame): DataFrame containing benchmark results with columns:
            - algorithm_name: Name of the algorithm
            - execution_time_seconds: Execution time in seconds
            - total_cost: Total cost of the solution
            - nodes_explored: Number of nodes in the solution path
            - success: Boolean indicating if the run was successful

        metric (str, optional): The metric to plot. Must be one of:
            - 'execution_time_seconds': Algorithm execution time
            - 'total_cost': Total cost of the solution found
            - 'nodes_explored': Number of nodes in the solution path
            Default: 'execution_time_seconds'

        save_path (Optional[str], optional): If provided, saves the plot to this
            path as a PNG file at 300 DPI. If None, displays the plot using
            plt.show(). Default: None.

    Returns:
        plt.Figure: The matplotlib figure object containing the plot.

    Raises:
        ValueError: If the specified metric is not in the DataFrame or if
            no successful runs are found.
        KeyError: If required columns are missing from the DataFrame.

    Example:
        >>> import pandas as pd
        >>> from src.evaluation.visualizations import plot_algorithm_comparison
        >>>
        >>> # Load benchmark results
        >>> df = pd.read_csv('results/benchmark_results.csv')
        >>>
        >>> # Compare execution times
        >>> fig = plot_algorithm_comparison(df, metric='execution_time_seconds')
        >>>
        >>> # Compare costs and save
        >>> fig = plot_algorithm_comparison(df, metric='total_cost',
        ...                                  save_path='results/cost_comparison.png')
    """
    _setup_style()

    # Validate inputs
    if "algorithm_name" not in results_df.columns:
        raise KeyError("DataFrame must contain 'algorithm_name' column")

    if metric not in results_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in DataFrame. "
            f"Available columns: {', '.join(results_df.columns)}"
        )

    # Filter successful runs and remove NaN values
    if "success" in results_df.columns:
        df_filtered = results_df[results_df["success"] == True].copy()
    else:
        df_filtered = results_df.copy()

    df_filtered = df_filtered.dropna(subset=[metric, "algorithm_name"])

    if df_filtered.empty:
        raise ValueError("No valid data to plot after filtering")

    # Calculate mean metric per algorithm
    grouped = (
        df_filtered.groupby("algorithm_name")[metric].agg(["mean", "std"]).reset_index()
    )
    grouped = grouped.sort_values("mean", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for algorithms
    colors = plt.cm.Set3(np.linspace(0, 1, len(grouped)))

    # Create bar chart
    bars = ax.bar(
        range(len(grouped)),
        grouped["mean"],
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add error bars if standard deviation is available
    if "std" in grouped.columns and not grouped["std"].isna().all():
        ax.errorbar(
            range(len(grouped)),
            grouped["mean"],
            yerr=grouped["std"],
            fmt="none",
            ecolor="black",
            capsize=5,
            capthick=2,
            alpha=0.6,
        )

    # Customize plot
    ax.set_xlabel("Algorithm", fontsize=12, fontweight="bold")

    # Set y-axis label based on metric
    ylabel_map = {
        "execution_time_seconds": "Execution Time (seconds)",
        "total_cost": "Total Cost",
        "nodes_explored": "Nodes Explored",
    }
    ylabel = ylabel_map.get(metric, metric.replace("_", " ").title())
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    # Set title
    title_metric = metric.replace("_", " ").title()
    ax.set_title(
        f"Algorithm Comparison - {title_metric}", fontsize=14, fontweight="bold", pad=20
    )

    # Set x-axis ticks
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped["algorithm_name"], rotation=45, ha="right")

    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, grouped["mean"])):
        height = bar.get_height()
        # Format based on metric type
        if metric == "execution_time_seconds":
            label = f"{value:.4f}s"
        elif metric == "total_cost":
            label = f"{value:.2f}"
        else:
            label = f"{value:.1f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add grid for readability
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig


def plot_execution_time_scalability(
    results_df: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Show how execution time scales with workflow size for different algorithms.

    This function creates a line plot showing the relationship between workflow
    size (number of nodes) and execution time for each algorithm. Useful for
    understanding algorithm scalability and performance characteristics.

    Args:
        results_df (pd.DataFrame): DataFrame containing benchmark results with columns:
            - algorithm_name: Name of the algorithm
            - execution_time_seconds: Execution time in seconds
            - nodes_explored or workflow_id: Used to determine workflow size
            - success: Boolean indicating if the run was successful

        save_path (Optional[str], optional): If provided, saves the plot to this
            path as a PNG file at 300 DPI. If None, displays the plot using
            plt.show(). Default: None.

    Returns:
        plt.Figure: The matplotlib figure object containing the plot.

    Raises:
        ValueError: If required columns are missing or no valid data is found.

    Note:
        - Uses log scale on y-axis if execution times span multiple orders of magnitude
        - Workflow size is determined from nodes_explored or extracted from workflow_id
        - Multiple runs for the same workflow size are averaged

    Example:
        >>> import pandas as pd
        >>> from src.evaluation.visualizations import plot_execution_time_scalability
        >>>
        >>> df = pd.read_csv('results/benchmark_results.csv')
        >>> fig = plot_execution_time_scalability(df)
        >>>
        >>> # Save to file
        >>> fig = plot_execution_time_scalability(df, save_path='results/scalability.png')
    """
    _setup_style()

    # Validate inputs
    required_cols = ["algorithm_name", "execution_time_seconds"]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Filter successful runs
    if "success" in results_df.columns:
        df_filtered = results_df[results_df["success"] == True].copy()
    else:
        df_filtered = results_df.copy()

    # Determine workflow size
    if "nodes_explored" in df_filtered.columns:
        df_filtered["workflow_size"] = df_filtered["nodes_explored"]
    elif "workflow_id" in df_filtered.columns:
        # Try to extract size from workflow metadata or use a proxy
        # For now, we'll group by workflow_id and use the index as a proxy
        workflow_sizes = df_filtered.groupby("workflow_id").size().to_dict()
        df_filtered["workflow_size"] = df_filtered["workflow_id"].map(
            {wf: idx * 10 + 10 for idx, wf in enumerate(workflow_sizes.keys())}
        )
    else:
        # Use row index as a fallback
        df_filtered["workflow_size"] = range(10, 10 + len(df_filtered) * 5, 5)

    # Remove NaN values
    df_filtered = df_filtered.dropna(
        subset=["workflow_size", "execution_time_seconds", "algorithm_name"]
    )

    if df_filtered.empty:
        raise ValueError("No valid data to plot after filtering")

    # Group by algorithm and workflow size, calculate mean
    grouped = (
        df_filtered.groupby(["algorithm_name", "workflow_size"])[
            "execution_time_seconds"
        ]
        .mean()
        .reset_index()
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique algorithms
    algorithms = grouped["algorithm_name"].unique()

    # Define colors and markers for each algorithm
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Plot each algorithm
    for i, algo in enumerate(algorithms):
        algo_data = grouped[grouped["algorithm_name"] == algo].sort_values(
            "workflow_size"
        )

        ax.plot(
            algo_data["workflow_size"],
            algo_data["execution_time_seconds"],
            marker=markers[i % len(markers)],
            markersize=8,
            linewidth=2,
            color=colors[i],
            label=algo,
            alpha=0.8,
        )

    # Customize plot
    ax.set_xlabel("Workflow Size (nodes)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Execution Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Algorithm Scalability", fontsize=14, fontweight="bold", pad=20)

    # Determine if log scale is appropriate
    time_range = (
        grouped["execution_time_seconds"].max()
        / grouped["execution_time_seconds"].min()
    )
    if time_range > 100:  # More than 2 orders of magnitude
        ax.set_yscale("log")
        ax.set_ylabel(
            "Execution Time (seconds, log scale)", fontsize=12, fontweight="bold"
        )

    # Add legend
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    # Add grid
    ax.grid(True, alpha=0.3, which="both")
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig


def plot_cost_comparison(
    results_df: pd.DataFrame, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare total costs found by each algorithm across different workflows.

    This function creates a grouped bar chart showing the solution cost found
    by each algorithm for each workflow. Differences in cost indicate that
    some algorithms found suboptimal solutions. Useful for comparing solution
    quality across algorithms.

    Args:
        results_df (pd.DataFrame): DataFrame containing benchmark results with columns:
            - algorithm_name: Name of the algorithm
            - workflow_id: Identifier for the workflow
            - total_cost: Total cost of the solution found
            - success: Boolean indicating if the run was successful

        save_path (Optional[str], optional): If provided, saves the plot to this
            path as a PNG file at 300 DPI. If None, displays the plot using
            plt.show(). Default: None.

    Returns:
        plt.Figure: The matplotlib figure object containing the plot.

    Raises:
        ValueError: If required columns are missing or no valid data is found.

    Note:
        - Highlights workflows where algorithms find different costs
        - Uses side-by-side bars for easy comparison
        - Optimal algorithms should have the lowest bars

    Example:
        >>> import pandas as pd
        >>> from src.evaluation.visualizations import plot_cost_comparison
        >>>
        >>> df = pd.read_csv('results/benchmark_results.csv')
        >>> fig = plot_cost_comparison(df)
        >>>
        >>> # Save comparison chart
        >>> fig = plot_cost_comparison(df, save_path='results/cost_comparison.png')
    """
    _setup_style()

    # Validate inputs
    required_cols = ["algorithm_name", "total_cost"]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Filter successful runs
    if "success" in results_df.columns:
        df_filtered = results_df[results_df["success"] == True].copy()
    else:
        df_filtered = results_df.copy()

    # Remove NaN values
    df_filtered = df_filtered.dropna(subset=["total_cost", "algorithm_name"])

    if df_filtered.empty:
        raise ValueError("No valid data to plot after filtering")

    # Check if we have workflow_id for grouping
    if "workflow_id" in df_filtered.columns:
        # Group by workflow and algorithm
        grouped = (
            df_filtered.groupby(["workflow_id", "algorithm_name"])["total_cost"]
            .mean()
            .reset_index()
        )

        # Get unique workflows and algorithms
        workflows = grouped["workflow_id"].unique()
        algorithms = grouped["algorithm_name"].unique()

        # Limit number of workflows to display for readability
        if len(workflows) > 10:
            workflows = workflows[:10]
            grouped = grouped[grouped["workflow_id"].isin(workflows)]
            print(
                f"Note: Displaying first 10 workflows out of {len(df_filtered['workflow_id'].unique())}"
            )

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate bar positions
        x = np.arange(len(workflows))
        width = 0.8 / len(algorithms)

        # Define colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))

        # Plot bars for each algorithm
        for i, algo in enumerate(algorithms):
            algo_data = grouped[grouped["algorithm_name"] == algo]

            # Create a mapping of workflow to cost
            cost_map = dict(zip(algo_data["workflow_id"], algo_data["total_cost"]))
            costs = [cost_map.get(wf, 0) for wf in workflows]

            offset = (i - len(algorithms) / 2) * width + width / 2
            bars = ax.bar(
                x + offset,
                costs,
                width,
                label=algo,
                color=colors[i],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.8,
            )

        # Customize plot
        ax.set_xlabel("Workflow", fontsize=12, fontweight="bold")
        ax.set_ylabel("Total Cost", fontsize=12, fontweight="bold")
        ax.set_title(
            "Cost Comparison Across Workflows", fontsize=14, fontweight="bold", pad=20
        )

        # Set x-axis ticks
        ax.set_xticks(x)
        # Shorten workflow names for display
        workflow_labels = [
            wf.split("_")[-1] if "_" in wf else wf[:15] for wf in workflows
        ]
        ax.set_xticklabels(workflow_labels, rotation=45, ha="right")

        # Highlight suboptimal solutions
        # For each workflow, find the minimum cost
        for workflow_idx, workflow in enumerate(workflows):
            workflow_costs = grouped[grouped["workflow_id"] == workflow]["total_cost"]
            if len(workflow_costs) > 1:
                min_cost = workflow_costs.min()
                max_cost = workflow_costs.max()
                if max_cost > min_cost * 1.01:  # More than 1% difference
                    # Add a subtle indicator
                    ax.axvspan(
                        workflow_idx - 0.4,
                        workflow_idx + 0.4,
                        alpha=0.1,
                        color="red",
                        zorder=0,
                    )

    else:
        # If no workflow_id, just compare algorithms directly
        grouped = (
            df_filtered.groupby("algorithm_name")["total_cost"].mean().reset_index()
        )
        grouped = grouped.sort_values("total_cost")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(grouped)))

        # Create bar chart
        bars = ax.bar(
            range(len(grouped)),
            grouped["total_cost"],
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
        )

        # Customize plot
        ax.set_xlabel("Algorithm", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Total Cost", fontsize=12, fontweight="bold")
        ax.set_title("Average Cost Comparison", fontsize=14, fontweight="bold", pad=20)

        # Set x-axis ticks
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped["algorithm_name"], rotation=45, ha="right")

        # Add value labels
        for bar, value in zip(bars, grouped["total_cost"]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Highlight the optimal (minimum cost) algorithm
        min_idx = grouped["total_cost"].idxmin()
        min_position = list(grouped.index).index(min_idx)
        bars[min_position].set_edgecolor("green")
        bars[min_position].set_linewidth(3)

    # Add legend
    if "workflow_id" in df_filtered.columns:
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig
