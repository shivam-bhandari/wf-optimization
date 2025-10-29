"""
Markdown Report Generator for Benchmark Results

This module provides a comprehensive report generator that creates professional
markdown reports from benchmark results, including executive summaries, detailed
analysis, and embedded visualizations.

Example:
    >>> import pandas as pd
    >>> from src.reporting.report_generator import ReportGenerator
    >>> 
    >>> df = pd.read_csv('results/benchmark_results.csv')
    >>> generator = ReportGenerator(df, output_dir='results/')
    >>> report_path = generator.generate_report()
    >>> print(f"Report saved to: {report_path}")
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

# Try to import tabulate, fallback to manual table generation if not available
try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class ReportGenerator:
    """
    Generate comprehensive markdown reports from benchmark results.

    This class creates detailed, professional markdown reports that include
    executive summaries, performance analysis, comparison tables, and embedded
    visualizations. The reports are suitable for documentation, presentations,
    and technical communication.

    Attributes:
        results_df (pd.DataFrame): DataFrame containing benchmark results
        output_dir (Path): Directory for saving reports and finding visualizations
        summary_stats (Dict[str, Any]): Pre-calculated summary statistics

    Example:
        >>> df = pd.read_csv('results/benchmark_results.csv')
        >>> generator = ReportGenerator(df, output_dir='results/')
        >>> report_path = generator.generate_report()
    """

    def __init__(self, results_df: pd.DataFrame, output_dir: str = "results/"):
        """
        Initialize the report generator.

        Args:
            results_df (pd.DataFrame): DataFrame with benchmark results containing:
                - algorithm_name: Name of the algorithm
                - execution_time_seconds: Execution time in seconds
                - total_cost: Total cost of solution
                - nodes_explored: Number of nodes in path
                - success: Boolean indicating success
                - workflow_id: Workflow identifier

            output_dir (str, optional): Directory for saving reports and finding
                visualizations. Default: 'results/'.

        Raises:
            ValueError: If results_df is empty or missing required columns.
        """
        if results_df.empty:
            raise ValueError("Results DataFrame cannot be empty")

        self.results_df = results_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate summary statistics on initialization
        self.summary_stats = self._calculate_summary_stats()

    def _create_markdown_table(self, data: list, headers: list) -> str:
        """
        Create a markdown table manually when tabulate is not available.

        Args:
            data: List of dictionaries containing row data
            headers: List of column headers

        Returns:
            str: Markdown formatted table
        """
        if not data:
            return ""

        # Create header row
        header_row = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---" for _ in headers]) + " |"

        # Create data rows
        rows = []
        for row_dict in data:
            row = "| " + " | ".join(str(row_dict.get(h, "")) for h in headers) + " |"
            rows.append(row)

        return "\n".join([header_row, separator] + rows)

    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate comprehensive summary statistics from results.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - total_runs: Total number of benchmark runs
                - successful_runs: Number of successful runs
                - failed_runs: Number of failed runs
                - success_rate: Overall success rate (0-1)
                - num_algorithms: Number of unique algorithms tested
                - num_workflows: Number of unique workflows tested
                - algorithms: List of algorithm names
                - workflows: List of workflow IDs
                - best_time_algo: Algorithm with fastest average time
                - best_cost_algo: Algorithm with lowest average cost
                - overall_best_algo: Best overall algorithm
        """
        stats = {}

        # Basic counts
        stats["total_runs"] = len(self.results_df)

        if "success" in self.results_df.columns:
            stats["successful_runs"] = self.results_df["success"].sum()
            stats["failed_runs"] = stats["total_runs"] - stats["successful_runs"]
            stats["success_rate"] = (
                stats["successful_runs"] / stats["total_runs"]
                if stats["total_runs"] > 0
                else 0
            )
        else:
            stats["successful_runs"] = stats["total_runs"]
            stats["failed_runs"] = 0
            stats["success_rate"] = 1.0

        # Filter to successful runs for analysis
        if "success" in self.results_df.columns:
            df_success = self.results_df[self.results_df["success"] == True]
        else:
            df_success = self.results_df

        # Algorithm and workflow counts
        if "algorithm_name" in self.results_df.columns:
            stats["num_algorithms"] = self.results_df["algorithm_name"].nunique()
            stats["algorithms"] = sorted(
                self.results_df["algorithm_name"].unique().tolist()
            )
        else:
            stats["num_algorithms"] = 0
            stats["algorithms"] = []

        if "workflow_id" in self.results_df.columns:
            stats["num_workflows"] = self.results_df["workflow_id"].nunique()
            stats["workflows"] = sorted(
                self.results_df["workflow_id"].unique().tolist()
            )
        else:
            stats["num_workflows"] = 0
            stats["workflows"] = []

        # Find best algorithms
        if not df_success.empty and "algorithm_name" in df_success.columns:
            if "execution_time_seconds" in df_success.columns:
                time_avg = df_success.groupby("algorithm_name")[
                    "execution_time_seconds"
                ].mean()
                stats["best_time_algo"] = time_avg.idxmin()
                stats["best_time"] = time_avg.min()

            if "total_cost" in df_success.columns:
                cost_avg = df_success.groupby("algorithm_name")["total_cost"].mean()
                stats["best_cost_algo"] = cost_avg.idxmin()
                stats["best_cost"] = cost_avg.min()

            # Overall best: combine time and cost
            if (
                "execution_time_seconds" in df_success.columns
                and "total_cost" in df_success.columns
            ):
                # Best is the one that's fastest AND has optimal cost
                if stats.get("best_time_algo") == stats.get("best_cost_algo"):
                    stats["overall_best_algo"] = stats["best_time_algo"]
                else:
                    # If different, prefer the one with optimal cost (correctness over speed)
                    stats["overall_best_algo"] = stats["best_cost_algo"]
            elif "execution_time_seconds" in df_success.columns:
                stats["overall_best_algo"] = stats.get("best_time_algo")
            elif "total_cost" in df_success.columns:
                stats["overall_best_algo"] = stats.get("best_cost_algo")

        return stats

    def generate_report(self) -> str:
        """
        Generate a comprehensive markdown report.

        This method creates a complete benchmark report including:
        - Executive summary with key findings
        - Test setup and configuration
        - Results table with performance metrics
        - Detailed algorithm analysis
        - Recommendations for production use
        - Embedded visualizations (if available)

        Returns:
            str: Path to the saved markdown file.

        Example:
            >>> generator = ReportGenerator(df)
            >>> report_path = generator.generate_report()
            >>> print(f"Report saved to: {report_path}")
        """
        # Generate all sections
        sections = [
            self._generate_header(),
            self._generate_executive_summary(),
            self._generate_setup_section(),
            self._generate_results_table(),
            self._generate_algorithm_analysis(),
            self._generate_recommendations(),
            self._embed_visualizations(),
            self._generate_footer(),
        ]

        # Combine sections
        report_content = "\n\n".join(sections)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_report_{timestamp}.md"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write(report_content)

        return str(filepath)

    def _generate_header(self) -> str:
        """
        Generate report header with title and metadata.

        Returns:
            str: Markdown header section.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = f"""# Workflow Optimization Benchmark Report

**Generated:** {timestamp}  
**Total Runs:** {self.summary_stats['total_runs']}  
**Success Rate:** {self.summary_stats['success_rate']:.1%}

---
"""
        return header

    def _generate_executive_summary(self) -> str:
        """
        Generate 2-3 sentence executive summary with key findings.

        Returns:
            str: Markdown executive summary section.
        """
        summary = "## Executive Summary\n\n"

        # Overall performance summary
        if self.summary_stats["success_rate"] >= 0.95:
            summary += f"This benchmark successfully evaluated **{self.summary_stats['num_algorithms']} algorithms** "
            summary += f"across **{self.summary_stats['num_workflows']} workflows** "
            summary += f"with a {self.summary_stats['success_rate']:.1%} success rate. "
        else:
            summary += f"This benchmark evaluated {self.summary_stats['num_algorithms']} algorithms "
            summary += f"across {self.summary_stats['num_workflows']} workflows "
            summary += f"with a {self.summary_stats['success_rate']:.1%} success rate. "

        # Best algorithm identification
        if "overall_best_algo" in self.summary_stats:
            best_algo = self.summary_stats["overall_best_algo"]
            summary += f"**{best_algo}** emerged as the best overall algorithm, "

            if "best_time" in self.summary_stats and "best_cost" in self.summary_stats:
                if (
                    self.summary_stats.get("best_time_algo") == best_algo
                    and self.summary_stats.get("best_cost_algo") == best_algo
                ):
                    summary += f"achieving both the fastest execution time ({self.summary_stats['best_time']:.4f}s average) "
                    summary += f"and optimal solution cost ({self.summary_stats['best_cost']:.2f} average). "
                elif self.summary_stats.get("best_cost_algo") == best_algo:
                    summary += f"consistently finding optimal solutions with an average cost of {self.summary_stats['best_cost']:.2f}. "
            elif "best_time" in self.summary_stats:
                summary += f"with an average execution time of {self.summary_stats['best_time']:.4f}s. "
            elif "best_cost" in self.summary_stats:
                summary += f"with an average solution cost of {self.summary_stats['best_cost']:.2f}. "

        # Key finding
        if (
            "best_time_algo" in self.summary_stats
            and "best_cost_algo" in self.summary_stats
        ):
            if (
                self.summary_stats["best_time_algo"]
                != self.summary_stats["best_cost_algo"]
            ):
                summary += f"Note that **{self.summary_stats['best_time_algo']}** was fastest, "
                summary += f"while **{self.summary_stats['best_cost_algo']}** found optimal solutions, "
                summary += "indicating a speed-quality tradeoff."

        return summary

    def _generate_setup_section(self) -> str:
        """
        Generate test setup and configuration section.

        Returns:
            str: Markdown setup section.
        """
        setup = "## Test Setup\n\n"

        setup += "### Configuration\n\n"
        setup += f"- **Algorithms Tested:** {self.summary_stats['num_algorithms']}\n"
        setup += f"- **Workflows Tested:** {self.summary_stats['num_workflows']}\n"

        # Calculate trials per combination
        if (
            self.summary_stats["num_algorithms"] > 0
            and self.summary_stats["num_workflows"] > 0
        ):
            expected_combinations = (
                self.summary_stats["num_algorithms"]
                * self.summary_stats["num_workflows"]
            )
            if expected_combinations > 0:
                trials_per = self.summary_stats["total_runs"] / expected_combinations
                setup += f"- **Trials per Combination:** {trials_per:.0f}\n"

        setup += f"- **Total Test Combinations:** {self.summary_stats['total_runs']}\n"

        # List algorithms
        if self.summary_stats["algorithms"]:
            setup += "\n### Algorithms\n\n"
            for algo in self.summary_stats["algorithms"]:
                setup += f"- {algo}\n"

        # List workflows (limit to first 10 if many)
        if self.summary_stats["workflows"]:
            setup += "\n### Workflows\n\n"
            workflows_to_show = self.summary_stats["workflows"][:10]
            for workflow in workflows_to_show:
                setup += f"- {workflow}\n"

            if len(self.summary_stats["workflows"]) > 10:
                remaining = len(self.summary_stats["workflows"]) - 10
                setup += f"- *... and {remaining} more workflows*\n"

        return setup

    def _generate_results_table(self) -> str:
        """
        Generate markdown table with performance metrics.

        Returns:
            str: Markdown results table section.
        """
        section = "## Performance Results\n\n"

        # Filter to successful runs
        if "success" in self.results_df.columns:
            df_success = self.results_df[self.results_df["success"] == True]
        else:
            df_success = self.results_df

        if df_success.empty:
            return section + "*No successful runs to report.*\n"

        # Group by algorithm and calculate metrics
        metrics = []

        if "algorithm_name" in df_success.columns:
            for algo in sorted(df_success["algorithm_name"].unique()):
                algo_data = df_success[df_success["algorithm_name"] == algo]

                row = {"Algorithm": algo}

                # Calculate average time
                if "execution_time_seconds" in algo_data.columns:
                    avg_time = algo_data["execution_time_seconds"].mean()
                    row["Avg Time (s)"] = f"{avg_time:.4f}"

                # Calculate average cost
                if "total_cost" in algo_data.columns:
                    avg_cost = algo_data["total_cost"].mean()
                    row["Avg Cost"] = f"{avg_cost:.2f}"

                # Calculate success rate
                if "success" in self.results_df.columns:
                    algo_total = len(
                        self.results_df[self.results_df["algorithm_name"] == algo]
                    )
                    algo_success = len(algo_data)
                    success_rate = algo_success / algo_total if algo_total > 0 else 0
                    row["Success Rate"] = f"{success_rate:.1%}"
                else:
                    row["Success Rate"] = "100.0%"

                metrics.append(row)

            # Find best values for highlighting
            if metrics:
                # Find best time
                if "Avg Time (s)" in metrics[0]:
                    times = [float(m["Avg Time (s)"]) for m in metrics]
                    best_time_idx = times.index(min(times))
                    metrics[best_time_idx][
                        "Avg Time (s)"
                    ] = f"**{metrics[best_time_idx]['Avg Time (s)']}**"

                # Find best cost
                if "Avg Cost" in metrics[0]:
                    costs = [float(m["Avg Cost"]) for m in metrics]
                    best_cost_idx = costs.index(min(costs))
                    metrics[best_cost_idx][
                        "Avg Cost"
                    ] = f"**{metrics[best_cost_idx]['Avg Cost']}**"

                # Find best success rate
                if "Success Rate" in metrics[0]:
                    rates = [float(m["Success Rate"].rstrip("%")) for m in metrics]
                    best_rate_idx = rates.index(max(rates))
                    metrics[best_rate_idx][
                        "Success Rate"
                    ] = f"**{metrics[best_rate_idx]['Success Rate']}**"

            # Create table
            if metrics:
                if HAS_TABULATE:
                    table = tabulate(metrics, headers="keys", tablefmt="pipe")
                else:
                    # Use manual table generation
                    headers = list(metrics[0].keys())
                    table = self._create_markdown_table(metrics, headers)

                section += table + "\n\n"
                section += "*Best values highlighted in bold*\n"

        return section

    def _generate_algorithm_analysis(self) -> str:
        """
        Generate detailed analysis of each algorithm's performance.

        Returns:
            str: Markdown algorithm analysis section.
        """
        section = "## Algorithm Analysis\n\n"

        # Filter to successful runs
        if "success" in self.results_df.columns:
            df_success = self.results_df[self.results_df["success"] == True]
        else:
            df_success = self.results_df

        if df_success.empty or "algorithm_name" not in df_success.columns:
            return section + "*No data available for analysis.*\n"

        # Analyze each algorithm
        for algo in sorted(df_success["algorithm_name"].unique()):
            algo_data = df_success[df_success["algorithm_name"] == algo]

            section += f"### {algo}\n\n"

            # Performance summary
            performance_notes = []

            if "execution_time_seconds" in algo_data.columns:
                avg_time = algo_data["execution_time_seconds"].mean()
                std_time = algo_data["execution_time_seconds"].std()
                min_time = algo_data["execution_time_seconds"].min()
                max_time = algo_data["execution_time_seconds"].max()

                performance_notes.append(
                    f"averaged {avg_time:.4f}s per workflow (σ={std_time:.4f}s, range: {min_time:.4f}s-{max_time:.4f}s)"
                )

                # Compare to best
                if "best_time_algo" in self.summary_stats:
                    if algo == self.summary_stats["best_time_algo"]:
                        performance_notes.append("**was consistently fastest**")

            if "total_cost" in algo_data.columns:
                avg_cost = algo_data["total_cost"].mean()
                std_cost = algo_data["total_cost"].std()

                performance_notes.append(
                    f"found solutions with average cost of {avg_cost:.2f} (σ={std_cost:.2f})"
                )

                # Compare to best
                if "best_cost_algo" in self.summary_stats:
                    if algo == self.summary_stats["best_cost_algo"]:
                        performance_notes.append(
                            "**consistently found optimal solutions**"
                        )

            if performance_notes:
                section += f"{algo} {', '.join(performance_notes)}. "

            # Success rate
            if "success" in self.results_df.columns:
                algo_total = len(
                    self.results_df[self.results_df["algorithm_name"] == algo]
                )
                algo_success = len(algo_data)
                success_rate = algo_success / algo_total if algo_total > 0 else 0

                if success_rate < 1.0:
                    section += f"Success rate was {success_rate:.1%}. "
                else:
                    section += "All runs completed successfully. "

            section += "\n\n"

        return section

    def _generate_recommendations(self) -> str:
        """
        Generate recommendations for production use.

        Returns:
            str: Markdown recommendations section.
        """
        section = "## Recommendations\n\n"

        if "overall_best_algo" not in self.summary_stats:
            return section + "*Insufficient data to make recommendations.*\n"

        best_algo = self.summary_stats["overall_best_algo"]

        section += f"### Primary Recommendation\n\n"
        section += f"**Use {best_algo} for production workflows.** "

        # Explain why
        reasons = []
        if (
            "best_time_algo" in self.summary_stats
            and "best_cost_algo" in self.summary_stats
        ):
            if (
                self.summary_stats["best_time_algo"] == best_algo
                and self.summary_stats["best_cost_algo"] == best_algo
            ):
                reasons.append(
                    "it provides both optimal solutions and fastest execution"
                )
            elif self.summary_stats["best_cost_algo"] == best_algo:
                reasons.append("it consistently finds optimal solutions")
                reasons.append("ensuring correct results")

        if reasons:
            section += "This algorithm " + ", ".join(reasons) + ". "

        section += "\n\n"

        # Alternative recommendations
        section += "### Alternative Scenarios\n\n"

        if (
            "best_time_algo" in self.summary_stats
            and "best_cost_algo" in self.summary_stats
        ):
            if (
                self.summary_stats["best_time_algo"]
                != self.summary_stats["best_cost_algo"]
            ):
                fast_algo = self.summary_stats["best_time_algo"]
                optimal_algo = self.summary_stats["best_cost_algo"]

                section += f"- **For time-critical applications:** Use {fast_algo} if speed is more important than optimality\n"
                section += f"- **For cost-sensitive workflows:** Use {optimal_algo} to guarantee optimal solutions\n"

        section += "\n### Tradeoffs to Consider\n\n"

        # Analyze tradeoffs
        if (
            "best_time_algo" in self.summary_stats
            and "best_cost_algo" in self.summary_stats
        ):
            if (
                self.summary_stats["best_time_algo"]
                != self.summary_stats["best_cost_algo"]
            ):
                section += "- **Speed vs. Quality:** Faster algorithms may sacrifice solution quality\n"
                section += "- **Resource Usage:** Consider memory and CPU requirements for production deployment\n"
            else:
                section += "- The recommended algorithm provides an excellent balance of speed and quality\n"

        section += "- **Scalability:** Test with production-scale workflows before deployment\n"
        section += (
            "- **Reliability:** Monitor success rates in production environments\n"
        )

        return section

    def _embed_visualizations(self) -> str:
        """
        Embed visualization images if they exist.

        Returns:
            str: Markdown section with embedded images.
        """
        section = "## Visualizations\n\n"

        # Define expected visualization files
        viz_files = [
            ("algorithm_comparison_time.png", "Algorithm Execution Time Comparison"),
            ("algorithm_comparison_cost.png", "Algorithm Cost Comparison"),
            ("algorithm_comparison_nodes.png", "Algorithm Path Length Comparison"),
            ("scalability.png", "Algorithm Scalability Analysis"),
            ("cost_comparison.png", "Cost Comparison Across Workflows"),
        ]

        # Check for visualizations directory
        viz_dir = self.output_dir / "visualizations"

        found_any = False

        for filename, title in viz_files:
            # Check in both output_dir and visualizations subdirectory
            file_path = self.output_dir / filename
            viz_path = viz_dir / filename

            if viz_path.exists():
                section += f"### {title}\n\n"
                section += f"![{title}](./visualizations/{filename})\n\n"
                found_any = True
            elif file_path.exists():
                section += f"### {title}\n\n"
                section += f"![{title}](./{filename})\n\n"
                found_any = True

        if not found_any:
            section += "*No visualizations found. Run the visualize command to generate charts.*\n\n"
            section += "```bash\n"
            section += "python -m src.cli visualize --results-file results/benchmark_results.csv\n"
            section += "```\n"

        return section

    def _generate_footer(self) -> str:
        """
        Generate report footer with metadata.

        Returns:
            str: Markdown footer section.
        """
        footer = "\n---\n\n"
        footer += "## Report Metadata\n\n"
        footer += f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        footer += f"- **Total Runs:** {self.summary_stats['total_runs']}\n"
        footer += f"- **Successful Runs:** {self.summary_stats['successful_runs']}\n"
        footer += f"- **Failed Runs:** {self.summary_stats['failed_runs']}\n"
        footer += f"- **Success Rate:** {self.summary_stats['success_rate']:.2%}\n"

        footer += "\n*This report was automatically generated by the Workflow Optimization Benchmark Suite.*\n"

        return footer
