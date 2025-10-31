"""
Workflow Optimization Benchmark

"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.datasets.healthcare import HealthcareWorkflowGenerator
from src.datasets.finance import FinancialWorkflowGenerator
from src.datasets.legal import LegalWorkflowGenerator

from src.algorithms import (
    DAGDynamicProgramming,
    DijkstraOptimizer,
    AStarOptimizer,
    BellmanFordOptimizer,
)

from src.benchmarking.runner import BenchmarkConfig, BenchmarkRunner

from src.evaluation.visualizations import (
    plot_algorithm_comparison,
    plot_execution_time_scalability,
    plot_cost_comparison
)
from src.reporting.report_generator import ReportGenerator


def print_step(step_num: int, total: int, description: str):
    """Print step header."""
    print(f"\n[STEP {step_num}/{total}] {description}")


def generate_workflows():
    """Generate sample workflows from each domain."""
    print_step(1, 4, "Generating workflows...")
    
    workflows = []
    
    healthcare_gen = HealthcareWorkflowGenerator(seed=2)
    healthcare_workflow = healthcare_gen.generate_medical_record_extraction()
    workflow_id = "healthcare_medical_record_extraction"
    workflows.append((workflow_id, healthcare_workflow))
    print(f"  [OK] Healthcare workflow generated ({healthcare_workflow.number_of_nodes()} nodes)")
    
    finance_gen = FinancialWorkflowGenerator(seed=2)
    finance_workflow = finance_gen.generate_loan_approval()
    workflow_id = "finance_loan_approval"
    workflows.append((workflow_id, finance_workflow))
    print(f"  [OK] Finance workflow generated ({finance_workflow.number_of_nodes()} nodes)")
    
    legal_gen = LegalWorkflowGenerator(seed=2)
    legal_workflow = legal_gen.generate_contract_review()
    workflow_id = "legal_contract_review"
    workflows.append((workflow_id, legal_workflow))
    print(f"  [OK] Legal workflow generated ({legal_workflow.number_of_nodes()} nodes)")
    
    return workflows


def create_algorithms():
    """Create algorithm instances."""
    algorithms = [
        DAGDynamicProgramming(source='start', target='end', weight_attr='cost_units'),
        DijkstraOptimizer(source='start', target='end', weight_attr='cost_units'),
        AStarOptimizer(source='start', target='end', weight_attr='cost_units'),
        BellmanFordOptimizer(source='start', target='end', weight_attr='cost_units'),
    ]
    return algorithms


def run_benchmarks(workflows, algorithms, output_dir):
    """Run benchmarks on workflows with algorithms."""
    print_step(2, 4, "Running algorithms...")
    
    config = BenchmarkConfig(
        trials_per_combination=2,
        timeout_seconds=60.0,
        objectives=['cost_units'],
        save_results=True,
        results_dir=output_dir,
        use_multiprocessing=True
    )
    
    runner = BenchmarkRunner(algorithms, workflows, config)
    
    total_runs = len(algorithms) * len(workflows) * 2
    print(f"  Running {len(algorithms)} algorithms on {len(workflows)} workflows...")
    print(f"  Total runs: {total_runs} (2 trials each)")
    
    results_df = runner.run_benchmarks()
    
    successful = results_df['success'].sum()
    success_rate = (successful / len(results_df)) * 100
    print(f"  [OK] Benchmark complete ({successful}/{len(results_df)} runs successful, {success_rate:.0f}% success rate)")
    
    return results_df


def generate_visualizations(results_df, output_dir):
    """Generate all visualization charts."""
    print_step(3, 4, "Generating visualizations...")
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    viz_count = 0
    
    plot_algorithm_comparison(
        results_df,
        metric='execution_time_seconds',
        save_path=str(viz_dir / 'algorithm_comparison_time.png')
    )
    viz_count += 1
    
    plot_algorithm_comparison(
        results_df,
        metric='total_cost',
        save_path=str(viz_dir / 'algorithm_comparison_cost.png')
    )
    print(f"  [OK] Comparison charts saved")
    viz_count += 1
    
    plot_execution_time_scalability(
        results_df,
        save_path=str(viz_dir / 'scalability.png')
    )
    print(f"  [OK] Scalability chart saved")
    viz_count += 1
    
    plot_cost_comparison(
        results_df,
        save_path=str(viz_dir / 'cost_comparison.png')
    )
    print(f"  [OK] Cost chart saved")
    viz_count += 1
    
    return viz_count


def generate_report(results_df, output_dir):
    """Generate comprehensive markdown report."""
    print_step(4, 4, "Generating report...")
    
    generator = ReportGenerator(results_df, output_dir=str(output_dir))
    report_path = generator.generate_report()
    print(f"  [OK] Report saved")
    return report_path


def analyze_results(results_df):
    """Analyze results and generate recommendation."""
    df_success = results_df[results_df['success'] == True]
    
    if df_success.empty:
        return "No successful runs to analyze", None, None
    
    time_avg = df_success.groupby('algorithm_name')['execution_time_seconds'].mean()
    cost_avg = df_success.groupby('algorithm_name')['total_cost'].mean()
    
    fastest_algo = time_avg.idxmin()
    fastest_time = time_avg.min()
    
    optimal_algo = cost_avg.idxmin()
    optimal_cost = cost_avg.min()
    
    if fastest_algo == optimal_algo:
        if len(time_avg) > 1:
            second_fastest = time_avg.nsmallest(2).iloc[1]
            speedup = second_fastest / fastest_time
            recommendation = f"Use {fastest_algo} for production ({speedup:.1f}× faster than alternatives, optimal solutions)"
        else:
            recommendation = f"Use {fastest_algo} for production (optimal solutions)"
    else:
        recommendation = f"Use {optimal_algo} for optimal solutions, or {fastest_algo} for speed"
    
    return recommendation, fastest_algo, optimal_algo


def print_summary(output_dir, report_path, viz_count, recommendation):
    """Print final summary."""
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80 + "\n")
    
    print("Results:")
    if viz_count > 0:
        print(f"  Charts: results/{output_dir.name}/visualizations/")
    if report_path:
        report_name = Path(report_path).name
        print(f"  Report: {report_name}")
    
    csv_files = list(output_dir.glob("benchmark_results_*.csv"))
    if csv_files:
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        csv_name = latest_csv.name
        print(f"  Data: {csv_name}")
    
    print(f"\nRecommendation: {recommendation}")
    print()


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path('results') / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    workflows = generate_workflows()
    
    algorithms = create_algorithms()
    results_df = run_benchmarks(workflows, algorithms, output_dir)
    
    viz_count = generate_visualizations(results_df, output_dir)
    
    report_path = generate_report(results_df, output_dir)
    
    recommendation, fastest, optimal = analyze_results(results_df)
    
    print_summary(output_dir, report_path, viz_count, recommendation)


if __name__ == "__main__":
    main()
