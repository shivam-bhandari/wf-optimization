"""
Demonstration of BenchmarkRunner for comprehensive algorithm benchmarking.

This example shows how to:
1. Set up multiple algorithms for benchmarking
2. Generate multiple test workflows
3. Configure benchmark parameters
4. Run comprehensive benchmarks
5. Analyze and visualize results
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx
from src.benchmarking.runner import BenchmarkConfig, BenchmarkRunner
from src.algorithms.dijkstra import DijkstraOptimizer
from src.algorithms.astar import AStarOptimizer
from src.algorithms.bellman_ford import BellmanFordOptimizer
from src.algorithms.dag_dp import DAGDynamicProgramming
from src.datasets.healthcare import HealthcareWorkflowGenerator


def main():
    """Run a comprehensive benchmark demonstration."""
    
    print("="*80)
    print("Benchmark Runner Demonstration")
    print("="*80)
    print()
    
    # Step 1: Generate test workflows
    print("Step 1: Generating test workflows...")
    generator = HealthcareWorkflowGenerator(seed=42)
    
    workflows = [
        ('ehr_extraction_1', generator.generate_ehr_extraction()),
        ('ehr_extraction_2', generator.generate_ehr_extraction()),
        ('insurance_claim_1', generator.generate_insurance_claim_processing()),
        ('patient_intake_1', generator.generate_patient_intake())
    ]
    
    print(f"  Generated {len(workflows)} workflows:")
    for workflow_id, graph in workflows:
        print(f"    - {workflow_id}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print()
    
    # Step 2: Create algorithm instances
    print("Step 2: Creating algorithm instances...")
    
    # Get source and target nodes from first workflow
    sample_graph = workflows[0][1]
    source = [n for n, d in sample_graph.in_degree() if d == 0][0]
    target = [n for n, d in sample_graph.out_degree() if d == 0][0]
    
    algorithms = [
        DijkstraOptimizer(name="Dijkstra", source=source, target=target),
        BellmanFordOptimizer(name="BellmanFord", source=source, target=target),
        DAGDynamicProgramming(name="DAG_DP", source=source, target=target),
    ]
    
    # Try to add A* if it has compatible interface
    try:
        from src.algorithms.astar import AStarOptimizer
        algorithms.append(AStarOptimizer(name="AStar", source=source, target=target))
    except Exception as e:
        print(f"  Note: A* optimizer not available: {e}")
    
    print(f"  Created {len(algorithms)} algorithms:")
    for algo in algorithms:
        print(f"    - {algo.name}")
    print()
    
    # Step 3: Configure benchmark
    print("Step 3: Configuring benchmark...")
    config = BenchmarkConfig(
        trials_per_combination=3,  # Run each combination 3 times
        timeout_seconds=60.0,      # 60 second timeout per run
        random_seed=42,
        objectives=['cost'],       # Optimize for cost
        save_results=True,
        results_dir=Path('results/')
    )
    
    print(f"  Configuration:")
    print(f"    - Trials per combination: {config.trials_per_combination}")
    print(f"    - Timeout: {config.timeout_seconds}s")
    print(f"    - Objectives: {config.objectives}")
    print(f"    - Results directory: {config.results_dir}")
    print()
    
    # Step 4: Run benchmarks
    print("Step 4: Running benchmarks...")
    print(f"  Total runs: {len(algorithms)} algorithms × {len(workflows)} workflows × {len(config.objectives)} objectives × {config.trials_per_combination} trials")
    print(f"            = {len(algorithms) * len(workflows) * len(config.objectives) * config.trials_per_combination} total benchmark runs")
    print()
    
    runner = BenchmarkRunner(algorithms, workflows, config)
    results_df = runner.run_benchmarks()
    
    # Step 5: Analyze results
    print("\n" + "="*80)
    print("Step 5: Analyzing Results")
    print("="*80)
    print()
    
    # Summary statistics
    print("Summary Statistics:")
    print("-" * 80)
    
    successful_runs = results_df[results_df['success']]
    failed_runs = results_df[~results_df['success']]
    
    print(f"Total runs: {len(results_df)}")
    print(f"Successful: {len(successful_runs)} ({len(successful_runs)/len(results_df)*100:.1f}%)")
    print(f"Failed: {len(failed_runs)} ({len(failed_runs)/len(results_df)*100:.1f}%)")
    print()
    
    if not successful_runs.empty:
        # Performance by algorithm
        print("Performance by Algorithm:")
        print("-" * 80)
        algo_stats = successful_runs.groupby('algorithm_name').agg({
            'total_cost': ['mean', 'std', 'min', 'max'],
            'execution_time_seconds': ['mean', 'std', 'min', 'max'],
            'nodes_explored': ['mean', 'std']
        }).round(4)
        print(algo_stats)
        print()
        
        # Performance by workflow
        print("Performance by Workflow:")
        print("-" * 80)
        workflow_stats = successful_runs.groupby('workflow_id').agg({
            'total_cost': ['mean', 'std', 'min', 'max'],
            'execution_time_seconds': ['mean', 'std'],
        }).round(4)
        print(workflow_stats)
        print()
        
        # Best performing algorithm per workflow
        print("Best Performing Algorithm per Workflow (by mean cost):")
        print("-" * 80)
        best_per_workflow = successful_runs.groupby(['workflow_id', 'algorithm_name'])['total_cost'].mean().reset_index()
        best_per_workflow = best_per_workflow.loc[best_per_workflow.groupby('workflow_id')['total_cost'].idxmin()]
        for _, row in best_per_workflow.iterrows():
            print(f"  {row['workflow_id']}: {row['algorithm_name']} (cost={row['total_cost']:.2f})")
        print()
    
    if not failed_runs.empty:
        print("Failed Runs:")
        print("-" * 80)
        for _, row in failed_runs.iterrows():
            print(f"  {row['algorithm_name']} on {row['workflow_id']}: {row['error_message']}")
        print()
    
    print("="*80)
    print("Benchmark demonstration completed!")
    print(f"Results saved to: {config.results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

