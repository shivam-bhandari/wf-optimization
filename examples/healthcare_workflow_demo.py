#!/usr/bin/env python3
"""
Healthcare Workflow Generator Demo

This script demonstrates the healthcare workflow generator with all three
workflow types and shows how to use them with the optimization algorithms.

Run with: python examples/healthcare_workflow_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from src.datasets import HealthcareWorkflowGenerator
from src.algorithms import (
    DAGDynamicProgramming,
    DijkstraOptimizer,
    AStarOptimizer
)


def print_banner(text: str) -> None:
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def visualize_workflow(workflow: nx.DiGraph, name: str) -> None:
    """Print detailed workflow information."""
    print(f"\n{name}")
    print("-" * 80)
    
    # Metadata
    print(f"Workflow ID: {workflow.graph['workflow_id']}")
    print(f"Domain: {workflow.graph['domain']}")
    print(f"Type: {workflow.graph['workflow_type']}")
    print(f"Generated: {workflow.graph['generated_at'][:19]}")
    
    # Statistics
    stats = workflow.graph['statistics']
    print(f"\n📊 Statistics:")
    print(f"  • Total Nodes: {stats['total_nodes']}")
    print(f"  • Total Edges: {stats['total_edges']}")
    print(f"  • Max Depth: {stats['max_depth']}")
    print(f"  • Avg Branching Factor: {stats['avg_branching_factor']}")
    print(f"  • Estimated Total Cost: ${stats['estimated_total_cost']:.2f}")
    print(f"  • Estimated Total Time: {stats['estimated_total_time_ms'] / 1000:.2f}s")
    
    # Show critical path
    try:
        path = nx.dag_longest_path(workflow)
        print(f"\n🔗 Critical Path ({len(path)} nodes):")
        if len(path) <= 8:
            print(f"  {' → '.join(path)}")
        else:
            print(f"  {' → '.join(path[:4])} ... → {' → '.join(path[-2:])}")
    except Exception:
        print("\n🔗 Critical Path: Could not compute")
    
    # Show task details
    regular_tasks = [n for n in workflow.nodes() if n not in ['start', 'end']]
    if len(regular_tasks) >= 3:
        print(f"\n📋 Sample Tasks:")
        for i, task in enumerate(regular_tasks[:3]):
            attrs = workflow.nodes[task]
            print(f"\n  {i+1}. {task}")
            print(f"     • Execution Time: {attrs['execution_time_ms']}ms")
            print(f"     • Cost: ${attrs['cost_units']}")
            print(f"     • CPU: {attrs['resource_requirements']['cpu_cores']} cores")
            print(f"     • Memory: {attrs['resource_requirements']['memory_gb']}GB")
            print(f"     • Failure Probability: {attrs['failure_probability'] * 100:.2f}%")


def run_optimization_demo(workflow: nx.DiGraph, workflow_name: str) -> None:
    """Demonstrate optimization algorithms on a workflow."""
    print(f"\n\n🔍 Running Optimization Algorithms on {workflow_name}")
    print("-" * 80)
    
    # Find source and target
    source = 'start'
    target = 'end'
    
    # Test algorithms
    algorithms = [
        ("DAG-DP", DAGDynamicProgramming(source=source, target=target)),
        ("Dijkstra", DijkstraOptimizer(source=source, target=target, weight_attr='cost_units')),
        ("A* (task_depth)", AStarOptimizer(source=source, target=target, heuristic_type='task_depth', weight_attr='cost_units')),
    ]
    
    print(f"\n{'Algorithm':<20} {'Path Length':<15} {'Total Cost':<15} {'Time (ms)':<15}")
    print("-" * 65)
    
    for name, algo in algorithms:
        try:
            # Convert node attributes to edge weights for path finding
            # Create a copy with edge weights based on target node costs
            G = nx.DiGraph()
            for u, v in workflow.edges():
                # Weight is the cost of the target node
                target_cost = workflow.nodes[v].get('cost_units', 0)
                G.add_edge(u, v, cost_units=target_cost, weight=target_cost)
            
            solution = algo.solve(G)
            print(f"{name:<20} {len(solution['path']):<15} ${solution['total_cost']:<14.2f} {solution.get('execution_time_seconds', 0)*1000:<15.3f}")
        except Exception as e:
            print(f"{name:<20} Failed: {str(e)[:40]}")


def main():
    """Main demo function."""
    print_banner("Healthcare Workflow Generator Demo")
    
    print("This demo generates three types of healthcare workflows:")
    print("  1. Medical Record Extraction")
    print("  2. Insurance Claim Processing")
    print("  3. Patient Intake Workflow")
    print("\nEach workflow includes realistic tasks with execution times,")
    print("costs, resource requirements, and failure probabilities.")
    
    # Create generator with seed for reproducibility
    print("\n📦 Initializing Healthcare Workflow Generator (seed=42)...")
    gen = HealthcareWorkflowGenerator(seed=42)
    
    # Generate workflows
    print_banner("Generated Healthcare Workflows")
    
    workflows = {
        "Medical Record Extraction": gen.generate_medical_record_extraction(),
        "Insurance Claim Processing": gen.generate_insurance_claim_processing(),
        "Patient Intake Workflow": gen.generate_patient_intake_workflow(),
    }
    
    # Visualize each workflow
    for name, workflow in workflows.items():
        visualize_workflow(workflow, name)
    
    # Run optimization demo on one workflow
    print_banner("Optimization Algorithm Demo")
    demo_workflow = workflows["Medical Record Extraction"]
    run_optimization_demo(demo_workflow, "Medical Record Extraction")
    
    # Summary
    print_banner("Demo Complete")
    print("✅ Successfully generated and analyzed healthcare workflows")
    print("📊 All workflows are valid DAGs with realistic characteristics")
    print("🚀 Ready for benchmarking and optimization studies")
    print("\nNext steps:")
    print("  • Use these workflows with optimization algorithms")
    print("  • Benchmark algorithm performance on healthcare scenarios")
    print("  • Analyze resource utilization and costs")
    print("  • Extend with additional healthcare workflow patterns")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

