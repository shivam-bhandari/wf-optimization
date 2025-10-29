#!/usr/bin/env python3
"""
Manual Phase 1 Test Script

This script provides a visual, user-friendly way to test all Phase 1 algorithms.
Run with: python tests/manual_phase1_test.py

Features:
- Generates a visual workflow
- Runs all 4 core algorithms
- Prints formatted results table
- Shows which algorithm was fastest
- Highlights any failures
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.algorithms import (
    DAGDynamicProgramming,
    DijkstraOptimizer,
    AStarOptimizer,
    BellmanFordOptimizer,
    NegativeCycleError
)
from src.datasets.generator import WorkflowGenerator


def print_banner(text: str, char: str = '=') -> None:
    """Print a formatted banner."""
    width = 80
    print()
    print(char * width)
    print(text.center(width))
    print(char * width)
    print()


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{text}")
    print("-" * 80)


def visualize_workflow(graph: nx.DiGraph, source: str, target: str) -> None:
    """Print a simple text visualization of the workflow."""
    print(f"\nWorkflow Structure:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(graph)}")
    
    # Show some sample edges
    print(f"\n  Sample edges:")
    for i, (u, v, data) in enumerate(graph.edges(data=True)):
        if i >= 5:  # Show first 5 edges
            print(f"    ... and {graph.number_of_edges() - 5} more")
            break
        weight = data.get('weight', 1)
        print(f"    {u} → {v} (weight={weight})")


def run_algorithm(
    name: str,
    algorithm_class,
    config: Dict[str, Any],
    graph: nx.DiGraph,
    source: str,
    target: str
) -> Dict[str, Any]:
    """
    Run a single algorithm and return results.
    
    Args:
        name: Algorithm name for display
        algorithm_class: Algorithm class to instantiate
        config: Configuration dict for algorithm
        graph: Workflow graph
        source: Source node
        target: Target node
    
    Returns:
        Dict with algorithm results or error information
    """
    result = {
        'name': name,
        'success': False,
        'error': None,
        'execution_time': 0.0,
        'path_length': 0,
        'total_cost': 0.0,
        'nodes_explored': 0
    }
    
    try:
        algo = algorithm_class(source=source, target=target, **config)
        
        start_time = time.perf_counter()
        solution = algo.solve(graph)
        end_time = time.perf_counter()
        
        result['success'] = True
        result['execution_time'] = end_time - start_time
        result['path_length'] = len(solution['path'])
        result['total_cost'] = solution['total_cost']
        result['nodes_explored'] = solution.get('nodes_explored', 0)
        result['path'] = solution['path']
        
    except NegativeCycleError as e:
        result['error'] = f"Negative cycle: {str(e)[:50]}..."
    except ValueError as e:
        result['error'] = str(e)
    except Exception as e:
        result['error'] = f"Error: {str(e)[:50]}..."
    
    return result


def print_results_table(results: List[Dict[str, Any]]) -> None:
    """
    Print a formatted results table.
    
    Args:
        results: List of algorithm results
    """
    print_section("RESULTS")
    
    # Header
    header = f"{'Algorithm':<25} {'Status':<10} {'Time (ms)':<12} {'Cost':<10} {'Path Len':<10} {'Nodes Exp.':<12}"
    print(header)
    print("-" * len(header))
    
    # Find fastest successful algorithm
    successful = [r for r in results if r['success']]
    fastest_time = min(r['execution_time'] for r in successful) if successful else float('inf')
    
    # Rows
    for result in results:
        if result['success']:
            status = "✓ SUCCESS"
            time_ms = result['execution_time'] * 1000
            cost = f"{result['total_cost']:.2f}"
            path_len = str(result['path_length'])
            nodes_exp = str(result['nodes_explored'])
            
            # Highlight fastest
            time_str = f"{time_ms:.3f}"
            if result['execution_time'] == fastest_time:
                time_str += " ⚡"
            
            row = f"{result['name']:<25} {status:<10} {time_str:<12} {cost:<10} {path_len:<10} {nodes_exp:<12}"
        else:
            status = "✗ FAILED"
            error = result['error'][:30] + "..." if len(result['error']) > 30 else result['error']
            row = f"{result['name']:<25} {status:<10} {error}"
        
        print(row)
    
    print()


def print_path_details(results: List[Dict[str, Any]]) -> None:
    """Print details about the paths found."""
    print_section("PATH DETAILS")
    
    for result in results:
        if result['success'] and 'path' in result:
            path = result['path']
            if len(path) <= 10:
                path_str = ' → '.join(str(n) for n in path)
            else:
                path_str = ' → '.join(str(n) for n in path[:5]) + ' ... → ' + ' → '.join(str(n) for n in path[-2:])
            
            print(f"\n{result['name']}:")
            print(f"  Path: {path_str}")
            print(f"  Cost: {result['total_cost']:.2f}")
            print(f"  Length: {result['path_length']} nodes")


def print_comparison_analysis(results: List[Dict[str, Any]]) -> None:
    """Print comparative analysis of algorithms."""
    print_section("COMPARATIVE ANALYSIS")
    
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("No successful results to compare.")
        return
    
    # Check if all found same optimal cost
    costs = [r['total_cost'] for r in successful]
    all_optimal = all(abs(c - costs[0]) < 0.01 for c in costs)
    
    if all_optimal:
        print(f"✓ All algorithms found the SAME optimal solution (cost={costs[0]:.2f})")
    else:
        print("⚠ Algorithms found DIFFERENT solutions:")
        for r in successful:
            print(f"  {r['name']}: {r['total_cost']:.2f}")
    
    # Performance ranking
    print(f"\nPerformance Ranking (by execution time):")
    sorted_results = sorted(successful, key=lambda x: x['execution_time'])
    for i, r in enumerate(sorted_results, 1):
        time_ms = r['execution_time'] * 1000
        speedup = sorted_results[-1]['execution_time'] / r['execution_time']
        print(f"  {i}. {r['name']:<25} {time_ms:>8.3f}ms  ({speedup:.1f}x speedup)" )
    
    # Efficiency (nodes explored)
    print(f"\nEfficiency Ranking (by nodes explored):")
    sorted_by_nodes = sorted(successful, key=lambda x: x['nodes_explored'])
    for i, r in enumerate(sorted_by_nodes, 1):
        print(f"  {i}. {r['name']:<25} {r['nodes_explored']:>8} nodes")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print final summary."""
    print_section("SUMMARY")
    
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total - successful
    
    print(f"Total Algorithms Tested: {total}")
    print(f"Successful: {successful} ✓")
    print(f"Failed: {failed} ✗")
    
    if successful == total:
        print("\n🎉 ALL ALGORITHMS PASSED!")
        print("✓ Phase 1 core algorithms are working correctly")
        print("✓ Ready to proceed to Phase 2")
    elif successful > 0:
        print(f"\n⚠ {failed} algorithm(s) failed")
        print("Review error messages above")
    else:
        print("\n❌ ALL ALGORITHMS FAILED")
        print("Please review errors and fix issues")


def main():
    """Main test execution function."""
    print_banner("PHASE 1 ALGORITHM TEST SUITE", '=')
    print("Testing core shortest path algorithms on generated workflow\n")
    
    # Generate test workflow
    print_section("GENERATING TEST WORKFLOW")
    generator = WorkflowGenerator(random_seed=42)
    graph = generator.generate_dag(
        num_nodes=30,
        edge_probability=0.15
    )
    
    # Find source and target
    sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    targets = [n for n in graph.nodes() if graph.out_degree(n) == 0]
    
    if not sources:
        print("❌ Generated graph has no source nodes")
        print("Try running again with a different seed")
        sys.exit(1)
    
    source = sources[0]
    
    # Find reachable targets from source
    reachable = nx.descendants(graph, source)
    reachable_targets = [t for t in targets if t in reachable]
    
    if not reachable_targets:
        # If no sink is reachable, just pick any reachable node
        if reachable:
            target = list(reachable)[-1]
        else:
            print("❌ No nodes reachable from source")
            print("Try running again with a different seed")
            sys.exit(1)
    else:
        target = reachable_targets[0]
    
    # Visualize workflow
    visualize_workflow(graph, source, target)
    
    # Define algorithms to test
    print_section("RUNNING ALGORITHMS")
    
    algorithms = [
        ("DAG Dynamic Programming", DAGDynamicProgramming, {}),
        ("Dijkstra's Algorithm", DijkstraOptimizer, {}),
        ("A* (zero heuristic)", AStarOptimizer, {'heuristic_type': 'zero'}),
        ("A* (task_depth heuristic)", AStarOptimizer, {'heuristic_type': 'task_depth'}),
        ("Bellman-Ford", BellmanFordOptimizer, {}),
    ]
    
    results = []
    
    for name, algo_class, config in algorithms:
        print(f"\nRunning {name}...", end=" ")
        result = run_algorithm(name, algo_class, config, graph, source, target)
        results.append(result)
        
        if result['success']:
            print(f"✓ ({result['execution_time']*1000:.3f}ms)")
        else:
            print(f"✗ {result['error'][:50]}")
    
    # Print results
    print()
    print_results_table(results)
    
    # Show path details for successful runs
    if any(r['success'] for r in results):
        print_path_details(results)
    
    # Comparative analysis
    print_comparison_analysis(results)
    
    # Summary
    print_summary(results)
    
    print_banner("TEST COMPLETE", '=')
    
    # Return exit code
    sys.exit(0 if all(r['success'] for r in results) else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

