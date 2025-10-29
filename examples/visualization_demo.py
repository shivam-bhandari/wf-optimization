"""
Visualization Demo

This script demonstrates the use of visualization functions for analyzing
benchmark results.

Usage:
    python examples/visualization_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.evaluation.visualizations import (
    plot_algorithm_comparison,
    plot_execution_time_scalability,
    plot_cost_comparison
)


def demo_visualizations():
    """
    Demonstrate all visualization functions.
    
    This function:
    1. Checks for existing benchmark results
    2. Creates all three types of visualizations
    3. Saves them to the results directory
    """
    print("=" * 80)
    print("Visualization Demo")
    print("=" * 80)
    print()
    
    # Find the most recent results file
    results_dir = project_root / "results"
    
    if not results_dir.exists():
        print("Error: No results directory found.")
        print("Please run benchmarks first:")
        print("  poetry run python -m src.cli run --trials 3")
        return
    
    # Find CSV files
    csv_files = list(results_dir.glob("benchmark_results_*.csv"))
    
    if not csv_files:
        print("Error: No benchmark results found.")
        print("Please run benchmarks first:")
        print("  poetry run python -m src.cli run --trials 3")
        return
    
    # Use the most recent file
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file.name}")
    print()
    
    # Load results
    df = pd.read_csv(latest_file)
    print(f"Loaded {len(df)} records")
    print(f"Algorithms: {', '.join(df['algorithm_name'].unique())}")
    print(f"Success rate: {df['success'].mean() * 100:.1f}%")
    print()
    
    # Create visualizations directory
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Algorithm comparison - Execution Time
    print("Creating visualization 1/6: Execution Time Comparison...")
    try:
        fig1 = plot_algorithm_comparison(
            df,
            metric='execution_time_seconds',
            save_path=str(viz_dir / 'algorithm_comparison_time.png')
        )
        print("✓ Saved: algorithm_comparison_time.png")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # 2. Algorithm comparison - Cost
    print("Creating visualization 2/6: Cost Comparison...")
    try:
        fig2 = plot_algorithm_comparison(
            df,
            metric='total_cost',
            save_path=str(viz_dir / 'algorithm_comparison_cost.png')
        )
        print("✓ Saved: algorithm_comparison_cost.png")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # 3. Algorithm comparison - Nodes Explored
    print("Creating visualization 3/6: Nodes Explored Comparison...")
    try:
        fig3 = plot_algorithm_comparison(
            df,
            metric='nodes_explored',
            save_path=str(viz_dir / 'algorithm_comparison_nodes.png')
        )
        print("✓ Saved: algorithm_comparison_nodes.png")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # 4. Execution Time Scalability
    print("Creating visualization 4/6: Execution Time Scalability...")
    try:
        fig4 = plot_execution_time_scalability(
            df,
            save_path=str(viz_dir / 'scalability.png')
        )
        print("✓ Saved: scalability.png")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # 5. Cost Comparison
    print("Creating visualization 5/6: Cost Comparison Across Workflows...")
    try:
        fig5 = plot_cost_comparison(
            df,
            save_path=str(viz_dir / 'cost_comparison.png')
        )
        print("✓ Saved: cost_comparison.png")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # Summary
    print("=" * 80)
    print("Visualization Summary")
    print("=" * 80)
    print(f"All visualizations saved to: {viz_dir}")
    print()
    print("Generated visualizations:")
    print("  1. algorithm_comparison_time.png - Execution time by algorithm")
    print("  2. algorithm_comparison_cost.png - Solution cost by algorithm")
    print("  3. algorithm_comparison_nodes.png - Path length by algorithm")
    print("  4. scalability.png - How algorithms scale with workflow size")
    print("  5. cost_comparison.png - Cost comparison across workflows")
    print()
    print("You can view these PNG files in any image viewer.")
    print("=" * 80)


if __name__ == "__main__":
    demo_visualizations()

