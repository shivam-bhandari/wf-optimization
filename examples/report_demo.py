"""
Report Generation Demo

This script demonstrates the use of the ReportGenerator to create
comprehensive markdown reports from benchmark results.

Usage:
    python examples/report_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.reporting.report_generator import ReportGenerator


def demo_report_generation():
    """
    Demonstrate report generation functionality.
    
    This function:
    1. Finds the most recent benchmark results
    2. Generates a comprehensive markdown report
    3. Displays the report path
    """
    print("=" * 80)
    print("Report Generation Demo")
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
    if 'success' in df.columns:
        print(f"Success rate: {df['success'].mean() * 100:.1f}%")
    print()
    
    # Create report generator
    print("Creating report generator...")
    generator = ReportGenerator(df, output_dir=str(results_dir))
    print()
    
    # Generate report
    print("Generating comprehensive markdown report...")
    report_path = generator.generate_report()
    print()
    
    # Summary
    print("=" * 80)
    print("Report Generation Complete")
    print("=" * 80)
    print(f"Report saved to: {report_path}")
    print()
    print("You can view the report in any markdown viewer or text editor.")
    print("The report includes:")
    print("  • Executive summary with key findings")
    print("  • Test setup and configuration")
    print("  • Performance results table")
    print("  • Detailed algorithm analysis")
    print("  • Production recommendations")
    print("  • Embedded visualizations (if available)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demo_report_generation()

