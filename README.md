# Workflow Optimization Benchmark Suite

A comprehensive benchmark suite for evaluating workflow optimization algorithms.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install
```

## Quick Start

Run the complete end-to-end demo:

```bash
poetry run python main.py
```

This will:
- Generate 3 sample workflows (healthcare, finance, legal)
- Run 4 algorithms (DAG-DP, Dijkstra, A*, Bellman-Ford) with 2 trials each
- Create 4 visualization charts
- Generate a comprehensive markdown report
- Display results with recommendations

## Usage

### Command-Line Interface

The benchmark suite includes a comprehensive CLI for running benchmarks, analyzing results, and generating workflows.

#### Quick Start

```bash
# Run all benchmarks
poetry run benchmark run

# Run specific algorithms on specific domain
poetry run benchmark run --algorithms dijkstra,astar --domains healthcare --trials 5

# Analyze results
poetry run benchmark analyze --results-file results/benchmark_results_20251031_112112.csv

# Generate visualizations
poetry run benchmark visualize --results-file results/benchmark_results_20251031_112112.csv

# Generate report
poetry run benchmark report --results-file results/benchmark_results_20251031_112112.csv

# Generate sample workflows
poetry run benchmark workflows --domain healthcare --count 3
```

#### Available Commands

**Note**: All CLI commands use the format `poetry run benchmark [COMMAND]` or `poetry run python -m src.cli benchmark [COMMAND]`

- **`run`** - Execute benchmarks across algorithms and domains
  - Options: `--algorithms`, `--domains`, `--trials`, `--timeout`, `--output-dir`
  - Algorithms: `dag_dp`, `dijkstra`, `astar`, `bellman_ford`
  - Domains: `healthcare`, `finance`, `legal`
  - Example: `poetry run benchmark run --algorithms dijkstra,astar --trials 5`

- **`analyze`** - Analyze benchmark results from CSV files
  - Options: `--results-file` (required)
  - Provides summary statistics, best algorithms, and recommendations
  - Example: `poetry run benchmark analyze --results-file results/benchmark_results_20251031_112112.csv`

- **`visualize`** - Generate visualizations from benchmark results
  - Options: `--results-file` (required), `--output-dir` (default: `results/visualizations/`)
  - Creates 5 high-resolution PNG plots (300 DPI):
    1. `algorithm_comparison_time.png` - Execution time by algorithm
    2. `algorithm_comparison_cost.png` - Solution cost by algorithm
    3. `algorithm_comparison_nodes.png` - Path length by algorithm
    4. `scalability.png` - How algorithms scale with workflow size
    5. `cost_comparison.png` - Cost comparison across workflows
  - Example: `poetry run benchmark visualize --results-file results/benchmark_results_20251031_112112.csv --output-dir plots/`

- **`report`** - Generate comprehensive markdown report
  - Options: `--results-file` (required), `--output-dir` (default: `results/`)
  - Creates detailed report with:
    - Executive summary with key findings
    - Test setup and configuration
    - Performance results table
    - Detailed algorithm analysis
    - Production recommendations
    - Embedded visualizations (if available)
  - Output: `benchmark_report_{timestamp}.md`
  - Example: `poetry run benchmark report --results-file results/benchmark_results_20251031_112112.csv`

- **`workflows`** - Generate sample workflows
  - Options: `--domain` (required), `--count`, `--output-dir` (default: `workflows/`)
  - Generates realistic workflows as JSON files
  - Example: `poetry run benchmark workflows --domain healthcare --count 3`

### Helper Scripts

- **`scripts/prepare_web_data.sh`** - Prepare web dashboard data
  - Copies latest benchmark results and visualizations to `web/data/` and `web/images/`
  - Updates `last_updated.json` timestamp
  - Usage: `bash scripts/prepare_web_data.sh`
  - Note: Run after generating benchmark results to update the web dashboard

- **`web/serve.py`** - Start local web server for dashboard
  - Serves web dashboard at `http://localhost:8000`
  - Automatically opens browser
  - Usage: `python web/serve.py`
  - Options:
    - `--port PORT` - Specify port (default: 8000)
    - `--no-open` - Don't open browser automatically

### Example Scripts

The `examples/` directory contains demonstration scripts:

- `examples/benchmark_runner_demo.py` - Benchmark runner usage example
- `examples/healthcare_workflow_demo.py` - Healthcare workflow generation example
- `examples/report_demo.py` - Report generation example
- `examples/visualization_demo.py` - Visualization creation example

Run examples with: `poetry run python examples/[script_name].py`

### Complete Workflow Example

```bash
# 1. Run benchmarks
poetry run benchmark run --trials 5

# 2. Analyze results
poetry run benchmark analyze --results-file results/benchmark_results_*.csv

# 3. Generate visualizations
poetry run benchmark visualize --results-file results/benchmark_results_*.csv

# 4. Generate report
poetry run benchmark report --results-file results/benchmark_results_*.csv

# 5. Prepare web dashboard
bash scripts/prepare_web_data.sh

# 6. View dashboard
python web/serve.py
```

For detailed CLI documentation, see [CLI_GUIDE.md](CLI_GUIDE.md).  
For visualization documentation, see [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md).
