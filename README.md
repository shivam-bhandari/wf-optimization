# Workflow Optimization Benchmark Suite

A comprehensive benchmark suite for evaluating workflow optimization algorithms.

## Project Structure

```
├── src/
│   ├── benchmarking/  # Core benchmarking code
│   ├── algorithms/    # Optimization algorithm implementations
│   ├── datasets/      # Workflow generation
│   ├── evaluation/    # Metrics and analysis
│   ├── reporting/     # Reports
│   └── cli.py         # Command-line interface
├── tests/             # Unit tests
├── configs/           # Benchmark configurations
├── docs/              # Documentation
├── examples/          # Example scripts
├── results/           # Benchmark output data
└── workflows/         # Generated workflow files

```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Install Dependencies

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

### Command-Line Interface (CLI)

The benchmark suite includes a comprehensive CLI for running benchmarks, analyzing results, and generating workflows.

#### Quick Start

```bash
# Run all benchmarks
poetry run python -m src.cli run

# Run specific algorithms on specific domain
poetry run python -m src.cli run --algorithms dijkstra,astar --domains healthcare --trials 5

# Analyze results
poetry run python -m src.cli analyze --results-file results/benchmark_results_20251029_153133.csv

# Generate sample workflows
poetry run python -m src.cli workflows --domain healthcare --count 3
```

#### Available Commands

- **`run`** - Execute benchmarks across algorithms and domains
  - Options: `--algorithms`, `--domains`, `--trials`, `--timeout`, `--output-dir`
  - Algorithms: `dag_dp`, `dijkstra`, `astar`, `bellman_ford`
  - Domains: `healthcare`, `finance`, `legal`

- **`analyze`** - Analyze benchmark results from CSV files
  - Options: `--results-file` (required)
  - Provides summary statistics, best algorithms, and recommendations

- **`visualize`** - Generate visualizations from benchmark results
  - Options: `--results-file` (required), `--output-dir`
  - Creates 5 high-resolution plots: time comparison, cost comparison, scalability, etc.

- **`report`** - Generate comprehensive markdown report
  - Options: `--results-file` (required), `--output-dir`
  - Creates detailed report with analysis, recommendations, and embedded visualizations

- **`workflows`** - Generate sample workflows
  - Options: `--domain` (required), `--count`, `--output-dir`
  - Generates realistic workflows as JSON files

For detailed CLI documentation, see [CLI_GUIDE.md](CLI_GUIDE.md).
For visualization documentation, see [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md).

#### Examples

```bash
# Run comprehensive benchmark
poetry run python -m src.cli run --trials 10 --timeout 600

# Compare DAG-DP and Dijkstra on healthcare workflows
poetry run python -m src.cli run --algorithms dag_dp,dijkstra --domains healthcare

# Generate workflows for testing
poetry run python -m src.cli workflows --domain finance --count 5

# Analyze with verbose output
poetry run python -m src.cli --verbose analyze --results-file results/latest.csv

# Generate visualizations
poetry run python -m src.cli visualize --results-file results/benchmark_results_20251029_153133.csv

# Generate comprehensive report
poetry run python -m src.cli report --results-file results/benchmark_results_20251029_153133.csv
```

### Running Tests

```bash
poetry run pytest
```

### Running with Coverage

```bash
poetry run pytest --cov=src
```
