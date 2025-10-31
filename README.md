# Workflow Optimization Benchmark Suite

A comprehensive benchmark suite for evaluating workflow optimization algorithms.

---

## 🚀 Live Demo
- **See it live:** [https://username.github.io/workflow-benchmark](https://username.github.io/workflow-benchmark)
- Direct link to our interactive dashboard comparing algorithm performance, charts, and expert recommendations.
- <img src="web/images/dashboard_screenshot.png" alt="Dashboard Screenshot" width="600" height="auto" style="box-shadow:0 2px 12px #2563eb28; margin:1em 0;"/>
- The dashboard is automatically updated whenever changes are pushed to `main`.

---

## Deployment
- For full, step-by-step deployment, see [DEPLOYMENT.md](DEPLOYMENT.md)
- **Automatic deployments**: Any push to main triggers a GitHub Actions workflow to re-run benchmarks, prepare artifacts, and update GitHub Pages.
- Manual and advanced options also documented.

---

## Screenshots

| Dashboard          | Charts Gallery                 | Algorithm Compare Tool      |
|--------------------|-------------------------------|----------------------------|
| ![dashboard](web/images/dashboard_screenshot.png) | ![charts](web/images/charts_gallery_screenshot.png) | ![compare](web/images/compare_tool_screenshot.png) |

> Replace placeholder PNGs in `web/images/` with sample screenshots of your interface for best effect.

---

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.
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
