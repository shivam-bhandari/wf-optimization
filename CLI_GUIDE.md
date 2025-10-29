# Benchmark CLI Guide

A comprehensive command-line interface for running benchmarks, analyzing results, and generating workflows using the Click framework.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [benchmark run](#benchmark-run)
  - [benchmark analyze](#benchmark-analyze)
  - [benchmark workflows](#benchmark-workflows)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

First, ensure all dependencies are installed:

```bash
# Install dependencies
poetry install

# Or if using pip directly
pip install click
```

## Quick Start

The CLI provides three main commands:

1. **`run`** - Execute benchmarks
2. **`analyze`** - Analyze benchmark results
3. **`workflows`** - Generate sample workflows

### Basic Usage

```bash
# Show help
poetry run python -m src.cli --help

# Show version
poetry run python -m src.cli --version

# Enable verbose logging
poetry run python -m src.cli --verbose [COMMAND]
```

## Commands

### benchmark run

Run a complete benchmark suite across specified algorithms and domains.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--algorithms` | str | all | Comma-separated list: `dijkstra,astar,dag_dp,bellman_ford` |
| `--domains` | str | all | Comma-separated list: `healthcare,finance,legal` |
| `--trials` | int | 3 | Number of trials per workflow-algorithm combination |
| `--timeout` | int | 300 | Timeout in seconds for each algorithm run |
| `--output-dir` | path | `results/` | Output directory for results |

#### Algorithm Options

- **`dag_dp`** - DAG Dynamic Programming (optimal for DAGs, O(V+E))
- **`dijkstra`** - Dijkstra's algorithm (optimal for non-negative weights, O((V+E)log V))
- **`astar`** - A* search (optimal with admissible heuristic)
- **`bellman_ford`** - Bellman-Ford algorithm (handles negative weights, O(VE))


#### Domain Options

- **`healthcare`** - Medical record extraction, insurance claims, patient intake
- **`finance`** - Loan approval, fraud detection, risk assessment
- **`legal`** - Contract review, compliance checking, document redlining

#### Usage Examples

```bash
# Run all algorithms on all domains (default)
poetry run python -m src.cli run

# Run with custom trials
poetry run python -m src.cli run --trials 5

# Run specific algorithms on specific domain
poetry run python -m src.cli run --algorithms dijkstra,astar --domains healthcare --trials 3

# Run with custom timeout and output directory
poetry run python -m src.cli run --timeout 600 --output-dir my_results/ --trials 10

# Run Bellman-Ford and DAG-DP on finance domain
poetry run python -m src.cli run --algorithms bellman_ford,dag_dp --domains finance
```

#### Output

The command generates:
- CSV and JSON files with raw results
- CSV and JSON files with aggregate statistics
- Console output with summary table showing:
  - Algorithm name
  - Average execution time
  - Average cost
  - Number of runs
  - Success rate

Files are saved with timestamps (e.g., `benchmark_results_20251029_153133.csv`).

---

### benchmark analyze

Analyze existing benchmark results from a CSV file.

#### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--results-file` | path | Yes | Path to CSV results file to analyze |

#### Usage Examples

```bash
# Analyze specific results file
poetry run python -m src.cli analyze --results-file results/benchmark_results_20251029_153133.csv

# With verbose output
poetry run python -m src.cli --verbose analyze --results-file results/benchmark_results.csv
```

#### Output

The analysis provides:

1. **Summary Statistics**
   - Execution time (mean, std, min, max)
   - Total cost (mean, std, min, max)
   - Path length (mean, std, min, max)

2. **Best Algorithms**
   - Fastest execution
   - Lowest cost
   - Most reliable (highest success rate)

3. **Recommendation**
   - Overall best algorithm or trade-offs

---

### benchmark workflows

Generate sample workflows for testing and demonstration.

#### Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `--domain` | choice | - | Yes | Domain: `healthcare`, `finance`, or `legal` |
| `--count` | int | 1 | No | Number of workflows to generate |
| `--output-dir` | path | `workflows/` | No | Output directory for workflow files |

#### Workflow Types by Domain

**Healthcare:**
- `medical_record_extraction` - Document processing, entity extraction, ICD coding
- `insurance_claim_processing` - Verification, fraud detection, payment
- `patient_intake` - Registration, insurance verification, scheduling

**Finance:**
- `loan_approval` - Credit checks, risk scoring, underwriting
- `fraud_detection` - Pattern analysis, ML scoring, analyst review
- `risk_assessment` - Portfolio analysis, VaR calculation, stress testing

**Legal:**
- `contract_review` - Clause extraction, risk detection, attorney review
- `compliance_check` - Regulation mapping, gap analysis, audit trails
- `document_redlining` - Version comparison, risk assessment, attorney approval

#### Usage Examples

```bash
# Generate 1 healthcare workflow
poetry run python -m src.cli workflows --domain healthcare

# Generate 5 finance workflows
poetry run python -m src.cli workflows --domain finance --count 5

# Generate legal workflows in custom directory
poetry run python -m src.cli workflows --domain legal --count 3 --output-dir my_workflows/
```

#### Output

Workflows are saved as JSON files with the naming pattern:
```
{domain}_{workflow_type}_{timestamp}.json
```

Each workflow file contains:
- **Metadata** - Workflow ID, type, domain, statistics
- **Nodes** - Task nodes with execution times, costs, resource requirements
- **Edges** - Dependencies with data transfer costs

---

## Examples

### Complete Workflow: Run, Analyze, Generate

```bash
# Step 1: Generate sample workflows
poetry run python -m src.cli workflows --domain healthcare --count 3
poetry run python -m src.cli workflows --domain finance --count 3

# Step 2: Run benchmarks on all algorithms and domains
poetry run python -m src.cli run --trials 5 --timeout 300

# Step 3: Analyze the results
poetry run python -m src.cli analyze --results-file results/benchmark_results_20251029_153133.csv
```

### Performance Comparison

```bash
# Compare DAG-DP vs Dijkstra on healthcare workflows
poetry run python -m src.cli run \
  --algorithms dag_dp,dijkstra \
  --domains healthcare \
  --trials 10

# Compare all algorithms on a single domain
poetry run python -m src.cli run \
  --domains finance \
  --trials 5 \
  --timeout 120
```

### Testing Different Configurations

```bash
# Quick test with minimal trials
poetry run python -m src.cli run --trials 1 --timeout 30

# Comprehensive benchmark with many trials
poetry run python -m src.cli run --trials 20 --timeout 600

# Test a subset of algorithms
poetry run python -m src.cli run --algorithms dag_dp,dijkstra --trials 10
```

---

## Troubleshooting

### Command Not Found

If you get `command not found` errors:

```bash
# Make sure you're using poetry run
poetry run python -m src.cli --help

# Or activate the virtual environment first
poetry shell
python -m src.cli --help
```

### Module Import Errors

If you get import errors:

```bash
# Reinstall dependencies
poetry install

# Or install click directly
poetry run pip install click
```

### Permission Errors

If you get permission errors for output directories:

```bash
# Create directory manually
mkdir -p results workflows

# Or specify a different directory
poetry run python -m src.cli run --output-dir ~/my_results
```

### Timeout Issues

If algorithms frequently timeout:

```bash
# Increase timeout
poetry run python -m src.cli run --timeout 600

# Or reduce workflow complexity
poetry run python -m src.cli run --domains healthcare --timeout 120
```

### Memory Issues

If you run out of memory:

```bash
# Reduce number of trials
poetry run python -m src.cli run --trials 1

# Run fewer algorithms at once
poetry run python -m src.cli run --algorithms dijkstra --trials 3
```

---

## Advanced Usage

### Scripting

You can use the CLI in shell scripts:

```bash
#!/bin/bash
# benchmark_script.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results_${TIMESTAMP}"

# Run benchmarks
poetry run python -m src.cli run \
  --output-dir "$OUTPUT_DIR" \
  --trials 10 \
  --timeout 300

# Find the latest results file
RESULTS_FILE=$(ls -t "$OUTPUT_DIR"/benchmark_results_*.csv | head -1)

# Analyze results
poetry run python -m src.cli analyze --results-file "$RESULTS_FILE"
```

### Batch Processing

Run multiple benchmark configurations:

```bash
# Benchmark each domain separately
for domain in healthcare finance legal; do
  poetry run python -m src.cli run \
    --domains $domain \
    --trials 5 \
    --output-dir "results_${domain}"
done

# Benchmark each algorithm separately
for algo in dag_dp dijkstra astar bellman_ford; do
  poetry run python -m src.cli run \
    --algorithms $algo \
    --trials 5 \
    --output-dir "results_${algo}"
done
```

---

## Output File Formats

### CSV Results File

The CSV file contains one row per trial with columns:
- `workflow_id` - Workflow identifier
- `algorithm_name` - Algorithm name
- `objective` - Optimization objective
- `trial_number` - Trial index
- `path` - Solution path (node IDs)
- `total_cost` - Total cost of solution
- `total_time_ms` - Total time in milliseconds
- `execution_time_seconds` - Algorithm execution time
- `nodes_explored` - Number of nodes in path
- `success` - Whether run completed successfully
- `error_message` - Error description (if failed)
- `timestamp` - ISO format timestamp

### JSON Workflow File

The JSON file contains:
- `metadata` - Workflow metadata (ID, type, domain, statistics)
- `nodes` - Dictionary of node ID to node attributes
- `edges` - List of edges with source, target, and attributes

---

## Tips and Best Practices

1. **Start Small** - Begin with `--trials 1` to test quickly
2. **Use Specific Algorithms** - Target specific algorithms when testing
3. **Increase Timeout** - For complex workflows, use longer timeouts
4. **Enable Verbose** - Use `--verbose` for debugging
5. **Save Results** - Results are automatically timestamped, so run freely
6. **Compare Algorithms** - Run the same configuration multiple times for consistency
7. **Check Success Rates** - If success rates are low, increase timeout
8. **Use Appropriate Algorithms** - DAG-DP for DAGs, Dijkstra for general graphs

---

## Color Output

The CLI uses colors to enhance readability:
- 🟢 **Green** - Success messages, positive metrics
- 🔴 **Red** - Error messages, failures
- 🟡 **Yellow** - Warnings, section headers
- 🔵 **Cyan** - Information, configuration values

---

## Version Information

Current version: **0.1.0**

Check version:
```bash
poetry run python -m src.cli --version
```

---

## Additional Resources

- **Main README**: See `README.md` for project overview
- **Benchmark Documentation**: See `docs/BENCHMARK_RUNNER_SUMMARY.md`
- **Test Files**: See `tests/` for usage examples
- **Example Scripts**: See `examples/` for demonstration code

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review example scripts in `examples/`
3. Run with `--verbose` flag for detailed output
4. Check the test files for usage patterns

