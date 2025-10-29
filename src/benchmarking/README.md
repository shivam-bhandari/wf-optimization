# Benchmarking Module

Comprehensive benchmarking framework for workflow optimization algorithms.

> **Note**: This module consolidates all benchmarking functionality. The previous `src/benchmarks/` folder has been merged into `src/benchmarking/` to eliminate redundancy.

## Overview

The benchmarking module provides a robust infrastructure for:
- Running multiple algorithms across multiple workflows
- Collecting detailed performance metrics
- Handling timeouts and errors gracefully
- Computing aggregate statistics
- Persisting results in multiple formats
- Comprehensive progress logging

## Components

### BenchmarkConfig

Configuration dataclass for controlling benchmark behavior.

```python
from src.benchmarking.runner import BenchmarkConfig

config = BenchmarkConfig(
    trials_per_combination=5,     # Number of trials per combination
    timeout_seconds=300.0,        # Timeout for each algorithm run
    random_seed=42,               # Random seed for reproducibility
    objectives=['cost'],          # Optimization objectives to test
    save_results=True,            # Auto-save results to disk
    results_dir=Path('results/')  # Directory for results
)
```

**Parameters:**
- `trials_per_combination` (int): Number of independent trials for each (workflow, algorithm, objective) combination. Default: 5
- `timeout_seconds` (float): Maximum execution time for a single algorithm run in seconds. Default: 300.0
- `random_seed` (int): Random seed for reproducible workflows and algorithms. Default: 42
- `objectives` (List[str]): List of optimization objectives (e.g., 'cost', 'time', 'weight'). Default: ['cost']
- `save_results` (bool): Whether to automatically save results to disk. Default: True
- `results_dir` (Path): Directory path for saving results. Default: Path('results/')

### BenchmarkRunner

Main class orchestrating benchmark execution.

```python
from src.benchmarking.runner import BenchmarkRunner
from src.algorithms.dijkstra import DijkstraOptimizer
from src.datasets.healthcare import HealthcareWorkflowGenerator

# Create workflows
generator = HealthcareWorkflowGenerator(seed=42)
workflows = [
    ('ehr_extraction_1', generator.generate_ehr_extraction()),
    ('ehr_extraction_2', generator.generate_ehr_extraction())
]

# Create algorithms
algorithms = [
    DijkstraOptimizer(name="Dijkstra", source='start', target='end')
]

# Run benchmarks
config = BenchmarkConfig(trials_per_combination=3)
runner = BenchmarkRunner(algorithms, workflows, config)
results_df = runner.run_benchmarks()
```

## Results Format

### Raw Results DataFrame

Columns:
- `workflow_id` (str): Unique workflow identifier
- `algorithm_name` (str): Algorithm name
- `objective` (str): Optimization objective
- `trial_number` (int): Trial index (0-based)
- `path` (List): Solution path as list of node IDs
- `total_cost` (float): Total cost/weight of the solution
- `total_time_ms` (float): Total time in milliseconds
- `execution_time_seconds` (float): Algorithm execution time
- `nodes_explored` (int): Number of nodes in the solution path
- `success` (bool): Whether the run completed successfully
- `error_message` (str): Error description (empty if successful)
- `timestamp` (str): ISO format timestamp of execution

### Aggregate Statistics DataFrame

Columns:
- `workflow_id` (str): Workflow identifier
- `algorithm_name` (str): Algorithm name
- `objective` (str): Optimization objective
- `num_trials` (int): Total number of trials
- `num_successful` (int): Number of successful trials
- `success_rate` (float): Success rate (0.0 to 1.0)
- `mean_cost` (float): Mean total cost
- `std_cost` (float): Standard deviation of cost
- `min_cost` (float): Minimum cost
- `max_cost` (float): Maximum cost
- `mean_execution_time` (float): Mean execution time in seconds
- `std_execution_time` (float): Standard deviation of execution time
- `min_execution_time` (float): Minimum execution time
- `max_execution_time` (float): Maximum execution time
- `mean_nodes_explored` (float): Mean number of nodes explored
- `std_nodes_explored` (float): Standard deviation of nodes explored

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from src.benchmarking.runner import BenchmarkConfig, BenchmarkRunner
from src.algorithms.dijkstra import DijkstraOptimizer
from src.algorithms.bellman_ford import BellmanFordOptimizer
from src.datasets.healthcare import HealthcareWorkflowGenerator

# Generate workflows
generator = HealthcareWorkflowGenerator(seed=42)
workflows = [
    ('ehr_1', generator.generate_ehr_extraction()),
    ('claim_1', generator.generate_insurance_claim_processing())
]

# Create algorithms
algorithms = [
    DijkstraOptimizer(source='start', target='end'),
    BellmanFordOptimizer(source='start', target='end')
]

# Configure and run
config = BenchmarkConfig(
    trials_per_combination=5,
    timeout_seconds=60.0,
    objectives=['cost'],
    save_results=True
)

runner = BenchmarkRunner(algorithms, workflows, config)
results = runner.run_benchmarks()

# Analyze results
print(results.groupby('algorithm_name')['total_cost'].mean())
```

### Multi-Objective Benchmarking

```python
# Test multiple objectives
config = BenchmarkConfig(
    trials_per_combination=10,
    objectives=['cost', 'time'],  # Test both cost and time optimization
    timeout_seconds=120.0
)

runner = BenchmarkRunner(algorithms, workflows, config)
results = runner.run_benchmarks()

# Compare objectives
cost_results = results[results['objective'] == 'cost']
time_results = results[results['objective'] == 'time']
```

### Large-Scale Benchmarking

```python
# Generate many workflows
workflows = []
for i in range(20):
    wf_id = f'ehr_{i}'
    graph = generator.generate_ehr_extraction()
    workflows.append((wf_id, graph))

# Test many algorithms
algorithms = [
    DijkstraOptimizer(source='start', target='end'),
    BellmanFordOptimizer(source='start', target='end'),
    DAGDynamicProgramming(source='start', target='end'),
    AStarOptimizer(source='start', target='end')
]

# Run with longer timeout for large workflows
config = BenchmarkConfig(
    trials_per_combination=10,
    timeout_seconds=600.0,  # 10 minutes
    objectives=['cost', 'time']
)

runner = BenchmarkRunner(algorithms, workflows, config)
results = runner.run_benchmarks()
```

## Output Files

When `save_results=True`, the following files are created:

1. **benchmark_results_YYYYMMDD_HHMMSS.csv** - Raw trial results in CSV format
2. **benchmark_results_YYYYMMDD_HHMMSS.json** - Raw trial results in JSON format
3. **benchmark_aggregates_YYYYMMDD_HHMMSS.csv** - Aggregate statistics in CSV format
4. **benchmark_aggregates_YYYYMMDD_HHMMSS.json** - Aggregate statistics in JSON format

All files include timestamps to prevent overwriting.

## Error Handling

The benchmark runner handles several types of errors:

### Timeouts
- Algorithms exceeding `timeout_seconds` are terminated
- Results marked as failed with error message
- Benchmarking continues for other combinations

```python
# Example with short timeout
config = BenchmarkConfig(timeout_seconds=1.0)  # Very short timeout
runner = BenchmarkRunner(algorithms, workflows, config)
results = runner.run_benchmarks()

# Check timeout failures
timeouts = results[results['error_message'].str.contains('Timeout', na=False)]
print(f"Timeouts: {len(timeouts)}")
```

### Algorithm Exceptions
- Exceptions from algorithm.solve() are caught
- Error message captured in results
- Stack trace logged at DEBUG level
- Execution continues

### Configuration Errors
- Invalid algorithms or workflows raise ValueError at initialization
- Empty algorithm or workflow lists rejected
- Missing required configuration parameters flagged

## Performance Tips

### Parallel Execution
The runner uses `ProcessPoolExecutor` for timeout handling. Each algorithm run executes in a separate process.

### Memory Management
For large-scale benchmarking:
- Use smaller `trials_per_combination` initially
- Test with subset of workflows first
- Monitor memory usage with many parallel runs

### Timeout Selection
Choose appropriate timeouts based on:
- Expected algorithm complexity
- Workflow size (nodes, edges)
- Hardware capabilities

## Logging

The module uses Python's logging framework:

```python
import logging

# Enable debug logging for detailed output
logging.getLogger('src.benchmarking.runner').setLevel(logging.DEBUG)

# Or configure globally
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Log Levels:**
- **INFO**: Progress updates, summary statistics
- **WARNING**: Timeouts, algorithm failures
- **DEBUG**: Detailed execution traces, full stack traces
- **ERROR**: Critical failures (rare)

## Advanced Usage

### Custom Result Analysis

```python
results = runner.run_benchmarks()

# Find best algorithm per workflow
best = results.loc[results.groupby('workflow_id')['total_cost'].idxmin()]
print(best[['workflow_id', 'algorithm_name', 'total_cost']])

# Calculate speedup ratios
baseline = results[results['algorithm_name'] == 'Dijkstra']['execution_time_seconds'].mean()
for algo in results['algorithm_name'].unique():
    algo_time = results[results['algorithm_name'] == algo]['execution_time_seconds'].mean()
    speedup = baseline / algo_time
    print(f"{algo} speedup: {speedup:.2f}x")
```

### Statistical Significance Testing

```python
from scipy import stats

# Compare two algorithms
dijkstra_costs = results[results['algorithm_name'] == 'Dijkstra']['total_cost']
astar_costs = results[results['algorithm_name'] == 'AStar']['total_cost']

t_stat, p_value = stats.ttest_ind(dijkstra_costs, astar_costs)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Box plot of costs by algorithm
plt.figure(figsize=(10, 6))
sns.boxplot(data=results, x='algorithm_name', y='total_cost')
plt.xticks(rotation=45)
plt.title('Cost Distribution by Algorithm')
plt.tight_layout()
plt.savefig('cost_comparison.png')

# Execution time vs cost scatter
plt.figure(figsize=(10, 6))
for algo in results['algorithm_name'].unique():
    algo_data = results[results['algorithm_name'] == algo]
    plt.scatter(algo_data['execution_time_seconds'], 
                algo_data['total_cost'], 
                label=algo, alpha=0.6)
plt.xlabel('Execution Time (s)')
plt.ylabel('Total Cost')
plt.legend()
plt.title('Execution Time vs Solution Cost')
plt.tight_layout()
plt.savefig('time_vs_cost.png')
```

## Testing

Run the example demonstration:

```bash
python examples/benchmark_runner_demo.py
```

## Integration with Existing Code

The benchmark runner integrates seamlessly with:
- All algorithm implementations in `src/algorithms/`
- All workflow generators in `src/datasets/`
- Existing evaluation metrics in `src/evaluation/`

## Future Enhancements

Potential future additions:
- Distributed execution across multiple machines
- Real-time progress dashboard
- Automatic hyperparameter tuning
- A/B testing framework
- Result comparison across runs
- Regression detection for performance

