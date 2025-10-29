# Benchmark Runner - Quick Start Guide

## Installation

The benchmark runner is already integrated into your project. No additional installation needed!

## 5-Minute Tutorial

### Step 1: Import Required Modules

```python
from src.benchmarking.runner import BenchmarkConfig, BenchmarkRunner
from src.algorithms.dijkstra import DijkstraOptimizer
from src.algorithms.bellman_ford import BellmanFordOptimizer
from src.datasets.healthcare import HealthcareWorkflowGenerator
```

### Step 2: Generate Test Workflows

```python
# Create workflow generator
generator = HealthcareWorkflowGenerator(seed=42)

# Generate some workflows
workflows = [
    ('ehr_extraction', generator.generate_ehr_extraction()),
    ('insurance_claim', generator.generate_insurance_claim_processing()),
]
```

### Step 3: Set Up Algorithms

```python
# Get source and target from workflow
graph = workflows[0][1]
source = [n for n, d in graph.in_degree() if d == 0][0]
target = [n for n, d in graph.out_degree() if d == 0][0]

# Create algorithm instances
algorithms = [
    DijkstraOptimizer(name="Dijkstra", source=source, target=target),
    BellmanFordOptimizer(name="BellmanFord", source=source, target=target),
]
```

### Step 4: Configure and Run

```python
# Configure benchmark
config = BenchmarkConfig(
    trials_per_combination=3,  # 3 trials per combination
    timeout_seconds=60.0,      # 60 second timeout
    objectives=['cost'],       # Optimize for cost
    save_results=True          # Save to results/
)

# Create runner and execute
runner = BenchmarkRunner(algorithms, workflows, config)
results_df = runner.run_benchmarks()
```

### Step 5: Analyze Results

```python
# View raw results
print(results_df.head())

# Summary by algorithm
print("\nMean cost by algorithm:")
print(results_df.groupby('algorithm_name')['total_cost'].mean())

# Success rate
print("\nSuccess rate:")
print(results_df.groupby('algorithm_name')['success'].mean())

# Execution time statistics
print("\nExecution time stats:")
print(results_df.groupby('algorithm_name')['execution_time_seconds'].describe())
```

## Complete Example

Save this as `my_benchmark.py`:

```python
from pathlib import Path
from src.benchmarking.runner import BenchmarkConfig, BenchmarkRunner
from src.algorithms.dijkstra import DijkstraOptimizer
from src.algorithms.bellman_ford import BellmanFordOptimizer
from src.datasets.healthcare import HealthcareWorkflowGenerator

def main():
    # Generate workflows
    generator = HealthcareWorkflowGenerator(seed=42)
    workflows = [
        ('ehr_1', generator.generate_ehr_extraction()),
        ('ehr_2', generator.generate_ehr_extraction()),
        ('claim_1', generator.generate_insurance_claim_processing()),
    ]
    
    # Get source/target from first workflow
    graph = workflows[0][1]
    source = [n for n, d in graph.in_degree() if d == 0][0]
    target = [n for n, d in graph.out_degree() if d == 0][0]
    
    # Create algorithms
    algorithms = [
        DijkstraOptimizer(name="Dijkstra", source=source, target=target),
        BellmanFordOptimizer(name="BellmanFord", source=source, target=target),
    ]
    
    # Configure
    config = BenchmarkConfig(
        trials_per_combination=5,
        timeout_seconds=60.0,
        objectives=['cost'],
        save_results=True,
        results_dir=Path('results/')
    )
    
    # Run
    print("Starting benchmark...")
    runner = BenchmarkRunner(algorithms, workflows, config)
    results = runner.run_benchmarks()
    
    # Analyze
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print("\nMean Cost by Algorithm:")
    print(results.groupby('algorithm_name')['total_cost'].mean())
    
    print("\nMean Execution Time (seconds):")
    print(results.groupby('algorithm_name')['execution_time_seconds'].mean())
    
    print("\nSuccess Rate:")
    print(results.groupby('algorithm_name')['success'].mean())
    
    print("\n" + "="*80)
    print(f"Results saved to: {config.results_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
```

Run it:
```bash
poetry run python my_benchmark.py
```

## Common Configurations

### Quick Test (Fast)
```python
config = BenchmarkConfig(
    trials_per_combination=1,
    timeout_seconds=10.0,
)
```

### Production Run (Comprehensive)
```python
config = BenchmarkConfig(
    trials_per_combination=10,
    timeout_seconds=300.0,
    objectives=['cost', 'time'],
)
```

### Large Scale (Many Workflows)
```python
config = BenchmarkConfig(
    trials_per_combination=5,
    timeout_seconds=600.0,
    use_multiprocessing=True,
)
```

### Testing Mode (Thread-safe)
```python
config = BenchmarkConfig(
    trials_per_combination=2,
    timeout_seconds=5.0,
    save_results=False,
    use_multiprocessing=False,
)
```

## Result File Locations

After running, find your results in:
```
results/
├── benchmark_results_20250128_143022.csv      # Raw data (CSV)
├── benchmark_results_20250128_143022.json     # Raw data (JSON)
├── benchmark_aggregates_20250128_143022.csv   # Statistics (CSV)
└── benchmark_aggregates_20250128_143022.json  # Statistics (JSON)
```

## Analyzing Results with Pandas

```python
import pandas as pd

# Load results
df = pd.read_csv('results/benchmark_results_20250128_143022.csv')

# Filter successful runs only
successful = df[df['success']]

# Best algorithm per workflow
best = successful.loc[successful.groupby('workflow_id')['total_cost'].idxmin()]
print(best[['workflow_id', 'algorithm_name', 'total_cost']])

# Algorithm comparison
comparison = successful.groupby('algorithm_name').agg({
    'total_cost': ['mean', 'std', 'min', 'max'],
    'execution_time_seconds': ['mean', 'std'],
    'nodes_explored': ['mean']
})
print(comparison)

# Visualization
import matplotlib.pyplot as plt
successful.boxplot(column='total_cost', by='algorithm_name')
plt.show()
```

## Troubleshooting

### Problem: Timeouts
**Symptom:** Many runs showing timeout errors
**Solution:** Increase `timeout_seconds` in config

### Problem: All runs fail
**Symptom:** All `success` values are False
**Solution:** 
1. Check algorithm source/target nodes match workflow
2. Check workflow has valid path from source to target
3. Review error messages in `error_message` column

### Problem: Process permission error
**Symptom:** "[Errno 1] Operation not permitted"
**Solution:** Set `use_multiprocessing=False` in config

### Problem: Out of memory
**Symptom:** System runs out of RAM
**Solution:** 
1. Reduce `trials_per_combination`
2. Process fewer workflows at once
3. Use `save_results=False` if not needed

## Next Steps

1. **Read Full Documentation:** `src/benchmarking/README.md`
2. **Run Example Demo:** `python examples/benchmark_runner_demo.py`
3. **Review Tests:** `tests/test_benchmark_runner.py`
4. **Customize for Your Needs:** Modify config and algorithms

## Getting Help

- Full documentation: `src/benchmarking/README.md`
- Implementation summary: `BENCHMARK_RUNNER_SUMMARY.md`
- Test examples: `tests/test_benchmark_runner.py`
- Demo script: `examples/benchmark_runner_demo.py`

Happy benchmarking! 🚀

