# Visualization Guide

Comprehensive guide for generating and customizing visualizations of benchmark results.

## Overview

The visualization module provides three powerful functions for analyzing benchmark results:

1. **`plot_algorithm_comparison`** - Compare algorithms across specific metrics
2. **`plot_execution_time_scalability`** - Analyze algorithm scalability
3. **`plot_cost_comparison`** - Compare solution quality across algorithms

All visualizations are generated using matplotlib and saved as high-resolution PNG files (300 DPI).

## Quick Start

### Using the CLI (Recommended)

The easiest way to generate visualizations is through the CLI:

```bash
# Generate all visualizations from benchmark results
poetry run python -m src.cli visualize --results-file results/benchmark_results_20251029_153133.csv

# Specify custom output directory
poetry run python -m src.cli visualize --results-file results/latest.csv --output-dir plots/
```

This will create 5 visualization files:
- `algorithm_comparison_time.png` - Execution time comparison
- `algorithm_comparison_cost.png` - Cost comparison
- `algorithm_comparison_nodes.png` - Path length comparison
- `scalability.png` - Scalability analysis
- `cost_comparison.png` - Cost comparison across workflows

### Using the Demo Script

Run the demonstration script:

```bash
poetry run python examples/visualization_demo.py
```

### Using Python API

For programmatic access:

```python
import pandas as pd
from src.evaluation.visualizations import (
    plot_algorithm_comparison,
    plot_execution_time_scalability,
    plot_cost_comparison
)

# Load benchmark results
df = pd.read_csv('results/benchmark_results.csv')

# Generate visualizations
plot_algorithm_comparison(df, metric='execution_time_seconds', 
                         save_path='plots/time_comparison.png')
plot_execution_time_scalability(df, save_path='plots/scalability.png')
plot_cost_comparison(df, save_path='plots/cost_comparison.png')
```

## Function Reference

### 1. plot_algorithm_comparison

Compare algorithms across a specific metric using a bar chart.

#### Signature

```python
plot_algorithm_comparison(
    results_df: pd.DataFrame,
    metric: str = 'execution_time_seconds',
    save_path: Optional[str] = None
) -> plt.Figure
```

#### Parameters

- **`results_df`** (pd.DataFrame): DataFrame with benchmark results containing:
  - `algorithm_name`: Algorithm identifier
  - `execution_time_seconds`: Execution time in seconds
  - `total_cost`: Total solution cost
  - `nodes_explored`: Number of nodes in solution path
  - `success`: Boolean indicating success

- **`metric`** (str, optional): Metric to plot. Options:
  - `'execution_time_seconds'` (default) - Algorithm execution time
  - `'total_cost'` - Total cost of solution
  - `'nodes_explored'` - Path length

- **`save_path`** (str, optional): File path to save PNG. If None, displays with `plt.show()`.

#### Returns

- **`plt.Figure`**: Matplotlib figure object

#### Features

- Grouped bar chart with different colors per algorithm
- Error bars showing standard deviation
- Value labels on top of each bar
- Grid lines for readability
- Automatic sorting by metric value

#### Example

```python
# Compare execution times
fig = plot_algorithm_comparison(
    df,
    metric='execution_time_seconds',
    save_path='plots/time_comparison.png'
)

# Compare costs
fig = plot_algorithm_comparison(
    df,
    metric='total_cost',
    save_path='plots/cost_comparison.png'
)

# Display instead of saving
fig = plot_algorithm_comparison(df, metric='nodes_explored')
```

#### Output Example

Bar chart showing:
- X-axis: Algorithm names
- Y-axis: Metric values (e.g., "Execution Time (seconds)")
- Different colored bars for each algorithm
- Value labels above each bar
- Error bars showing variance

---

### 2. plot_execution_time_scalability

Analyze how execution time scales with workflow size.

#### Signature

```python
plot_execution_time_scalability(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure
```

#### Parameters

- **`results_df`** (pd.DataFrame): DataFrame with benchmark results containing:
  - `algorithm_name`: Algorithm identifier
  - `execution_time_seconds`: Execution time
  - `nodes_explored` or `workflow_id`: Used to determine workflow size

- **`save_path`** (str, optional): File path to save PNG. If None, displays with `plt.show()`.

#### Returns

- **`plt.Figure`**: Matplotlib figure object

#### Features

- Line plot with different colors per algorithm
- Markers at each data point
- Automatic log scale if time range > 100x
- Legend showing all algorithms
- Grid lines for readability

#### Example

```python
# Generate scalability plot
fig = plot_execution_time_scalability(
    df,
    save_path='plots/scalability.png'
)

# Display interactively
fig = plot_execution_time_scalability(df)
```

#### Output Example

Line chart showing:
- X-axis: Workflow size (number of nodes)
- Y-axis: Execution time (seconds, possibly log scale)
- Different colored lines for each algorithm
- Markers at data points
- Legend identifying each algorithm

---

### 3. plot_cost_comparison

Compare solution costs across algorithms and workflows.

#### Signature

```python
plot_cost_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure
```

#### Parameters

- **`results_df`** (pd.DataFrame): DataFrame with benchmark results containing:
  - `algorithm_name`: Algorithm identifier
  - `total_cost`: Total solution cost
  - `workflow_id` (optional): Workflow identifier for grouping

- **`save_path`** (str, optional): File path to save PNG. If None, displays with `plt.show()`.

#### Returns

- **`plt.Figure`**: Matplotlib figure object

#### Features

- Grouped bar chart showing costs per workflow
- Side-by-side bars for different algorithms
- Highlights workflows where algorithms differ (suboptimal solutions)
- Red shading indicates cost discrepancies
- Green border on optimal algorithm

#### Example

```python
# Compare costs across workflows
fig = plot_cost_comparison(
    df,
    save_path='plots/cost_comparison.png'
)

# Display interactively
fig = plot_cost_comparison(df)
```

#### Output Example

Bar chart showing:
- X-axis: Workflows
- Y-axis: Total cost
- Grouped bars for each algorithm per workflow
- Red shading where algorithms disagree (>1% difference)
- Optimal algorithm highlighted with green border

---

## Customization

### Changing Figure Size

All functions use `figsize=(10, 6)` by default. To customize, modify the source:

```python
# In src/evaluation/visualizations.py
fig, ax = plt.subplots(figsize=(12, 8))  # Larger figure
```

### Changing Colors

Colors are automatically assigned using matplotlib colormaps:

```python
# Bar charts use Set3
colors = plt.cm.Set3(np.linspace(0, 1, len(grouped)))

# Line charts use tab10
colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
```

To customize:

```python
# Custom colors
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
ax.bar(x, y, color=custom_colors)
```

### Changing Style

The default style is `seaborn-v0_8-darkgrid` with fallback to `default`. To change:

```python
import matplotlib.pyplot as plt

# Use a different style
plt.style.use('ggplot')
# or
plt.style.use('bmh')
# or
plt.style.use('fivethirtyeight')
```

### Adding Custom Annotations

```python
# After creating the plot
fig = plot_algorithm_comparison(df, metric='total_cost')
ax = fig.axes[0]

# Add a horizontal line for target
ax.axhline(y=100, color='r', linestyle='--', label='Target')

# Add text annotation
ax.text(0.5, 0.95, 'Important Note', 
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat'))

plt.savefig('custom_plot.png', dpi=300)
```

---

## Complete Workflow Example

### 1. Run Benchmarks

```bash
# Run comprehensive benchmark
poetry run python -m src.cli run --trials 10 --timeout 300
```

### 2. Generate Visualizations

```bash
# Create all visualizations
poetry run python -m src.cli visualize --results-file results/benchmark_results_20251029_153133.csv
```

### 3. Custom Analysis (Python)

```python
import pandas as pd
from src.evaluation.visualizations import *

# Load results
df = pd.read_csv('results/benchmark_results_20251029_153133.csv')

# Filter to successful runs only
df = df[df['success'] == True]

# Compare specific algorithms
df_subset = df[df['algorithm_name'].isin(['dijkstra_optimizer', 'dag_dynamic_programming'])]

# Generate comparison
fig = plot_algorithm_comparison(
    df_subset,
    metric='execution_time_seconds',
    save_path='plots/dijkstra_vs_dag_dp.png'
)

# Analyze scalability
fig = plot_execution_time_scalability(
    df_subset,
    save_path='plots/scalability_subset.png'
)

# Compare costs
fig = plot_cost_comparison(
    df_subset,
    save_path='plots/cost_subset.png'
)
```

---

## Troubleshooting

### Issue: "No valid data to plot"

**Cause:** All runs failed or have NaN values

**Solution:**
```python
# Check for successful runs
print(df['success'].sum())

# Check for NaN values
print(df[['execution_time_seconds', 'total_cost']].isna().sum())

# Filter and retry
df_filtered = df[df['success'] == True].dropna()
plot_algorithm_comparison(df_filtered)
```

### Issue: "Column not found"

**Cause:** DataFrame missing required columns

**Solution:**
```python
# Check available columns
print(df.columns.tolist())

# Ensure required columns exist
required = ['algorithm_name', 'execution_time_seconds', 'total_cost']
missing = [col for col in required if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
```

### Issue: matplotlib style warnings

**Cause:** Seaborn style not available

**Solution:** The code automatically falls back to 'default' style. To manually set:
```python
import matplotlib.pyplot as plt
plt.style.use('default')
```

### Issue: Plots too small/large

**Solution:** Adjust DPI or figure size:
```python
# Higher resolution
plt.savefig('plot.png', dpi=600)  # Instead of 300

# Larger figure
fig, ax = plt.subplots(figsize=(14, 10))  # Instead of (10, 6)
```

---

## Advanced Usage

### Batch Generation

Generate visualizations for multiple result files:

```python
from pathlib import Path
import pandas as pd
from src.evaluation.visualizations import *

results_dir = Path('results')
output_dir = Path('plots')
output_dir.mkdir(exist_ok=True)

# Process all result files
for results_file in results_dir.glob('benchmark_results_*.csv'):
    timestamp = results_file.stem.replace('benchmark_results_', '')
    df = pd.read_csv(results_file)
    
    # Create subdirectory for this run
    run_dir = output_dir / timestamp
    run_dir.mkdir(exist_ok=True)
    
    # Generate all plots
    plot_algorithm_comparison(df, metric='execution_time_seconds',
                            save_path=run_dir / 'time.png')
    plot_algorithm_comparison(df, metric='total_cost',
                            save_path=run_dir / 'cost.png')
    plot_execution_time_scalability(df, save_path=run_dir / 'scalability.png')
    plot_cost_comparison(df, save_path=run_dir / 'cost_detail.png')
    
    print(f"Processed {results_file.name}")
```

### Comparing Multiple Benchmark Runs

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load multiple result files
df1 = pd.read_csv('results/run1.csv')
df2 = pd.read_csv('results/run2.csv')

# Add run identifier
df1['run'] = 'Run 1'
df2['run'] = 'Run 2'

# Combine
df_combined = pd.concat([df1, df2], ignore_index=True)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for run_name, ax in zip(['Run 1', 'Run 2'], axes):
    df_run = df_combined[df_combined['run'] == run_name]
    
    # Group by algorithm
    grouped = df_run.groupby('algorithm_name')['execution_time_seconds'].mean()
    
    ax.bar(range(len(grouped)), grouped.values)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.index, rotation=45, ha='right')
    ax.set_title(run_name)
    ax.set_ylabel('Execution Time (seconds)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/run_comparison.png', dpi=300)
```

---

## Best Practices

1. **Always filter successful runs first**
   ```python
   df = df[df['success'] == True]
   ```

2. **Check data quality before plotting**
   ```python
   print(df.describe())
   print(df['algorithm_name'].value_counts())
   ```

3. **Use consistent naming conventions**
   - Save plots with descriptive names
   - Include timestamps in filenames
   - Organize by experiment/run

4. **Generate multiple views**
   - Time comparison
   - Cost comparison
   - Scalability analysis
   - Per-workflow details

5. **Document experiments**
   - Save configuration with plots
   - Include README with explanations
   - Version control your analysis scripts

6. **High-quality outputs**
   - Use 300+ DPI for publications
   - Save as both PNG and PDF if needed
   - Include proper titles and labels

---

## Output Formats

### PNG (Default)

```python
plot_algorithm_comparison(df, save_path='plot.png')
```

**Pros:** Universal support, good for web/presentations
**Cons:** Raster format (pixels, not scalable)

### PDF (Vector Graphics)

```python
fig = plot_algorithm_comparison(df)
fig.savefig('plot.pdf', format='pdf', bbox_inches='tight')
```

**Pros:** Scalable, perfect for publications
**Cons:** Larger file size

### SVG (Web Graphics)

```python
fig = plot_algorithm_comparison(df)
fig.savefig('plot.svg', format='svg', bbox_inches='tight')
```

**Pros:** Scalable, editable in Illustrator/Inkscape
**Cons:** May have font embedding issues

---

## Integration with Reports

### Jupyter Notebook

```python
import pandas as pd
from src.evaluation.visualizations import *

# Load results
df = pd.read_csv('results/benchmark_results.csv')

# Display inline
fig = plot_algorithm_comparison(df, metric='execution_time_seconds')
plt.show()

# Save for later
plot_cost_comparison(df, save_path='report_figure1.png')
```

### LaTeX Document

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{plots/algorithm_comparison_time.png}
  \caption{Execution time comparison across algorithms}
  \label{fig:time_comparison}
\end{figure}
```

### Markdown Report

```markdown
## Results

The execution time comparison shows DAG-DP outperforms other algorithms:

![Execution Time Comparison](plots/algorithm_comparison_time.png)

For detailed scalability analysis, see:

![Scalability](plots/scalability.png)
```

---

## FAQ

**Q: Can I customize colors?**
A: Yes, modify the colormap in the function or use custom color lists.

**Q: How do I add more metrics?**
A: Extend `plot_algorithm_comparison` to accept additional metrics.

**Q: Can I plot specific algorithms only?**
A: Yes, filter the DataFrame before passing to functions:
```python
df_subset = df[df['algorithm_name'].isin(['dijkstra_optimizer', 'astar_optimizer'])]
plot_algorithm_comparison(df_subset)
```

**Q: How do I change font sizes?**
A: Use matplotlib rcParams:
```python
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
```

**Q: Can I export to Excel with embedded plots?**
A: Use pandas ExcelWriter with xlsxwriter:
```python
import pandas as pd
import xlsxwriter

# Save plot
plot_algorithm_comparison(df, save_path='temp.png')

# Create Excel with image
with pd.ExcelWriter('report.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Data')
    workbook = writer.book
    worksheet = writer.sheets['Data']
    worksheet.insert_image('K2', 'temp.png')
```

---

## Support

For issues or questions:
1. Check this guide for common solutions
2. Review function docstrings: `help(plot_algorithm_comparison)`
3. Examine example scripts in `examples/`
4. Check matplotlib documentation: https://matplotlib.org/

---

## References

- **Matplotlib Documentation**: https://matplotlib.org/stable/
- **Pandas Visualization**: https://pandas.pydata.org/docs/user_guide/visualization.html
- **Seaborn**: https://seaborn.pydata.org/
- **Color Palettes**: https://matplotlib.org/stable/tutorials/colors/colormaps.html

