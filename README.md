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

### Quick Deployment

```bash
# Run the interactive deployment helper
./deploy.sh
```

Choose from:
- 🌐 **Local Network** - Share on same WiFi/LAN
- 🔗 **ngrok** - Share with anyone via tunnel
- 📄 **GitHub Pages** - Free public hosting
- ☁️ **Netlify** - Quick cloud deployment
- 📦 **Docker** - Containerized deployment

### Detailed Deployment Guide

For comprehensive deployment instructions, see **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**

The guide includes:
- Step-by-step instructions for all deployment methods
- Cloud platform deployments (AWS, GCP, Azure)
- Docker containerization
- Troubleshooting tips
- Security considerations

### Quick Start Options

**Option 1: Local Network Sharing**
```bash
# Generate results and start server
./run_and_serve.sh

# Share URL with people on your network
# http://YOUR_IP_ADDRESS:8000
```

**Option 2: ngrok (Share with Anyone)**
```bash
# Start server
cd web && python serve.py --port 8000 &

# Create tunnel
ngrok http 8000

# Share the ngrok URL (e.g., https://abc123.ngrok.io)
```

**Option 3: GitHub Pages**
```bash
# Deploy to GitHub Pages
./deploy.sh
# Select option 3, then follow prompts
```

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

### 🚀 One-Command Pipeline (Recommended)

Run the complete pipeline and view results in your browser:

```bash
./run_and_serve.sh
```

This single command will:
1. ✅ Generate 3 sample workflows (healthcare, finance, legal)
2. ✅ Run 4 algorithms (DAG-DP, Dijkstra, A*, Bellman-Ford) with 2 trials each
3. ✅ Create 4 visualization charts
4. ✅ Generate a comprehensive markdown report
5. ✅ Start web server and open dashboard in your browser

**Press `Ctrl+C` to stop the server when done.**

### 📊 View Existing Results

If you already have benchmark results and just want to view them:

```bash
./serve_only.sh
```

### 🔧 Manual Step-by-Step

Run the complete end-to-end demo manually:

```bash
poetry run python main.py
bash scripts/prepare_web_data.sh
python web/serve.py
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

---

## Troubleshooting

### Server Won't Start / Port Already in Use

If you see "Port 8000 is already in use":

**Option 1: Use a different port**
```bash
python web/serve.py --port 8080
```

**Option 2: Find and kill the process using port 8000**
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Or find the process first
lsof -i:8000
```

**Option 3: The server automatically finds a free port**
The `serve.py` script will automatically try ports 8000-8009 if 8000 is busy.

### Browser Doesn't Open Automatically

If the browser doesn't open automatically, manually visit:
- **http://localhost:8000** (or the port shown in the terminal)

### Dashboard Shows "No Data" or Empty Results

1. Make sure benchmarks ran successfully:
   ```bash
   ls -la results/*/web_results.json
   ```

2. Prepare web data:
   ```bash
   bash scripts/prepare_web_data.sh
   ```

3. Check if `web/data/web_results.json` exists:
   ```bash
   ls -la web/data/web_results.json
   ```

### Server Connection Refused

If you see "Connection refused" when accessing the dashboard:

1. Check if the server is actually running:
   ```bash
   ps aux | grep "python.*serve.py"
   ```

2. Try accessing from a different browser or incognito mode

3. Check firewall settings (macOS may block Python servers)

4. Verify the server is binding to the correct interface:
   ```bash
   # Server should show: "Serving web dashboard at http://localhost:XXXX"
   python web/serve.py
   ```
