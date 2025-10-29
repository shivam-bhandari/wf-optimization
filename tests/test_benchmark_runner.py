"""
Unit tests for the BenchmarkRunner class.

Tests cover:
- Configuration validation
- Basic benchmark execution
- Timeout handling
- Error handling
- Result aggregation
- File persistence
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import networkx as nx
import pandas as pd

from src.benchmarking.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    _run_algorithm_subprocess
)
from src.algorithms.base import OptimizationAlgorithm


class MockAlgorithm(OptimizationAlgorithm):
    """Mock algorithm for testing."""
    
    def __init__(self, name: str = "MockAlgorithm", should_fail: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.should_fail = should_fail
        self.solve_calls = 0
    
    def solve(self, workflow_graph: nx.DiGraph):
        """Mock solve method."""
        self.solve_calls += 1
        
        if self.should_fail:
            raise ValueError("Mock algorithm failure")
        
        # Return a simple mock solution
        nodes = list(workflow_graph.nodes())
        return {
            'path': nodes[:2] if len(nodes) >= 2 else nodes,
            'total_cost': 100.0,
            'total_time': 50.0,
            'nodes_explored': min(2, len(nodes)),
            'execution_time_seconds': 0.01,
            'resource_utilization': 0.5,
            'algorithm': self.name,
            'weight_attr': 'cost'
        }
    
    def validate_solution(self, solution, workflow_graph):
        """Mock validation."""
        return True


def create_simple_graph():
    """Create a simple test graph."""
    G = nx.DiGraph()
    G.add_edge('A', 'B', cost=10, weight=10)
    G.add_edge('B', 'C', cost=20, weight=20)
    return G


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig()
        
        assert config.trials_per_combination == 5
        assert config.timeout_seconds == 300.0
        assert config.random_seed == 42
        assert config.objectives == ['cost']
        assert config.save_results == True
        assert config.results_dir == Path('results/')
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BenchmarkConfig(
            trials_per_combination=10,
            timeout_seconds=60.0,
            random_seed=123,
            objectives=['cost', 'time'],
            save_results=False,
            results_dir=Path('custom_results/')
        )
        
        assert config.trials_per_combination == 10
        assert config.timeout_seconds == 60.0
        assert config.random_seed == 123
        assert config.objectives == ['cost', 'time']
        assert config.save_results == False
        assert config.results_dir == Path('custom_results/')
    
    def test_validation(self):
        """Test configuration validation."""
        # Invalid trials (must be >= 1)
        with pytest.raises(Exception):
            BenchmarkConfig(trials_per_combination=0)
        
        # Invalid timeout (must be > 0)
        with pytest.raises(Exception):
            BenchmarkConfig(timeout_seconds=0)
        
        with pytest.raises(Exception):
            BenchmarkConfig(timeout_seconds=-1.0)


class TestBenchmarkRunner:
    """Test BenchmarkRunner class."""
    
    def test_initialization(self):
        """Test runner initialization."""
        algorithms = [MockAlgorithm(name="Algo1")]
        workflows = [('workflow1', create_simple_graph())]
        config = BenchmarkConfig(trials_per_combination=2)
        
        runner = BenchmarkRunner(algorithms, workflows, config)
        
        assert len(runner.algorithms) == 1
        assert len(runner.workflows) == 1
        assert runner.config.trials_per_combination == 2
        assert runner.results == []
    
    def test_empty_algorithms_error(self):
        """Test error when no algorithms provided."""
        workflows = [('workflow1', create_simple_graph())]
        config = BenchmarkConfig()
        
        with pytest.raises(ValueError, match="At least one algorithm"):
            BenchmarkRunner([], workflows, config)
    
    def test_empty_workflows_error(self):
        """Test error when no workflows provided."""
        algorithms = [MockAlgorithm()]
        config = BenchmarkConfig()
        
        with pytest.raises(ValueError, match="At least one workflow"):
            BenchmarkRunner(algorithms, [], config)
    
    def test_basic_benchmark_execution(self):
        """Test basic benchmark execution."""
        algorithms = [MockAlgorithm(name="Algo1")]
        workflows = [('workflow1', create_simple_graph())]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                trials_per_combination=2,
                timeout_seconds=10.0,
                save_results=False,
                results_dir=Path(tmpdir),
                use_multiprocessing=False  # Use threads for testing
            )
            
            runner = BenchmarkRunner(algorithms, workflows, config)
            results_df = runner.run_benchmarks()
            
            # Check results
            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) == 2  # 1 workflow × 1 algo × 1 objective × 2 trials
            assert all(results_df['success'])
            assert all(results_df['algorithm_name'] == 'Algo1')
            assert all(results_df['workflow_id'] == 'workflow1')
            assert all(results_df['total_cost'] == 100.0)
    
    def test_multiple_algorithms_and_workflows(self):
        """Test with multiple algorithms and workflows."""
        algorithms = [
            MockAlgorithm(name="Algo1"),
            MockAlgorithm(name="Algo2")
        ]
        workflows = [
            ('workflow1', create_simple_graph()),
            ('workflow2', create_simple_graph())
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                trials_per_combination=2,
                save_results=False,
                results_dir=Path(tmpdir),
                use_multiprocessing=False
            )
            
            runner = BenchmarkRunner(algorithms, workflows, config)
            results_df = runner.run_benchmarks()
            
            # 2 workflows × 2 algos × 1 objective × 2 trials = 8 runs
            assert len(results_df) == 8
            assert set(results_df['algorithm_name']) == {'Algo1', 'Algo2'}
            assert set(results_df['workflow_id']) == {'workflow1', 'workflow2'}
    
    def test_multiple_objectives(self):
        """Test with multiple optimization objectives."""
        algorithms = [MockAlgorithm(name="Algo1")]
        workflows = [('workflow1', create_simple_graph())]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                trials_per_combination=2,
                objectives=['cost', 'time'],
                save_results=False,
                results_dir=Path(tmpdir),
                use_multiprocessing=False
            )
            
            runner = BenchmarkRunner(algorithms, workflows, config)
            results_df = runner.run_benchmarks()
            
            # 1 workflow × 1 algo × 2 objectives × 2 trials = 4 runs
            assert len(results_df) == 4
            assert set(results_df['objective']) == {'cost', 'time'}
    
    def test_algorithm_failure_handling(self):
        """Test handling of algorithm failures."""
        algorithms = [MockAlgorithm(name="FailingAlgo", should_fail=True)]
        workflows = [('workflow1', create_simple_graph())]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                trials_per_combination=2,
                save_results=False,
                results_dir=Path(tmpdir),
                use_multiprocessing=False
            )
            
            runner = BenchmarkRunner(algorithms, workflows, config)
            results_df = runner.run_benchmarks()
            
            # Check all runs failed
            assert len(results_df) == 2
            assert all(~results_df['success'])
            assert all(results_df['error_message'].str.contains('Mock algorithm failure'))
            assert all(pd.isna(results_df['total_cost']))
    
    def test_result_columns(self):
        """Test that results contain all required columns."""
        algorithms = [MockAlgorithm(name="Algo1")]
        workflows = [('workflow1', create_simple_graph())]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                trials_per_combination=1,
                save_results=False,
                results_dir=Path(tmpdir),
                use_multiprocessing=False
            )
            
            runner = BenchmarkRunner(algorithms, workflows, config)
            results_df = runner.run_benchmarks()
            
            required_columns = {
                'workflow_id', 'algorithm_name', 'objective', 'trial_number',
                'path', 'total_cost', 'total_time_ms', 'execution_time_seconds',
                'nodes_explored', 'success', 'error_message', 'timestamp'
            }
            
            assert required_columns.issubset(set(results_df.columns))
    
    def test_aggregate_statistics(self):
        """Test aggregate statistics calculation."""
        algorithms = [MockAlgorithm(name="Algo1")]
        workflows = [('workflow1', create_simple_graph())]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                trials_per_combination=5,
                save_results=False,
                results_dir=Path(tmpdir),
                use_multiprocessing=False
            )
            
            runner = BenchmarkRunner(algorithms, workflows, config)
            results_df = runner.run_benchmarks()
            agg_df = runner._aggregate_statistics(results_df)
            
            # Check aggregate dataframe
            assert len(agg_df) == 1  # 1 workflow × 1 algo × 1 objective
            assert 'mean_cost' in agg_df.columns
            assert 'std_cost' in agg_df.columns
            assert 'min_cost' in agg_df.columns
            assert 'max_cost' in agg_df.columns
            assert 'success_rate' in agg_df.columns
            
            # All runs should succeed
            assert agg_df.iloc[0]['success_rate'] == 1.0
            assert agg_df.iloc[0]['num_trials'] == 5
            assert agg_df.iloc[0]['num_successful'] == 5
    
    def test_save_results(self):
        """Test result saving to files."""
        algorithms = [MockAlgorithm(name="Algo1")]
        workflows = [('workflow1', create_simple_graph())]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                trials_per_combination=2,
                save_results=True,
                results_dir=Path(tmpdir),
                use_multiprocessing=False
            )
            
            runner = BenchmarkRunner(algorithms, workflows, config)
            results_df = runner.run_benchmarks()
            
            # Check files were created
            result_dir = Path(tmpdir)
            csv_files = list(result_dir.glob('benchmark_results_*.csv'))
            json_files = list(result_dir.glob('benchmark_results_*.json'))
            agg_csv_files = list(result_dir.glob('benchmark_aggregates_*.csv'))
            agg_json_files = list(result_dir.glob('benchmark_aggregates_*.json'))
            
            assert len(csv_files) == 1
            assert len(json_files) == 1
            assert len(agg_csv_files) == 1
            assert len(agg_json_files) == 1
    
    def test_no_save_results(self):
        """Test that results are not saved when save_results=False."""
        algorithms = [MockAlgorithm(name="Algo1")]
        workflows = [('workflow1', create_simple_graph())]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                trials_per_combination=2,
                save_results=False,
                results_dir=Path(tmpdir)
            )
            
            runner = BenchmarkRunner(algorithms, workflows, config)
            results_df = runner.run_benchmarks()
            
            # Check no files were created
            result_dir = Path(tmpdir)
            csv_files = list(result_dir.glob('*.csv'))
            json_files = list(result_dir.glob('*.json'))
            
            assert len(csv_files) == 0
            assert len(json_files) == 0


class TestSubprocessRunner:
    """Test the subprocess runner function."""
    
    def test_subprocess_runner_success(self):
        """Test successful algorithm execution in subprocess."""
        algorithm = MockAlgorithm(name="Algo1")
        graph = create_simple_graph()
        
        result = _run_algorithm_subprocess(algorithm, graph)
        
        assert result['path'] == ['A', 'B']
        assert result['total_cost'] == 100.0
        assert result['algorithm'] == 'Algo1'
    
    def test_subprocess_runner_failure(self):
        """Test algorithm failure in subprocess."""
        algorithm = MockAlgorithm(name="FailingAlgo", should_fail=True)
        graph = create_simple_graph()
        
        with pytest.raises(ValueError, match="Mock algorithm failure"):
            _run_algorithm_subprocess(algorithm, graph)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

