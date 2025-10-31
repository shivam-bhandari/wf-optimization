"""
Financial Workflow Generator

This module generates realistic financial services workflow DAGs representing common
financial processes such as loan approval, fraud detection, and risk assessment.

Each workflow includes:
- Task nodes with execution time, cost, and resource requirements
- Dependency edges with data transfer costs
- Conditional branches for realistic workflow patterns
- Comprehensive metadata for benchmarking

Why tasks are expensive in financial workflows:
- External API calls to credit bureaus, identity verification services
- Machine learning model inference (fraud scoring, risk models)
- Human review steps for high-risk cases
- Regulatory compliance checks
- Real-time data retrieval from market feeds
"""

import networkx as nx
import random
from typing import Dict, List, Tuple, Optional
import uuid
from datetime import datetime


class FinancialWorkflowGenerator:
    """
    Generator for realistic financial services workflow DAGs.

    This class creates directed acyclic graphs (DAGs) representing various
    financial processes with realistic task characteristics including execution
    times, costs, resource requirements, and failure probabilities.

    Financial workflows are characterized by:
    - External API dependencies (credit bureaus, identity verification)
    - Machine learning model inference (fraud detection, risk scoring)
    - Human review for edge cases
    - Regulatory compliance requirements
    - Real-time decision making

    Attributes:
        random (random.Random): Random number generator instance
        seed (Optional[int]): Random seed for reproducibility
        domain (str): Domain identifier ("finance")
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize financial workflow generator.

        Args:
            seed: Random seed for reproducibility. If None, uses system time.
        """
        self.random = random.Random(seed)
        self.seed = seed
        self.domain = "finance"

    def add_task_node(
        self,
        graph: nx.DiGraph,
        node_id: str,
        task_type: str,
        exec_time_range: Tuple[int, int],
        cost_range: Tuple[float, float],
        cpu_cores: int,
        memory_gb: int,
        storage_gb: int = 10,
    ) -> None:
        """
        Add a task node with all required attributes to the workflow graph.

        Creates a node representing a financial task with realistic execution
        characteristics including timing, cost, resource requirements, and
        failure probability.

        Args:
            graph: The workflow graph to add the node to
            node_id: Unique identifier for the node
            task_type: Type/name of the task (e.g., 'credit_score_retrieval')
            exec_time_range: Tuple of (min_ms, max_ms) for execution time
            cost_range: Tuple of (min_cost, max_cost) in cost units
            cpu_cores: Number of CPU cores required
            memory_gb: Memory requirement in GB
            storage_gb: Storage requirement in GB (default: 10)

        Node Attributes Set:
            - node_id: Unique node identifier
            - task_type: Task type string
            - execution_time_ms: Randomized execution time in milliseconds
            - cost_units: Randomized cost in units
            - resource_requirements: Dict with CPU, memory, storage specs
            - failure_probability: Random failure rate between 0.01-0.05
            - description: Human-readable task description
        """
        graph.add_node(
            node_id,
            node_id=node_id,
            task_type=task_type,
            execution_time_ms=self.random.randint(*exec_time_range),
            cost_units=round(self.random.uniform(*cost_range), 2),
            resource_requirements={
                "cpu_cores": cpu_cores,
                "memory_gb": memory_gb,
                "storage_gb": storage_gb,
            },
            failure_probability=round(self.random.uniform(0.01, 0.05), 4),
            description=f"{task_type.replace('_', ' ').title()} task",
        )

    def _add_dependency_edge(
        self, graph: nx.DiGraph, source: str, target: str, data_size_mb: float = 10.0
    ) -> None:
        """
        Add a dependency edge with transition costs between two tasks.

        Creates an edge representing a sequential dependency between tasks,
        including data transfer time and costs.

        Args:
            graph: The workflow graph to add the edge to
            source: Source node ID
            target: Target node ID
            data_size_mb: Size of data transferred in MB (default: 10.0)

        Edge Attributes Set:
            - transition_cost: Cost of transitioning between tasks
            - data_transfer_time_ms: Time to transfer data in milliseconds
            - data_size_mb: Amount of data transferred
            - edge_type: Type of dependency ('sequential_dependency')
        """
        graph.add_edge(
            source,
            target,
            transition_cost=round(self.random.uniform(0.05, 0.15), 2),
            data_transfer_time_ms=int(data_size_mb * self.random.uniform(5, 15)),
            data_size_mb=data_size_mb,
            edge_type="sequential_dependency",
        )

    def _calculate_workflow_stats(self, graph: nx.DiGraph) -> Dict:
        """
        Calculate comprehensive statistics for the workflow graph.

        Computes aggregate metrics including total cost, execution time,
        branching factor, and graph depth.

        Args:
            graph: The workflow graph to analyze

        Returns:
            Dictionary containing workflow statistics:
                - total_nodes: Number of nodes in the workflow
                - total_edges: Number of edges in the workflow
                - avg_branching_factor: Average out-degree of nodes
                - estimated_total_cost: Sum of all task costs
                - estimated_total_time_ms: Sum of all execution times
                - max_depth: Longest path length in the DAG
        """
        total_cost = sum(graph.nodes[n]["cost_units"] for n in graph.nodes())
        total_time = sum(graph.nodes[n]["execution_time_ms"] for n in graph.nodes())
        avg_branching = (
            sum(graph.out_degree(n) for n in graph.nodes()) / graph.number_of_nodes()
        )

        return {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "avg_branching_factor": round(avg_branching, 2),
            "estimated_total_cost": round(total_cost, 2),
            "estimated_total_time_ms": total_time,
            "max_depth": nx.dag_longest_path_length(graph)
            if graph.number_of_nodes() > 0 and nx.is_directed_acyclic_graph(graph)
            else 0,
        }

    def generate_loan_approval(self) -> nx.DiGraph:
        """
        Generate a loan approval workflow.

        This workflow represents the end-to-end process of evaluating and approving
        a loan application in the financial services industry. It includes identity
        verification, credit analysis, income verification, risk assessment, and
        automated decision making.

        Business Process:
        The loan approval process combines multiple data sources (credit bureaus,
        employment databases, asset verification services) with sophisticated risk
        models to make lending decisions. This workflow is used by banks, credit
        unions, and online lenders.

        Why Tasks Are Expensive:
        - Identity verification: External API calls to verification services ($1-2)
        - Credit score retrieval: Credit bureau API fees ($0.80-1.50 per pull)
        - Employment verification: Third-party employment database queries ($1.50-2.50)
        - Risk scoring model: Complex ML model inference on large datasets ($0.50-1.00)
        - Fraud detection: Real-time fraud detection service ($1.20-2.50)
        - Manual review: Human underwriter time for edge cases ($10-20 per review)

        Workflow Steps:
            1. Application intake
            2. Identity verification (external API)
            3. Credit score retrieval (credit bureau API)
            4. Credit report analysis
            5. Income verification (external service)
            6. Employment verification (third-party database)
            7. Debt-to-income calculation
            8. Asset verification
            9. Risk scoring model (ML inference)
            10. Fraud detection (ML model + rules)
            11. Automated underwriting
            12. Approval decision
            13. Offer generation
            14. Notification dispatch

        Conditional Branches:
            - After risk scoring: 20% chance requires manual underwriter review
              (30-60 seconds, $10-20) for borderline cases
            - After fraud detection: 5% chance triggers fraud investigation
              (3-5 minutes, $50-100) for suspicious applications

        Returns:
            NetworkX DiGraph representing the loan approval workflow
            with comprehensive node and edge attributes.

        Graph Metadata:
            - workflow_id: Unique UUID for this workflow instance
            - domain: "finance"
            - workflow_type: "loan_approval"
            - description: Business process description
            - generated_at: ISO timestamp
            - statistics: Computed workflow statistics

        Typical Metrics:
            - Nodes: 16 (base) + up to 2 conditional branches
            - Estimated Cost: $10-25 (with conditional branches)
            - Estimated Time: 20-40 seconds (excluding manual review)
        """
        graph = nx.DiGraph()

        # Add start and end nodes
        graph.add_node(
            "start",
            node_id="start",
            task_type="start",
            execution_time_ms=0,
            cost_units=0.0,
            resource_requirements={},
            failure_probability=0.0,
            description="Workflow start node",
        )
        graph.add_node(
            "end",
            node_id="end",
            task_type="end",
            execution_time_ms=0,
            cost_units=0.0,
            resource_requirements={},
            failure_probability=0.0,
            description="Workflow end node",
        )

        # Define tasks in sequence
        tasks = [
            ("application_intake", (300, 600), (0.05, 0.10), 1, 2),
            ("identity_verification", (1500, 2500), (1.00, 2.00), 2, 4),
            ("credit_score_retrieval", (1000, 2000), (0.80, 1.50), 1, 2),
            ("credit_report_analysis", (1800, 3000), (0.50, 1.00), 4, 8),
            ("income_verification", (1500, 3000), (1.00, 2.00), 2, 4),
            ("employment_verification", (2000, 3500), (1.50, 2.50), 2, 4),
            ("debt_to_income_calculation", (200, 500), (0.05, 0.10), 1, 2),
            ("asset_verification", (1800, 3200), (0.80, 1.50), 2, 4),
            ("risk_scoring_model", (2000, 4000), (0.50, 1.00), 8, 16),
            ("fraud_detection", (2500, 4500), (1.20, 2.50), 4, 12),
            ("automated_underwriting", (1000, 2000), (0.30, 0.60), 4, 8),
            ("approval_decision", (500, 1000), (0.10, 0.20), 1, 2),
            ("offer_generation", (800, 1500), (0.15, 0.30), 2, 4),
            ("notification_dispatch", (200, 400), (0.05, 0.10), 1, 2),
        ]

        # Add all task nodes
        for task_type, exec_time_range, cost_range, cpu_cores, memory_gb in tasks:
            self.add_task_node(
                graph,
                task_type,
                task_type,
                exec_time_range,
                cost_range,
                cpu_cores,
                memory_gb,
            )

        # Build main pipeline
        self._add_dependency_edge(
            graph, "start", "application_intake", data_size_mb=2.0
        )
        self._add_dependency_edge(
            graph, "application_intake", "identity_verification", data_size_mb=5.0
        )
        self._add_dependency_edge(
            graph, "identity_verification", "credit_score_retrieval", data_size_mb=3.0
        )
        self._add_dependency_edge(
            graph, "credit_score_retrieval", "credit_report_analysis", data_size_mb=15.0
        )
        self._add_dependency_edge(
            graph, "credit_report_analysis", "income_verification", data_size_mb=8.0
        )
        self._add_dependency_edge(
            graph, "income_verification", "employment_verification", data_size_mb=6.0
        )
        self._add_dependency_edge(
            graph,
            "employment_verification",
            "debt_to_income_calculation",
            data_size_mb=4.0,
        )
        self._add_dependency_edge(
            graph, "debt_to_income_calculation", "asset_verification", data_size_mb=5.0
        )
        self._add_dependency_edge(
            graph, "asset_verification", "risk_scoring_model", data_size_mb=20.0
        )

        # Conditional branch 1: 20% chance of manual underwriter review after risk scoring
        if self.random.random() < 0.2:
            self.add_task_node(
                graph,
                "manual_underwriter_review",
                "manual_underwriter_review",
                (30000, 60000),
                (10.0, 20.0),
                1,
                4,
            )
            self._add_dependency_edge(
                graph,
                "risk_scoring_model",
                "manual_underwriter_review",
                data_size_mb=25.0,
            )
            self._add_dependency_edge(
                graph, "manual_underwriter_review", "fraud_detection", data_size_mb=25.0
            )
        else:
            self._add_dependency_edge(
                graph, "risk_scoring_model", "fraud_detection", data_size_mb=20.0
            )

        # Conditional branch 2: 5% chance of fraud investigation after fraud detection
        if self.random.random() < 0.05:
            self.add_task_node(
                graph,
                "fraud_investigation",
                "fraud_investigation",
                (180000, 300000),
                (50.0, 100.0),
                2,
                8,
            )
            self._add_dependency_edge(
                graph, "fraud_detection", "fraud_investigation", data_size_mb=30.0
            )
            self._add_dependency_edge(
                graph,
                "fraud_investigation",
                "automated_underwriting",
                data_size_mb=30.0,
            )
        else:
            self._add_dependency_edge(
                graph, "fraud_detection", "automated_underwriting", data_size_mb=22.0
            )

        # Continue main pipeline
        self._add_dependency_edge(
            graph, "automated_underwriting", "approval_decision", data_size_mb=10.0
        )
        self._add_dependency_edge(
            graph, "approval_decision", "offer_generation", data_size_mb=8.0
        )
        self._add_dependency_edge(
            graph, "offer_generation", "notification_dispatch", data_size_mb=5.0
        )
        self._add_dependency_edge(
            graph, "notification_dispatch", "end", data_size_mb=2.0
        )

        # Set graph metadata
        graph.graph["workflow_id"] = str(uuid.uuid4())
        graph.graph["domain"] = self.domain
        graph.graph["workflow_type"] = "loan_approval"
        graph.graph["description"] = (
            "End-to-end loan application evaluation and approval process combining "
            "identity verification, credit analysis, income verification, risk assessment, "
            "and automated decision making with optional manual review for borderline cases."
        )
        graph.graph["generated_at"] = datetime.now().isoformat()
        graph.graph["statistics"] = self._calculate_workflow_stats(graph)

        # Verify it's a DAG
        assert nx.is_directed_acyclic_graph(graph), "Generated graph contains cycles"

        return graph

    def generate_fraud_detection(self) -> nx.DiGraph:
        """
        Generate a fraud detection workflow.

        This workflow represents the real-time analysis of financial transactions
        to detect fraudulent activity. It combines multiple detection techniques
        including pattern matching, velocity checks, geolocation analysis, device
        fingerprinting, and machine learning models.

        Business Process:
        Used by payment processors, banks, and fintech companies to prevent
        fraudulent transactions in real-time. The system analyzes transaction
        characteristics, user behavior patterns, and contextual signals to
        assign a fraud risk score and make accept/reject decisions within
        milliseconds to seconds.

        Why Tasks Are Expensive:
        - Geolocation analysis: IP geolocation and address verification APIs ($0.30-0.60)
        - Device fingerprinting: Browser/device identification service ($0.40-0.80)
        - ML fraud scoring: Complex deep learning model inference ($0.80-1.50)
        - Historical pattern lookup: Real-time database queries across large datasets
        - Analyst review: Human fraud analyst for high-risk cases ($15-40 per review)

        Workflow Steps:
            1. Transaction ingestion
            2. Feature extraction (amount, merchant, time, etc.)
            3. Historical pattern lookup (user's transaction history)
            4. Velocity checks (transaction frequency limits)
            5. Geolocation analysis (IP/billing address matching)
            6. Device fingerprinting (browser/device identification)
            7. ML fraud scoring (deep learning model)
            8. Rules engine evaluation (business rules)
            9. Risk score calculation (composite score)
            10. Threshold decision (approve/review/decline)
            11. Alert generation (if flagged)

        Conditional Branch:
            - After threshold decision: 15% chance requires analyst review
              (1-3 minutes, $15-40) for borderline or high-value transactions

        Returns:
            NetworkX DiGraph representing the fraud detection workflow
            with comprehensive node and edge attributes.

        Graph Metadata:
            - workflow_id: Unique UUID for this workflow instance
            - domain: "finance"
            - workflow_type: "fraud_detection"
            - description: Business process description
            - generated_at: ISO timestamp
            - statistics: Computed workflow statistics

        Typical Metrics:
            - Nodes: 13 (base) + up to 1 conditional branch
            - Estimated Cost: $4-8 (with analyst review up to $48)
            - Estimated Time: 8-15 seconds (excluding analyst review)
        """
        graph = nx.DiGraph()

        # Add start and end nodes
        graph.add_node(
            "start",
            node_id="start",
            task_type="start",
            execution_time_ms=0,
            cost_units=0.0,
            resource_requirements={},
            failure_probability=0.0,
            description="Workflow start node",
        )
        graph.add_node(
            "end",
            node_id="end",
            task_type="end",
            execution_time_ms=0,
            cost_units=0.0,
            resource_requirements={},
            failure_probability=0.0,
            description="Workflow end node",
        )

        # Define tasks in sequence
        tasks = [
            ("transaction_ingestion", (200, 400), (0.03, 0.06), 1, 2),
            ("feature_extraction", (400, 800), (0.10, 0.20), 2, 4),
            ("historical_pattern_lookup", (800, 1500), (0.20, 0.40), 2, 4),
            ("velocity_checks", (600, 1200), (0.15, 0.30), 2, 4),
            ("geolocation_analysis", (1000, 2000), (0.30, 0.60), 2, 4),
            ("device_fingerprinting", (1200, 2000), (0.40, 0.80), 2, 4),
            ("ml_fraud_scoring", (1500, 3000), (0.80, 1.50), 4, 12),
            ("rules_engine_evaluation", (800, 1500), (0.20, 0.40), 2, 4),
            ("risk_score_calculation", (500, 1000), (0.12, 0.25), 2, 4),
            ("threshold_decision", (200, 400), (0.05, 0.10), 1, 2),
            ("alert_generation", (300, 600), (0.08, 0.15), 1, 2),
        ]

        # Add all task nodes
        for task_type, exec_time_range, cost_range, cpu_cores, memory_gb in tasks:
            self.add_task_node(
                graph,
                task_type,
                task_type,
                exec_time_range,
                cost_range,
                cpu_cores,
                memory_gb,
            )

        # Build main pipeline
        self._add_dependency_edge(
            graph, "start", "transaction_ingestion", data_size_mb=1.0
        )
        self._add_dependency_edge(
            graph, "transaction_ingestion", "feature_extraction", data_size_mb=2.0
        )
        self._add_dependency_edge(
            graph, "feature_extraction", "historical_pattern_lookup", data_size_mb=5.0
        )
        self._add_dependency_edge(
            graph, "historical_pattern_lookup", "velocity_checks", data_size_mb=8.0
        )
        self._add_dependency_edge(
            graph, "velocity_checks", "geolocation_analysis", data_size_mb=4.0
        )
        self._add_dependency_edge(
            graph, "geolocation_analysis", "device_fingerprinting", data_size_mb=6.0
        )
        self._add_dependency_edge(
            graph, "device_fingerprinting", "ml_fraud_scoring", data_size_mb=12.0
        )
        self._add_dependency_edge(
            graph, "ml_fraud_scoring", "rules_engine_evaluation", data_size_mb=10.0
        )
        self._add_dependency_edge(
            graph, "rules_engine_evaluation", "risk_score_calculation", data_size_mb=8.0
        )
        self._add_dependency_edge(
            graph, "risk_score_calculation", "threshold_decision", data_size_mb=5.0
        )

        # Conditional branch: 15% chance of analyst review after threshold decision
        if self.random.random() < 0.15:
            self.add_task_node(
                graph,
                "analyst_review",
                "analyst_review",
                (60000, 180000),
                (15.0, 40.0),
                1,
                4,
            )
            self._add_dependency_edge(
                graph, "threshold_decision", "analyst_review", data_size_mb=15.0
            )
            self._add_dependency_edge(
                graph, "analyst_review", "alert_generation", data_size_mb=15.0
            )
        else:
            self._add_dependency_edge(
                graph, "threshold_decision", "alert_generation", data_size_mb=8.0
            )

        # Continue to end
        self._add_dependency_edge(graph, "alert_generation", "end", data_size_mb=3.0)

        # Set graph metadata
        graph.graph["workflow_id"] = str(uuid.uuid4())
        graph.graph["domain"] = self.domain
        graph.graph["workflow_type"] = "fraud_detection"
        graph.graph["description"] = (
            "Real-time transaction fraud detection workflow combining pattern matching, "
            "velocity checks, geolocation analysis, device fingerprinting, and machine "
            "learning models to identify and prevent fraudulent transactions."
        )
        graph.graph["generated_at"] = datetime.now().isoformat()
        graph.graph["statistics"] = self._calculate_workflow_stats(graph)

        # Verify it's a DAG
        assert nx.is_directed_acyclic_graph(graph), "Generated graph contains cycles"

        return graph

    def generate_risk_assessment(self) -> nx.DiGraph:
        """
        Generate a risk assessment workflow.

        This workflow represents the comprehensive risk analysis process used by
        financial institutions to evaluate portfolio risk, market exposure, and
        regulatory capital requirements. It includes portfolio analysis, Value-at-Risk
        (VaR) calculations, stress testing, scenario analysis, and compliance checks.

        Business Process:
        Used by investment banks, hedge funds, asset managers, and trading desks
        to measure and manage financial risk. This workflow runs periodically
        (daily, weekly) or on-demand to assess exposure to market risk, credit
        risk, and operational risk. Results are used for risk limits, capital
        allocation, and regulatory reporting.

        Why Tasks Are Expensive:
        - Market data retrieval: Real-time market data feed subscriptions ($0.80-1.50)
        - VaR calculation: Complex Monte Carlo simulations on large portfolios ($1.20-2.50)
        - Stress testing: Running multiple stress scenarios requires significant compute ($1.50-3.00)
        - Scenario analysis: Simulating various market conditions ($1.30-2.60)
        - Portfolio analysis: Analyzing large portfolios with complex instruments ($1.00-2.00)
        - All use high-memory compute for large-scale matrix operations

        Workflow Steps:
            1. Data collection (portfolio positions)
            2. Market data retrieval (prices, rates, volatilities)
            3. Portfolio analysis (holdings, exposures)
            4. Exposure calculation (net positions by risk factor)
            5. VaR calculation (Value-at-Risk via Monte Carlo)
            6. Stress testing (extreme market scenarios)
            7. Scenario analysis (economic scenario simulations)
            8. Correlation analysis (cross-asset correlations)
            9. Risk metric aggregation (combining all risk measures)
            10. Report generation (risk dashboards and reports)
            11. Compliance checks (regulatory requirements)
            12. Risk rating assignment (overall risk score)

        No Conditional Branches:
            This is a deterministic analytical workflow that always executes
            all steps. Risk assessment is comprehensive by design.

        Returns:
            NetworkX DiGraph representing the risk assessment workflow
            with comprehensive node and edge attributes.

        Graph Metadata:
            - workflow_id: Unique UUID for this workflow instance
            - domain: "finance"
            - workflow_type: "risk_assessment"
            - description: Business process description
            - generated_at: ISO timestamp
            - statistics: Computed workflow statistics

        Typical Metrics:
            - Nodes: 14 (deterministic, no branches)
            - Estimated Cost: $10-20
            - Estimated Time: 20-40 seconds (compute-intensive)
        """
        graph = nx.DiGraph()

        # Add start and end nodes
        graph.add_node(
            "start",
            node_id="start",
            task_type="start",
            execution_time_ms=0,
            cost_units=0.0,
            resource_requirements={},
            failure_probability=0.0,
            description="Workflow start node",
        )
        graph.add_node(
            "end",
            node_id="end",
            task_type="end",
            execution_time_ms=0,
            cost_units=0.0,
            resource_requirements={},
            failure_probability=0.0,
            description="Workflow end node",
        )

        # Define tasks in sequence
        tasks = [
            ("data_collection", (500, 1000), (0.12, 0.25), 2, 4),
            ("market_data_retrieval", (1500, 2500), (0.80, 1.50), 2, 4),
            ("portfolio_analysis", (2000, 4000), (1.00, 2.00), 4, 8),
            ("exposure_calculation", (1800, 3200), (0.60, 1.20), 4, 8),
            ("var_calculation", (2500, 4500), (1.20, 2.50), 8, 16),
            ("stress_testing", (3000, 6000), (1.50, 3.00), 8, 16),
            ("scenario_analysis", (2500, 5000), (1.30, 2.60), 8, 16),
            ("correlation_analysis", (1500, 3000), (0.70, 1.40), 4, 8),
            ("risk_metric_aggregation", (1000, 2000), (0.30, 0.60), 2, 4),
            ("report_generation", (1500, 2500), (0.40, 0.80), 2, 4),
            ("compliance_checks", (1200, 2200), (0.50, 1.00), 2, 4),
            ("risk_rating_assignment", (600, 1200), (0.15, 0.30), 1, 2),
        ]

        # Add all task nodes
        for task_type, exec_time_range, cost_range, cpu_cores, memory_gb in tasks:
            self.add_task_node(
                graph,
                task_type,
                task_type,
                exec_time_range,
                cost_range,
                cpu_cores,
                memory_gb,
            )

        # Build main pipeline (linear, no branches)
        self._add_dependency_edge(graph, "start", "data_collection", data_size_mb=5.0)
        self._add_dependency_edge(
            graph, "data_collection", "market_data_retrieval", data_size_mb=20.0
        )
        self._add_dependency_edge(
            graph, "market_data_retrieval", "portfolio_analysis", data_size_mb=50.0
        )
        self._add_dependency_edge(
            graph, "portfolio_analysis", "exposure_calculation", data_size_mb=40.0
        )
        self._add_dependency_edge(
            graph, "exposure_calculation", "var_calculation", data_size_mb=35.0
        )
        self._add_dependency_edge(
            graph, "var_calculation", "stress_testing", data_size_mb=30.0
        )
        self._add_dependency_edge(
            graph, "stress_testing", "scenario_analysis", data_size_mb=45.0
        )
        self._add_dependency_edge(
            graph, "scenario_analysis", "correlation_analysis", data_size_mb=40.0
        )
        self._add_dependency_edge(
            graph, "correlation_analysis", "risk_metric_aggregation", data_size_mb=25.0
        )
        self._add_dependency_edge(
            graph, "risk_metric_aggregation", "report_generation", data_size_mb=15.0
        )
        self._add_dependency_edge(
            graph, "report_generation", "compliance_checks", data_size_mb=20.0
        )
        self._add_dependency_edge(
            graph, "compliance_checks", "risk_rating_assignment", data_size_mb=10.0
        )
        self._add_dependency_edge(
            graph, "risk_rating_assignment", "end", data_size_mb=5.0
        )

        # Set graph metadata
        graph.graph["workflow_id"] = str(uuid.uuid4())
        graph.graph["domain"] = self.domain
        graph.graph["workflow_type"] = "risk_assessment"
        graph.graph["description"] = (
            "Comprehensive financial risk assessment workflow for portfolio risk analysis, "
            "including Value-at-Risk calculations, stress testing, scenario analysis, and "
            "regulatory compliance checks for investment portfolios."
        )
        graph.graph["generated_at"] = datetime.now().isoformat()
        graph.graph["statistics"] = self._calculate_workflow_stats(graph)

        # Verify it's a DAG
        assert nx.is_directed_acyclic_graph(graph), "Generated graph contains cycles"

        return graph
