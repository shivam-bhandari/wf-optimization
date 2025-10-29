"""
Legal Workflow Generator

This module generates realistic legal services workflow DAGs representing common
legal processes such as contract review, compliance checking, and document redlining.

Each workflow includes:
- Task nodes with execution time, cost, and resource requirements
- Dependency edges with data transfer costs
- Multi-tier human review processes (junior associate → senior attorney → partner)
- Comprehensive metadata for benchmarking

Why legal workflows are extremely expensive:
- Attorney billable hours ($200-500+ per hour in reality)
- Multiple tiers of review (junior, senior, partner - each more expensive)
- High human oversight requirements for risk mitigation
- Complex analysis requiring legal expertise and judgment
- Liability concerns require careful review at multiple levels
- Partner approval for high-risk or high-value matters
- Precedent research and legal database access fees
- Specialized ML models for legal document analysis
"""

import networkx as nx
import random
from typing import Dict, List, Tuple, Optional
import uuid
from datetime import datetime


class LegalWorkflowGenerator:
    """
    Generator for realistic legal services workflow DAGs.

    This class creates directed acyclic graphs (DAGs) representing various
    legal processes with realistic task characteristics including execution
    times, costs, resource requirements, and failure probabilities.

    Legal workflows are characterized by:
    - Multi-tier human review processes (junior → senior → partner)
    - High human oversight costs (attorney billable hours)
    - Complex document analysis and ML-based legal tech
    - Precedent research and legal database access
    - Risk assessment and liability analysis
    - Compliance validation and regulatory checks

    The cost structure reflects:
    - Junior associate review: $50-100 (2-5 minutes at $200-300/hour)
    - Senior attorney review: $100-200 (3-7 minutes at $300-400/hour)
    - Partner approval: $150-250 (1-2 minutes at $500-600/hour)

    Attributes:
        random (random.Random): Random number generator instance
        seed (Optional[int]): Random seed for reproducibility
        domain (str): Domain identifier ("legal")
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize legal workflow generator.

        Args:
            seed: Random seed for reproducibility. If None, uses system time.
        """
        self.random = random.Random(seed)
        self.seed = seed
        self.domain = "legal"

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

        Creates a node representing a legal task with realistic execution
        characteristics including timing, cost, resource requirements, and
        failure probability.

        Args:
            graph: The workflow graph to add the node to
            node_id: Unique identifier for the node
            task_type: Type/name of the task (e.g., 'clause_extraction')
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

    def generate_contract_review(self, num_tasks: int) -> nx.DiGraph:
        """
        Generate a contract review workflow with multi-tier attorney review.

        This workflow represents the comprehensive legal review process for contracts,
        including automated analysis, clause extraction, risk assessment, and multi-tier
        human review by junior associates, senior attorneys, and partners.

        Business Process:
        Law firms, corporate legal departments, and legal tech companies use this
        workflow to review contracts before execution. The process combines automated
        document analysis with human legal expertise to identify risks, compliance
        issues, and unfavorable terms.

        Multi-Tier Review Process:
        1. **Automated Analysis** - ML models extract clauses, identify risks, score documents
        2. **Junior Associate Review** (ALWAYS required, $50-100)
           - First-pass review by junior attorney (1-3 years experience)
           - Reviews automated findings, identifies issues
           - Billable at $200-300/hour → 2-5 minutes = $50-100

        3. **Senior Attorney Review** (30% of cases, $100-200)
           - Second-tier review for complex or high-risk contracts
           - More experienced attorney (5-10+ years)
           - Billable at $300-400/hour → 3-7 minutes = $100-200

        4. **Partner Approval** (10% of senior reviews, $150-250)
           - Final approval for highest-risk or highest-value matters
           - Partner-level review (15+ years experience)
           - Billable at $500-600/hour → 1-2 minutes = $150-250

        Why Tasks Are Expensive:
        - ML clause extraction: Complex NLP models on legal text ($0.80-1.50)
        - Precedent comparison: Legal database access fees ($1.20-2.50)
        - Risk detection: Specialized legal AI models ($1.00-2.00)
        - Human review layers: Attorney billable hours ($50-250 per review)
        - Total workflow cost can exceed $500 with all review tiers

        Workflow Steps:
            1. Document upload and metadata extraction
            2. PDF parsing and document classification
            3. Clause extraction (ML-based)
            4. Key term identification
            5. Risk clause detection (legal AI)
            6. Obligation extraction
            7. Liability analysis
            8. Compliance validation
            9. Jurisdiction check
            10. Precedent comparison (legal database)
            11. Automated risk scoring
            12. Redlining generation
            13. Final automated approval
            14. Junior associate review (ALWAYS)
            15. Senior attorney review (30% chance)
            16. Partner approval (10% of senior reviews)

        Conditional Review Branches:
            - Junior associate review: 100% (always required)
            - Senior attorney review: 30% (complex/high-risk contracts)
            - Partner approval: 10% (highest-risk matters requiring partner sign-off)

        Args:
            num_tasks: Unused parameter (kept for interface compatibility)

        Returns:
            NetworkX DiGraph representing the contract review workflow
            with comprehensive node and edge attributes.

        Graph Metadata:
            - workflow_id: Unique UUID for this workflow instance
            - domain: "legal"
            - workflow_type: "contract_review"
            - description: Business process description
            - generated_at: ISO timestamp
            - statistics: Computed workflow statistics

        Typical Metrics:
            - Nodes: 17-20 (depending on review tiers activated)
            - Estimated Cost: $15-30 (base) + $50-100 (junior) + up to $350 (senior+partner)
            - Estimated Time: 30-50s (base) + 2-10 minutes (reviews)
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
            ("document_upload", (400, 800), (0.10, 0.15), 1, 2),
            ("metadata_extraction", (600, 1200), (0.15, 0.25), 2, 4),
            ("pdf_parsing", (1500, 3000), (0.30, 0.50), 4, 8),
            ("document_classification", (1200, 2000), (0.40, 0.80), 4, 8),
            ("clause_extraction", (3000, 6000), (0.80, 1.50), 4, 12),
            ("key_term_identification", (2000, 4000), (0.60, 1.20), 4, 8),
            ("risk_clause_detection", (2500, 5000), (1.00, 2.00), 4, 12),
            ("obligation_extraction", (2000, 4000), (0.70, 1.40), 4, 8),
            ("liability_analysis", (1800, 3500), (0.60, 1.20), 2, 6),
            ("compliance_validation", (2500, 5000), (1.00, 2.00), 4, 8),
            ("jurisdiction_check", (1000, 2000), (0.30, 0.60), 2, 4),
            ("precedent_comparison", (3000, 6000), (1.20, 2.50), 4, 12),
            ("automated_risk_scoring", (1500, 3000), (0.50, 1.00), 4, 8),
            ("redlining_generation", (1000, 2000), (0.40, 0.80), 2, 4),
            ("final_approval", (500, 1000), (0.20, 0.40), 1, 2),
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
        self._add_dependency_edge(graph, "start", "document_upload", data_size_mb=5.0)
        self._add_dependency_edge(
            graph, "document_upload", "metadata_extraction", data_size_mb=8.0
        )
        self._add_dependency_edge(
            graph, "metadata_extraction", "pdf_parsing", data_size_mb=15.0
        )
        self._add_dependency_edge(
            graph, "pdf_parsing", "document_classification", data_size_mb=25.0
        )
        self._add_dependency_edge(
            graph, "document_classification", "clause_extraction", data_size_mb=30.0
        )
        self._add_dependency_edge(
            graph, "clause_extraction", "key_term_identification", data_size_mb=20.0
        )
        self._add_dependency_edge(
            graph, "key_term_identification", "risk_clause_detection", data_size_mb=18.0
        )
        self._add_dependency_edge(
            graph, "risk_clause_detection", "obligation_extraction", data_size_mb=15.0
        )
        self._add_dependency_edge(
            graph, "obligation_extraction", "liability_analysis", data_size_mb=12.0
        )
        self._add_dependency_edge(
            graph, "liability_analysis", "compliance_validation", data_size_mb=14.0
        )
        self._add_dependency_edge(
            graph, "compliance_validation", "jurisdiction_check", data_size_mb=10.0
        )
        self._add_dependency_edge(
            graph, "jurisdiction_check", "precedent_comparison", data_size_mb=20.0
        )
        self._add_dependency_edge(
            graph, "precedent_comparison", "automated_risk_scoring", data_size_mb=22.0
        )
        self._add_dependency_edge(
            graph, "automated_risk_scoring", "redlining_generation", data_size_mb=18.0
        )
        self._add_dependency_edge(
            graph, "redlining_generation", "final_approval", data_size_mb=15.0
        )

        # Multi-tier review process
        # Tier 1: Junior Associate Review (ALWAYS required - 100% of contracts)
        self.add_task_node(
            graph,
            "junior_associate_review",
            "junior_associate_review",
            (120000, 300000),
            (50.0, 100.0),
            1,
            4,
        )
        self._add_dependency_edge(
            graph, "final_approval", "junior_associate_review", data_size_mb=25.0
        )

        # Tier 2: Senior Attorney Review (30% of contracts - complex/high-risk)
        needs_senior_review = self.random.random() < 0.3
        if needs_senior_review:
            self.add_task_node(
                graph,
                "senior_attorney_review",
                "senior_attorney_review",
                (180000, 450000),
                (100.0, 200.0),
                1,
                4,
            )
            self._add_dependency_edge(
                graph,
                "junior_associate_review",
                "senior_attorney_review",
                data_size_mb=30.0,
            )

            # Tier 3: Partner Approval (10% of senior reviews - highest-risk matters)
            needs_partner_approval = self.random.random() < 0.1
            if needs_partner_approval:
                self.add_task_node(
                    graph,
                    "partner_approval",
                    "partner_approval",
                    (60000, 120000),
                    (150.0, 250.0),
                    1,
                    4,
                )
                self._add_dependency_edge(
                    graph,
                    "senior_attorney_review",
                    "partner_approval",
                    data_size_mb=35.0,
                )
                self._add_dependency_edge(
                    graph, "partner_approval", "end", data_size_mb=5.0
                )
            else:
                self._add_dependency_edge(
                    graph, "senior_attorney_review", "end", data_size_mb=5.0
                )
        else:
            self._add_dependency_edge(
                graph, "junior_associate_review", "end", data_size_mb=5.0
            )

        # Set graph metadata
        graph.graph["workflow_id"] = str(uuid.uuid4())
        graph.graph["domain"] = self.domain
        graph.graph["workflow_type"] = "contract_review"
        graph.graph["description"] = (
            "Comprehensive legal contract review workflow combining automated document "
            "analysis, ML-based clause extraction and risk detection, with multi-tier "
            "human review by junior associates, senior attorneys, and partners for "
            "high-risk or high-value contracts."
        )
        graph.graph["generated_at"] = datetime.now().isoformat()
        graph.graph["statistics"] = self._calculate_workflow_stats(graph)

        # Verify it's a DAG
        assert nx.is_directed_acyclic_graph(graph), "Generated graph contains cycles"

        return graph

    def generate_compliance_check(self, num_tasks: int) -> nx.DiGraph:
        """
        Generate a compliance checking workflow.

        This workflow represents the regulatory compliance validation process used
        by legal departments and compliance officers to ensure documents, policies,
        or business practices meet regulatory requirements.

        Business Process:
        Organizations use this workflow to validate compliance with regulations such as
        GDPR, HIPAA, SOX, industry-specific regulations, and internal policies. The
        system identifies applicable regulations, maps requirements to documentation,
        performs gap analysis, and generates compliance reports.

        Why Tasks Are Expensive:
        - Regulation databases: Access to regulatory databases and updates ($0.60-1.20)
        - Requirements extraction: Complex analysis of regulatory text ($0.80-1.50)
        - Document analysis: Comprehensive document review ($1.00-2.00)
        - Gap analysis: Identifying compliance gaps ($0.70-1.40)
        - Evidence collection: Document and proof gathering ($1.00-2.00)
        - Legal counsel review: Attorney review for 20% of reports ($80-150)

        Legal Counsel Review:
        - Triggered for 20% of compliance checks (complex regulations or gaps found)
        - Senior compliance attorney or legal counsel review
        - Billable at $300-400/hour → 3-6 minutes = $80-150

        Workflow Steps:
            1. Document ingestion
            2. Regulation identification (which regulations apply)
            3. Requirements extraction (what the regulations require)
            4. Document analysis (review current state)
            5. Compliance mapping (map requirements to evidence)
            6. Gap analysis (identify non-compliance)
            7. Risk assessment (evaluate compliance risk)
            8. Checklist validation (verify all requirements checked)
            9. Evidence collection (gather proof of compliance)
            10. Audit trail generation (create compliance record)
            11. Compliance scoring (overall compliance rating)
            12. Report generation (compliance report)
            13. Legal counsel review (20% chance - complex cases)

        Conditional Branch:
            - Legal counsel review: 20% (complex regulations or gaps identified)

        Args:
            num_tasks: Unused parameter (kept for interface compatibility)

        Returns:
            NetworkX DiGraph representing the compliance checking workflow
            with comprehensive node and edge attributes.

        Graph Metadata:
            - workflow_id: Unique UUID for this workflow instance
            - domain: "legal"
            - workflow_type: "compliance_check"
            - description: Business process description
            - generated_at: ISO timestamp
            - statistics: Computed workflow statistics

        Typical Metrics:
            - Nodes: 14 (base) + 1 (if legal review)
            - Estimated Cost: $8-16 (base) + up to $150 (legal review)
            - Estimated Time: 20-35s (base) + 3-6 minutes (legal review)
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
            ("document_ingestion", (500, 1000), (0.12, 0.25), 2, 4),
            ("regulation_identification", (1500, 2500), (0.60, 1.20), 2, 4),
            ("requirements_extraction", (2000, 3500), (0.80, 1.50), 4, 8),
            ("document_analysis", (2500, 4500), (1.00, 2.00), 4, 8),
            ("compliance_mapping", (2000, 4000), (0.80, 1.60), 4, 8),
            ("gap_analysis", (1800, 3500), (0.70, 1.40), 4, 8),
            ("risk_assessment", (2000, 4000), (0.90, 1.80), 4, 8),
            ("checklist_validation", (1500, 2800), (0.50, 1.00), 2, 4),
            ("evidence_collection", (2500, 4500), (1.00, 2.00), 4, 8),
            ("audit_trail_generation", (1000, 2000), (0.30, 0.60), 2, 4),
            ("compliance_scoring", (1200, 2200), (0.40, 0.80), 2, 4),
            ("report_generation", (1800, 3200), (0.60, 1.20), 2, 4),
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
            graph, "start", "document_ingestion", data_size_mb=10.0
        )
        self._add_dependency_edge(
            graph, "document_ingestion", "regulation_identification", data_size_mb=15.0
        )
        self._add_dependency_edge(
            graph,
            "regulation_identification",
            "requirements_extraction",
            data_size_mb=12.0,
        )
        self._add_dependency_edge(
            graph, "requirements_extraction", "document_analysis", data_size_mb=25.0
        )
        self._add_dependency_edge(
            graph, "document_analysis", "compliance_mapping", data_size_mb=20.0
        )
        self._add_dependency_edge(
            graph, "compliance_mapping", "gap_analysis", data_size_mb=18.0
        )
        self._add_dependency_edge(
            graph, "gap_analysis", "risk_assessment", data_size_mb=15.0
        )
        self._add_dependency_edge(
            graph, "risk_assessment", "checklist_validation", data_size_mb=12.0
        )
        self._add_dependency_edge(
            graph, "checklist_validation", "evidence_collection", data_size_mb=20.0
        )
        self._add_dependency_edge(
            graph, "evidence_collection", "audit_trail_generation", data_size_mb=25.0
        )
        self._add_dependency_edge(
            graph, "audit_trail_generation", "compliance_scoring", data_size_mb=10.0
        )
        self._add_dependency_edge(
            graph, "compliance_scoring", "report_generation", data_size_mb=15.0
        )

        # Conditional branch: 20% chance of legal counsel review
        if self.random.random() < 0.2:
            self.add_task_node(
                graph,
                "legal_counsel_review",
                "legal_counsel_review",
                (180000, 360000),
                (80.0, 150.0),
                1,
                4,
            )
            self._add_dependency_edge(
                graph, "report_generation", "legal_counsel_review", data_size_mb=30.0
            )
            self._add_dependency_edge(
                graph, "legal_counsel_review", "end", data_size_mb=5.0
            )
        else:
            self._add_dependency_edge(
                graph, "report_generation", "end", data_size_mb=5.0
            )

        # Set graph metadata
        graph.graph["workflow_id"] = str(uuid.uuid4())
        graph.graph["domain"] = self.domain
        graph.graph["workflow_type"] = "compliance_check"
        graph.graph["description"] = (
            "Regulatory compliance validation workflow that identifies applicable "
            "regulations, extracts requirements, performs gap analysis, and generates "
            "compliance reports with optional legal counsel review for complex cases."
        )
        graph.graph["generated_at"] = datetime.now().isoformat()
        graph.graph["statistics"] = self._calculate_workflow_stats(graph)

        # Verify it's a DAG
        assert nx.is_directed_acyclic_graph(graph), "Generated graph contains cycles"

        return graph

    def generate_document_redlining(self, num_tasks: int) -> nx.DiGraph:
        """
        Generate a document redlining workflow.

        This workflow represents the legal document revision and negotiation process,
        where attorneys review proposed contract changes, assess risk impacts, generate
        alternative language, and track revisions through multiple negotiation rounds.

        Business Process:
        Used during contract negotiations when one party proposes changes to a draft
        contract. The system compares versions, identifies changes, assesses risk
        impact of each change, suggests alternative language based on precedents,
        and routes through attorney review before responding to the counterparty.

        Why Tasks Are Expensive:
        - Clause analysis: Detailed review of each changed clause ($1.00-2.00)
        - Risk impact assessment: Legal analysis of change implications ($1.20-2.50)
        - Alternative language generation: AI-powered drafting suggestions ($1.00-2.00)
        - Precedent suggestions: Legal database queries for favorable language ($1.50-3.00)
        - Attorney review: Mandatory human review of all changes ($60-120)

        Attorney Review Process:
        - ALWAYS required for document redlining (100% of cases)
        - Attorney reviews all proposed changes and alternative language
        - Makes final decisions on accepting, rejecting, or countering changes
        - Billable at $250-350/hour → 2-5 minutes = $60-120

        Workflow Steps:
            1. Original document parsing
            2. Version comparison (redlined vs. original)
            3. Change detection (identify all modifications)
            4. Clause-by-clause analysis (review each change)
            5. Risk impact assessment (evaluate legal risk of changes)
            6. Attorney review of changes (ALWAYS required)
            7. Alternative language generation (counter-proposals)
            8. Precedent suggestions (favorable language from past deals)
            9. Revision tracking (maintain version history)
            10. Final document generation (clean copy with revisions)

        Conditional Branch:
            - Attorney review changes: 100% (always required for all redlines)

        Args:
            num_tasks: Unused parameter (kept for interface compatibility)

        Returns:
            NetworkX DiGraph representing the document redlining workflow
            with comprehensive node and edge attributes.

        Graph Metadata:
            - workflow_id: Unique UUID for this workflow instance
            - domain: "legal"
            - workflow_type: "document_redlining"
            - description: Business process description
            - generated_at: ISO timestamp
            - statistics: Computed workflow statistics

        Typical Metrics:
            - Nodes: 12 (always includes attorney review)
            - Estimated Cost: $10-18 (base) + $60-120 (attorney review)
            - Estimated Time: 18-35s (base) + 2-5 minutes (attorney review)
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
            ("original_document_parsing", (1500, 3000), (0.30, 0.60), 4, 8),
            ("version_comparison", (2000, 4000), (0.60, 1.20), 4, 8),
            ("change_detection", (1800, 3500), (0.50, 1.00), 4, 8),
            ("clause_by_clause_analysis", (3000, 6000), (1.00, 2.00), 4, 12),
            ("risk_impact_assessment", (2500, 5000), (1.20, 2.50), 4, 12),
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

        # Build main pipeline up to attorney review
        self._add_dependency_edge(
            graph, "start", "original_document_parsing", data_size_mb=20.0
        )
        self._add_dependency_edge(
            graph, "original_document_parsing", "version_comparison", data_size_mb=25.0
        )
        self._add_dependency_edge(
            graph, "version_comparison", "change_detection", data_size_mb=15.0
        )
        self._add_dependency_edge(
            graph, "change_detection", "clause_by_clause_analysis", data_size_mb=20.0
        )
        self._add_dependency_edge(
            graph,
            "clause_by_clause_analysis",
            "risk_impact_assessment",
            data_size_mb=18.0,
        )

        # Attorney review is ALWAYS required for redlining
        self.add_task_node(
            graph,
            "attorney_review_changes",
            "attorney_review_changes",
            (120000, 300000),
            (60.0, 120.0),
            1,
            4,
        )
        self._add_dependency_edge(
            graph,
            "risk_impact_assessment",
            "attorney_review_changes",
            data_size_mb=25.0,
        )

        # Continue pipeline after attorney review
        remaining_tasks = [
            ("alternative_language_generation", (2500, 5000), (1.00, 2.00), 4, 8),
            ("precedent_suggestions", (3000, 6000), (1.50, 3.00), 4, 12),
            ("revision_tracking", (800, 1500), (0.20, 0.40), 2, 4),
            ("final_document_generation", (1200, 2400), (0.35, 0.70), 2, 4),
        ]

        for (
            task_type,
            exec_time_range,
            cost_range,
            cpu_cores,
            memory_gb,
        ) in remaining_tasks:
            self.add_task_node(
                graph,
                task_type,
                task_type,
                exec_time_range,
                cost_range,
                cpu_cores,
                memory_gb,
            )

        self._add_dependency_edge(
            graph,
            "attorney_review_changes",
            "alternative_language_generation",
            data_size_mb=22.0,
        )
        self._add_dependency_edge(
            graph,
            "alternative_language_generation",
            "precedent_suggestions",
            data_size_mb=20.0,
        )
        self._add_dependency_edge(
            graph, "precedent_suggestions", "revision_tracking", data_size_mb=15.0
        )
        self._add_dependency_edge(
            graph, "revision_tracking", "final_document_generation", data_size_mb=18.0
        )
        self._add_dependency_edge(
            graph, "final_document_generation", "end", data_size_mb=5.0
        )

        # Set graph metadata
        graph.graph["workflow_id"] = str(uuid.uuid4())
        graph.graph["domain"] = self.domain
        graph.graph["workflow_type"] = "document_redlining"
        graph.graph["description"] = (
            "Legal document redlining and negotiation workflow that compares contract "
            "versions, analyzes proposed changes, assesses risk impacts, and generates "
            "alternative language with mandatory attorney review of all modifications."
        )
        graph.graph["generated_at"] = datetime.now().isoformat()
        graph.graph["statistics"] = self._calculate_workflow_stats(graph)

        # Verify it's a DAG
        assert nx.is_directed_acyclic_graph(graph), "Generated graph contains cycles"

        return graph
