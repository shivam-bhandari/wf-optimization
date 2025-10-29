"""
Healthcare Workflow Generator

This module generates realistic medical workflow DAGs representing common healthcare
processes such as medical record extraction, insurance claim processing, and patient intake.

Each workflow includes:
- Task nodes with execution time, cost, and resource requirements
- Dependency edges with data transfer costs
- Conditional branches for realistic workflow patterns
- Comprehensive metadata for benchmarking
"""

import networkx as nx
import random
from typing import Dict, List, Tuple, Optional
import uuid
from datetime import datetime


class HealthcareWorkflowGenerator:
    """
    Generator for realistic healthcare workflow DAGs.

    This class creates directed acyclic graphs (DAGs) representing various
    healthcare processes with realistic task characteristics including execution
    times, costs, resource requirements, and failure probabilities.

    Attributes:
        random (random.Random): Random number generator instance
        seed (Optional[int]): Random seed for reproducibility
        domain (str): Domain identifier ("healthcare")
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize healthcare workflow generator.

        Args:
            seed: Random seed for reproducibility. If None, uses system time.
        """
        self.random = random.Random(seed)
        self.seed = seed
        self.domain = "healthcare"

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

        Creates a node representing a healthcare task with realistic execution
        characteristics including timing, cost, resource requirements, and
        failure probability.

        Args:
            graph: The workflow graph to add the node to
            node_id: Unique identifier for the node
            task_type: Type/name of the task (e.g., 'document_ingestion')
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

    def generate_medical_record_extraction(self, num_tasks: int) -> nx.DiGraph:
        """
        Generate a medical record extraction workflow.

        This workflow represents the process of extracting structured data from
        medical documents, including document ingestion, OCR processing, entity
        extraction, ICD code mapping, and quality validation.

        Workflow Steps:
            1. Document ingestion
            2. PDF parsing
            3. OCR processing
            4. Text extraction
            5. Entity extraction (medical terms, medications, etc.)
            6. ICD code mapping
            7. Data validation
            8. Quality check
            9. Database storage
            10. Audit logging

        Conditional Branch:
            - After OCR processing, 15% chance of manual review (60-120 seconds)

        Args:
            num_tasks: Unused parameter (kept for interface compatibility)

        Returns:
            NetworkX DiGraph representing the medical record extraction workflow
            with comprehensive node and edge attributes.

        Graph Metadata:
            - workflow_id: Unique UUID for this workflow instance
            - domain: "healthcare"
            - workflow_type: "medical_record_extraction"
            - generated_at: ISO timestamp
            - statistics: Computed workflow statistics
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
            ("document_ingestion", (500, 1000), (0.10, 0.20), 2, 4),
            ("pdf_parsing", (1200, 2000), (0.25, 0.40), 4, 8),
            ("ocr_processing", (2000, 5000), (0.50, 1.00), 8, 16),
            ("text_extraction", (800, 1500), (0.20, 0.35), 2, 4),
            ("entity_extraction", (1500, 3000), (0.40, 0.80), 4, 8),
            ("icd_code_mapping", (1500, 2500), (0.40, 0.70), 2, 6),
            ("data_validation", (500, 1500), (0.15, 0.30), 2, 4),
            ("quality_check", (600, 1200), (0.18, 0.35), 2, 4),
            ("database_storage", (300, 800), (0.10, 0.20), 1, 2),
            ("audit_logging", (200, 400), (0.05, 0.10), 1, 2),
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
            graph, "start", "document_ingestion", data_size_mb=5.0
        )
        self._add_dependency_edge(
            graph, "document_ingestion", "pdf_parsing", data_size_mb=20.0
        )
        self._add_dependency_edge(
            graph, "pdf_parsing", "ocr_processing", data_size_mb=50.0
        )

        # Conditional branch: 15% chance of manual review after OCR
        if self.random.random() < 0.15:
            self.add_task_node(
                graph,
                "manual_review",
                "manual_review",
                (60000, 120000),
                (15.0, 30.0),
                1,
                4,
            )
            self._add_dependency_edge(
                graph, "ocr_processing", "manual_review", data_size_mb=30.0
            )
            self._add_dependency_edge(
                graph, "manual_review", "text_extraction", data_size_mb=30.0
            )
        else:
            self._add_dependency_edge(
                graph, "ocr_processing", "text_extraction", data_size_mb=30.0
            )

        # Continue main pipeline
        self._add_dependency_edge(
            graph, "text_extraction", "entity_extraction", data_size_mb=15.0
        )
        self._add_dependency_edge(
            graph, "entity_extraction", "icd_code_mapping", data_size_mb=5.0
        )
        self._add_dependency_edge(
            graph, "icd_code_mapping", "data_validation", data_size_mb=8.0
        )
        self._add_dependency_edge(
            graph, "data_validation", "quality_check", data_size_mb=8.0
        )
        self._add_dependency_edge(
            graph, "quality_check", "database_storage", data_size_mb=10.0
        )
        self._add_dependency_edge(
            graph, "database_storage", "audit_logging", data_size_mb=2.0
        )
        self._add_dependency_edge(graph, "audit_logging", "end", data_size_mb=1.0)

        # Set graph metadata
        graph.graph["workflow_id"] = str(uuid.uuid4())
        graph.graph["domain"] = self.domain
        graph.graph["workflow_type"] = "medical_record_extraction"
        graph.graph["generated_at"] = datetime.now().isoformat()
        graph.graph["statistics"] = self._calculate_workflow_stats(graph)

        # Verify it's a DAG
        assert nx.is_directed_acyclic_graph(graph), "Generated graph contains cycles"

        return graph

    def generate_insurance_claim_processing(self, num_tasks: int) -> nx.DiGraph:
        """
        Generate an insurance claim processing workflow.

        This workflow represents the end-to-end process of processing a healthcare
        insurance claim, including verification, validation, fraud detection,
        medical necessity review, and payment processing.

        Workflow Steps:
            1. Claim intake
            2. Patient verification
            3. Coverage check
            4. Eligibility verification
            5. Claim validation
            6. Fraud detection
            7. Medical necessity review
            8. Pricing calculation
            9. Approval decision
            10. Payment processing
            11. Notification sending

        Conditional Branch:
            - After fraud detection, 10% chance of fraud investigation (2-4 minutes)

        Args:
            num_tasks: Unused parameter (kept for interface compatibility)

        Returns:
            NetworkX DiGraph representing the insurance claim processing workflow
            with comprehensive node and edge attributes.

        Graph Metadata:
            - workflow_id: Unique UUID for this workflow instance
            - domain: "healthcare"
            - workflow_type: "insurance_claim_processing"
            - generated_at: ISO timestamp
            - statistics: Computed workflow statistics
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
            ("claim_intake", (400, 800), (0.08, 0.15), 1, 2),
            ("patient_verification", (1000, 2000), (0.50, 1.00), 2, 4),
            ("coverage_check", (1500, 2500), (0.80, 1.50), 2, 4),
            ("eligibility_verification", (1200, 2200), (0.60, 1.20), 2, 4),
            ("claim_validation", (800, 1500), (0.25, 0.50), 2, 4),
            ("fraud_detection", (2000, 4000), (1.00, 2.00), 4, 8),
            ("medical_necessity_review", (1800, 3500), (0.80, 1.50), 4, 8),
            ("pricing_calculation", (1000, 1800), (0.30, 0.60), 2, 4),
            ("approval_decision", (500, 1000), (0.15, 0.30), 1, 2),
            ("payment_processing", (1500, 2500), (0.50, 1.00), 2, 4),
            ("notification_sending", (300, 600), (0.10, 0.20), 1, 2),
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
        self._add_dependency_edge(graph, "start", "claim_intake", data_size_mb=2.0)
        self._add_dependency_edge(
            graph, "claim_intake", "patient_verification", data_size_mb=5.0
        )
        self._add_dependency_edge(
            graph, "patient_verification", "coverage_check", data_size_mb=3.0
        )
        self._add_dependency_edge(
            graph, "coverage_check", "eligibility_verification", data_size_mb=4.0
        )
        self._add_dependency_edge(
            graph, "eligibility_verification", "claim_validation", data_size_mb=8.0
        )
        self._add_dependency_edge(
            graph, "claim_validation", "fraud_detection", data_size_mb=10.0
        )

        # Conditional branch: 10% chance of fraud investigation after fraud detection
        if self.random.random() < 0.1:
            self.add_task_node(
                graph,
                "fraud_investigation",
                "fraud_investigation",
                (120000, 240000),
                (50.0, 100.0),
                2,
                4,
            )
            self._add_dependency_edge(
                graph, "fraud_detection", "fraud_investigation", data_size_mb=15.0
            )
            self._add_dependency_edge(
                graph,
                "fraud_investigation",
                "medical_necessity_review",
                data_size_mb=15.0,
            )
        else:
            self._add_dependency_edge(
                graph, "fraud_detection", "medical_necessity_review", data_size_mb=12.0
            )

        # Continue main pipeline
        self._add_dependency_edge(
            graph, "medical_necessity_review", "pricing_calculation", data_size_mb=8.0
        )
        self._add_dependency_edge(
            graph, "pricing_calculation", "approval_decision", data_size_mb=5.0
        )
        self._add_dependency_edge(
            graph, "approval_decision", "payment_processing", data_size_mb=6.0
        )
        self._add_dependency_edge(
            graph, "payment_processing", "notification_sending", data_size_mb=2.0
        )
        self._add_dependency_edge(
            graph, "notification_sending", "end", data_size_mb=1.0
        )

        # Set graph metadata
        graph.graph["workflow_id"] = str(uuid.uuid4())
        graph.graph["domain"] = self.domain
        graph.graph["workflow_type"] = "insurance_claim_processing"
        graph.graph["generated_at"] = datetime.now().isoformat()
        graph.graph["statistics"] = self._calculate_workflow_stats(graph)

        # Verify it's a DAG
        assert nx.is_directed_acyclic_graph(graph), "Generated graph contains cycles"

        return graph

    def generate_patient_intake_workflow(self, num_tasks: int) -> nx.DiGraph:
        """
        Generate a patient intake workflow.

        This workflow represents the process of onboarding a new patient or
        preparing an existing patient for a medical visit, including registration,
        insurance verification, medical history collection, and appointment scheduling.

        Workflow Steps:
            1. Registration
            2. Insurance verification
            3. Medical history collection
            4. Consent form processing
            5. Appointment scheduling
            6. Pre-visit questionnaire
            7. Record retrieval
            8. Provider assignment

        Args:
            num_tasks: Unused parameter (kept for interface compatibility)

        Returns:
            NetworkX DiGraph representing the patient intake workflow
            with comprehensive node and edge attributes.

        Graph Metadata:
            - workflow_id: Unique UUID for this workflow instance
            - domain: "healthcare"
            - workflow_type: "patient_intake"
            - generated_at: ISO timestamp
            - statistics: Computed workflow statistics
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
            ("registration", (600, 1200), (0.15, 0.30), 1, 2),
            ("insurance_verification", (1500, 2500), (0.70, 1.30), 2, 4),
            ("medical_history_collection", (1000, 2000), (0.30, 0.60), 2, 4),
            ("consent_form_processing", (800, 1500), (0.20, 0.40), 2, 4),
            ("appointment_scheduling", (1200, 2000), (0.40, 0.80), 2, 4),
            ("pre_visit_questionnaire", (500, 1000), (0.15, 0.30), 1, 2),
            ("record_retrieval", (2000, 3500), (0.80, 1.50), 4, 8),
            ("provider_assignment", (600, 1200), (0.20, 0.40), 2, 4),
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
        self._add_dependency_edge(graph, "start", "registration", data_size_mb=1.0)
        self._add_dependency_edge(
            graph, "registration", "insurance_verification", data_size_mb=3.0
        )
        self._add_dependency_edge(
            graph,
            "insurance_verification",
            "medical_history_collection",
            data_size_mb=5.0,
        )
        self._add_dependency_edge(
            graph,
            "medical_history_collection",
            "consent_form_processing",
            data_size_mb=8.0,
        )
        self._add_dependency_edge(
            graph, "consent_form_processing", "appointment_scheduling", data_size_mb=4.0
        )
        self._add_dependency_edge(
            graph, "appointment_scheduling", "pre_visit_questionnaire", data_size_mb=2.0
        )
        self._add_dependency_edge(
            graph, "pre_visit_questionnaire", "record_retrieval", data_size_mb=10.0
        )
        self._add_dependency_edge(
            graph, "record_retrieval", "provider_assignment", data_size_mb=15.0
        )
        self._add_dependency_edge(graph, "provider_assignment", "end", data_size_mb=3.0)

        # Set graph metadata
        graph.graph["workflow_id"] = str(uuid.uuid4())
        graph.graph["domain"] = self.domain
        graph.graph["workflow_type"] = "patient_intake"
        graph.graph["generated_at"] = datetime.now().isoformat()
        graph.graph["statistics"] = self._calculate_workflow_stats(graph)

        # Verify it's a DAG
        assert nx.is_directed_acyclic_graph(graph), "Generated graph contains cycles"

        return graph
