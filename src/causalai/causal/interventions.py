"""
Intervention operations for the do() operator.

This module implements the do() operator for causal interventions,
allowing users to set variables to specific values and observe
the causal effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from causalai.core.nodes import ConversationNode, NodeType, NodeMetadata

if TYPE_CHECKING:
    from causalai.core.dag import ConversationDAG


@dataclass
class Intervention:
    """Represents a do() intervention on the conversation graph.

    An intervention do(X=x) fixes variable X to value x, breaking
    all causal influences on X while preserving X's influence on descendants.

    Attributes:
        id: Unique identifier for this intervention
        target_node_id: The node being intervened upon
        intervention_value: The value being set
        description: Human-readable description
        timestamp: When the intervention was created
    """

    target_node_id: UUID
    intervention_value: Any
    description: str = ""
    id: UUID = field(default_factory=uuid4)

    def __str__(self) -> str:
        """String representation of the intervention."""
        return f"do({self.target_node_id}) = {self.intervention_value}"


@dataclass
class InterventionResult:
    """Result of applying an intervention.

    Attributes:
        intervention: The intervention that was applied
        intervention_node_id: ID of the intervention marker node
        mutilated_graph: The graph with intervention applied
        affected_nodes: Nodes whose values may change due to intervention
    """

    intervention: Intervention
    intervention_node_id: UUID
    affected_nodes: set[UUID] = field(default_factory=set)


class InterventionEngine:
    """Manages interventions on the conversation graph.

    The InterventionEngine allows applying do() operations to the graph,
    which:
    1. Creates an intervention marker node
    2. Tracks the intervention for do-calculus analysis
    3. Provides methods for counterfactual reasoning

    Example:
        >>> engine = InterventionEngine(dag)
        >>> result = engine.do(node_id, "new_value", "Setting X to new value")
        >>> # The graph now has an intervention marker
        >>> # Get the mutilated graph for causal analysis
        >>> mutilated = engine.get_interventional_graph()
    """

    def __init__(self, dag: ConversationDAG):
        """Initialize the engine with a DAG.

        Args:
            dag: The ConversationDAG to manage interventions on
        """
        self.dag = dag
        self._active_interventions: dict[UUID, Intervention] = {}
        self._intervention_history: list[Intervention] = []

    def do(
        self,
        node_id: UUID,
        new_value: Any,
        description: str = "",
    ) -> InterventionResult:
        """Apply do(X=x) intervention.

        This:
        1. Creates an intervention marker node
        2. Records the intervention for do-calculus analysis
        3. Returns information about affected nodes

        Args:
            node_id: The node to intervene on
            new_value: The value to set
            description: Human-readable description

        Returns:
            InterventionResult with details about the intervention

        Raises:
            ValueError: If the node doesn't exist
        """
        if not self.dag.has_node(node_id):
            raise ValueError(f"Node {node_id} not in graph")

        intervention = Intervention(
            target_node_id=node_id,
            intervention_value=new_value,
            description=description or f"do({node_id}) = {new_value}",
        )

        # Create intervention marker node
        intervention_node = ConversationNode(
            node_type=NodeType.INTERVENTION,
            content=str(intervention),
            is_intervention=True,
            intervened_variables=[str(node_id)],
            metadata=NodeMetadata(
                causal_annotations={
                    "intervention_id": str(intervention.id),
                    "target_node": str(node_id),
                    "value": str(new_value),
                }
            ),
        )

        new_node_id = self.dag.add_node(intervention_node)

        # Record intervention
        self._active_interventions[intervention.id] = intervention
        self._intervention_history.append(intervention)

        # Find affected nodes (descendants of the intervened node)
        affected_nodes = self.dag.get_descendants(node_id)

        return InterventionResult(
            intervention=intervention,
            intervention_node_id=new_node_id,
            affected_nodes=affected_nodes,
        )

    def undo(self, intervention_id: UUID) -> bool:
        """Remove an active intervention.

        Args:
            intervention_id: ID of the intervention to remove

        Returns:
            True if the intervention was removed, False if not found
        """
        if intervention_id in self._active_interventions:
            del self._active_interventions[intervention_id]
            return True
        return False

    def get_active_interventions(self) -> list[Intervention]:
        """Get all currently active interventions."""
        return list(self._active_interventions.values())

    def get_intervention_history(self) -> list[Intervention]:
        """Get the history of all interventions (including undone ones)."""
        return self._intervention_history.copy()

    def get_interventional_graph(self) -> ConversationDAG:
        """Get the graph with all active interventions applied.

        Returns G with incoming edges to all intervened nodes removed.
        This is the mutilated graph used for causal effect computation.

        Returns:
            A new ConversationDAG with interventions applied
        """
        intervened_nodes = {
            i.target_node_id for i in self._active_interventions.values()
        }
        return self.dag.get_mutilated_graph(intervened_nodes)

    def clear_all(self) -> None:
        """Clear all active interventions."""
        self._active_interventions.clear()

    def is_intervened(self, node_id: UUID) -> bool:
        """Check if a node is currently under intervention."""
        return any(
            i.target_node_id == node_id
            for i in self._active_interventions.values()
        )

    def get_intervention_for_node(self, node_id: UUID) -> Intervention | None:
        """Get the active intervention for a node, if any."""
        for intervention in self._active_interventions.values():
            if intervention.target_node_id == node_id:
                return intervention
        return None


class CounterfactualEngine:
    """Engine for counterfactual queries.

    Counterfactual reasoning asks "What would Y have been if we had done X=x?"
    given that we observed certain evidence.

    This requires:
    1. Abduction: Update the model based on observed evidence
    2. Action: Apply the hypothetical intervention
    3. Prediction: Compute the counterfactual outcome

    Note: Full counterfactual computation requires a structural causal model
    with specific functional forms. This implementation provides the
    graphical framework for counterfactual analysis.
    """

    def __init__(self, dag: ConversationDAG):
        """Initialize the counterfactual engine.

        Args:
            dag: The ConversationDAG to analyze
        """
        self.dag = dag
        self.intervention_engine = InterventionEngine(dag)

    def query(
        self,
        outcome_node: UUID,
        intervention_node: UUID,
        intervention_value: Any,
        evidence: dict[UUID, Any] | None = None,
    ) -> dict[str, Any]:
        """Answer a counterfactual query.

        Query: "What would outcome_node be if we had set intervention_node
        to intervention_value, given the evidence?"

        Args:
            outcome_node: The node whose counterfactual value we want
            intervention_node: The node to hypothetically intervene on
            intervention_value: The hypothetical intervention value
            evidence: Observed evidence (node_id -> observed_value)

        Returns:
            Dictionary with counterfactual analysis results
        """
        evidence = evidence or {}

        # This is a structural analysis - actual computation would require
        # specific functional forms in a structural causal model

        result = {
            "query": f"Y_{{{intervention_node}={intervention_value}}} | evidence",
            "outcome_node": str(outcome_node),
            "intervention": f"do({intervention_node}) = {intervention_value}",
            "evidence": {str(k): str(v) for k, v in evidence.items()},
            "analysis": self._analyze_counterfactual(
                outcome_node, intervention_node, evidence
            ),
        }

        return result

    def _analyze_counterfactual(
        self,
        outcome_node: UUID,
        intervention_node: UUID,
        evidence: dict[UUID, Any],
    ) -> dict[str, Any]:
        """Analyze the counterfactual query structure.

        Determines which variables are affected and which paths are relevant.
        """
        # Get the causal paths from intervention to outcome
        try:
            import networkx as nx
            paths = list(nx.all_simple_paths(
                self.dag._graph,
                intervention_node,
                outcome_node,
            ))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            paths = []

        # Determine ancestors of outcome (potential confounders)
        ancestors = self.dag.get_ancestors(outcome_node)

        # Determine which evidence nodes are relevant
        relevant_evidence = {
            k: v for k, v in evidence.items()
            if k in ancestors or k == outcome_node
        }

        return {
            "causal_paths": len(paths),
            "path_lengths": [len(p) for p in paths],
            "num_ancestors": len(ancestors),
            "relevant_evidence_nodes": len(relevant_evidence),
            "requires_abduction": bool(evidence),
            "identifiable": len(paths) > 0,
        }

    def twin_network(
        self,
        factual_evidence: dict[UUID, Any],
        counterfactual_intervention: tuple[UUID, Any],
    ) -> dict[str, Any]:
        """Create a twin network for counterfactual analysis.

        A twin network represents both the factual world (what happened)
        and the counterfactual world (what would have happened under
        different circumstances) in a single graph.

        Args:
            factual_evidence: What was observed in the factual world
            counterfactual_intervention: (node_id, value) for the counterfactual

        Returns:
            Description of the twin network structure
        """
        intervention_node, intervention_value = counterfactual_intervention

        return {
            "factual_world": {
                "evidence": {str(k): str(v) for k, v in factual_evidence.items()},
                "nodes": [str(n) for n in self.dag._graph.nodes()],
            },
            "counterfactual_world": {
                "intervention": f"do({intervention_node}) = {intervention_value}",
                "affected_descendants": [
                    str(n) for n in self.dag.get_descendants(intervention_node)
                ],
            },
            "shared_exogenous": "Exogenous variables are shared between worlds",
            "note": "Full twin network computation requires structural equations",
        }
