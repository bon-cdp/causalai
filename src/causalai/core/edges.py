"""
Conversation edge types representing causal relationships.

This module defines the edge types used to connect nodes in the
conversation DAG, representing various causal relationships.
"""

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class EdgeType(str, Enum):
    """Types of edges representing causal relationships.

    Each edge type has specific causal semantics:
    - TEMPORAL: Sequential order (A then B)
    - CAUSAL: A causes B (direct causation)
    - CONTEXTUAL: A provides context for B
    - INTERVENTION: do(A) applied to produce B
    - ALTERNATIVE: B is an alternative to A (parallel candidates)
    - REPLACEMENT: B replaces A
    - REFERENCE: B references content from A
    - CONFOUNDING: A is a confounding variable for the A→B relationship
    """

    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CONTEXTUAL = "contextual"
    INTERVENTION = "intervention"
    ALTERNATIVE = "alternative"
    REPLACEMENT = "replacement"
    REFERENCE = "reference"
    CONFOUNDING = "confounding"


class EdgeStrength(str, Enum):
    """Strength of causal relationship.

    Used for weighted causal analysis and identifying
    strong vs weak causal connections.
    """

    STRONG = "strong"  # Direct causation
    MODERATE = "moderate"  # Contributing factor
    WEAK = "weak"  # Minor influence
    UNKNOWN = "unknown"  # Not yet determined


class ConversationEdge(BaseModel):
    """An edge in the conversation DAG representing a causal relationship.

    Edges connect nodes and encode the nature of their relationship.
    Each edge can be marked for d-separation analysis and has
    optional conditioning information.

    Attributes:
        source_id: UUID of the source (cause) node
        target_id: UUID of the target (effect) node
        edge_type: The type of causal relationship
        strength: Strength of the causal connection
        is_blocked: Whether this edge is blocked for d-separation
        conditioned_on: Node IDs this edge is conditioned on
        weight: Numerical weight for weighted analyses
        annotations: Additional metadata about the edge

    Example:
        >>> edge = ConversationEdge(
        ...     source_id=user_node.id,
        ...     target_id=response_node.id,
        ...     edge_type=EdgeType.CAUSAL,
        ...     strength=EdgeStrength.STRONG,
        ... )
    """

    source_id: UUID
    target_id: UUID
    edge_type: EdgeType
    strength: EdgeStrength = EdgeStrength.UNKNOWN

    # Causal properties for d-separation analysis
    is_blocked: bool = False
    conditioned_on: list[UUID] = Field(default_factory=list)

    # Additional metadata
    weight: float = 1.0
    annotations: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True

    def block(self) -> "ConversationEdge":
        """Create a copy of this edge marked as blocked."""
        return self.model_copy(update={"is_blocked": True})

    def unblock(self) -> "ConversationEdge":
        """Create a copy of this edge marked as unblocked."""
        return self.model_copy(update={"is_blocked": False})

    def condition_on(self, node_ids: list[UUID]) -> "ConversationEdge":
        """Create a copy of this edge conditioned on additional nodes."""
        new_conditions = self.conditioned_on + node_ids
        return self.model_copy(update={"conditioned_on": new_conditions})

    def with_strength(self, strength: EdgeStrength) -> "ConversationEdge":
        """Create a copy of this edge with a new strength."""
        return self.model_copy(update={"strength": strength})

    def with_weight(self, weight: float) -> "ConversationEdge":
        """Create a copy of this edge with a new weight."""
        return self.model_copy(update={"weight": weight})

    def __hash__(self) -> int:
        """Hash based on source and target for set/dict usage."""
        return hash((self.source_id, self.target_id))

    def __eq__(self, other: object) -> bool:
        """Equality based on source and target IDs."""
        if not isinstance(other, ConversationEdge):
            return False
        return self.source_id == other.source_id and self.target_id == other.target_id


def create_temporal_edge(source_id: UUID, target_id: UUID) -> ConversationEdge:
    """Create a temporal (sequential) edge between nodes."""
    return ConversationEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.TEMPORAL,
    )


def create_causal_edge(
    source_id: UUID,
    target_id: UUID,
    strength: EdgeStrength = EdgeStrength.STRONG,
) -> ConversationEdge:
    """Create a causal edge between nodes.

    Args:
        source_id: The cause node
        target_id: The effect node
        strength: How strong the causal relationship is

    Returns:
        A ConversationEdge with CAUSAL type
    """
    return ConversationEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.CAUSAL,
        strength=strength,
    )


def create_intervention_edge(source_id: UUID, target_id: UUID) -> ConversationEdge:
    """Create an intervention edge (do() operation applied).

    Intervention edges represent the application of do() operations
    where incoming causal influences are severed.
    """
    return ConversationEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.INTERVENTION,
        strength=EdgeStrength.STRONG,
    )


def create_contextual_edge(source_id: UUID, target_id: UUID) -> ConversationEdge:
    """Create a contextual edge (provides context but not direct cause)."""
    return ConversationEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.CONTEXTUAL,
        strength=EdgeStrength.MODERATE,
    )


def create_alternative_edge(source_id: UUID, target_id: UUID) -> ConversationEdge:
    """Create an alternative edge (parallel candidate relationship)."""
    return ConversationEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=EdgeType.ALTERNATIVE,
    )


def create_replacement_edge(
    original_id: UUID,
    replacement_id: UUID,
) -> ConversationEdge:
    """Create a replacement edge (node supersedes another)."""
    return ConversationEdge(
        source_id=original_id,
        target_id=replacement_id,
        edge_type=EdgeType.REPLACEMENT,
    )
