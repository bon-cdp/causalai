"""
Conversation node types and data structures.

This module defines the node types used in the conversation DAG,
including prompts, responses, interventions, and forks.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the conversation DAG.

    Each type has specific causal semantics:
    - SYSTEM_PROMPT: Background/confounding variable
    - USER_MESSAGE: Treatment/intervention trigger
    - ASSISTANT_RESPONSE: Outcome variable
    - TOOL_CALL: Mediating variable
    - TOOL_RESULT: Intermediate outcome
    - INTERVENTION: do() operation marker
    - OBSERVATION: Observed variable
    - FORK_POINT: Counterfactual branch
    - AGGREGATE: Merged/combined node
    """

    SYSTEM_PROMPT = "system_prompt"
    USER_MESSAGE = "user_message"
    ASSISTANT_RESPONSE = "assistant_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    INTERVENTION = "intervention"
    OBSERVATION = "observation"
    FORK_POINT = "fork_point"
    AGGREGATE = "aggregate"


class NodeStatus(str, Enum):
    """Lifecycle status of a node.

    Nodes progress through statuses as they are generated and evaluated:
    - PENDING: Awaiting generation
    - GENERATING: LLM is producing output
    - COMPLETED: Successfully generated
    - SELECTED: Chosen from parallel options
    - REJECTED: Not selected from parallel options
    - REPLACED: Superseded by another node
    - ARCHIVED: Soft-deleted
    """

    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    SELECTED = "selected"
    REJECTED = "rejected"
    REPLACED = "replaced"
    ARCHIVED = "archived"


class NodeMetadata(BaseModel):
    """Metadata attached to each node.

    Stores information about how the node was generated,
    test results, and user-defined tags.
    """

    model_id: str | None = None
    temperature: float | None = None
    token_count: int | None = None
    latency_ms: int | None = None
    test_results: dict[str, Any] = Field(default_factory=dict)
    custom_tags: list[str] = Field(default_factory=list)
    causal_annotations: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class ConversationNode(BaseModel):
    """A node in the conversation DAG.

    Represents a single event in the conversation, such as a user message,
    assistant response, or intervention. Each node has causal properties
    that determine how it relates to other nodes in the graph.

    Attributes:
        id: Unique identifier for the node
        node_type: The type of conversation event
        content: The actual content (text) of the node
        status: Current lifecycle status
        created_at: When the node was created
        metadata: Additional information about the node
        is_intervention: True if this is a do() node
        intervened_variables: Variables affected by this intervention
        observed_variables: Variables observed/mentioned in this node
        is_fork_point: True if this node is a branching point
        branch_label: Label for the branch if this is a fork
        replaced_by: ID of node that replaced this one

    Example:
        >>> node = ConversationNode(
        ...     node_type=NodeType.USER_MESSAGE,
        ...     content="What is causal inference?",
        ... )
        >>> node.id  # UUID automatically generated
        >>> node.status  # Defaults to COMPLETED
    """

    id: UUID = Field(default_factory=uuid4)
    node_type: NodeType
    content: str
    status: NodeStatus = NodeStatus.COMPLETED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)

    # Causal properties
    is_intervention: bool = False
    intervened_variables: list[str] = Field(default_factory=list)
    observed_variables: list[str] = Field(default_factory=list)

    # Navigation properties
    is_fork_point: bool = False
    branch_label: str | None = None
    replaced_by: UUID | None = None

    class Config:
        frozen = True  # Immutable after creation

    def with_status(self, status: NodeStatus) -> "ConversationNode":
        """Create a copy of this node with a new status.

        Since nodes are immutable, this creates a new node with the updated status.
        """
        return self.model_copy(update={"status": status})

    def with_metadata(self, **updates: Any) -> "ConversationNode":
        """Create a copy of this node with updated metadata."""
        new_metadata = self.metadata.model_copy(update=updates)
        return self.model_copy(update={"metadata": new_metadata})

    def add_tag(self, tag: str) -> "ConversationNode":
        """Create a copy of this node with an additional tag."""
        new_tags = self.metadata.custom_tags + [tag]
        return self.with_metadata(custom_tags=new_tags)

    def mark_as_intervention(self, variables: list[str]) -> "ConversationNode":
        """Mark this node as an intervention on the specified variables."""
        return self.model_copy(
            update={
                "is_intervention": True,
                "intervened_variables": variables,
            }
        )

    def __hash__(self) -> int:
        """Hash based on node ID for set/dict usage."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on node ID."""
        if not isinstance(other, ConversationNode):
            return False
        return self.id == other.id


def create_user_message(content: str, **metadata: Any) -> ConversationNode:
    """Factory function to create a user message node."""
    return ConversationNode(
        node_type=NodeType.USER_MESSAGE,
        content=content,
        metadata=NodeMetadata(**metadata),
    )


def create_assistant_response(content: str, **metadata: Any) -> ConversationNode:
    """Factory function to create an assistant response node."""
    return ConversationNode(
        node_type=NodeType.ASSISTANT_RESPONSE,
        content=content,
        metadata=NodeMetadata(**metadata),
    )


def create_system_prompt(content: str) -> ConversationNode:
    """Factory function to create a system prompt node."""
    return ConversationNode(
        node_type=NodeType.SYSTEM_PROMPT,
        content=content,
    )


def create_intervention(
    target_content: str,
    intervened_variables: list[str],
    description: str = "",
) -> ConversationNode:
    """Factory function to create an intervention (do()) node.

    Args:
        target_content: The content/value being set by the intervention
        intervened_variables: List of variable names being intervened upon
        description: Human-readable description of the intervention

    Returns:
        A ConversationNode marked as an intervention
    """
    content = description or f"do({', '.join(intervened_variables)}) = {target_content}"
    return ConversationNode(
        node_type=NodeType.INTERVENTION,
        content=content,
        is_intervention=True,
        intervened_variables=intervened_variables,
    )


def create_fork_point(label: str, reason: str = "") -> ConversationNode:
    """Factory function to create a fork point node.

    Fork points represent counterfactual branches in the conversation.

    Args:
        label: Label for this branch (e.g., "branch_a", "alternative_1")
        reason: Why this fork was created

    Returns:
        A ConversationNode marked as a fork point
    """
    return ConversationNode(
        node_type=NodeType.FORK_POINT,
        content=reason or f"Fork: {label}",
        is_fork_point=True,
        branch_label=label,
    )
