"""Unit tests for ConversationNode and related classes."""

import pytest
from uuid import UUID

from causalai.core.nodes import (
    ConversationNode,
    NodeType,
    NodeStatus,
    NodeMetadata,
    create_user_message,
    create_assistant_response,
    create_system_prompt,
    create_intervention,
    create_fork_point,
)


class TestNodeType:
    """Tests for NodeType enum."""

    def test_all_types_defined(self):
        """Verify all expected node types exist."""
        expected_types = [
            "SYSTEM_PROMPT",
            "USER_MESSAGE",
            "ASSISTANT_RESPONSE",
            "TOOL_CALL",
            "TOOL_RESULT",
            "INTERVENTION",
            "OBSERVATION",
            "FORK_POINT",
            "AGGREGATE",
        ]
        for type_name in expected_types:
            assert hasattr(NodeType, type_name)

    def test_type_values_are_strings(self):
        """Node type values should be strings for serialization."""
        assert NodeType.USER_MESSAGE.value == "user_message"
        assert NodeType.ASSISTANT_RESPONSE.value == "assistant_response"


class TestNodeStatus:
    """Tests for NodeStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all expected statuses exist."""
        expected_statuses = [
            "PENDING",
            "GENERATING",
            "COMPLETED",
            "SELECTED",
            "REJECTED",
            "REPLACED",
            "ARCHIVED",
        ]
        for status_name in expected_statuses:
            assert hasattr(NodeStatus, status_name)


class TestNodeMetadata:
    """Tests for NodeMetadata."""

    def test_default_values(self):
        """Metadata should have sensible defaults."""
        meta = NodeMetadata()
        assert meta.model_id is None
        assert meta.temperature is None
        assert meta.test_results == {}
        assert meta.custom_tags == []

    def test_custom_values(self):
        """Metadata should accept custom values."""
        meta = NodeMetadata(
            model_id="qwen-plus",
            temperature=0.7,
            custom_tags=["test", "example"],
        )
        assert meta.model_id == "qwen-plus"
        assert meta.temperature == 0.7
        assert "test" in meta.custom_tags


class TestConversationNode:
    """Tests for ConversationNode."""

    def test_create_basic_node(self):
        """Create a basic conversation node."""
        node = ConversationNode(
            node_type=NodeType.USER_MESSAGE,
            content="Hello, world!",
        )
        assert isinstance(node.id, UUID)
        assert node.node_type == NodeType.USER_MESSAGE
        assert node.content == "Hello, world!"
        assert node.status == NodeStatus.COMPLETED

    def test_node_immutability(self):
        """Nodes should be immutable (frozen)."""
        node = ConversationNode(
            node_type=NodeType.USER_MESSAGE,
            content="Test",
        )
        with pytest.raises(Exception):  # ValidationError or similar
            node.content = "Modified"

    def test_with_status(self):
        """with_status should create a copy with new status."""
        node = ConversationNode(
            node_type=NodeType.ASSISTANT_RESPONSE,
            content="Response",
        )
        updated = node.with_status(NodeStatus.SELECTED)

        assert updated.status == NodeStatus.SELECTED
        assert node.status == NodeStatus.COMPLETED  # Original unchanged
        assert updated.content == node.content
        assert updated.id == node.id

    def test_add_tag(self):
        """add_tag should create a copy with additional tag."""
        node = ConversationNode(
            node_type=NodeType.USER_MESSAGE,
            content="Test",
        )
        tagged = node.add_tag("important")

        assert "important" in tagged.metadata.custom_tags
        assert "important" not in node.metadata.custom_tags

    def test_mark_as_intervention(self):
        """mark_as_intervention should set intervention properties."""
        node = ConversationNode(
            node_type=NodeType.USER_MESSAGE,
            content="Change the topic",
        )
        intervention = node.mark_as_intervention(["topic"])

        assert intervention.is_intervention is True
        assert "topic" in intervention.intervened_variables
        assert node.is_intervention is False  # Original unchanged

    def test_node_equality(self):
        """Nodes with same ID should be equal."""
        node1 = ConversationNode(
            node_type=NodeType.USER_MESSAGE,
            content="Test",
        )
        # Create a copy (same ID)
        node2 = node1.with_status(NodeStatus.SELECTED)

        assert node1 == node2
        assert hash(node1) == hash(node2)

    def test_node_inequality(self):
        """Nodes with different IDs should not be equal."""
        node1 = ConversationNode(node_type=NodeType.USER_MESSAGE, content="A")
        node2 = ConversationNode(node_type=NodeType.USER_MESSAGE, content="A")

        assert node1 != node2  # Different UUIDs


class TestNodeFactoryFunctions:
    """Tests for node factory functions."""

    def test_create_user_message(self):
        """create_user_message should create correct node type."""
        node = create_user_message("What is AI?")

        assert node.node_type == NodeType.USER_MESSAGE
        assert node.content == "What is AI?"

    def test_create_user_message_with_metadata(self):
        """create_user_message should accept metadata."""
        node = create_user_message(
            "Hello",
            model_id="test-model",
            custom_tags=["greeting"],
        )

        assert node.metadata.model_id == "test-model"
        assert "greeting" in node.metadata.custom_tags

    def test_create_assistant_response(self):
        """create_assistant_response should create correct node type."""
        node = create_assistant_response("AI stands for Artificial Intelligence.")

        assert node.node_type == NodeType.ASSISTANT_RESPONSE
        assert "Artificial Intelligence" in node.content

    def test_create_system_prompt(self):
        """create_system_prompt should create correct node type."""
        node = create_system_prompt("You are a helpful assistant.")

        assert node.node_type == NodeType.SYSTEM_PROMPT
        assert "helpful" in node.content

    def test_create_intervention(self):
        """create_intervention should create intervention node."""
        node = create_intervention(
            target_content="new_value",
            intervened_variables=["temperature"],
            description="Setting temperature to new value",
        )

        assert node.node_type == NodeType.INTERVENTION
        assert node.is_intervention is True
        assert "temperature" in node.intervened_variables

    def test_create_fork_point(self):
        """create_fork_point should create fork point node."""
        node = create_fork_point("alternative_a", "Testing different approach")

        assert node.node_type == NodeType.FORK_POINT
        assert node.is_fork_point is True
        assert node.branch_label == "alternative_a"


class TestNodeSerialization:
    """Tests for node serialization."""

    def test_model_dump(self):
        """Node should serialize to dictionary."""
        node = ConversationNode(
            node_type=NodeType.USER_MESSAGE,
            content="Test content",
        )
        data = node.model_dump()

        assert data["node_type"] == "user_message"
        assert data["content"] == "Test content"
        assert "id" in data

    def test_model_validate(self):
        """Node should deserialize from dictionary."""
        original = ConversationNode(
            node_type=NodeType.USER_MESSAGE,
            content="Test",
        )
        data = original.model_dump()
        restored = ConversationNode.model_validate(data)

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.node_type == original.node_type
