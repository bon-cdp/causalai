"""Unit tests for ConversationDAG."""

import pytest
from uuid import uuid4

from causalai.core.dag import ConversationDAG
from causalai.core.nodes import ConversationNode, NodeType, create_user_message
from causalai.core.edges import (
    ConversationEdge,
    EdgeType,
    create_causal_edge,
    create_temporal_edge,
)


class TestConversationDAGBasics:
    """Basic tests for ConversationDAG."""

    def test_create_empty_dag(self):
        """Create an empty DAG."""
        dag = ConversationDAG()

        assert dag.node_count == 0
        assert dag.edge_count == 0
        assert dag.session_id is not None

    def test_add_node(self):
        """Add a node to the DAG."""
        dag = ConversationDAG()
        node = create_user_message("Hello")

        node_id = dag.add_node(node)

        assert node_id == node.id
        assert dag.node_count == 1
        assert dag.has_node(node.id)

    def test_get_node(self):
        """Retrieve a node by ID."""
        dag = ConversationDAG()
        node = create_user_message("Test")
        dag.add_node(node)

        retrieved = dag.get_node(node.id)

        assert retrieved is not None
        assert retrieved.content == "Test"

    def test_get_nonexistent_node(self):
        """Getting a nonexistent node returns None."""
        dag = ConversationDAG()

        assert dag.get_node(uuid4()) is None

    def test_remove_node(self):
        """Remove a node from the DAG."""
        dag = ConversationDAG()
        node = create_user_message("To be removed")
        dag.add_node(node)

        result = dag.remove_node(node.id)

        assert result is True
        assert dag.node_count == 0
        assert not dag.has_node(node.id)


class TestDAGEdges:
    """Tests for edge operations."""

    def test_add_edge(self):
        """Add an edge between nodes."""
        dag = ConversationDAG()
        node1 = create_user_message("Question")
        node2 = ConversationNode(
            node_type=NodeType.ASSISTANT_RESPONSE,
            content="Answer",
        )
        dag.add_node(node1)
        dag.add_node(node2)

        edge = create_causal_edge(node1.id, node2.id)
        dag.add_edge(edge)

        assert dag.edge_count == 1
        assert dag.has_edge(node1.id, node2.id)

    def test_add_edge_missing_node(self):
        """Adding edge with missing node raises error."""
        dag = ConversationDAG()
        node1 = create_user_message("Test")
        dag.add_node(node1)

        edge = ConversationEdge(
            source_id=node1.id,
            target_id=uuid4(),  # Nonexistent
            edge_type=EdgeType.CAUSAL,
        )

        with pytest.raises(ValueError):
            dag.add_edge(edge)

    def test_get_parents(self):
        """Get parent nodes."""
        dag = ConversationDAG()
        parent = create_user_message("Parent")
        child = ConversationNode(
            node_type=NodeType.ASSISTANT_RESPONSE,
            content="Child",
        )
        dag.add_node(parent)
        dag.add_node(child)
        dag.add_edge(create_causal_edge(parent.id, child.id))

        parents = dag.get_parents(child.id)

        assert len(parents) == 1
        assert parents[0].id == parent.id

    def test_get_children(self):
        """Get child nodes."""
        dag = ConversationDAG()
        parent = create_user_message("Parent")
        child = ConversationNode(
            node_type=NodeType.ASSISTANT_RESPONSE,
            content="Child",
        )
        dag.add_node(parent)
        dag.add_node(child)
        dag.add_edge(create_causal_edge(parent.id, child.id))

        children = dag.get_children(parent.id)

        assert len(children) == 1
        assert children[0].id == child.id


class TestDAGCausalProperties:
    """Tests for causal graph properties."""

    def test_get_ancestors(self):
        """Get all ancestors of a node."""
        dag = ConversationDAG()
        # Create chain: A → B → C
        a = create_user_message("A")
        b = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="B")
        c = ConversationNode(node_type=NodeType.USER_MESSAGE, content="C")

        dag.add_node(a)
        dag.add_node(b)
        dag.add_node(c)
        dag.add_edge(create_causal_edge(a.id, b.id))
        dag.add_edge(create_causal_edge(b.id, c.id))

        ancestors_of_c = dag.get_ancestors(c.id)

        assert a.id in ancestors_of_c
        assert b.id in ancestors_of_c
        assert len(ancestors_of_c) == 2

    def test_get_descendants(self):
        """Get all descendants of a node."""
        dag = ConversationDAG()
        # Create chain: A → B → C
        a = create_user_message("A")
        b = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="B")
        c = ConversationNode(node_type=NodeType.USER_MESSAGE, content="C")

        dag.add_node(a)
        dag.add_node(b)
        dag.add_node(c)
        dag.add_edge(create_causal_edge(a.id, b.id))
        dag.add_edge(create_causal_edge(b.id, c.id))

        descendants_of_a = dag.get_descendants(a.id)

        assert b.id in descendants_of_a
        assert c.id in descendants_of_a
        assert len(descendants_of_a) == 2

    def test_is_valid_dag(self):
        """Check if graph is a valid DAG."""
        dag = ConversationDAG()
        a = create_user_message("A")
        b = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="B")

        dag.add_node(a)
        dag.add_node(b)
        dag.add_edge(create_causal_edge(a.id, b.id))

        assert dag.is_valid_dag() is True

    def test_topological_order(self):
        """Get nodes in topological order."""
        dag = ConversationDAG()
        a = create_user_message("A")
        b = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="B")
        c = ConversationNode(node_type=NodeType.USER_MESSAGE, content="C")

        dag.add_node(a)
        dag.add_node(b)
        dag.add_node(c)
        dag.add_edge(create_causal_edge(a.id, b.id))
        dag.add_edge(create_causal_edge(b.id, c.id))

        order = dag.topological_order()

        assert order.index(a.id) < order.index(b.id)
        assert order.index(b.id) < order.index(c.id)

    def test_get_roots(self):
        """Get root nodes (no incoming edges)."""
        dag = ConversationDAG()
        root1 = create_user_message("Root1")
        root2 = create_user_message("Root2")
        child = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="Child")

        dag.add_node(root1)
        dag.add_node(root2)
        dag.add_node(child)
        dag.add_edge(create_causal_edge(root1.id, child.id))
        dag.add_edge(create_causal_edge(root2.id, child.id))

        roots = dag.get_roots()

        assert len(roots) == 2
        assert root1.id in roots
        assert root2.id in roots


class TestGraphSurgery:
    """Tests for graph surgery operations used in do-calculus."""

    def create_confounding_dag(self):
        """Create a DAG with confounding: Z → X, Z → Y, X → Y.

        This represents a classic confounding scenario where Z is
        a common cause of both X and Y.
        """
        dag = ConversationDAG()

        # Z: confounder (e.g., socioeconomic status)
        z = ConversationNode(node_type=NodeType.SYSTEM_PROMPT, content="Z (confounder)")
        # X: treatment (e.g., education)
        x = ConversationNode(node_type=NodeType.USER_MESSAGE, content="X (treatment)")
        # Y: outcome (e.g., income)
        y = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="Y (outcome)")

        dag.add_node(z)
        dag.add_node(x)
        dag.add_node(y)

        # Z → X (confounder affects treatment)
        dag.add_edge(create_causal_edge(z.id, x.id))
        # Z → Y (confounder affects outcome)
        dag.add_edge(create_causal_edge(z.id, y.id))
        # X → Y (treatment affects outcome)
        dag.add_edge(create_causal_edge(x.id, y.id))

        return dag, z, x, y

    def test_mutilated_graph_removes_incoming_edges(self):
        """Mutilated graph should remove incoming edges to intervention nodes."""
        dag, z, x, y = self.create_confounding_dag()

        # Intervene on X: create G_X̄
        mutilated = dag.get_mutilated_graph({x.id})

        # X should have no parents in mutilated graph
        x_parents = mutilated.get_parent_ids(x.id)
        assert len(x_parents) == 0

        # But Z → Y should still exist
        assert mutilated.has_edge(z.id, y.id)

        # And X → Y should still exist
        assert mutilated.has_edge(x.id, y.id)

    def test_edge_deleted_graph_removes_outgoing_edges(self):
        """Edge-deleted graph should remove outgoing edges from nodes."""
        dag, z, x, y = self.create_confounding_dag()

        # Remove outgoing edges from Z
        edge_deleted = dag.get_edge_deleted_graph({z.id})

        # Z should have no children
        z_children = edge_deleted.get_children_ids(z.id)
        assert len(z_children) == 0

        # But X → Y should still exist
        assert edge_deleted.has_edge(x.id, y.id)

    def test_double_mutilated_graph(self):
        """Double mutilated graph for Rule 2 of do-calculus."""
        dag, z, x, y = self.create_confounding_dag()

        # Create G_X̄,Z̲: remove incoming to X, outgoing from Z
        double_mut = dag.get_double_mutilated_graph({x.id}, {z.id})

        # X should have no parents
        assert len(double_mut.get_parent_ids(x.id)) == 0

        # Z should have no children
        assert len(double_mut.get_children_ids(z.id)) == 0

        # X → Y should still exist
        assert double_mut.has_edge(x.id, y.id)

    def test_rule_3_graph_removes_non_ancestor_z(self):
        """Rule 3 graph should remove Z nodes that aren't ancestors of W."""
        dag = ConversationDAG()

        # Create: W ← X → Y, Z (isolated)
        w = ConversationNode(node_type=NodeType.USER_MESSAGE, content="W")
        x = ConversationNode(node_type=NodeType.USER_MESSAGE, content="X")
        y = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="Y")
        z = ConversationNode(node_type=NodeType.USER_MESSAGE, content="Z (isolated)")

        dag.add_node(w)
        dag.add_node(x)
        dag.add_node(y)
        dag.add_node(z)

        dag.add_edge(create_causal_edge(x.id, w.id))
        dag.add_edge(create_causal_edge(x.id, y.id))

        # Create Rule 3 graph with intervention on X, Z nodes, conditioning on W
        rule3_graph = dag.get_rule_3_graph({x.id}, {z.id}, {w.id})

        # Z should be removed (not an ancestor of W)
        assert not rule3_graph.has_node(z.id)

        # Other nodes should remain
        assert rule3_graph.has_node(w.id)
        assert rule3_graph.has_node(x.id)
        assert rule3_graph.has_node(y.id)


class TestDAGSerialization:
    """Tests for DAG serialization."""

    def test_to_dict(self):
        """DAG should serialize to dictionary."""
        dag = ConversationDAG()
        node = create_user_message("Test")
        dag.add_node(node)

        data = dag.to_dict()

        assert "session_id" in data
        assert "nodes" in data
        assert len(data["nodes"]) == 1

    def test_from_dict(self):
        """DAG should deserialize from dictionary."""
        original = ConversationDAG()
        node1 = create_user_message("Question")
        node2 = ConversationNode(
            node_type=NodeType.ASSISTANT_RESPONSE,
            content="Answer",
        )
        original.add_node(node1)
        original.add_node(node2)
        original.add_edge(create_causal_edge(node1.id, node2.id))

        data = original.to_dict()
        restored = ConversationDAG.from_dict(data)

        assert restored.node_count == 2
        assert restored.edge_count == 1
        assert restored.session_id == original.session_id

    def test_iteration(self):
        """DAG should be iterable in topological order."""
        dag = ConversationDAG()
        a = create_user_message("A")
        b = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="B")
        dag.add_node(a)
        dag.add_node(b)
        dag.add_edge(create_causal_edge(a.id, b.id))

        nodes = list(dag)

        assert len(nodes) == 2
        assert nodes[0].id == a.id  # A comes before B

    def test_contains(self):
        """DAG should support 'in' operator."""
        dag = ConversationDAG()
        node = create_user_message("Test")
        dag.add_node(node)

        assert node.id in dag
        assert uuid4() not in dag

    def test_len(self):
        """DAG should support len()."""
        dag = ConversationDAG()
        dag.add_node(create_user_message("A"))
        dag.add_node(create_user_message("B"))

        assert len(dag) == 2
