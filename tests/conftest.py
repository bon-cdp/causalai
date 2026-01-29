"""Pytest configuration and fixtures for CausalAI tests."""

import pytest

from causalai.core.dag import ConversationDAG
from causalai.core.nodes import ConversationNode, NodeType, create_user_message
from causalai.core.edges import create_causal_edge


@pytest.fixture
def empty_dag():
    """Create an empty ConversationDAG."""
    return ConversationDAG()


@pytest.fixture
def simple_chain_dag():
    """Create a simple chain DAG: A → B → C.

    This is useful for testing basic d-separation in chains.
    """
    dag = ConversationDAG()

    a = ConversationNode(node_type=NodeType.USER_MESSAGE, content="A")
    b = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="B")
    c = ConversationNode(node_type=NodeType.USER_MESSAGE, content="C")

    dag.add_node(a)
    dag.add_node(b)
    dag.add_node(c)

    dag.add_edge(create_causal_edge(a.id, b.id))
    dag.add_edge(create_causal_edge(b.id, c.id))

    return dag, {"a": a, "b": b, "c": c}


@pytest.fixture
def fork_dag():
    """Create a fork DAG: A ← B → C.

    This is useful for testing d-separation with common causes.
    """
    dag = ConversationDAG()

    a = ConversationNode(node_type=NodeType.USER_MESSAGE, content="A")
    b = ConversationNode(node_type=NodeType.SYSTEM_PROMPT, content="B (common cause)")
    c = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="C")

    dag.add_node(a)
    dag.add_node(b)
    dag.add_node(c)

    dag.add_edge(create_causal_edge(b.id, a.id))
    dag.add_edge(create_causal_edge(b.id, c.id))

    return dag, {"a": a, "b": b, "c": c}


@pytest.fixture
def collider_dag():
    """Create a collider DAG: A → B ← C.

    This is useful for testing the 'explaining away' phenomenon.
    """
    dag = ConversationDAG()

    a = ConversationNode(node_type=NodeType.USER_MESSAGE, content="A")
    b = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="B (collider)")
    c = ConversationNode(node_type=NodeType.USER_MESSAGE, content="C")

    dag.add_node(a)
    dag.add_node(b)
    dag.add_node(c)

    dag.add_edge(create_causal_edge(a.id, b.id))
    dag.add_edge(create_causal_edge(c.id, b.id))

    return dag, {"a": a, "b": b, "c": c}


@pytest.fixture
def confounding_dag():
    """Create a confounding DAG: Z → X, Z → Y, X → Y.

    This is the classic confounding scenario where Z confounds
    the treatment X and outcome Y relationship.
    """
    dag = ConversationDAG()

    z = ConversationNode(node_type=NodeType.SYSTEM_PROMPT, content="Z (confounder)")
    x = ConversationNode(node_type=NodeType.USER_MESSAGE, content="X (treatment)")
    y = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="Y (outcome)")

    dag.add_node(z)
    dag.add_node(x)
    dag.add_node(y)

    dag.add_edge(create_causal_edge(z.id, x.id))
    dag.add_edge(create_causal_edge(z.id, y.id))
    dag.add_edge(create_causal_edge(x.id, y.id))

    return dag, {"z": z, "x": x, "y": y}


@pytest.fixture
def frontdoor_dag():
    """Create a front-door DAG: Z → X → M → Y, Z → Y.

    This is useful for testing the front-door criterion.
    """
    dag = ConversationDAG()

    z = ConversationNode(node_type=NodeType.SYSTEM_PROMPT, content="Z (confounder)")
    x = ConversationNode(node_type=NodeType.USER_MESSAGE, content="X (treatment)")
    m = ConversationNode(node_type=NodeType.TOOL_CALL, content="M (mediator)")
    y = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="Y (outcome)")

    dag.add_node(z)
    dag.add_node(x)
    dag.add_node(m)
    dag.add_node(y)

    dag.add_edge(create_causal_edge(z.id, x.id))
    dag.add_edge(create_causal_edge(z.id, y.id))
    dag.add_edge(create_causal_edge(x.id, m.id))
    dag.add_edge(create_causal_edge(m.id, y.id))

    return dag, {"z": z, "x": x, "m": m, "y": y}


@pytest.fixture
def conversation_dag():
    """Create a typical conversation DAG.

    System → User1 → Assistant1 → User2 → Assistant2
    """
    dag = ConversationDAG()

    system = ConversationNode(
        node_type=NodeType.SYSTEM_PROMPT,
        content="You are a helpful assistant.",
    )
    user1 = ConversationNode(
        node_type=NodeType.USER_MESSAGE,
        content="What is causal inference?",
    )
    assistant1 = ConversationNode(
        node_type=NodeType.ASSISTANT_RESPONSE,
        content="Causal inference is the process of determining cause-effect relationships.",
    )
    user2 = ConversationNode(
        node_type=NodeType.USER_MESSAGE,
        content="Can you give an example?",
    )
    assistant2 = ConversationNode(
        node_type=NodeType.ASSISTANT_RESPONSE,
        content="Sure! A classic example is determining if smoking causes cancer.",
    )

    dag.add_node(system)
    dag.add_node(user1)
    dag.add_node(assistant1)
    dag.add_node(user2)
    dag.add_node(assistant2)

    dag.add_edge(create_causal_edge(system.id, user1.id))
    dag.add_edge(create_causal_edge(user1.id, assistant1.id))
    dag.add_edge(create_causal_edge(assistant1.id, user2.id))
    dag.add_edge(create_causal_edge(user2.id, assistant2.id))

    return dag, {
        "system": system,
        "user1": user1,
        "assistant1": assistant1,
        "user2": user2,
        "assistant2": assistant2,
    }
