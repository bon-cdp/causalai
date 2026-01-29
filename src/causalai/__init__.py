"""
CausalAI: Pearl's Causal Inference Framework for AI Conversations

A meta-control system that models AI conversations as causal DAGs,
applying do-calculus rules to enable interventions, forks, and
alignment testing at each node.
"""

__version__ = "0.1.0"

from causalai.core.nodes import ConversationNode, NodeType, NodeStatus
from causalai.core.edges import ConversationEdge, EdgeType
from causalai.core.dag import ConversationDAG

__all__ = [
    "ConversationNode",
    "ConversationEdge",
    "ConversationDAG",
    "NodeType",
    "NodeStatus",
    "EdgeType",
]
