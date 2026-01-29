"""
Conversation DAG (Directed Acyclic Graph) implementation.

This module provides the ConversationDAG class which wraps NetworkX
and provides causal graph operations including graph surgery for
do-calculus.
"""

from __future__ import annotations

from typing import Any, Iterator
from uuid import UUID, uuid4

import networkx as nx

from causalai.core.edges import ConversationEdge, EdgeType, create_causal_edge
from causalai.core.nodes import ConversationNode, NodeType


class ConversationDAG:
    """A Directed Acyclic Graph representing a conversation with causal structure.

    This class wraps NetworkX DiGraph and provides:
    - Conversation-specific node/edge operations
    - Causal graph manipulations (graph surgery)
    - D-separation queries
    - Navigation (fork, branch, merge)

    The DAG represents conversation events as nodes and causal relationships
    as edges. It supports the do-calculus operations through graph surgery
    methods that create modified versions of the graph.

    Attributes:
        session_id: Unique identifier for this conversation session

    Example:
        >>> dag = ConversationDAG()
        >>> user_node = create_user_message("Hello")
        >>> dag.add_node(user_node)
        >>> response_node = create_assistant_response("Hi there!")
        >>> dag.add_node(response_node)
        >>> dag.add_edge(create_causal_edge(user_node.id, response_node.id))
    """

    def __init__(self, session_id: UUID | None = None):
        """Initialize a new ConversationDAG.

        Args:
            session_id: Optional session ID. If not provided, one will be generated.
        """
        self._graph: nx.DiGraph = nx.DiGraph()
        self.session_id = session_id or uuid4()
        self._current_head: UUID | None = None  # Active position in graph
        self._branches: dict[str, UUID] = {}  # Named branches

    # --- Node Operations ---

    def add_node(self, node: ConversationNode) -> UUID:
        """Add a node to the DAG.

        Args:
            node: The ConversationNode to add

        Returns:
            The UUID of the added node
        """
        self._graph.add_node(
            node.id,
            data=node,
            node_type=node.node_type.value,
            is_intervention=node.is_intervention,
        )
        # Update current head to most recently added node
        self._current_head = node.id
        return node.id

    def get_node(self, node_id: UUID) -> ConversationNode | None:
        """Retrieve a node by ID.

        Args:
            node_id: The UUID of the node to retrieve

        Returns:
            The ConversationNode if found, None otherwise
        """
        if node_id in self._graph:
            return self._graph.nodes[node_id]["data"]
        return None

    def has_node(self, node_id: UUID) -> bool:
        """Check if a node exists in the DAG."""
        return node_id in self._graph

    def remove_node(self, node_id: UUID) -> bool:
        """Remove a node from the DAG.

        Args:
            node_id: The UUID of the node to remove

        Returns:
            True if the node was removed, False if it didn't exist
        """
        if node_id in self._graph:
            self._graph.remove_node(node_id)
            return True
        return False

    def get_nodes_by_type(self, node_type: NodeType) -> list[ConversationNode]:
        """Get all nodes of a specific type.

        Args:
            node_type: The type of nodes to retrieve

        Returns:
            List of ConversationNodes matching the type
        """
        return [
            self._graph.nodes[n]["data"]
            for n in self._graph.nodes
            if self._graph.nodes[n]["node_type"] == node_type.value
        ]

    def get_all_nodes(self) -> list[ConversationNode]:
        """Get all nodes in the DAG."""
        return [self._graph.nodes[n]["data"] for n in self._graph.nodes]

    def get_intervention_nodes(self) -> list[ConversationNode]:
        """Get all nodes that are interventions."""
        return [
            self._graph.nodes[n]["data"]
            for n in self._graph.nodes
            if self._graph.nodes[n].get("is_intervention", False)
        ]

    @property
    def node_count(self) -> int:
        """Get the number of nodes in the DAG."""
        return self._graph.number_of_nodes()

    # --- Edge Operations ---

    def add_edge(self, edge: ConversationEdge) -> None:
        """Add an edge between nodes.

        Args:
            edge: The ConversationEdge to add

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if edge.source_id not in self._graph:
            raise ValueError(f"Source node {edge.source_id} not in graph")
        if edge.target_id not in self._graph:
            raise ValueError(f"Target node {edge.target_id} not in graph")

        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            data=edge,
            edge_type=edge.edge_type.value,
        )

    def get_edge(self, source_id: UUID, target_id: UUID) -> ConversationEdge | None:
        """Get an edge between two nodes."""
        if self._graph.has_edge(source_id, target_id):
            return self._graph.edges[source_id, target_id]["data"]
        return None

    def has_edge(self, source_id: UUID, target_id: UUID) -> bool:
        """Check if an edge exists between two nodes."""
        return self._graph.has_edge(source_id, target_id)

    def remove_edge(self, source_id: UUID, target_id: UUID) -> bool:
        """Remove an edge between two nodes.

        Returns:
            True if the edge was removed, False if it didn't exist
        """
        if self._graph.has_edge(source_id, target_id):
            self._graph.remove_edge(source_id, target_id)
            return True
        return False

    def get_parents(self, node_id: UUID) -> list[ConversationNode]:
        """Get all parent nodes (predecessors).

        Args:
            node_id: The node to get parents for

        Returns:
            List of parent ConversationNodes
        """
        return [self._graph.nodes[p]["data"] for p in self._graph.predecessors(node_id)]

    def get_children(self, node_id: UUID) -> list[ConversationNode]:
        """Get all child nodes (successors).

        Args:
            node_id: The node to get children for

        Returns:
            List of child ConversationNodes
        """
        return [self._graph.nodes[c]["data"] for c in self._graph.successors(node_id)]

    def get_parent_ids(self, node_id: UUID) -> set[UUID]:
        """Get UUIDs of all parent nodes."""
        return set(self._graph.predecessors(node_id))

    def get_children_ids(self, node_id: UUID) -> set[UUID]:
        """Get UUIDs of all child nodes."""
        return set(self._graph.successors(node_id))

    @property
    def edge_count(self) -> int:
        """Get the number of edges in the DAG."""
        return self._graph.number_of_edges()

    # --- Causal Graph Properties ---

    def get_ancestors(self, node_id: UUID) -> set[UUID]:
        """Get all ancestors of a node (recursive parents)."""
        return nx.ancestors(self._graph, node_id)

    def get_descendants(self, node_id: UUID) -> set[UUID]:
        """Get all descendants of a node (recursive children)."""
        return nx.descendants(self._graph, node_id)

    def is_valid_dag(self) -> bool:
        """Check if graph is a valid DAG (no cycles)."""
        return nx.is_directed_acyclic_graph(self._graph)

    def topological_order(self) -> list[UUID]:
        """Get nodes in topological order.

        Returns:
            List of node UUIDs in topological order

        Raises:
            NetworkXUnfeasible: If the graph has cycles
        """
        return list(nx.topological_sort(self._graph))

    def get_roots(self) -> list[UUID]:
        """Get all root nodes (nodes with no incoming edges)."""
        return [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]

    def get_leaves(self) -> list[UUID]:
        """Get all leaf nodes (nodes with no outgoing edges)."""
        return [n for n in self._graph.nodes if self._graph.out_degree(n) == 0]

    # --- Graph Surgery for Do-Calculus ---

    def get_mutilated_graph(self, intervention_nodes: set[UUID]) -> ConversationDAG:
        """Create G_X̄ - graph with incoming edges to intervention nodes removed.

        Used for Rule 1 and Rule 2 of do-calculus.
        This simulates the effect of do(X) by removing all arrows into X.

        Args:
            intervention_nodes: Set of node IDs that are being intervened upon

        Returns:
            A new ConversationDAG with incoming edges to intervention nodes removed
        """
        mutilated = ConversationDAG(session_id=self.session_id)
        mutilated._graph = self._graph.copy()

        # Remove all incoming edges to intervention nodes
        for node_id in intervention_nodes:
            if node_id in mutilated._graph:
                parents = list(mutilated._graph.predecessors(node_id))
                for parent in parents:
                    mutilated._graph.remove_edge(parent, node_id)

        return mutilated

    def get_edge_deleted_graph(self, observation_nodes: set[UUID]) -> ConversationDAG:
        """Create G_Z̲ - graph with outgoing edges from observation nodes removed.

        Used as part of Rule 2 of do-calculus.
        This removes the causal effect of Z on its descendants.

        Args:
            observation_nodes: Set of node IDs whose outgoing edges to remove

        Returns:
            A new ConversationDAG with outgoing edges from observation nodes removed
        """
        edge_deleted = ConversationDAG(session_id=self.session_id)
        edge_deleted._graph = self._graph.copy()

        # Remove all outgoing edges from observation nodes
        for node_id in observation_nodes:
            if node_id in edge_deleted._graph:
                children = list(edge_deleted._graph.successors(node_id))
                for child in children:
                    edge_deleted._graph.remove_edge(node_id, child)

        return edge_deleted

    def get_double_mutilated_graph(
        self,
        intervention_nodes: set[UUID],
        observation_nodes: set[UUID],
    ) -> ConversationDAG:
        """Create G_X̄,Z̲ - graph with incoming edges to X removed and outgoing from Z removed.

        Used for Rule 2 of do-calculus:
        P(y|do(x),do(z),w) = P(y|do(x),z,w) if Y⊥Z|X,W in G_X̄,Z̲

        Args:
            intervention_nodes: Nodes being intervened upon (X)
            observation_nodes: Nodes for observation exchange (Z)

        Returns:
            A new ConversationDAG with both modifications applied
        """
        mutilated = self.get_mutilated_graph(intervention_nodes)

        # Remove all outgoing edges from observation nodes
        for node_id in observation_nodes:
            if node_id in mutilated._graph:
                children = list(mutilated._graph.successors(node_id))
                for child in children:
                    mutilated._graph.remove_edge(node_id, child)

        return mutilated

    def get_rule_3_graph(
        self,
        intervention_nodes: set[UUID],
        z_nodes: set[UUID],
        w_nodes: set[UUID],
    ) -> ConversationDAG:
        """Create G_X̄,Z̄(W) - graph for Rule 3 of do-calculus.

        Graph with incoming edges to X removed and Z-nodes removed
        that are not ancestors of any W-node.

        Used for Rule 3:
        P(y|do(x),do(z),w) = P(y|do(x),w) if Y⊥Z|X,W in G_X̄,Z̄(W)

        Args:
            intervention_nodes: Nodes being intervened upon (X)
            z_nodes: Nodes whose intervention to potentially remove (Z)
            w_nodes: Conditioning nodes (W)

        Returns:
            A new ConversationDAG with Rule 3 modifications applied
        """
        # Start with G_X̄
        modified = self.get_mutilated_graph(intervention_nodes)

        # Find Z nodes that are ancestors of W
        w_ancestors: set[UUID] = set()
        for w_node in w_nodes:
            if w_node in modified._graph:
                w_ancestors.update(nx.ancestors(modified._graph, w_node))

        # Remove Z nodes that are NOT ancestors of W
        z_to_remove = z_nodes - w_ancestors
        for node_id in z_to_remove:
            if node_id in modified._graph:
                modified._graph.remove_node(node_id)

        return modified

    # --- Navigation ---

    def get_path_to_node(self, node_id: UUID) -> list[ConversationNode]:
        """Get the conversation path from root to specified node.

        Args:
            node_id: The target node

        Returns:
            List of ConversationNodes from root to target, or empty list if no path
        """
        roots = self.get_roots()
        if not roots:
            return []

        # Find shortest path from any root
        for root in roots:
            try:
                path = nx.shortest_path(self._graph, root, node_id)
                return [self._graph.nodes[n]["data"] for n in path]
            except nx.NetworkXNoPath:
                continue

        return []

    def get_current_head(self) -> ConversationNode | None:
        """Get the current head node (most recent position)."""
        if self._current_head:
            return self.get_node(self._current_head)
        return None

    def set_current_head(self, node_id: UUID) -> None:
        """Set the current head to a specific node.

        Args:
            node_id: The node to set as current head

        Raises:
            ValueError: If the node doesn't exist
        """
        if node_id not in self._graph:
            raise ValueError(f"Node {node_id} not in graph")
        self._current_head = node_id

    def create_branch(self, name: str, node_id: UUID | None = None) -> None:
        """Create a named branch at the specified node.

        Args:
            name: Name for the branch
            node_id: Node to branch from. Defaults to current head.

        Raises:
            ValueError: If the node doesn't exist
        """
        branch_point = node_id or self._current_head
        if branch_point is None:
            raise ValueError("No node specified and no current head")
        if branch_point not in self._graph:
            raise ValueError(f"Node {branch_point} not in graph")
        self._branches[name] = branch_point

    def get_branch(self, name: str) -> UUID | None:
        """Get the node ID for a named branch."""
        return self._branches.get(name)

    def list_branches(self) -> dict[str, UUID]:
        """Get all named branches."""
        return self._branches.copy()

    # --- Serialization ---

    def to_networkx(self) -> nx.DiGraph:
        """Get a copy of the underlying NetworkX graph."""
        return self._graph.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the DAG to a dictionary."""
        return {
            "session_id": str(self.session_id),
            "nodes": [self._graph.nodes[n]["data"].model_dump() for n in self._graph.nodes],
            "edges": [
                self._graph.edges[e]["data"].model_dump()
                for e in self._graph.edges
                if "data" in self._graph.edges[e]
            ],
            "current_head": str(self._current_head) if self._current_head else None,
            "branches": {k: str(v) for k, v in self._branches.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationDAG:
        """Create a ConversationDAG from a dictionary.

        Args:
            data: Dictionary representation of the DAG

        Returns:
            A new ConversationDAG instance
        """
        from uuid import UUID as UUIDClass

        dag = cls(session_id=UUIDClass(data["session_id"]))

        # Add nodes
        for node_data in data["nodes"]:
            node = ConversationNode.model_validate(node_data)
            dag.add_node(node)

        # Add edges
        for edge_data in data["edges"]:
            edge = ConversationEdge.model_validate(edge_data)
            dag.add_edge(edge)

        # Restore state
        if data.get("current_head"):
            dag._current_head = UUIDClass(data["current_head"])
        dag._branches = {k: UUIDClass(v) for k, v in data.get("branches", {}).items()}

        return dag

    def __len__(self) -> int:
        """Return the number of nodes in the DAG."""
        return self.node_count

    def __contains__(self, node_id: UUID) -> bool:
        """Check if a node is in the DAG."""
        return self.has_node(node_id)

    def __iter__(self) -> Iterator[ConversationNode]:
        """Iterate over nodes in topological order."""
        for node_id in self.topological_order():
            yield self._graph.nodes[node_id]["data"]

    def __repr__(self) -> str:
        """String representation of the DAG."""
        return f"ConversationDAG(session_id={self.session_id}, nodes={self.node_count}, edges={self.edge_count})"
