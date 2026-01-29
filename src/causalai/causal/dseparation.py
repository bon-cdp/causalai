"""
D-separation algorithms for conditional independence testing.

D-separation (directed separation) is a criterion for determining
conditional independence relationships in directed acyclic graphs.
It is fundamental to Pearl's causal inference framework.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING
from uuid import UUID

import networkx as nx
from networkx.algorithms.d_separation import is_d_separator, find_minimal_d_separator

if TYPE_CHECKING:
    from causalai.core.dag import ConversationDAG


class DSeparationAnalyzer:
    """Implements d-separation algorithms for conditional independence testing.

    D-separation is the key graphical criterion for determining when
    variables are conditionally independent given a set of observed variables.

    Two nodes X and Y are d-separated by a set Z if all paths between X and Y
    are "blocked" by Z. A path is blocked if it contains:
    1. A chain (A→B→C) where B is in Z
    2. A fork (A←B→C) where B is in Z
    3. A collider (A→B←C) where B is NOT in Z and no descendant of B is in Z

    This implementation uses NetworkX's efficient d-separation algorithms.

    Example:
        >>> analyzer = DSeparationAnalyzer(dag)
        >>> # Check if X and Y are independent given Z
        >>> is_indep = analyzer.is_d_separated({x_id}, {y_id}, {z_id})
    """

    def __init__(self, dag: ConversationDAG):
        """Initialize the analyzer with a DAG.

        Args:
            dag: The ConversationDAG to analyze
        """
        self.dag = dag
        self._nx_graph = dag.to_networkx()

    def refresh(self) -> None:
        """Refresh the internal NetworkX graph from the DAG.

        Call this if the DAG has been modified since the analyzer was created.
        """
        self._nx_graph = self.dag.to_networkx()

    def is_d_separated(
        self,
        x: set[UUID],
        y: set[UUID],
        z: set[UUID] | None = None,
    ) -> bool:
        """Test if X and Y are d-separated given Z.

        Two sets of nodes X and Y are d-separated by Z if every path
        between any node in X and any node in Y is blocked by Z.

        Args:
            x: Source node set
            y: Target node set
            z: Conditioning set (optional, defaults to empty set)

        Returns:
            True if X and Y are d-separated given Z

        Example:
            >>> # In a chain A → B → C, A and C are d-separated by B
            >>> analyzer.is_d_separated({a_id}, {c_id}, {b_id})
            True
            >>> # But not without conditioning on B
            >>> analyzer.is_d_separated({a_id}, {c_id})
            False
        """
        z = z or set()

        # Validate inputs
        if x & y:
            raise ValueError("X and Y must be disjoint")
        if x & z or y & z:
            raise ValueError("Z must be disjoint from both X and Y")

        # Use NetworkX's built-in d-separation (efficient O(m) algorithm)
        try:
            return is_d_separator(self._nx_graph, x, y, z)
        except nx.NetworkXError as e:
            # Node not in graph or other error
            raise ValueError(f"D-separation check failed: {e}") from e

    def is_d_connected(
        self,
        x: set[UUID],
        y: set[UUID],
        z: set[UUID] | None = None,
    ) -> bool:
        """Test if X and Y are d-connected given Z.

        D-connection is the opposite of d-separation. X and Y are
        d-connected if there exists at least one unblocked path.

        Args:
            x: Source node set
            y: Target node set
            z: Conditioning set (optional)

        Returns:
            True if X and Y are d-connected given Z
        """
        return not self.is_d_separated(x, y, z)

    def find_find_minimal_d_separator(
        self,
        x: set[UUID],
        y: set[UUID],
    ) -> set[UUID] | None:
        """Find a minimal d-separator between X and Y if one exists.

        A minimal d-separator is a set Z such that:
        1. X and Y are d-separated given Z
        2. No proper subset of Z also d-separates X and Y

        Args:
            x: Source node set
            y: Target node set

        Returns:
            A minimal separating set, or None if X and Y cannot be separated
            (i.e., they are always d-connected)
        """
        try:
            # NetworkX provides an efficient algorithm for this
            return set(find_minimal_d_separator(self._nx_graph, x, y))
        except nx.NetworkXError:
            return None

    def get_all_d_separators(
        self,
        x: set[UUID],
        y: set[UUID],
        max_size: int | None = None,
    ) -> list[set[UUID]]:
        """Find all d-separators between X and Y up to a maximum size.

        Args:
            x: Source node set
            y: Target node set
            max_size: Maximum size of separator sets to consider

        Returns:
            List of all d-separator sets found
        """
        separators: list[set[UUID]] = []
        candidates = self._get_candidate_separator_nodes(x, y)

        max_size = max_size or len(candidates)

        # Check empty set first
        if self.is_d_separated(x, y, set()):
            separators.append(set())
            return separators  # If empty set works, it's the only minimal one

        # Enumerate subsets of candidates
        for size in range(1, min(max_size, len(candidates)) + 1):
            for subset in combinations(candidates, size):
                z = set(subset)
                if self.is_d_separated(x, y, z):
                    separators.append(z)

        return separators

    def _get_candidate_separator_nodes(
        self,
        x: set[UUID],
        y: set[UUID],
    ) -> set[UUID]:
        """Get nodes that could potentially be in a d-separator.

        Candidates are nodes that are:
        - Not in X or Y
        - Ancestors of X or Y (or the nodes themselves)

        This reduces the search space for finding separators.
        """
        candidates: set[UUID] = set()

        for node in x | y:
            if node in self._nx_graph:
                candidates.update(nx.ancestors(self._nx_graph, node))

        # Remove X and Y themselves
        candidates -= x
        candidates -= y

        return candidates

    def check_conditional_independence(
        self,
        x: UUID,
        y: UUID,
        given: set[UUID] | None = None,
    ) -> dict[str, any]:
        """Check conditional independence and return detailed results.

        Args:
            x: First node
            y: Second node
            given: Conditioning set

        Returns:
            Dictionary with:
            - is_independent: Whether X ⊥ Y | given
            - given: The conditioning set used
            - explanation: Human-readable explanation
        """
        given = given or set()
        is_sep = self.is_d_separated({x}, {y}, given)

        if is_sep:
            explanation = (
                f"X and Y are conditionally independent given the conditioning set. "
                f"All paths between them are blocked."
            )
        else:
            explanation = (
                f"X and Y are NOT conditionally independent given the conditioning set. "
                f"There exists at least one unblocked (d-connected) path."
            )

        return {
            "is_independent": is_sep,
            "given": given,
            "explanation": explanation,
        }

    def find_backdoor_paths(
        self,
        treatment: UUID,
        outcome: UUID,
    ) -> list[list[UUID]]:
        """Find all backdoor paths between treatment and outcome.

        A backdoor path is a path from treatment to outcome that starts
        with an arrow INTO the treatment (i.e., ← from treatment).

        These are the confounding paths that need to be blocked for
        causal identification.

        Args:
            treatment: The treatment/intervention node
            outcome: The outcome node

        Returns:
            List of backdoor paths (each path is a list of node IDs)
        """
        backdoor_paths: list[list[UUID]] = []

        # Get parents of treatment (backdoor paths go through these)
        parents = list(self._nx_graph.predecessors(treatment))

        # Find all paths from each parent to outcome
        for parent in parents:
            try:
                # Use undirected graph for path finding (backdoor paths can be any direction)
                undirected = self._nx_graph.to_undirected()
                for path in nx.all_simple_paths(undirected, parent, outcome):
                    # Reconstruct full path including treatment
                    full_path = [treatment] + list(path)
                    backdoor_paths.append(full_path)
            except nx.NetworkXNoPath:
                continue

        return backdoor_paths

    def is_valid_adjustment_set(
        self,
        treatment: UUID,
        outcome: UUID,
        adjustment_set: set[UUID],
    ) -> bool:
        """Check if a set is a valid adjustment set (backdoor criterion).

        An adjustment set Z is valid if:
        1. Z blocks all backdoor paths from treatment to outcome
        2. Z does not include any descendant of treatment

        Args:
            treatment: The treatment/intervention node
            outcome: The outcome node
            adjustment_set: The proposed adjustment set

        Returns:
            True if the adjustment set satisfies the backdoor criterion
        """
        # Check condition 2: Z should not contain descendants of treatment
        descendants = nx.descendants(self._nx_graph, treatment)
        if adjustment_set & descendants:
            return False

        # Check condition 1: Z blocks all backdoor paths
        # This is equivalent to: treatment ⊥ outcome | Z in the mutilated graph
        # where we remove outgoing edges from treatment

        # Create mutilated graph (remove outgoing edges from treatment)
        mutilated = self._nx_graph.copy()
        children = list(mutilated.successors(treatment))
        for child in children:
            mutilated.remove_edge(treatment, child)

        # Check d-separation in mutilated graph
        try:
            return is_d_separator(mutilated, {treatment}, {outcome}, adjustment_set)
        except nx.NetworkXError:
            return False

    def find_valid_adjustment_sets(
        self,
        treatment: UUID,
        outcome: UUID,
        max_size: int | None = None,
    ) -> list[set[UUID]]:
        """Find all valid adjustment sets up to a maximum size.

        Args:
            treatment: The treatment/intervention node
            outcome: The outcome node
            max_size: Maximum size of adjustment sets to consider

        Returns:
            List of valid adjustment sets
        """
        valid_sets: list[set[UUID]] = []

        # Candidates are non-descendants of treatment
        all_nodes = set(self._nx_graph.nodes())
        descendants = nx.descendants(self._nx_graph, treatment)
        candidates = all_nodes - descendants - {treatment, outcome}

        max_size = max_size or len(candidates)

        # Check empty set
        if self.is_valid_adjustment_set(treatment, outcome, set()):
            valid_sets.append(set())

        # Enumerate subsets
        for size in range(1, min(max_size, len(candidates)) + 1):
            for subset in combinations(candidates, size):
                z = set(subset)
                if self.is_valid_adjustment_set(treatment, outcome, z):
                    valid_sets.append(z)

        return valid_sets
