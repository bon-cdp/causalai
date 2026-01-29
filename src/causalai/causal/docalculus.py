"""
Pearl's Do-Calculus implementation.

The do-calculus consists of three rules that allow transforming
interventional distributions P(Y|do(X),Z) into observational
distributions P(Y|X,Z) under certain graphical conditions.

These rules are sound and complete for causal inference.

Reference: Pearl, J. (2009). Causality (2nd ed.). Cambridge University Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from causalai.causal.dseparation import DSeparationAnalyzer

if TYPE_CHECKING:
    from causalai.core.dag import ConversationDAG


class DoCalculusRule(Enum):
    """The three rules of do-calculus."""

    RULE_1 = "insertion_deletion_observations"
    RULE_2 = "action_observation_exchange"
    RULE_3 = "insertion_deletion_interventions"


@dataclass
class RuleApplicationResult:
    """Result of attempting to apply a do-calculus rule.

    Attributes:
        rule: Which rule was attempted
        applicable: Whether the rule can be applied
        original_expression: The original probabilistic expression
        transformed_expression: The transformed expression (if applicable)
        d_separation_holds: Whether the required d-separation condition holds
        modified_graph_type: Description of the graph modification used
        explanation: Human-readable explanation of the result
    """

    rule: DoCalculusRule
    applicable: bool
    original_expression: str
    transformed_expression: str | None
    d_separation_holds: bool
    modified_graph_type: str
    explanation: str

    def __str__(self) -> str:
        """String representation of the result."""
        if self.applicable:
            return (
                f"{self.rule.value}: APPLICABLE\n"
                f"  {self.original_expression} = {self.transformed_expression}\n"
                f"  {self.explanation}"
            )
        return (
            f"{self.rule.value}: NOT APPLICABLE\n"
            f"  {self.original_expression}\n"
            f"  {self.explanation}"
        )


class DoCalculusEngine:
    """Implements the three rules of do-calculus for causal inference.

    Do-calculus provides a complete set of inference rules for determining
    when causal effects can be identified from observational data and the
    causal graph.

    The three rules are:
    - Rule 1: Insertion/deletion of observations
    - Rule 2: Action/observation exchange
    - Rule 3: Insertion/deletion of interventions

    Example:
        >>> engine = DoCalculusEngine(dag)
        >>> # Check if we can remove observation Z
        >>> result = engine.apply_rule_1(y={y_id}, x={x_id}, z={z_id})
        >>> if result.applicable:
        ...     print(f"Can simplify: {result.transformed_expression}")

    Reference:
        Pearl, J. (2009). Causality (2nd ed.). Chapter 3.
    """

    def __init__(self, dag: ConversationDAG):
        """Initialize the engine with a DAG.

        Args:
            dag: The ConversationDAG to analyze
        """
        self.dag = dag
        self.dsep = DSeparationAnalyzer(dag)

    def apply_rule_1(
        self,
        y: set[UUID],
        x: set[UUID],
        z: set[UUID],
        w: set[UUID] | None = None,
    ) -> RuleApplicationResult:
        """Rule 1: Insertion/deletion of observations.

        P(y | do(x), z, w) = P(y | do(x), w)

        Condition: Y ⊥ Z | X, W in G_X̄
        (graph with incoming edges to X removed)

        This rule allows adding/removing observations Z if Y and Z are
        d-separated in the mutilated graph where we intervene on X.

        Args:
            y: Outcome variables
            x: Intervention variables (do(x))
            z: Variables to insert/delete from observations
            w: Additional conditioning variables (optional)

        Returns:
            RuleApplicationResult indicating if the rule applies
        """
        w = w or set()

        # Create mutilated graph G_X̄ (remove incoming edges to X)
        mutilated_dag = self.dag.get_mutilated_graph(x)
        mutilated_dsep = DSeparationAnalyzer(mutilated_dag)

        # Check d-separation: Y ⊥ Z | X ∪ W in G_X̄
        conditioning_set = x | w
        is_d_separated = mutilated_dsep.is_d_separated(y, z, conditioning_set)

        # Build expression strings
        x_str = self._format_vars(x, "X")
        y_str = self._format_vars(y, "Y")
        z_str = self._format_vars(z, "Z")
        w_str = self._format_vars(w, "W") if w else ""

        if w:
            original = f"P({y_str} | do({x_str}), {z_str}, {w_str})"
            transformed = f"P({y_str} | do({x_str}), {w_str})" if is_d_separated else None
        else:
            original = f"P({y_str} | do({x_str}), {z_str})"
            transformed = f"P({y_str} | do({x_str}))" if is_d_separated else None

        return RuleApplicationResult(
            rule=DoCalculusRule.RULE_1,
            applicable=is_d_separated,
            original_expression=original,
            transformed_expression=transformed,
            d_separation_holds=is_d_separated,
            modified_graph_type="G_X̄ (incoming edges to X removed)",
            explanation=self._explain_rule_1(is_d_separated),
        )

    def apply_rule_2(
        self,
        y: set[UUID],
        x: set[UUID],
        z: set[UUID],
        w: set[UUID] | None = None,
    ) -> RuleApplicationResult:
        """Rule 2: Action/observation exchange.

        P(y | do(x), do(z), w) = P(y | do(x), z, w)

        Condition: Y ⊥ Z | X, W in G_X̄,Z̲
        (graph with incoming edges to X removed AND outgoing edges from Z removed)

        This rule allows replacing an intervention do(z) with an observation z
        if Y and Z are d-separated in the double-mutilated graph.

        Args:
            y: Outcome variables
            x: Intervention variables (do(x))
            z: Variables for action/observation exchange
            w: Additional conditioning variables (optional)

        Returns:
            RuleApplicationResult indicating if the rule applies
        """
        w = w or set()

        # Create double-mutilated graph G_X̄,Z̲
        double_mutilated = self.dag.get_double_mutilated_graph(x, z)
        mutilated_dsep = DSeparationAnalyzer(double_mutilated)

        # Check d-separation: Y ⊥ Z | X ∪ W in G_X̄,Z̲
        conditioning_set = x | w
        is_d_separated = mutilated_dsep.is_d_separated(y, z, conditioning_set)

        # Build expression strings
        x_str = self._format_vars(x, "X")
        y_str = self._format_vars(y, "Y")
        z_str = self._format_vars(z, "Z")
        w_str = self._format_vars(w, "W") if w else ""

        if w:
            original = f"P({y_str} | do({x_str}), do({z_str}), {w_str})"
            transformed = f"P({y_str} | do({x_str}), {z_str}, {w_str})" if is_d_separated else None
        else:
            original = f"P({y_str} | do({x_str}), do({z_str}))"
            transformed = f"P({y_str} | do({x_str}), {z_str})" if is_d_separated else None

        return RuleApplicationResult(
            rule=DoCalculusRule.RULE_2,
            applicable=is_d_separated,
            original_expression=original,
            transformed_expression=transformed,
            d_separation_holds=is_d_separated,
            modified_graph_type="G_X̄,Z̲ (incoming to X and outgoing from Z removed)",
            explanation=self._explain_rule_2(is_d_separated),
        )

    def apply_rule_3(
        self,
        y: set[UUID],
        x: set[UUID],
        z: set[UUID],
        w: set[UUID] | None = None,
    ) -> RuleApplicationResult:
        """Rule 3: Insertion/deletion of interventions.

        P(y | do(x), do(z), w) = P(y | do(x), w)

        Condition: Y ⊥ Z | X, W in G_X̄,Z̄(W)
        (graph with incoming edges to X removed and Z-nodes removed
        that are not ancestors of any W-node)

        This rule allows removing an intervention do(z) entirely if there
        are no causal paths from Z to Y that aren't blocked.

        Args:
            y: Outcome variables
            x: Intervention variables (do(x))
            z: Intervention to potentially remove (do(z))
            w: Additional conditioning variables (optional)

        Returns:
            RuleApplicationResult indicating if the rule applies
        """
        w = w or set()

        # Create G_X̄,Z̄(W) - complex graph surgery
        modified_dag = self.dag.get_rule_3_graph(x, z, w)
        modified_dsep = DSeparationAnalyzer(modified_dag)

        # Get remaining Z nodes (some may have been removed)
        remaining_z = z & set(modified_dag._graph.nodes())

        if not remaining_z:
            # If all Z nodes were removed, rule applies trivially
            is_d_separated = True
        else:
            # Check d-separation in modified graph
            conditioning_set = x | w
            is_d_separated = modified_dsep.is_d_separated(y, remaining_z, conditioning_set)

        # Build expression strings
        x_str = self._format_vars(x, "X")
        y_str = self._format_vars(y, "Y")
        z_str = self._format_vars(z, "Z")
        w_str = self._format_vars(w, "W") if w else ""

        if w:
            original = f"P({y_str} | do({x_str}), do({z_str}), {w_str})"
            transformed = f"P({y_str} | do({x_str}), {w_str})" if is_d_separated else None
        else:
            original = f"P({y_str} | do({x_str}), do({z_str}))"
            transformed = f"P({y_str} | do({x_str}))" if is_d_separated else None

        return RuleApplicationResult(
            rule=DoCalculusRule.RULE_3,
            applicable=is_d_separated,
            original_expression=original,
            transformed_expression=transformed,
            d_separation_holds=is_d_separated,
            modified_graph_type="G_X̄,Z̄(W) (X incoming removed, non-ancestor Z removed)",
            explanation=self._explain_rule_3(is_d_separated),
        )

    def check_identifiability(
        self,
        y: set[UUID],
        x: set[UUID],
    ) -> dict[str, any]:
        """Check if the causal effect P(Y|do(X)) is identifiable.

        A causal effect is identifiable if it can be computed from
        observational data and the causal graph using do-calculus.

        This is a simplified check. Full identifiability requires the
        Shpitser-Pearl ID algorithm.

        Args:
            y: Outcome variables
            x: Intervention variables

        Returns:
            Dictionary with identifiability results
        """
        results = {
            "is_identifiable": False,
            "method": None,
            "adjustment_sets": [],
            "explanation": "",
        }

        # Check backdoor criterion first (most common)
        dsep = DSeparationAnalyzer(self.dag)

        # Find valid adjustment sets
        if len(x) == 1 and len(y) == 1:
            treatment = next(iter(x))
            outcome = next(iter(y))
            adjustment_sets = dsep.find_valid_adjustment_sets(treatment, outcome, max_size=3)

            if adjustment_sets:
                results["is_identifiable"] = True
                results["method"] = "backdoor_criterion"
                results["adjustment_sets"] = adjustment_sets
                results["explanation"] = (
                    f"Causal effect is identifiable via backdoor criterion. "
                    f"Found {len(adjustment_sets)} valid adjustment set(s)."
                )
                return results

        # Try front-door criterion (simplified check)
        # Full implementation would use the ID algorithm

        results["explanation"] = (
            "Could not identify causal effect using backdoor criterion. "
            "Full identifiability analysis requires the ID algorithm."
        )
        return results

    def find_identifying_sequence(
        self,
        y: set[UUID],
        x: set[UUID],
    ) -> list[RuleApplicationResult] | None:
        """Attempt to find a sequence of rule applications that identifies P(Y|do(X)).

        This is a simplified search. Full implementation would use the
        Shpitser-Pearl ID algorithm.

        Args:
            y: Outcome variables
            x: Intervention variables

        Returns:
            List of rule applications if found, None if not identifiable
        """
        sequence: list[RuleApplicationResult] = []

        # Try simple direct applications
        # This is a placeholder - real implementation needs ID algorithm

        # Try Rule 2 to convert intervention to observation
        rule2_result = self.apply_rule_2(y, set(), x)
        if rule2_result.applicable:
            sequence.append(rule2_result)
            return sequence

        # Try Rule 3 to remove intervention
        rule3_result = self.apply_rule_3(y, set(), x)
        if rule3_result.applicable:
            sequence.append(rule3_result)
            return sequence

        return None if not sequence else sequence

    def _format_vars(self, vars: set[UUID], name: str) -> str:
        """Format a set of variables for display."""
        if len(vars) == 0:
            return ""
        if len(vars) == 1:
            return name.lower()
        return f"{name.lower()}₁,...,{name.lower()}ₙ"

    def _explain_rule_1(self, holds: bool) -> str:
        """Generate explanation for Rule 1 result."""
        if holds:
            return (
                "Rule 1 applies: Y and Z are d-separated given X and W "
                "in graph G_X̄. The observation Z can be removed from the expression "
                "because it provides no additional information about Y once we "
                "intervene on X and condition on W."
            )
        return (
            "Rule 1 does not apply: Y and Z are NOT d-separated given X and W "
            "in graph G_X̄. There exists an open (unblocked) path between Y and Z, "
            "so Z contains information about Y that cannot be removed."
        )

    def _explain_rule_2(self, holds: bool) -> str:
        """Generate explanation for Rule 2 result."""
        if holds:
            return (
                "Rule 2 applies: Y and Z are d-separated given X and W "
                "in graph G_X̄,Z̲. The intervention do(Z) can be replaced with "
                "observation Z because Z's causal effect on Y is blocked."
            )
        return (
            "Rule 2 does not apply: Y and Z are NOT d-separated given X and W "
            "in graph G_X̄,Z̲. The intervention do(Z) and observation Z are not "
            "exchangeable because there exists an unblocked path."
        )

    def _explain_rule_3(self, holds: bool) -> str:
        """Generate explanation for Rule 3 result."""
        if holds:
            return (
                "Rule 3 applies: Y and Z are d-separated in G_X̄,Z̄(W). "
                "The intervention do(Z) can be removed entirely because "
                "Z has no causal effect on Y in this modified graph."
            )
        return (
            "Rule 3 does not apply: There exist causal paths from Z to Y "
            "that cannot be blocked. The intervention do(Z) is required "
            "and cannot be removed from the expression."
        )


def demonstrate_do_calculus(dag: ConversationDAG) -> None:
    """Demonstrate do-calculus rules on a sample DAG.

    This function prints detailed information about rule applications
    for educational purposes.
    """
    engine = DoCalculusEngine(dag)

    print("Do-Calculus Demonstration")
    print("=" * 50)

    nodes = list(dag._graph.nodes())
    if len(nodes) < 3:
        print("Need at least 3 nodes for demonstration")
        return

    x = {nodes[0]}
    y = {nodes[-1]}
    z = {nodes[1]}

    print("\nRule 1: Insertion/deletion of observations")
    print("-" * 40)
    result1 = engine.apply_rule_1(y, x, z)
    print(result1)

    print("\nRule 2: Action/observation exchange")
    print("-" * 40)
    result2 = engine.apply_rule_2(y, x, z)
    print(result2)

    print("\nRule 3: Insertion/deletion of interventions")
    print("-" * 40)
    result3 = engine.apply_rule_3(y, x, z)
    print(result3)
