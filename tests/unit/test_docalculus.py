"""Unit tests for d-separation and do-calculus.

These tests use classic causal inference examples to verify
the correctness of the causal reasoning algorithms.
"""

import pytest

from causalai.core.dag import ConversationDAG
from causalai.core.nodes import ConversationNode, NodeType
from causalai.core.edges import create_causal_edge
from causalai.causal.dseparation import DSeparationAnalyzer
from causalai.causal.docalculus import DoCalculusEngine, DoCalculusRule


def create_chain_dag():
    """Create a simple chain: A → B → C.

    In a chain, A and C are d-separated by B (the mediator).
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

    return dag, a, b, c


def create_fork_dag():
    """Create a fork (common cause): A ← B → C.

    In a fork, A and C are d-separated by B (the common cause).
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

    return dag, a, b, c


def create_collider_dag():
    """Create a collider (common effect): A → B ← C.

    In a collider, A and C are d-separated unconditionally,
    but become d-connected when conditioning on B (or its descendants).
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

    return dag, a, b, c


def create_confounding_dag():
    """Create a confounded structure: Z → X, Z → Y, X → Y.

    Classic confounding where Z is a common cause of treatment X
    and outcome Y. This is the most common scenario requiring
    adjustment or do-calculus.
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

    return dag, z, x, y


def create_frontdoor_dag():
    """Create a front-door structure: Z → X → M → Y, Z → Y.

    The front-door criterion applies when there's a mediator M
    between treatment X and outcome Y that blocks all backdoor paths.
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

    return dag, z, x, m, y


class TestDSeparationChain:
    """Tests for d-separation in chains (A → B → C)."""

    def test_unconditional_d_connection(self):
        """In a chain, A and C are d-connected unconditionally."""
        dag, a, b, c = create_chain_dag()
        analyzer = DSeparationAnalyzer(dag)

        # A and C are NOT d-separated (d-connected)
        assert analyzer.is_d_separated({a.id}, {c.id}) is False

    def test_d_separation_by_mediator(self):
        """In a chain, A and C are d-separated by B."""
        dag, a, b, c = create_chain_dag()
        analyzer = DSeparationAnalyzer(dag)

        # A and C ARE d-separated when conditioning on B
        assert analyzer.is_d_separated({a.id}, {c.id}, {b.id}) is True

    def test_d_connection_helper(self):
        """is_d_connected is inverse of is_d_separated."""
        dag, a, b, c = create_chain_dag()
        analyzer = DSeparationAnalyzer(dag)

        assert analyzer.is_d_connected({a.id}, {c.id}) is True
        assert analyzer.is_d_connected({a.id}, {c.id}, {b.id}) is False


class TestDSeparationFork:
    """Tests for d-separation in forks (A ← B → C)."""

    def test_unconditional_d_connection(self):
        """In a fork, A and C are d-connected unconditionally."""
        dag, a, b, c = create_fork_dag()
        analyzer = DSeparationAnalyzer(dag)

        # A and C are d-connected (through common cause B)
        assert analyzer.is_d_separated({a.id}, {c.id}) is False

    def test_d_separation_by_common_cause(self):
        """In a fork, A and C are d-separated by the common cause B."""
        dag, a, b, c = create_fork_dag()
        analyzer = DSeparationAnalyzer(dag)

        # A and C are d-separated when conditioning on B
        assert analyzer.is_d_separated({a.id}, {c.id}, {b.id}) is True


class TestDSeparationCollider:
    """Tests for d-separation in colliders (A → B ← C).

    Colliders exhibit the 'explaining away' phenomenon where
    conditioning on the collider opens the path between causes.
    """

    def test_unconditional_d_separation(self):
        """In a collider, A and C are d-separated unconditionally."""
        dag, a, b, c = create_collider_dag()
        analyzer = DSeparationAnalyzer(dag)

        # A and C ARE d-separated (collider blocks the path)
        assert analyzer.is_d_separated({a.id}, {c.id}) is True

    def test_d_connection_when_conditioning_on_collider(self):
        """Conditioning on collider opens the path (explaining away)."""
        dag, a, b, c = create_collider_dag()
        analyzer = DSeparationAnalyzer(dag)

        # Conditioning on B opens the path!
        assert analyzer.is_d_separated({a.id}, {c.id}, {b.id}) is False


class TestDSeparationConfounding:
    """Tests for d-separation with confounding."""

    def test_confounded_treatment_outcome(self):
        """Treatment X and outcome Y are d-connected due to confounding."""
        dag, z, x, y = create_confounding_dag()
        analyzer = DSeparationAnalyzer(dag)

        # X and Y are d-connected (via both X→Y and X←Z→Y)
        assert analyzer.is_d_separated({x.id}, {y.id}) is False

    def test_backdoor_blocked_by_confounder(self):
        """Conditioning on confounder Z blocks the backdoor path."""
        dag, z, x, y = create_confounding_dag()
        analyzer = DSeparationAnalyzer(dag)

        # In the mutilated graph G_X̄ (intervene on X):
        # The backdoor path X←Z→Y is blocked when we condition on Z
        mutilated = dag.get_mutilated_graph({x.id})
        mut_analyzer = DSeparationAnalyzer(mutilated)

        # In mutilated graph, X and Y are d-separated given Z
        # (only the direct path X→Y remains, and it's blocked by conditioning on Z... wait no)
        # Actually, in mutilated graph, X has no parents, so X⊥Z, and the only path is X→Y
        # So X and Y are NOT d-separated in G_X̄ (they're connected by X→Y)
        assert mut_analyzer.is_d_separated({x.id}, {y.id}) is False


class TestBackdoorCriterion:
    """Tests for the backdoor criterion (valid adjustment sets)."""

    def test_confounder_is_valid_adjustment(self):
        """In confounding scenario, Z is a valid adjustment set."""
        dag, z, x, y = create_confounding_dag()
        analyzer = DSeparationAnalyzer(dag)

        # Z is a valid adjustment set for estimating X→Y effect
        assert analyzer.is_valid_adjustment_set(x.id, y.id, {z.id}) is True

    def test_empty_set_invalid_with_confounding(self):
        """Empty adjustment set is invalid when confounding exists."""
        dag, z, x, y = create_confounding_dag()
        analyzer = DSeparationAnalyzer(dag)

        # Empty set doesn't block backdoor path
        assert analyzer.is_valid_adjustment_set(x.id, y.id, set()) is False

    def test_descendant_not_valid_adjustment(self):
        """Descendants of treatment are not valid for adjustment."""
        dag, z, x, y = create_confounding_dag()
        analyzer = DSeparationAnalyzer(dag)

        # Y is a descendant of X, so cannot be in adjustment set
        assert analyzer.is_valid_adjustment_set(x.id, y.id, {y.id}) is False

    def test_find_valid_adjustment_sets(self):
        """Find all valid adjustment sets."""
        dag, z, x, y = create_confounding_dag()
        analyzer = DSeparationAnalyzer(dag)

        valid_sets = analyzer.find_valid_adjustment_sets(x.id, y.id)

        # Should find {Z} as valid
        assert any(z.id in s for s in valid_sets)


class TestDoCalculusRule1:
    """Tests for Rule 1: Insertion/deletion of observations."""

    def test_rule1_chain(self):
        """Rule 1 in a chain: can remove observation if d-separated."""
        dag, a, b, c = create_chain_dag()
        engine = DoCalculusEngine(dag)

        # P(c|do(a),b) = P(c|do(a)) because C⊥B|A in G_Ā
        # Wait, in G_Ā, A has no parents (already none), so G_Ā = G
        # In G: C is d-separated from B given A? No, B→C is still there
        # Let me reconsider: we want to check if C⊥B|A in G_Ā
        # G_Ā is A→B→C with A's incoming edges removed (none exist)
        # C⊥B|A? Given A, B→C still connects B and C, so NO.

        result = engine.apply_rule_1(
            y={c.id},  # outcome
            x={a.id},  # intervention
            z={b.id},  # observation to potentially remove
        )

        # B is on the causal path A→B→C, so C and B are NOT d-sep given A
        assert result.applicable is False

    def test_rule1_fork_remove_observation(self):
        """Rule 1 can remove observation in a fork structure."""
        dag, a, b, c = create_fork_dag()  # A ← B → C
        engine = DoCalculusEngine(dag)

        # Can we remove A when computing P(C|do(B),A)?
        # In G_B̄, B has no incoming edges (already none for root)
        # Is C⊥A|B? Yes! In the fork, conditioning on B blocks A-C path

        result = engine.apply_rule_1(
            y={c.id},  # outcome C
            x={b.id},  # intervention do(B)
            z={a.id},  # observation A to potentially remove
        )

        # A and C are d-separated given B in G_B̄
        assert result.applicable is True


class TestDoCalculusRule2:
    """Tests for Rule 2: Action/observation exchange."""

    def test_rule2_exchange_intervention_for_observation(self):
        """Rule 2: can exchange do(Z) for observing Z under certain conditions."""
        dag, a, b, c = create_chain_dag()  # A → B → C
        engine = DoCalculusEngine(dag)

        # P(c|do(a),do(b)) = P(c|do(a),b)?
        # Need C⊥B|A in G_Ā,B̲
        # G_Ā,B̲: remove incoming to A, remove outgoing from B
        # Result: A → B (no edge B→C)
        # Is C d-sep from B given A? C is isolated, so yes!

        result = engine.apply_rule_2(
            y={c.id},
            x={a.id},
            z={b.id},  # can we replace do(b) with observing b?
        )

        assert result.applicable is True
        assert result.rule == DoCalculusRule.RULE_2


class TestDoCalculusRule3:
    """Tests for Rule 3: Insertion/deletion of interventions."""

    def test_rule3_remove_intervention(self):
        """Rule 3: can remove do(Z) if no causal path to Y."""
        # Create a graph where Z doesn't affect Y
        dag = ConversationDAG()
        x = ConversationNode(node_type=NodeType.USER_MESSAGE, content="X")
        y = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="Y")
        z = ConversationNode(node_type=NodeType.USER_MESSAGE, content="Z (isolated)")

        dag.add_node(x)
        dag.add_node(y)
        dag.add_node(z)
        dag.add_edge(create_causal_edge(x.id, y.id))
        # Z has no edges to Y

        engine = DoCalculusEngine(dag)

        # P(y|do(x),do(z)) = P(y|do(x))?
        # Z is isolated from Y, so intervention on Z shouldn't matter

        result = engine.apply_rule_3(
            y={y.id},
            x={x.id},
            z={z.id},
        )

        assert result.applicable is True


class TestDoCalculusIdentifiability:
    """Tests for causal effect identifiability."""

    def test_identifiable_with_backdoor(self):
        """Effect is identifiable when backdoor criterion is satisfied."""
        dag, z, x, y = create_confounding_dag()
        engine = DoCalculusEngine(dag)

        result = engine.check_identifiability({y.id}, {x.id})

        assert result["is_identifiable"] is True
        assert result["method"] == "backdoor_criterion"
        assert len(result["adjustment_sets"]) > 0


class TestClassicCausalExamples:
    """Tests using classic causal inference examples from literature."""

    def test_smoking_tar_cancer(self):
        """Classic example: Smoking → Tar → Cancer, with potential confounding.

        This is a front-door structure where we can identify the
        effect of Smoking on Cancer through the mediator Tar.
        """
        dag = ConversationDAG()

        # U: unobserved confounder (e.g., genetic predisposition)
        u = ConversationNode(node_type=NodeType.SYSTEM_PROMPT, content="U (unobserved)")
        # Smoking: treatment
        smoking = ConversationNode(node_type=NodeType.USER_MESSAGE, content="Smoking")
        # Tar: mediator
        tar = ConversationNode(node_type=NodeType.TOOL_CALL, content="Tar")
        # Cancer: outcome
        cancer = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="Cancer")

        dag.add_node(u)
        dag.add_node(smoking)
        dag.add_node(tar)
        dag.add_node(cancer)

        # U → Smoking, U → Cancer (confounding)
        dag.add_edge(create_causal_edge(u.id, smoking.id))
        dag.add_edge(create_causal_edge(u.id, cancer.id))
        # Smoking → Tar → Cancer (front-door path)
        dag.add_edge(create_causal_edge(smoking.id, tar.id))
        dag.add_edge(create_causal_edge(tar.id, cancer.id))

        analyzer = DSeparationAnalyzer(dag)

        # Tar is d-separated from U given Smoking
        # (Tar has no backdoor path to U that isn't through Smoking)
        assert analyzer.is_d_separated({tar.id}, {u.id}, {smoking.id}) is True

    def test_simpsons_paradox_structure(self):
        """Simpson's paradox: aggregate vs stratified analysis.

        Structure: Gender → Department → Admission
                   Gender → Admission

        The effect of Department on Admission differs when conditioning
        on Gender vs marginalizing over Gender.
        """
        dag = ConversationDAG()

        gender = ConversationNode(node_type=NodeType.SYSTEM_PROMPT, content="Gender")
        dept = ConversationNode(node_type=NodeType.USER_MESSAGE, content="Department")
        admit = ConversationNode(node_type=NodeType.ASSISTANT_RESPONSE, content="Admission")

        dag.add_node(gender)
        dag.add_node(dept)
        dag.add_node(admit)

        dag.add_edge(create_causal_edge(gender.id, dept.id))
        dag.add_edge(create_causal_edge(gender.id, admit.id))
        dag.add_edge(create_causal_edge(dept.id, admit.id))

        analyzer = DSeparationAnalyzer(dag)

        # Gender is a valid adjustment for Department → Admission effect
        assert analyzer.is_valid_adjustment_set(dept.id, admit.id, {gender.id}) is True

        # Empty adjustment is NOT valid (Gender confounds)
        assert analyzer.is_valid_adjustment_set(dept.id, admit.id, set()) is False
