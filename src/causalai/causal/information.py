"""
Information-theoretic primitives for causal inference.

This module implements information-theoretic measures that bridge
Pearl's causal framework with information theory:

- Causal Entropy: H(Y|do(X)) - entropy remaining after intervention
- Causal Mutual Information: I(X;Y) under causal assumptions
- Transfer Entropy: Information transfer over time (Granger-style)
- Directed Information: For quantifying causal effects

Reference:
- Information Theoretic Causal Effect Quantification (Entropy, 2019)
- A unified theory of information transfer and causal relation (arXiv:2204.13598)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from uuid import UUID

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from causalai.core.dag import ConversationDAG


@dataclass
class InformationMetrics:
    """Container for information-theoretic metrics.

    Attributes:
        entropy: Shannon entropy H(X)
        conditional_entropy: Conditional entropy H(Y|X)
        mutual_information: Mutual information I(X;Y)
        causal_entropy: Causal entropy H(Y|do(X))
        transfer_entropy: Transfer entropy T(X→Y)
    """

    entropy: float | None = None
    conditional_entropy: float | None = None
    mutual_information: float | None = None
    causal_entropy: float | None = None
    transfer_entropy: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert to dictionary."""
        return {
            "entropy": self.entropy,
            "conditional_entropy": self.conditional_entropy,
            "mutual_information": self.mutual_information,
            "causal_entropy": self.causal_entropy,
            "transfer_entropy": self.transfer_entropy,
        }


class InformationTheoreticAnalyzer:
    """Computes information-theoretic measures for causal analysis.

    This class bridges Pearl's structural causal models with information
    theory by computing quantities like causal entropy and directed
    information that have both information-theoretic and causal interpretations.

    The key insight is that mutual information I(X;Y) can capture causal
    effects in unconfounded settings, while causal entropy H(Y|do(X))
    quantifies the remaining uncertainty after intervention.

    Example:
        >>> analyzer = InformationTheoreticAnalyzer(dag)
        >>> # Compute causal entropy for intervention
        >>> result = analyzer.compute_causal_entropy(y_id, x_id, distribution)
    """

    def __init__(self, dag: ConversationDAG):
        """Initialize the analyzer.

        Args:
            dag: The ConversationDAG to analyze
        """
        self.dag = dag

    @staticmethod
    def entropy(probabilities: np.ndarray, base: float = 2) -> float:
        """Compute Shannon entropy H(X).

        H(X) = -Σ p(x) log p(x)

        Args:
            probabilities: Array of probabilities (must sum to 1)
            base: Logarithm base (2 for bits, e for nats)

        Returns:
            Shannon entropy in specified units
        """
        # Filter out zero probabilities to avoid log(0)
        p = probabilities[probabilities > 0]

        if len(p) == 0:
            return 0.0

        if base == 2:
            return -np.sum(p * np.log2(p))
        elif base == math.e:
            return -np.sum(p * np.log(p))
        else:
            return -np.sum(p * np.log(p)) / np.log(base)

    @staticmethod
    def conditional_entropy(
        joint_probs: np.ndarray,
        marginal_probs: np.ndarray,
        base: float = 2,
    ) -> float:
        """Compute conditional entropy H(Y|X).

        H(Y|X) = -Σ p(x,y) log p(y|x)
               = H(X,Y) - H(X)

        Args:
            joint_probs: 2D array of joint probabilities P(X,Y)
            marginal_probs: 1D array of marginal probabilities P(X)
            base: Logarithm base

        Returns:
            Conditional entropy H(Y|X)
        """
        joint_entropy = InformationTheoreticAnalyzer.entropy(
            joint_probs.flatten(), base
        )
        marginal_entropy = InformationTheoreticAnalyzer.entropy(marginal_probs, base)
        return joint_entropy - marginal_entropy

    @staticmethod
    def mutual_information(
        joint_probs: np.ndarray,
        marginal_x: np.ndarray,
        marginal_y: np.ndarray,
        base: float = 2,
    ) -> float:
        """Compute mutual information I(X;Y).

        I(X;Y) = H(X) + H(Y) - H(X,Y)
               = Σ p(x,y) log(p(x,y) / (p(x)p(y)))

        Mutual information quantifies the amount of information that
        X and Y share - how much knowing X reduces uncertainty about Y.

        Args:
            joint_probs: 2D array of joint probabilities P(X,Y)
            marginal_x: 1D array of marginal probabilities P(X)
            marginal_y: 1D array of marginal probabilities P(Y)
            base: Logarithm base

        Returns:
            Mutual information I(X;Y)
        """
        h_x = InformationTheoreticAnalyzer.entropy(marginal_x, base)
        h_y = InformationTheoreticAnalyzer.entropy(marginal_y, base)
        h_xy = InformationTheoreticAnalyzer.entropy(joint_probs.flatten(), base)
        return h_x + h_y - h_xy

    def causal_entropy(
        self,
        y_node: UUID,
        x_node: UUID,
        intervention_distribution: Callable[[any], np.ndarray],
        outcome_distribution: Callable[[any, any], np.ndarray],
        x_values: np.ndarray,
    ) -> float:
        """Compute causal entropy H(Y|do(X)).

        Causal entropy is the entropy of Y that remains, on average,
        after atomically intervening on X:

        H(Y|do(X)) = E_x[H(Y|do(X=x))]
                   = Σ_x P(X=x) H(Y|do(X=x))

        This differs from conditional entropy H(Y|X) because it considers
        the interventional distribution, not the observational.

        Args:
            y_node: The outcome node
            x_node: The intervention node
            intervention_distribution: Function that returns P(do(X=x))
            outcome_distribution: Function(x, y_values) returning P(Y|do(X=x))
            x_values: Possible values of X

        Returns:
            Causal entropy H(Y|do(X))
        """
        total_entropy = 0.0

        for x in x_values:
            # Get P(do(X=x)) - the probability of intervening with this value
            p_do_x = intervention_distribution(x)

            # Get P(Y|do(X=x)) - the interventional distribution of Y
            p_y_given_do_x = outcome_distribution(x, None)

            # Compute H(Y|do(X=x))
            h_y_given_do_x = self.entropy(p_y_given_do_x)

            # Weight by intervention probability
            total_entropy += p_do_x * h_y_given_do_x

        return total_entropy

    def causal_mutual_information(
        self,
        x_node: UUID,
        y_node: UUID,
        adjustment_set: set[UUID] | None = None,
    ) -> dict[str, any]:
        """Compute causal mutual information.

        In an unconfounded setting (or after adjusting for confounders),
        conditional mutual information I(X;Y|Z) captures the average
        causal effect analogous to the Average Treatment Effect (ATE).

        This method identifies valid adjustment sets and describes how
        to compute the causal mutual information.

        Args:
            x_node: Treatment/cause node
            y_node: Outcome/effect node
            adjustment_set: Optional adjustment set Z

        Returns:
            Dictionary with causal MI analysis results
        """
        from causalai.causal.dseparation import DSeparationAnalyzer

        dsep = DSeparationAnalyzer(self.dag)

        # Find valid adjustment sets if not provided
        if adjustment_set is None:
            valid_sets = dsep.find_valid_adjustment_sets(x_node, y_node, max_size=3)
            adjustment_set = valid_sets[0] if valid_sets else set()

        # Check if adjustment set is valid
        is_valid = dsep.is_valid_adjustment_set(x_node, y_node, adjustment_set)

        return {
            "x_node": str(x_node),
            "y_node": str(y_node),
            "adjustment_set": [str(n) for n in adjustment_set],
            "is_valid_adjustment": is_valid,
            "formula": "I(X;Y|Z) where Z is valid adjustment set",
            "interpretation": (
                "If adjustment is valid, I(X;Y|Z) captures causal dependence. "
                "This is analogous to the Average Causal Effect when properly adjusted."
            ),
            "note": "Actual computation requires data from the specified distributions",
        }

    @staticmethod
    def transfer_entropy(
        x_series: np.ndarray,
        y_series: np.ndarray,
        lag: int = 1,
        bins: int = 10,
    ) -> float:
        """Compute transfer entropy T(X→Y).

        Transfer entropy measures the amount of directed information transfer
        from X to Y, accounting for Y's own past:

        T(X→Y) = I(Y_t ; X_{t-lag} | Y_{t-1})

        This is a measure of Granger causality in information-theoretic terms.

        Args:
            x_series: Time series of X values
            y_series: Time series of Y values (same length as x_series)
            lag: Time lag to consider
            bins: Number of bins for discretization

        Returns:
            Transfer entropy from X to Y
        """
        if len(x_series) != len(y_series):
            raise ValueError("Time series must have same length")

        n = len(x_series)
        if n < lag + 2:
            raise ValueError("Time series too short for specified lag")

        # Create lagged versions
        y_current = y_series[lag + 1:]  # Y_t
        y_past = y_series[lag:-1]  # Y_{t-1}
        x_lagged = x_series[:-lag - 1]  # X_{t-lag}

        # Discretize for histogram-based estimation
        y_curr_disc = np.digitize(y_current, np.linspace(y_current.min(), y_current.max(), bins))
        y_past_disc = np.digitize(y_past, np.linspace(y_past.min(), y_past.max(), bins))
        x_lag_disc = np.digitize(x_lagged, np.linspace(x_lagged.min(), x_lagged.max(), bins))

        # Compute joint and marginal probabilities using histograms
        # P(Y_t, Y_{t-1}, X_{t-lag})
        joint_xyz, _ = np.histogramdd(
            [y_curr_disc, y_past_disc, x_lag_disc],
            bins=bins,
        )
        joint_xyz = joint_xyz / joint_xyz.sum()

        # P(Y_t, Y_{t-1})
        joint_yy, _ = np.histogramdd([y_curr_disc, y_past_disc], bins=bins)
        joint_yy = joint_yy / joint_yy.sum()

        # P(Y_{t-1}, X_{t-lag})
        joint_yx, _ = np.histogramdd([y_past_disc, x_lag_disc], bins=bins)
        joint_yx = joint_yx / joint_yx.sum()

        # P(Y_{t-1})
        p_y_past = np.histogram(y_past_disc, bins=bins)[0]
        p_y_past = p_y_past / p_y_past.sum()

        # Compute transfer entropy using chain rule
        # T(X→Y) = H(Y_t|Y_{t-1}) - H(Y_t|Y_{t-1}, X_{t-lag})

        # H(Y_t, Y_{t-1})
        h_yy = InformationTheoreticAnalyzer.entropy(joint_yy.flatten())
        # H(Y_{t-1})
        h_y_past = InformationTheoreticAnalyzer.entropy(p_y_past)
        # H(Y_t | Y_{t-1})
        h_ycurr_given_ypast = h_yy - h_y_past

        # H(Y_t, Y_{t-1}, X_{t-lag})
        h_xyz = InformationTheoreticAnalyzer.entropy(joint_xyz.flatten())
        # H(Y_{t-1}, X_{t-lag})
        h_yx = InformationTheoreticAnalyzer.entropy(joint_yx.flatten())
        # H(Y_t | Y_{t-1}, X_{t-lag})
        h_ycurr_given_ypast_x = h_xyz - h_yx

        # Transfer entropy
        te = h_ycurr_given_ypast - h_ycurr_given_ypast_x

        return max(0, te)  # TE should be non-negative

    @staticmethod
    def directed_information(
        x_series: np.ndarray,
        y_series: np.ndarray,
        bins: int = 10,
    ) -> float:
        """Compute directed information I(X^n → Y^n).

        Directed information is the causal version of mutual information
        for time series:

        I(X^n → Y^n) = Σ_{t=1}^n I(X^t ; Y_t | Y^{t-1})

        This captures the total information that X causally provides to Y
        over the entire time series.

        Args:
            x_series: Time series of X values
            y_series: Time series of Y values
            bins: Number of bins for discretization

        Returns:
            Directed information from X to Y
        """
        n = len(x_series)
        total_di = 0.0

        for t in range(1, n):
            # Compute I(X^t ; Y_t | Y^{t-1}) for each t
            # Simplified: use transfer entropy as approximation
            if t >= 2:
                te = InformationTheoreticAnalyzer.transfer_entropy(
                    x_series[:t + 1],
                    y_series[:t + 1],
                    lag=1,
                    bins=min(bins, t),
                )
                total_di += te

        return total_di

    def analyze_causal_information_flow(
        self,
        source_node: UUID,
        target_node: UUID,
    ) -> dict[str, any]:
        """Analyze information flow between nodes in the causal graph.

        This combines graphical criteria with information-theoretic concepts
        to characterize the causal information flow from source to target.

        Args:
            source_node: The source/cause node
            target_node: The target/effect node

        Returns:
            Dictionary with information flow analysis
        """
        import networkx as nx

        # Check if there's a directed path (causal path)
        has_causal_path = nx.has_path(self.dag._graph, source_node, target_node)

        # Find all causal paths
        causal_paths = []
        if has_causal_path:
            causal_paths = list(nx.all_simple_paths(
                self.dag._graph, source_node, target_node
            ))

        # Check for confounding (common ancestors)
        source_ancestors = self.dag.get_ancestors(source_node)
        target_ancestors = self.dag.get_ancestors(target_node)
        common_ancestors = source_ancestors & target_ancestors

        # Analyze the structure
        return {
            "source": str(source_node),
            "target": str(target_node),
            "has_causal_path": has_causal_path,
            "num_causal_paths": len(causal_paths),
            "shortest_path_length": min(len(p) for p in causal_paths) if causal_paths else None,
            "has_confounding": len(common_ancestors) > 0,
            "num_confounders": len(common_ancestors),
            "information_theoretic_interpretation": {
                "if_no_confounding": (
                    "I(X;Y) = causal information flow when no confounders exist"
                ),
                "if_confounding": (
                    "I(X;Y|Z) = causal information flow after adjusting for confounders Z"
                ),
                "causal_entropy": (
                    "H(Y|do(X)) measures remaining uncertainty after intervention"
                ),
            },
        }


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence D_KL(P || Q).

    D_KL(P || Q) = Σ p(x) log(p(x) / q(x))

    Args:
        p: True distribution
        q: Approximating distribution

    Returns:
        KL divergence (non-negative, 0 iff P = Q)
    """
    # Avoid division by zero
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence.

    JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)

    where M = 0.5 * (P + Q)

    JSD is symmetric and bounded between 0 and 1.

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        Jensen-Shannon divergence
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
