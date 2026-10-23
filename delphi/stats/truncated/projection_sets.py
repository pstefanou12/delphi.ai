# Author: pstefanou12@
"""Convex sets that projected SGD keeps a truncated linear model's weight in.

Each set is built from the training data once, before the descent starts,
and holds a feasible anchor. Projecting a point outside moves it along the
segment to the anchor until it re-enters the set. That is not the Euclidean
projection, but the point it returns lies in the set, which is where the
truncated NLL is strongly convex and the stochastic gradient has bounded
variance; for a convex set the segment never leaves it.
"""

import abc
import math
from typing import Optional

import torch as ch

# Halvings of the segment; 2^-40 of its length is below float32 precision.
BISECTION_STEPS = 40


class ProjectionSet(abc.ABC):
    """A convex set of weights with a feasible anchor.

    Attributes:
        anchor: A weight inside the set, which projections move towards.
    """

    def __init__(self, anchor: ch.Tensor):
        """Initialize the set.

        Args:
            anchor: A weight inside the set.
        """
        self.anchor = anchor.detach().clone()

    @abc.abstractmethod
    def contains(self, weight: ch.Tensor) -> bool:
        """Return whether weight lies in the set."""

    @abc.abstractmethod
    def describe(self, weight: ch.Tensor) -> list[tuple[str, str]]:
        """Return labelled rows describing the set and weight's place in it."""

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        """Return weight if it lies in the set, else its retraction onto it.

        Args:
            weight: The point to project.

        Returns:
            The point of the segment from the anchor to weight that is
            furthest from the anchor while inside the set.
        """
        if self.contains(weight):
            return weight
        inside, outside = 0.0, 1.0
        direction = weight.detach() - self.anchor
        for _ in range(BISECTION_STEPS):
            middle = (inside + outside) / 2.0
            if self.contains(self.anchor + middle * direction):
                inside = middle
            else:
                outside = middle
        return self.anchor + inside * direction


class Unconstrained(ProjectionSet):
    """The whole space: projected SGD without a projection."""

    def contains(self, weight: ch.Tensor) -> bool:
        return True

    def describe(self, weight: ch.Tensor) -> list[tuple[str, str]]:
        return [("set", "the whole space")]

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        return weight


class ResidualSpectralSet(ProjectionSet):
    """The projection set D_{r,B} of Daskalakis et al. (2020), Definition 2.

    ``{w : sum_i ((y_i - w^T x_i) / sigma)^2 x_i x_i^T <= r sum_i x_i x_i^T,
    ||w|| <= B}``, in the Loewner order. In coordinates where
    ``sum_i x_i x_i^T`` is the identity, the first constraint reads
    ``lambda_max(sum_i rho_i^2 u_i u_i^T) <= r``, which costs one
    O(n d^2) product and a d x d eigenvalue per membership test.

    Attributes:
        r: Bound on the residual-weighted covariance, relative to the
            covariance.
        bound: Bound B on the weight's norm, or None for no ball.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        design: ch.Tensor,
        targ: ch.Tensor,
        noise_var: float,
        r: float,
        anchor: ch.Tensor,
        bound: Optional[float] = None,
    ):
        """Build D_{r,B} from the training data.

        Args:
            design: Design matrix, shaped (n, d).
            targ: Responses, shaped (n, 1).
            noise_var: Known noise variance sigma^2.
            r: The set's r; the paper's r* is ``4 log(2 / alpha) + 7``.
            anchor: Weight of shape (d, 1) to retract towards.
            bound: The set's B, or None for no ball.
        """
        super().__init__(anchor)
        self.design = design
        self.targ = targ
        self.scale = math.sqrt(noise_var)
        self.r = r
        self.bound = bound
        chol = ch.linalg.cholesky(design.T @ design)  # pylint: disable=not-callable
        self.whitened = ch.linalg.solve_triangular(  # pylint: disable=not-callable
            chol, design.T, upper=False
        ).T

    @staticmethod
    def paper_radius(alpha: float) -> float:
        """Return r* = 4 log(2 / alpha) + 7, which holds the truth w.h.p."""
        return 4.0 * math.log(2.0 / alpha) + 7.0

    def spectral_radius(self, weight: ch.Tensor) -> float:
        """Return lambda_max of the whitened residual-weighted covariance.

        Args:
            weight: Weight of shape (d, 1).

        Returns:
            The value the set bounds by r.
        """
        resid = ((self.targ - self.design @ weight) / self.scale).pow(2)
        weighted = (self.whitened * resid).T @ self.whitened
        return float(ch.linalg.eigvalsh(weighted)[-1])  # pylint: disable=not-callable

    def contains(self, weight: ch.Tensor) -> bool:
        if self.bound is not None and float(weight.norm()) > self.bound:
            return False
        return self.spectral_radius(weight) <= self.r

    def describe(self, weight: ch.Tensor) -> list[tuple[str, str]]:
        bound = "none" if self.bound is None else f"{self.bound:.4f}"
        return [
            ("set", "D_{r,B}, Daskalakis et al. (2020)"),
            ("r", f"{self.r:.4f}"),
            ("B", bound),
            ("lambda_max at the estimate", f"{self.spectral_radius(weight):.4f}"),
            ("||w|| at the estimate", f"{float(weight.norm()):.4f}"),
        ]
