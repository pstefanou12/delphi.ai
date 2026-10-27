# Author: pstefanou12@
"""Convex sets that projected SGD keeps a model's parameters in.

Each set holds a feasible anchor.
A point outside moves along the segment to the anchor until it re-enters the
set. That is not the Euclidean projection, but the point it returns lies in
the set, where the truncated NLL is strongly convex and the stochastic
gradient has bounded variance.
"""

import abc
import math
from typing import Optional, cast

import torch as ch


class ProjectionSet(abc.ABC):
    """A convex set of weights with a feasible anchor.

    Attributes:
        anchor: A weight inside the set, which projections move towards.
        radius: The set's size, for sets that have one.
        bisection_steps: Halvings of the segment a projection searches.
    """

    def __init__(
        self,
        anchor: ch.Tensor,
        radius: Optional[float] = None,
        bisection_steps: int = 40,
    ):
        """Initialize the set.

        Args:
            anchor: A weight inside the set.
            radius: The set's size, or None.
            bisection_steps: Halvings of the segment a projection searches;
                40 leaves 2^-40 of it, below float32 precision.
        """
        self.anchor = anchor.detach().clone()
        self.radius = radius
        self.bisection_steps = bisection_steps

    @abc.abstractmethod
    def contains(self, weight: ch.Tensor) -> bool:
        """Return whether weight lies in the set."""

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
        for _ in range(self.bisection_steps):
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

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        return weight


class ResidualSpectralSet(ProjectionSet):
    """The projection set D_{r,B} of Daskalakis et al. (2020), Definition 2.

    ``{w : sum_i ((y_i - w^T x_i) / sigma)^2 x_i x_i^T <= r sum_i x_i x_i^T,
    ||w|| <= B}``, in the Loewner order. In coordinates where
    ``sum_i x_i x_i^T`` is the identity the first constraint bounds the
    largest eigenvalue of ``sum_i rho_i^2 u_i u_i^T`` by r. ``rho_i`` is
    linear in ``v = (-w, 1)``, so that matrix is ``sum_pq v_p v_q Q_pq`` for
    a tensor Q summed over the data once, and a membership test costs
    O(d^4) instead of O(n d^2) whenever d^2 is below n and Q fits in memory.

    Attributes:
        radius: The set's r, bounding the residual-weighted covariance
            relative to the covariance.
        bound: Bound B on the weight's norm, or None for no ball.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        noise_var: float,
        radius: float,
        anchor: ch.Tensor,
        bound: Optional[float] = None,
        bisection_steps: int = 40,
        max_quadratic_entries: int = 2**24,
    ):
        """Build D_{r,B} from the training data.

        Args:
            X: Design matrix, shaped (n, d).
            y: Responses, shaped (n, 1).
            noise_var: Known noise variance sigma^2.
            radius: The set's r.
            anchor: Weight of shape (d, 1) to retract towards.
            bound: The set's B, or None for no ball.
            bisection_steps: As ``ProjectionSet``.
            max_quadratic_entries: Largest precomputed quadratic form, in
                float64 entries; 2^24 is 128 MB. A larger one is skipped for
                the O(n d^2) test.
        """
        super().__init__(anchor, radius, bisection_steps)
        self.bound = bound
        scale = math.sqrt(noise_var)
        chol = ch.linalg.cholesky(X.T @ X)  # pylint: disable=not-callable
        whitened = ch.linalg.solve_triangular(chol, X.T, upper=False).T  # pylint: disable=not-callable
        joint = ch.cat([-X, y], dim=1) / scale
        self._direct: Optional[tuple[ch.Tensor, ch.Tensor]] = None
        self._quadratic: Optional[ch.Tensor] = None
        entries = ((X.size(1) + 1) * X.size(1)) ** 2
        if (X.size(1) + 1) ** 2 <= X.size(0) and entries <= max_quadratic_entries:
            whitened, joint = whitened.double(), joint.double()
            self._quadratic = ch.einsum(
                "np,nq,na,nb->pqab", joint, joint, whitened, whitened
            )
        else:
            self._direct = (joint, whitened)

    def contains(self, weight: ch.Tensor) -> bool:
        if self.bound is not None and float(weight.norm()) > self.bound:
            return False
        v = ch.cat([weight.flatten(), ch.ones(1)])
        if self._quadratic is not None:
            v = v.double()
            weighted = ch.einsum("p,q,pqab->ab", v, v, self._quadratic)
        else:
            joint, whitened = cast(tuple[ch.Tensor, ch.Tensor], self._direct)
            weighted = (whitened * (joint @ v)[:, None].pow(2)).T @ whitened
        return float(ch.linalg.eigvalsh(weighted)[-1]) <= cast(float, self.radius)  # pylint: disable=not-callable
