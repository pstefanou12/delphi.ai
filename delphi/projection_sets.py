# Author: pstefanou12@
"""Convex sets that projected SGD keeps a model's parameters in.

Each set holds a feasible anchor.
A point outside moves along the segment to the anchor until it re-enters the
set. That is not the Euclidean projection, but the point it returns lies in
the set, where the truncated NLL is strongly convex and the stochastic
gradient has bounded variance.
"""

import abc
from typing import Optional

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
