# Author: pstefanou12@
"""The transformation interface, and a pipeline that chains transformations."""

import abc
from collections.abc import Sequence

import torch as ch


class Transformation(abc.ABC):
    """A change of coordinates, fit to data and invertible."""

    @abc.abstractmethod
    def fit(self, x: ch.Tensor) -> "Transformation":
        """Fit the transformation to data of shape (n, dims); return self."""

    @abc.abstractmethod
    def transform(self, x: ch.Tensor) -> ch.Tensor:
        """Map data of shape (n, dims) into the new coordinates."""

    @abc.abstractmethod
    def inverse_transform(self, z: ch.Tensor) -> ch.Tensor:
        """Map data of shape (n, dims) back to the original coordinates."""


class Pipeline(Transformation):
    """Transformations applied in order, and undone in reverse.

    Attributes:
        steps: The transformations, first applied first.
    """

    def __init__(self, steps: Sequence[Transformation]):
        """Initialize the pipeline.

        Args:
            steps: The transformations, first applied first.
        """
        self.steps = list(steps)

    def fit(self, x: ch.Tensor) -> "Pipeline":
        """Fit each step to the output of the steps before it."""
        for step in self.steps:
            x = step.fit(x).transform(x)
        return self

    def transform(self, x: ch.Tensor) -> ch.Tensor:
        """Apply every step in order."""
        for step in self.steps:
            x = step.transform(x)
        return x

    def inverse_transform(self, z: ch.Tensor) -> ch.Tensor:
        """Undo every step in reverse order."""
        for step in reversed(self.steps):
            z = step.inverse_transform(z)
        return z
