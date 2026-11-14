# Author: pstefanou12@
"""Models fit by projected SGD, in phases that widen the projection set.

Each phase keeps the iterates in a projection set, and the next widens its
radius until a phase never projects.
"""

import abc
from typing import ClassVar, Generic, Optional, TypeVar

import torch as ch

from delphi.configs.models import psgd
from delphi.models.statistical import StatisticalModel
import delphi.projection_sets
import delphi.trainer

_PSGDConfigT = TypeVar("_PSGDConfigT", bound=psgd.PSGDModelConfig)


class PSGDModel(StatisticalModel[_PSGDConfigT], Generic[_PSGDConfigT], abc.ABC):
    """Projected SGD in phases, each widening the projection set's radius.

    The first phase builds the set from ``default_projection_set``. With alpha
    known its radius holds the truth and one phase runs. Without it the
    radius grows by ``rate`` per phase, until a phase never projects, the
    subclass's ``_phase_converged`` stops it, or it reaches ``max_radius``.
    A subclass projects its parameters in ``step_post_hook`` through
    ``project``, which counts each move.

    Attributes:
        alpha: Survival mass of the truncation set, if known.
        projection_set: The set the iterates are kept in, built on the first
            phase.
        radius: The projection set's radius in the current phase.
    """

    # The stop reason when a phase never projected.
    not_binding_reason: ClassVar[str] = "radius_not_binding"

    def __init__(self, *, hyperparameters: Optional[dict] = None):
        """Initialize the model.

        Args:
            hyperparameters: Hyperparameter dict, validated by the subclass
                ``defaults`` config.
        """
        super().__init__(hyperparameters=hyperparameters)
        self.alpha: Optional[float] = None
        self.projection_set: Optional[delphi.projection_sets.ProjectionSet] = None
        self.radius: Optional[float] = None
        self._projections: int = 0
        self._projection_distance: float = 0.0

    @abc.abstractmethod
    def default_projection_set(self) -> delphi.projection_sets.ProjectionSet:
        """Return the set the first phase keeps the iterates in, at its radius."""

    def pre_phase_hook(self) -> None:
        """Build the projection set on the first phase, resize it after."""
        self._projections, self._projection_distance = 0, 0.0
        if self.projection_set is not None:
            self.projection_set.radius = self.radius
            return
        self.projection_set = self.default_projection_set()
        self.radius = self.projection_set.radius

    def project(self, value: ch.Tensor) -> ch.Tensor:
        """Return value's projection onto the projection set, counting a move.

        Raises:
            RuntimeError: Before the first phase builds the set.
        """
        if self.projection_set is None:
            raise RuntimeError(f"{type(self).__name__} has no projection set yet.")
        projected = self.projection_set.project(value)
        if projected is not value:
            self._projections += 1
            self._projection_distance += float((projected - value.detach()).norm())
        return projected

    def post_phase_hook(
        self, result: delphi.trainer.RunResult
    ) -> tuple[bool, Optional[str]]:
        """Widen the projection radius, or say why the procedure is done.

        A radius the phase never projected onto constrained nothing, so
        widening it would only rerun the same fit.
        """
        if self.alpha is not None:
            return True, "alpha_known"
        if result.metrics["projections"] == 0:
            return True, self.not_binding_reason
        stop, reason = self._phase_converged(result)
        if stop:
            return stop, reason
        if self.radius is None:
            raise RuntimeError("the first phase sets the radius.")
        if self.radius >= self.hyperparameters.max_radius:
            return True, "max_radius"
        self.radius = min(
            self.radius * self.hyperparameters.rate, self.hyperparameters.max_radius
        )
        return False, None

    def _phase_converged(
        self, result: delphi.trainer.RunResult
    ) -> tuple[bool, Optional[str]]:
        """Return whether to stop before widening; never, by default."""
        del result
        return False, None

    def run_metrics(self) -> dict:
        """Return the run's radius and how often and how far it projected."""
        return {
            "radius": self.radius,
            "projections": self._projections,
            "projection_distance": self._projection_distance,
        }
