# Author: pstefanou12@
"""Statistical models: delphi models whose estimates come from their fit.

A statistical model is fit by the trainer and may then refine that fit with
Newton steps, which neural networks, lacking a tractable Hessian, cannot.
``PSGDModel`` fits by projected SGD, in phases that widen the projection
set until it stops binding.
"""

import abc
from collections.abc import Callable
from typing import ClassVar, Generic, Optional, TypeVar

import torch as ch
import torch.utils.data

from delphi.configs import base
from delphi.delphi import DelphiModel
import delphi.projection_sets
import delphi.trainer

_StatisticalConfigT = TypeVar("_StatisticalConfigT", bound=base.StatisticalModelConfig)
_PSGDConfigT = TypeVar("_PSGDConfigT", bound=base.PSGDModelConfig)


class StatisticalModel(DelphiModel[_StatisticalConfigT], Generic[_StatisticalConfigT]):
    """A delphi model fit by the trainer, then optionally refined.

    A model setting ``refine_steps`` trains one parameter tensor and supplies
    ``hessian``. Its estimates come from ``final_run``.

    Attributes:
        refinement: The refinement's record, or None if it did not run.
    """

    def __init__(self, *, hyperparameters: Optional[dict] = None):
        """Initialize the model.

        Args:
            hyperparameters: Hyperparameter dict, validated by the subclass
                ``defaults`` config.
        """
        super().__init__(hyperparameters=hyperparameters)
        self.refinement: Optional[delphi.trainer.ProcedureResult] = None

    def hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the Hessian of the criterion at theta.

        Raises:
            NotImplementedError: Unless a subclass supplies it.
        """
        raise NotImplementedError(f"{type(self).__name__} has no hessian().")

    @property
    def final_run(self) -> delphi.trainer.RunResult:
        """The run the estimates come from: the refinement's, else the fit's.

        Raises:
            RuntimeError: If the model has not been fit.
        """
        record = self.refinement or self.procedure
        if record is None:
            raise RuntimeError(f"{type(self).__name__} has not been fit.")
        return record.final

    def refine(
        self, train_loader: torch.utils.data.DataLoader, rand_seed: int = 0
    ) -> Optional[delphi.trainer.ProcedureResult]:
        """Take ``refine_steps`` Newton steps on the full sample.

        Starts from the fit's averaged iterate. Each step moves by
        ``H^-1 grad``, with ``H`` the Hessian at the current iterate, damped
        when ``refine_damped`` is set; the model's own hooks, such as a
        projection, still apply.

        Args:
            train_loader: The fit's training loader, whose dataset the steps
                take in one batch.
            rand_seed: Seed applied at the start of the refinement.

        Returns:
            The refinement's record, or None when ``refine_steps`` is 0.

        Raises:
            RuntimeError: If the model has not been fit.
            ValueError: If the model trains more than one parameter tensor.
        """
        self.refinement = None
        main = self.hyperparameters
        if main.refine_steps == 0:
            return None
        if self.procedure is None:
            raise RuntimeError(f"{type(self).__name__} has not been fit.")
        params = [p for p in self.parameters() if p.requires_grad]
        if len(params) != 1:
            raise ValueError(
                f"the refinement steps one parameter tensor; {self} trains "
                f"{len(params)}."
            )
        (param,) = params
        start = self.procedure.final.avg_params
        if start is not None:
            with ch.no_grad():
                param.copy_(start.reshape(param.shape))
        dataset = train_loader.dataset
        full_batch = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),  # type: ignore[arg-type]
            collate_fn=train_loader.collate_fn,
        )
        # A tensor hook runs after backward, inside optimizer.step's closure.
        hook = param.register_hook(self._newton_direction(param))
        try:
            self.refinement = delphi.trainer.Trainer(
                self, main.refinement_overlay()
            ).train_model(full_batch, None, rand_seed=rand_seed)
        finally:
            hook.remove()
            self.hyperparameters = main
        return self.refinement

    def _newton_direction(self, param: ch.Tensor) -> Callable[[ch.Tensor], ch.Tensor]:
        """Return a gradient hook mapping grad to H^-1 grad at param."""

        def direction(grad: ch.Tensor) -> ch.Tensor:
            hess = self.hessian(param.detach())
            flat = grad.reshape(-1, 1).to(hess.dtype)
            step = ch.linalg.solve(hess, flat)  # pylint: disable=not-callable
            if self.hyperparameters.refine_damped:
                step = step / (1.0 + float((flat * step).sum().clamp_min(0.0).sqrt()))
            return step.reshape(grad.shape).to(grad.dtype)

        return direction


class PSGDModel(StatisticalModel[_PSGDConfigT], Generic[_PSGDConfigT], abc.ABC):
    """Projected SGD in phases, each widening the projection set's radius.

    The first phase builds the set from ``build_projection_set``. With alpha
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
        self._projections, self._projection_distance = 0, 0.0

    @abc.abstractmethod
    def build_first_projection_set(self) -> delphi.projection_sets.ProjectionSet:
        """Return the projection set of the first phase, at its first radius."""

    def pre_phase_hook(self) -> None:
        """Build the projection set on the first phase, resize it after."""
        self._projections, self._projection_distance = 0, 0.0
        if self.projection_set is not None:
            self.projection_set.radius = self.radius
            return
        self.projection_set = self.build_first_projection_set()
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
