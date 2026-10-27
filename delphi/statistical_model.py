# Author: pstefanou12@
"""Statistical models: delphi models whose estimates come from their fit.

A statistical model is fit by the trainer and may then refine that fit with
Newton steps, which neural networks, lacking a tractable Hessian, cannot.
"""

from collections.abc import Callable
from typing import Generic, Optional, TypeVar

import torch as ch
import torch.utils.data

from delphi import configs, trainer
from delphi.delphi import delphi

_StatisticalConfigT = TypeVar(
    "_StatisticalConfigT", bound=configs.StatisticalModelConfig
)


class StatisticalModel(delphi[_StatisticalConfigT], Generic[_StatisticalConfigT]):
    """A delphi model fit by the trainer, then optionally refined.

    A model setting ``refine_steps`` trains one parameter tensor and supplies
    ``hessian``. Its estimates come from ``final_run``.

    Attributes:
        procedure: The trainer's record of the fit, or None before it.
        refinement: The refinement's record, or None if it did not run.
    """

    def __init__(self, hyperparameters: dict):
        """Initialize the model.

        Args:
            hyperparameters: Hyperparameter dict, validated by the subclass
                ``defaults`` config.
        """
        super().__init__(hyperparameters)
        self.procedure: Optional[trainer.ProcedureResult] = None
        self.refinement: Optional[trainer.ProcedureResult] = None

    def hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the Hessian of the criterion at theta.

        Raises:
            NotImplementedError: Unless a subclass supplies it.
        """
        raise NotImplementedError(f"{type(self).__name__} has no hessian().")

    @property
    def final_run(self) -> trainer.RunResult:
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
    ) -> Optional[trainer.ProcedureResult]:
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
            self.refinement = trainer.Trainer(
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
