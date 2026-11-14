# Author: pstefanou12@
"""Configuration models for statistical models and their refinement."""

import pydantic

from delphi.configs.base import DelphiConfig


class RefinementConfig(pydantic.BaseModel):
    """Configuration of a statistical model's refinement, after its fit.

    Setting ``refine_steps`` enables it. From the fit's averaged iterate,
    each step is ``theta - H^-1 grad`` on the full sample, with ``H`` the
    model's ``hessian`` at the current iterate; see
    ``StatisticalModel.refine``.

    Attributes:
        refine_steps: Newton steps; 0 skips the stage.
        refine_avg_burn_in: Fraction of the steps left out of the average,
            which cancels the Monte Carlo noise in a sampled gradient.
        refine_damped: Shrink each step by ``1 / (1 + delta)``, for delta
            the Newton decrement ``sqrt(grad^T H^-1 grad)``: Nesterov's
            damped Newton, which converges from a start where full steps
            overshoot.
    """

    refine_steps: int = pydantic.Field(default=0, ge=0)
    refine_avg_burn_in: float = pydantic.Field(default=0.5, ge=0.0, lt=1.0)
    refine_damped: bool = False


class StatisticalModelConfig(DelphiConfig, RefinementConfig):
    """Base config of the statistical models, which may refine their fit."""

    def refinement_overlay(self) -> "StatisticalModelConfig":
        """Return this config as the refinement's: unit-rate, unscheduled SGD.

        Returns:
            A copy that takes ``refine_steps`` Newton steps.
        """
        return self.model_copy(
            update={
                "optimizer": "sgd",
                "lr": 1.0,
                "sgd_momentum": 0.0,
                "weight_decay": 0.0,
                "scheduler": None,
                "constant": True,
                "max_grad_norm": None,
                "epochs": None,
                "iterations": self.refine_steps,
                "avg_burn_in": self.refine_avg_burn_in,
                "max_phases": 1,
                "warm_start_iterations": None,
                "warm_start_epochs": None,
            }
        )
