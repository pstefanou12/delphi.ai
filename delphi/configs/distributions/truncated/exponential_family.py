# Author: pstefanou12@
"""Configuration models for truncated distribution algorithms."""

from typing import Optional

import pydantic

from delphi.configs.base import DelphiConfig


class TruncatedExponentialFamilyDistributionConfig(DelphiConfig):
    """Configuration for truncated exponential family distribution algorithms.

    Attributes:
        eps: Numerical stability constant for the NLL criterion.
        min_radius: Initial NLL budget above the empirical initialization
            for the sublevel-set projection (phase 1).
        max_radius: Maximum NLL budget; the procedure stops when reached.
        rate: Multiplicative budget expansion factor per phase.
        batch_size: Mini-batch size for training.
        num_samples: Monte Carlo samples drawn per NLL evaluation.
        max_phases: Maximum number of radius-expansion phases.
        loss_convergence_tol: Absolute loss improvement threshold for
            stopping between phases.
        relative_loss_tol: Relative loss improvement threshold between phases.
        loss_increase_tol: Loss increase threshold for detecting overshoot.
        project: Enable per-step sublevel-set projection.

    The training defaults are the values the test suite settled on across
    families: 2500 iterations at a learning rate of 1e-2 under the Fisher
    schedule, with 1000 rejection samples per gradient step.
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    batch_size: int = 10
    tol: float = pydantic.Field(default=1e-1, ge=0.0)
    record_params_every: int = pydantic.Field(default=1, ge=1)
    epochs: Optional[int] = pydantic.Field(default=None, ge=1)
    iterations: Optional[int] = pydantic.Field(default=2500, ge=1)
    lr: float = pydantic.Field(default=1e-2, gt=0.0)
    scheduler: Optional[str] = "fisher"

    eps: float = pydantic.Field(default=1e-5, gt=0.0)
    min_radius: float = pydantic.Field(default=3.0, ge=0.0)
    max_radius: float = pydantic.Field(default=10.0, ge=0.0)
    rate: float = pydantic.Field(default=1.1, gt=1.0)
    num_samples: int = pydantic.Field(default=1000, ge=1)
    max_phases: int = pydantic.Field(default=1, ge=1)
    loss_convergence_tol: float = pydantic.Field(default=1e-3, ge=0.0)
    relative_loss_tol: float = pydantic.Field(default=float("inf"), ge=0.0)
    loss_increase_tol: float = pydantic.Field(default=float("inf"), ge=0.0)
    project: bool = True

    @pydantic.model_validator(mode="after")
    def check_radius(self) -> "TruncatedExponentialFamilyDistributionConfig":
        """Validate that min_radius does not exceed max_radius."""
        if self.min_radius > self.max_radius:
            raise ValueError(
                f"min_radius ({self.min_radius}) must be <= "
                f"max_radius ({self.max_radius})."
            )
        return self

    @pydantic.model_validator(mode="after")
    def resolve_epochs_iterations(
        self,
    ) -> "TruncatedExponentialFamilyDistributionConfig":
        """Clear the default iterations when epochs is explicitly provided.

        The trainer uses exactly one stopping criterion, so a caller choosing
        epochs gets epochs rather than both.
        """
        if (
            "epochs" in self.model_fields_set
            and "iterations" not in self.model_fields_set
        ):
            object.__setattr__(self, "iterations", None)
        return self
