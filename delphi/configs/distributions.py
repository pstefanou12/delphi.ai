# Author: pstefanou12@
"""Configuration models for truncated distribution algorithms."""

from typing import Optional

import pydantic

from delphi.configs.base import OptimizerConfig, TrainerConfig


class TruncatedExponentialFamilyDistributionConfig(TrainerConfig, OptimizerConfig):
    """Configuration for truncated exponential family distribution algorithms.

    Attributes:
        val: Fraction of data held out for validation.
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
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    # Override parent defaults for distribution training.
    tol: float = pydantic.Field(default=1e-1, ge=0.0)
    record_params_every: int = pydantic.Field(default=1, ge=1)
    epochs: Optional[int] = pydantic.Field(default=1, ge=1)

    # Distribution-specific fields.
    val: float = pydantic.Field(default=0.2, ge=0.0, le=1.0)
    eps: float = pydantic.Field(default=1e-5, gt=0.0)
    min_radius: float = pydantic.Field(default=3.0, ge=0.0)
    max_radius: float = pydantic.Field(default=10.0, ge=0.0)
    rate: float = pydantic.Field(default=1.1, gt=1.0)
    batch_size: int = pydantic.Field(default=10, ge=1)
    num_samples: int = pydantic.Field(default=10000, ge=1)
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
        """Clear the default epochs when iterations is explicitly provided.

        The trainer uses exactly one stopping criterion. When the user
        supplies iterations, the per-phase epochs default is cleared so
        the trainer stops on iterations instead.
        """
        if (
            "iterations" in self.model_fields_set
            and "epochs" not in self.model_fields_set
        ):
            object.__setattr__(self, "epochs", None)
        return self


class TruncatedMultivariateNormalConfig(TruncatedExponentialFamilyDistributionConfig):
    """Configuration for truncated multivariate normal distributions.

    Attributes:
        eigenvalue_lower_bound: Minimum eigenvalue enforced during the
            negative-definite cone projection of the precision matrix T.
        covariance_matrix_lr: Optional separate learning rate for the
            covariance matrix parameter; falls back to lr when None.
    """

    eigenvalue_lower_bound: float = pydantic.Field(default=1e-2, gt=0.0)
    covariance_matrix_lr: Optional[float] = pydantic.Field(default=None, gt=0.0)
