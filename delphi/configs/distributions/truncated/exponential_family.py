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
        rate: Multiplicative budget expansion factor per phase, applied
            only when alpha is unknown.
        batch_size: Mini-batch size for training.
        num_samples: Monte Carlo samples drawn per NLL evaluation.
        project: Enable per-step sublevel-set projection.
        phase_loss_tol: Stop widening the budget when a phase improves the
            best loss by less than this; None disables it.
        phase_relative_loss_tol: As phase_loss_tol, relative to the previous
            phase's best loss; None disables it.
        phase_loss_increase_tol: Stop widening the budget when a phase raises
            the best loss by more than this; None disables it.

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
    project: bool = True
    phase_loss_tol: Optional[float] = pydantic.Field(default=1e-3, ge=0.0)
    phase_relative_loss_tol: Optional[float] = pydantic.Field(default=None, ge=0.0)
    phase_loss_increase_tol: Optional[float] = pydantic.Field(default=None, ge=0.0)

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


class WarmStartedExponentialFamilyDistributionConfig(
    TruncatedExponentialFamilyDistributionConfig
):
    """Configuration for families whose untruncated MLE has no closed form.

    Their rough start is refined to the untruncated MLE first, by a warm
    start of ``warm_start_iterations`` steps.
    """

    warm_start_iterations: Optional[int] = pydantic.Field(
        default=1000, ge=1, serialization_alias="iterations"
    )
