# Author: pstefanou12@
"""Configuration models for the truncated regression estimators."""

from typing import Optional

import pydantic

from delphi.configs.stats.truncated.linear import TruncatedLinearModelConfig


class TruncatedRegressionConfig(TruncatedLinearModelConfig):
    """Configuration for truncated linear regression with known variance.

    One shuffled pass, one small batch per step, steps of 1 / (lambda i)
    from the Fisher schedule, and every iterate averaged; Newton steps on
    the full sample then take that average to the empirical MLE.

    Attributes:
        weight_bound: The B of the projection set, on the weight in the
            normalized coordinates the descent runs in; None drops the ball.
        refine_steps: Newton steps after the descent, as
            ``delphi.configs.base.RefinementConfig``; on by default here.
        information_samples: Kept draws per sample for the conditional
            moments behind ``covariance``'s information matrix; a sample's
            moments from m draws shrink its variances by about 1 / m.
    """

    batch_size: int = 10
    epochs: Optional[int] = pydantic.Field(default=1, ge=1)
    val: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    scheduler: Optional[str] = "fisher"
    fisher_refresh_every: int = pydantic.Field(
        default=0, ge=0, serialization_alias="refresh_every"
    )
    avg_burn_in: float = pydantic.Field(default=0.0, ge=0.0, lt=1.0)
    weight_bound: Optional[float] = pydantic.Field(default=None, gt=0.0)
    refine_steps: int = pydantic.Field(default=30, ge=0)
    information_samples: int = pydantic.Field(default=1000, ge=2)

    @pydantic.model_validator(mode="after")
    def drop_default_schedule_off_sgd(self) -> "TruncatedRegressionConfig":
        """Leave a non-SGD optimizer unscheduled unless a schedule is set.

        The 1 / (lambda i) schedule is SGD's; L-BFGS and Adam scale their
        own steps, and decaying them as well stalls the descent.
        """
        if self.optimizer.lower() != "sgd" and "scheduler" not in (
            self.model_fields_set
        ):
            object.__setattr__(self, "scheduler", None)
        return self


class TruncatedUnknownVarianceRegressionConfig(TruncatedRegressionConfig):
    """Configuration for truncated linear regression with the variance learned.

    The descent runs a fixed number of gradient steps rather than passes
    over the data, and converges on its own: its tail average lands far
    closer to the MLE's likelihood than the true parameters do, so the
    sampled Newton steps would only add noise.

    Attributes:
        batch_size: Larger than the known variance's: the precision's
            gradient ``(y^2 - E[z^2 | S]) / 2`` is noisy enough at 10 to
            jitter it by about 2%.
        iterations: Gradient steps of the descent.
        avg_burn_in: Half, since the descent starts at OLS, far from the
            optimum along the precision, and an average over every iterate
            keeps that transient.
        refine_steps: Off; see above.
        refine_damped: On, for refine_steps set, since a full Newton step
            can overshoot lambda to where phi keeps almost no mass.
        variance_lr_scale: The precision's learning rate as a multiple of
            the weight's, at every step of the descent's schedule.
    """

    batch_size: int = 50
    epochs: Optional[int] = pydantic.Field(default=None, ge=1)
    iterations: Optional[int] = pydantic.Field(default=5000, ge=1)
    avg_burn_in: float = pydantic.Field(default=0.5, ge=0.0, lt=1.0)
    refine_steps: int = pydantic.Field(default=0, ge=0)
    refine_damped: bool = True
    variance_lr_scale: float = pydantic.Field(default=1.0, gt=0.0)


class TruncatedLinearDynamicalSystemConfig(TruncatedRegressionConfig):
    """Configuration for a truncated linear dynamical system.

    One pass over the pairs in temporal order, one pair per step, at a
    constant step: an iterate must not have seen a later pair, or its
    gradient is biased.

    Attributes:
        c_s: Sets ``s = c_s (sqrt(log(1 / alpha)) + 1)``; the warm-up's
            covariance divided by s bounds the confidence ellipsoid.
        c_eta: The step is ``(2 / alpha) ** c_eta``.
        c_gamma: A step is censor-aware when its sampled survival mass
            reaches ``2 gamma``, for ``gamma = (alpha / 2) ** c_gamma``.
        warm_start_fraction: Share of the leading pairs fit by least squares.
    """

    c_s: float = pydantic.Field(default=10.0, gt=0.0)
    c_eta: float = pydantic.Field(default=0.5, ge=0.0)
    c_gamma: float = pydantic.Field(default=2.0, gt=0.0)
    warm_start_fraction: float = pydantic.Field(default=0.25, gt=0.0, lt=1.0)

    batch_size: int = 1
    epochs: Optional[int] = pydantic.Field(default=1, ge=1)
    val: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    shuffle: bool = False
    constant: bool = True
    scheduler: Optional[str] = None
    avg_burn_in: float = pydantic.Field(default=0.5, ge=0.0, lt=1.0)
    refine_steps: int = pydantic.Field(default=0, ge=0)


class TruncatedLassoRegressionConfig(TruncatedLinearModelConfig):
    """Configuration for truncated lasso regression.

    One sample per step at a constant step size, as their Algorithm 1, on
    the unscaled design matrix. The iterates after the first half are
    averaged, since the descent starts at the smallest-L1 point of E_r,
    which is often zero.
    """

    batch_size: int = 1
    iterations: Optional[int] = pydantic.Field(default=20000, ge=1)
    val: float = pydantic.Field(default=0.0, ge=0.0, le=1.0)
    constant: bool = True
    scale_covariates: bool = False
