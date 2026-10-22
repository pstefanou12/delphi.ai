# Author: pstefanou12@
"""Configuration model for the normal distribution under unknown truncation."""

from typing import Literal, Optional

import pydantic

from delphi.configs.base import DelphiConfig


class UnknownTruncationNormalConfig(DelphiConfig):
    """Configuration for the unknown-truncation normal estimators.

    Kontonis, Tzamos and Zampetakis (2019), Algorithm 1: a degree-``degree``
    Hermite expansion of the weighted indicator of the set, then projected
    descent on the convex objective it defines.

    The descent defaults to full-batch L-BFGS rather than the paper's SGD.
    Each term is an exponential of a quadratic, so minibatch gradients are
    heavy-tailed: on a ten-dimensional ball, SGD in batches of 100 ended
    above its starting objective, at KL 0.53 against 0.19 at the minimizer.
    The objective is a fixed average over the descent's samples, smooth and
    well conditioned (condition number about 7 there), so L-BFGS reaches its
    minimizer in a few steps. Converged steps repeat the minimizer, so the
    averaged and moving-average estimates agree with it.

    Attributes:
        degree: Total degree k of the Hermite expansion. Any degree is
            honored, with a warning when its basis has more terms than there
            are samples to estimate them. When None, the largest whose basis
            fits ``max_terms``, up to ``max_degree``. The expansion's bias
            falls with k, but each coefficient adds its own sampling noise:
            on 9000 coefficient samples, a two-dimensional half-space went
            from KL 0.087 at degree 10 to 0.009 at 100 (5151 terms), then
            rose to 0.013 at 128 (8385 terms), where the noise overtook the
            bias. More samples move that point: a four-dimensional ball went
            from 0.121 to 0.092 at degree 26 with four times the samples.
        max_terms: Term budget that sizes the default degree; None is the
            number of samples estimating the coefficients, about one term
            per sample. With 20000 samples that is 9000: degree 1200 (the
            cap) in one dimension, 132 in two, 35 in three, 19 in four and 6
            in ten.
        max_degree: Cap on the default degree. A half-line and an interval
            were still improving at 1200, and no larger degree was measured.
        hermite_fraction: Share of the training samples that estimate the
            Hermite coefficients. The rest drive the descent, which the
            paper draws independently of the expansion.
        min_precision: Smallest eigenvalue the precision matrix may take in
            the whitened frame; from alpha when None.
        max_precision: Largest eigenvalue the precision matrix may take in
            the whitened frame; from alpha when None.
        loc_radius: Bound on the norm of the natural mean parameter in the
            whitened frame; from alpha and max_precision when None.
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    # None is the whole descent sample.
    batch_size: Optional[int] = pydantic.Field(default=None, ge=1)  # type: ignore[assignment]
    val: float = pydantic.Field(default=0.1, ge=0.0, le=1.0)
    record_params_every: int = pydantic.Field(default=1, ge=1)
    iterations: Optional[int] = pydantic.Field(default=100, ge=1)
    optimizer: str = "lbfgs"
    lr: float = pydantic.Field(default=1.0, gt=0.0)
    scheduler: Optional[str] = None
    lbfgs_max_iter: int = pydantic.Field(
        default=50, ge=1, serialization_alias="max_iter"
    )
    lbfgs_line_search_fn: Optional[Literal["strong_wolfe"]] = pydantic.Field(
        default="strong_wolfe", serialization_alias="line_search_fn"
    )

    degree: Optional[int] = pydantic.Field(default=None, ge=1)
    max_terms: Optional[int] = pydantic.Field(default=None, ge=2)
    max_degree: int = pydantic.Field(default=1200, ge=1)
    hermite_fraction: float = pydantic.Field(default=0.5, gt=0.0, lt=1.0)
    min_precision: Optional[float] = pydantic.Field(default=None, gt=0.0)
    max_precision: Optional[float] = pydantic.Field(default=None, gt=0.0)
    loc_radius: Optional[float] = pydantic.Field(default=None, gt=0.0)

    @pydantic.model_validator(mode="after")
    def check_precision_bounds(self) -> "UnknownTruncationNormalConfig":
        """Validate that min_precision does not exceed max_precision."""
        if (
            self.min_precision is not None
            and self.max_precision is not None
            and self.min_precision > self.max_precision
        ):
            raise ValueError(
                f"min_precision ({self.min_precision}) must be <= "
                f"max_precision ({self.max_precision})."
            )
        return self
