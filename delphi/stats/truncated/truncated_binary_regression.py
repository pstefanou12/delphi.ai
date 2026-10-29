# Author: pstefanou12@
"""Truncated binary regression: a latent-variable classifier seen through phi.

The latent ``z = w^T x + b + e`` labels the sample ``y = 1{z >= 0}``, and
the pair (x, y) is kept with probability ``phi(z)``. Probit and logistic
regression differ only in the noise e.
"""

from collections.abc import Callable
import math
from typing import Generic, Optional, TypeVar

import torch as ch

from delphi import configs
import delphi.projection_sets
from delphi.stats.truncated.truncated_classification import (
    TruncatedClassification,
)
from delphi.truncated import sampling

_BinaryConfigT = TypeVar(
    "_BinaryConfigT", bound=configs.TruncatedBinaryRegressionConfig
)

# Fraction of the segment a LatentIntervalSet projection stops short of
# its boundary.
_INTERIOR = 1e-6


class TruncatedBinaryRegression(
    TruncatedClassification[_BinaryConfigT], Generic[_BinaryConfigT]
):
    """Truncated binary regression with a known noise law.

    A subclass sets ``noise`` and ``loss``, the criterion over that noise.
    ``phi`` sees latents shaped (n, m, 1); alpha is at most 1/2.
    """

    noise: sampling.LatentNoise
    sampler: sampling.LatentSampler
    default_sampler = sampling.StratifiedLatentSampler

    @staticmethod
    def _check_inputs(X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Raise unless y holds both labels 0 and 1, in one column."""
        TruncatedClassification._check_inputs(X, y)
        if y.size(1) != 1 or not ((y == 0) | (y == 1)).all():
            raise ValueError("expected labels in {0, 1} shaped (n, 1).")
        if y.min() == y.max():
            raise ValueError("y holds one label; both are needed.")

    def _untruncated_nll(self, latent: ch.Tensor, y: ch.Tensor) -> ch.Tensor:
        return -self.noise.log_cdf((2.0 * y.double() - 1.0) * latent).mean()

    def _label_mass(self, linear: ch.Tensor) -> ch.Tensor:
        """Return each latent mean's per-label mass, from fixed draws."""
        with ch.no_grad(), ch.random.fork_rng():
            ch.manual_seed(self.rand_seed)
            return self.sampler.label_moments(linear, self.noise, self.phi).mass

    def latent_interval(self, radius: float) -> tuple[float, float]:
        """Return the hull of the latent means keeping e^-radius per label.

        Each label keeps at most its noise mass, so the search covers the
        means at which both noise masses reach e^-radius.

        Args:
            radius: The set's ``log(1 / alpha')``.

        Returns:
            (lower, upper), one grid step wider than the feasible grid
            points; (inf, -inf) when none is feasible.
        """
        floor = math.exp(-radius)
        if floor >= 0.5:
            return math.inf, -math.inf
        reach = -float(self.noise.icdf(ch.tensor(floor, dtype=ch.float64)))
        grid = ch.linspace(
            -reach, reach, self.hyperparameters.interval_grid, dtype=ch.float64
        )
        feasible = (self._label_mass(grid[:, None]).min(1).values >= floor).nonzero()
        if feasible.numel() == 0:
            return math.inf, -math.inf
        step = float(grid[1] - grid[0])
        return (
            max(float(grid[feasible.min()]) - step, -reach),
            min(float(grid[feasible.max()]) + step, reach),
        )

    def _projection_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        anchor: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        return LatentIntervalSet(X, anchor, radius, self.latent_interval)

    def predict_proba(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return ``P(y = 1 | x)`` without truncation, shaped (n, 1).

        Args:
            X: Covariates, shaped (n, d).
        """
        return self.noise.cdf(super().predict(X))

    def predict(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return the more likely label without truncation, shaped (n, 1).

        Args:
            X: Covariates, shaped (n, d).
        """
        return (super().predict(X) >= 0).float()


class LatentIntervalSet(delphi.projection_sets.ProjectionSet):
    """Weights whose latent means ``x_i^T w`` all lie in one interval.

    The interval is a function of the radius, recomputed when it changes;
    R_alpha, the weights at which every sample keeps mass alpha on each
    label, takes this form. The projection is exact along the segment to
    the anchor, so it skips the bisection.

    Attributes:
        lower: The interval's lower end at the current radius.
        upper: The interval's upper end at the current radius.
    """

    def __init__(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        anchor: ch.Tensor,
        radius: float,
        interval: Callable[[float], tuple[float, float]],
    ):
        """Build the set over the training covariates.

        Args:
            X: Design matrix, shaped (n, d).
            anchor: Weight of shape (d, 1) to retract towards.
            radius: The set's size, which interval maps to its bounds.
            interval: Maps a radius to the (lower, upper) bounds on every
                latent mean; lower > upper describes the empty set.
        """
        self._design = X.double()
        self._interval = interval
        self.lower, self.upper = interval(radius)
        super().__init__(anchor, radius)

    @property
    def radius(self) -> Optional[float]:
        """The set's size; setting it recomputes the interval."""
        return self._radius

    @radius.setter
    def radius(self, value: Optional[float]) -> None:
        self._radius = value
        if value is not None:
            self.lower, self.upper = self._interval(value)

    def contains(self, weight: ch.Tensor) -> bool:
        latent = self._design @ weight.detach().double().reshape(-1, 1)
        return bool(((latent >= self.lower) & (latent <= self.upper)).all())

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        if self.contains(weight):
            return weight
        start = self._design @ self.anchor.double().reshape(-1, 1)
        step = self._design @ (weight.detach() - self.anchor).double().reshape(-1, 1)
        bound = ch.where(step > 0, self.upper, self.lower)
        ratio = ch.where(step != 0, (bound - start) / step, math.inf)
        # Inside the boundary, so rounding cannot leave the result outside.
        reach = (ratio.min().clamp(0.0, 1.0) * (1.0 - _INTERIOR)).item()
        return self.anchor + reach * (weight.detach() - self.anchor)
