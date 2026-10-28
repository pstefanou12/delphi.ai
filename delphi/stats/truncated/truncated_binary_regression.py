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
import torch.optim

from delphi import configs
import delphi.projection_sets
from delphi.stats import linear_model
from delphi.stats.truncated.truncated_linear_model import TruncatedLinearModel
from delphi.truncated import sampling

_ClassificationConfigT = TypeVar(
    "_ClassificationConfigT", bound=configs.TruncatedClassificationConfig
)

# Above this radius e^-radius underflows float64, so a set that still
# leaves out the start never will.
_MAX_START_RADIUS = 700.0
_NAIVE_ITERATIONS = 200
# Fraction of the segment a LatentIntervalSet projection stops short of
# its boundary.
_INTERIOR = 1e-6


class TruncatedBinaryRegression(
    TruncatedLinearModel[_ClassificationConfigT], Generic[_ClassificationConfigT]
):
    """Truncated binary regression with a known noise law.

    A subclass sets ``noise`` and ``loss``, the criterion over that noise.

    Attributes:
        sampler: Estimates the per-label masses and scores of the latent.
    """

    noise: sampling.LatentNoise
    loss: type[ch.autograd.Function]

    def __init__(  # pylint: disable=too-many-arguments
        self,
        hyperparameters: dict,
        phi: Callable,
        alpha: Optional[float] = None,
        *,
        sampler: Optional[sampling.LatentSampler] = None,
        fit_intercept: bool = True,
        emp_weight: Optional[ch.Tensor] = None,
        rand_seed: int = 0,
    ):
        """Initialize the model.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation oracle over latents shaped (n, m, 1), returning
                a mask or a survival probability.
            alpha: Lower bound in (0, 1/2] on every training sample's mass
                on each label, ``P(z in S, label y)`` at the truth; None if
                unknown.
            sampler: Estimates the loss's per-label moments; stratified
                sampling at ``num_samples`` by default.
            fit_intercept: Whether to fit an intercept.
            emp_weight: Start in place of the naive fit, which ignores
                phi; see ``TruncatedLinearModel``.
            rand_seed: Seed applied at the start of every run.

        Raises:
            ValueError: If alpha exceeds 1/2, which no two label masses
                summing to at most 1 can both reach.
        """
        if alpha is not None and alpha > 0.5:
            raise ValueError(f"alpha bounds both label masses, so <= 1/2; got {alpha}.")
        super().__init__(
            hyperparameters, phi, alpha, fit_intercept, emp_weight, rand_seed
        )
        self.sampler = sampler or sampling.StratifiedLatentSampler(
            self.hyperparameters.num_samples
        )
        self.criterion = self.loss.apply

    def criterion_parameters(self, X: ch.Tensor, y: ch.Tensor) -> list:  # pylint: disable=invalid-name
        return [self.phi, self.sampler, self.hyperparameters.eps]

    @staticmethod
    def _check_inputs(X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Raise unless y holds both labels 0 and 1, in one column."""
        TruncatedLinearModel._check_inputs(X, y)
        if y.size(1) != 1 or not ((y == 0) | (y == 1)).all():
            raise ValueError("expected labels in {0, 1} shaped (n, 1).")
        if y.min() == y.max():
            raise ValueError("y holds one label; both are needed.")

    def _calc_emp_model(self, X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Set emp_weight to the naive fit unless given, and register it."""
        if self.emp_weight is None:
            self.emp_weight = self._naive_fit(X, y) / self.covariate_scale
        self.register_weight(self.emp_start)

    def _naive_fit(self, X: ch.Tensor, y: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return the untruncated MLE, which ignores phi, shaped (d + 1, 1)."""
        design, sign = X.double(), 2.0 * y.double() - 1.0
        weight = ch.zeros(X.size(1), 1, dtype=ch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weight],
            max_iter=_NAIVE_ITERATIONS,
            tolerance_grad=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure() -> ch.Tensor:
            optimizer.zero_grad()
            loss = -self.noise.log_cdf(sign * (design @ weight)).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        return weight.detach().float()

    def compute_loss(self, batch, criterion=None) -> ch.Tensor:
        """Average the NLL, which is per sample, over the batch."""
        return super().compute_loss(batch, criterion).mean()

    def default_radius(self) -> float:
        """Return ``2 log(1 / alpha)``, or the start's when alpha is unknown.

        R_alpha holds the truth, and so lies on the boundary at alpha =
        alpha*; R_{alpha^2} holds the sample's MLE around it as well.
        """
        if self.alpha is not None:
            return 2.0 * math.log(1.0 / self.alpha)
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        mass = self._label_mass(X @ self.emp_start).min()
        return min(-math.log(max(float(mass), 1e-300)), _MAX_START_RADIUS)

    def _label_mass(self, latent: ch.Tensor) -> ch.Tensor:
        """Return each latent mean's per-label mass, from fixed draws."""
        with ch.no_grad(), ch.random.fork_rng():
            ch.manual_seed(self.rand_seed)
            return self.sampler.label_moments(latent, self.noise, self.phi).mass

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

    def build_projection_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        """Return R_alpha anchored at emp_weight, its radius grown to hold it.

        Raises:
            ValueError: If no radius holds emp_weight, as when phi leaves a
                label no mass at its latent means.
        """
        del y
        anchor = self.emp_start
        projection = LatentIntervalSet(X, anchor, radius, self.latent_interval)
        while not projection.contains(anchor):
            if radius >= _MAX_START_RADIUS:
                raise ValueError(
                    "no R_alpha holds emp_weight: phi leaves a label no mass "
                    "at its latent means."
                )
            radius = min(radius * self.hyperparameters.rate, _MAX_START_RADIUS)
            projection.radius = radius
            self.logger.warning(f"emp_weight lies outside R_alpha; r -> {radius:.4f}.")
        return projection

    def hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the NLL's Fisher information, by default at the iterate."""
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        weight = self.weight.detach() if theta is None else theta
        return self.loss.hessian(  # type: ignore[attr-defined]
            X @ weight.reshape(-1, 1), X, self.phi, self.sampler
        )

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
