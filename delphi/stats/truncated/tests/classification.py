# Author: pstefanou12@
"""Test harness for the truncated classifiers.

The binary latent ``z = w^T x + b + e`` labels a sample ``y = 1{z >= 0}``,
and the multinomial latents ``z_k = w_k^T x + b_k + g_k``, with Gumbel g
and class 0's weights zero, label it ``y = argmax_k z_k``; either is kept
with probability ``phi(z)``. ``simulate`` and ``simulate_multinomial`` draw
such a sample with known parameters, and ``assert_recovers`` fits a model
to it and checks every estimate the model returns against the truth.
"""

from collections.abc import Callable, Sequence
import dataclasses
import math
from typing import Optional

import torch as ch
import torch.distributions
import torch.nn.functional

from delphi.stats import linear_model
from delphi.truncated import sampling

# Draws per block while collecting survivors.
_BLOCK = 100_000


class PiecewiseSurvival:
    """A truncation oracle keeping a latent with a probability per interval.

    The probability is constant on each of a few disjoint intervals and zero
    elsewhere, so each label's survival mass has a closed form.

    Attributes:
        pieces: (low, high, probability) per interval.
    """

    def __init__(self, *pieces: tuple[float, float, float]):
        """Initialize the oracle.

        Args:
            *pieces: Disjoint intervals (low, high) and the probability a
                latent in each survives; elsewhere none does.
        """
        self.pieces = pieces

    def __call__(self, z: ch.Tensor) -> ch.Tensor:
        """Return a mask, or survival probabilities if any is below 1."""
        survival = ch.zeros_like(z)
        for low, high, probability in self.pieces:
            survival = ch.where((z > low) & (z < high), probability, survival)
        if all(probability == 1.0 for _, _, probability in self.pieces):
            return survival > 0.0
        return survival

    def label_mass(self, latent: ch.Tensor, noise: sampling.LatentNoise) -> ch.Tensor:
        """Return ``P(z in S, label c)`` for ``z = latent + e``, shaped (n, 2).

        Args:
            latent: Latent means, shaped (n, 1).
            noise: The law of e.
        """
        columns = []
        for label_low, label_high in ((-math.inf, 0.0), (0.0, math.inf)):
            mass = ch.zeros_like(latent)
            for low, high, probability in self.pieces:
                start, end = max(low, label_low), min(high, label_high)
                if start < end:
                    mass = mass + probability * _interval_mass(
                        start, end, latent, noise
                    )
            columns.append(mass)
        return ch.cat(columns, dim=1)


def _interval_mass(start, end, latent, noise) -> ch.Tensor:
    """Return ``P(start < latent + e < end)``, accurate in either tail."""
    # An infinite end would give the unused branch of where a nan gradient.
    if math.isinf(start) and math.isinf(end):
        return ch.ones_like(latent)
    if math.isinf(start):
        return noise.cdf(end - latent)
    if math.isinf(end):
        return noise.cdf(latent - start)
    upper_tail = noise.cdf(latent - start) - noise.cdf(latent - end)
    lower_tail = noise.cdf(end - latent) - noise.cdf(start - latent)
    return ch.where(latent < start, upper_tail, lower_tail)


class CoordinateSurvival:
    """A truncation oracle on one coordinate of a Gumbel latent vector.

    Piecewise in that coordinate as ``PiecewiseSurvival``. With the others
    maximized out in closed form, so is each label's survival mass.

    Attributes:
        coordinate: The latent coordinate phi reads.
        survival: The oracle on that coordinate.
    """

    def __init__(self, coordinate: int, *pieces: tuple[float, float, float]):
        """Initialize the oracle.

        Args:
            coordinate: The latent coordinate phi reads.
            *pieces: As ``PiecewiseSurvival``.
        """
        self.coordinate = coordinate
        self.survival = PiecewiseSurvival(*pieces)

    def __call__(self, z: ch.Tensor) -> ch.Tensor:
        """Return a mask, or survival probabilities, shaped z's batch dims."""
        return self.survival(z[..., self.coordinate])

    def label_mass(self, logits: ch.Tensor) -> ch.Tensor:
        """Return ``P(z in S, argmax z = c)`` for ``z = logits + g``.

        With j the coordinate read, p the softmax and A the set, label j
        keeps ``p_j P(Gumbel(logsumexp) in A)`` and another label c a share
        ``p_c / (1 - p_j)`` of ``P(z_j in A)`` less that.

        Args:
            logits: Latent means, shaped (n, K).

        Returns:
            The masses, shaped (n, K).
        """
        j = self.coordinate
        prob = ch.softmax(logits, dim=1)
        top = ch.logsumexp(logits, dim=1, keepdim=True)
        winner = ch.zeros_like(top)
        own = ch.zeros_like(top)
        for low, high, probability in self.survival.pieces:
            winner = winner + probability * _gumbel_mass(low, high, top)
            own = own + probability * _gumbel_mass(low, high, logits[:, j : j + 1])
        share = prob / (1.0 - prob[:, j : j + 1])
        mass = share * (own - prob[:, j : j + 1] * winner)
        return ch.cat([mass[:, :j], prob[:, j : j + 1] * winner, mass[:, j + 1 :]], 1)


def _gumbel_mass(low: float, high: float, loc: ch.Tensor) -> ch.Tensor:
    """Return ``P(low < loc + g < high)`` for standard Gumbel g, in either tail."""

    def cdf(t):
        return ch.exp(-ch.exp(-t))

    def survival(t):
        return -ch.expm1(-ch.exp(-t))

    if math.isinf(low) and math.isinf(high):
        return ch.ones_like(loc)
    if math.isinf(low):
        return cdf(high - loc)
    if math.isinf(high):
        return survival(low - loc)
    upper_tail = survival(low - loc) - survival(high - loc)
    return ch.where(loc < low, upper_tail, cdf(high - loc) - cdf(low - loc))


@dataclasses.dataclass(frozen=True)
class Simulation:
    """A truncated classification sample and the parameters that produced it.

    Attributes:
        X: Observed covariates, shaped (n, d).
        y: Observed labels, shaped (n, 1).
        coef: True coefficients, shaped (k, d).
        intercept: True intercept, shaped (k,).
        phi: The truncation the sample passed.
        alpha: The smallest per-label mass of any observed sample at the
            truth, the alpha of R_alpha.
    """

    X: ch.Tensor  # pylint: disable=invalid-name
    y: ch.Tensor
    coef: ch.Tensor
    intercept: ch.Tensor
    phi: Callable
    alpha: float

    def error(self, coef: ch.Tensor, intercept: Optional[ch.Tensor]) -> float:
        """Return the L2 distance of (coef, intercept) from the truth."""
        estimate = coef.flatten()
        truth = self.coef.flatten()
        if intercept is not None:
            estimate = ch.cat([estimate, intercept.flatten()])
            truth = ch.cat([truth, self.intercept])
        return float((estimate - truth).norm())


def _survivors(
    latent: Callable[[ch.Tensor], ch.Tensor],
    phi: Callable,
    n: int,
    covariance: ch.Tensor,
    seed: int,
) -> tuple[ch.Tensor, ch.Tensor]:
    """Draw x ~ N(0, covariance) and its latent until n pass phi.

    Returns:
        The first n survivors' covariates and latents.
    """
    ch.manual_seed(seed)
    sampler = ch.distributions.MultivariateNormal(
        ch.zeros(covariance.size(0)), covariance
    )
    kept_X: list[ch.Tensor] = []  # pylint: disable=invalid-name
    kept_z: list[ch.Tensor] = []
    survived = 0
    while survived < n:
        X = sampler.sample([_BLOCK])  # pylint: disable=invalid-name
        z = latent(X)
        kept = phi(z[:, None, :]).reshape(-1)
        kept = ch.bernoulli(kept).bool() if kept.is_floating_point() else kept
        kept_X.append(X[kept])
        kept_z.append(z[kept])
        survived += int(kept.sum())
    return ch.cat(kept_X)[:n], ch.cat(kept_z)[:n]


def simulate(  # pylint: disable=too-many-arguments
    coef: Sequence[float],
    intercept: float,
    phi: PiecewiseSurvival,
    n: int,
    *,
    noise: sampling.LatentNoise,
    covariance: Optional[ch.Tensor] = None,
    seed: int = 0,
) -> Simulation:
    """Draw ``y = 1{coef^T x + intercept + e >= 0}`` until n samples pass phi.

    Args:
        coef: True coefficients.
        intercept: True intercept.
        phi: The truncation, applied to the latent.
        n: Survivors to collect.
        noise: The law of e.
        covariance: Covariance of x ~ N(0, covariance); identity if None.
        seed: Seed for the draws.

    Returns:
        The first n survivors.
    """
    weight = ch.tensor(coef)[:, None]

    def latent(X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        uniform = ch.rand(X.size(0), 1, dtype=ch.float64).clamp(1e-15, 1.0 - 1e-15)
        return X @ weight + intercept + noise.icdf(uniform).float()

    covariance = ch.eye(weight.size(0)) if covariance is None else covariance
    X, z = _survivors(latent, phi, n, covariance, seed)  # pylint: disable=invalid-name
    mean = (X @ weight + intercept).double()
    return Simulation(
        X=X,
        y=(z >= 0).float(),
        coef=weight.T,
        intercept=ch.tensor([intercept]),
        phi=phi,
        alpha=float(phi.label_mass(mean, noise).min()),
    )


def simulate_multinomial(  # pylint: disable=too-many-arguments
    coef: Sequence[Sequence[float]],
    intercept: Sequence[float],
    phi: CoordinateSurvival,
    n: int,
    *,
    covariance: Optional[ch.Tensor] = None,
    seed: int = 0,
) -> Simulation:
    """Draw ``y = argmax_k (w_k^T x + b_k + g_k)`` until n samples pass phi.

    Args:
        coef: True coefficients of classes 1 to K - 1, shaped (K - 1, d);
            class 0's are zero.
        intercept: True intercepts of classes 1 to K - 1.
        phi: The truncation, applied to the latent vector.
        n: Survivors to collect.
        covariance: Covariance of x ~ N(0, covariance); identity if None.
        seed: Seed for the draws.

    Returns:
        The first n survivors, labeled in {0, ..., K - 1}.
    """
    weight, bias = ch.tensor(coef), ch.tensor(intercept)

    def logits(X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        return torch.nn.functional.pad(X @ weight.T + bias, (1, 0))

    def latent(X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        mean = logits(X)
        return mean + torch.distributions.Gumbel(0.0, 1.0).sample(mean.shape)

    covariance = ch.eye(weight.size(1)) if covariance is None else covariance
    X, z = _survivors(latent, phi, n, covariance, seed)  # pylint: disable=invalid-name
    return Simulation(
        X=X,
        y=z.argmax(1, keepdim=True),
        coef=weight,
        intercept=bias,
        phi=phi,
        alpha=float(phi.label_mass(logits(X).double()).min()),
    )


def assert_recovers(
    model,
    simulation: Simulation,
    tolerance: float,
    *,
    naive_margin: Optional[float] = 0.5,
) -> None:
    """Fit a classifier to a simulation and check each estimate it returns.

    Args:
        model: An unfitted truncated classifier, exposing
            ``{avg,ema,final,best,emp}_coef_`` and ``_intercept_``.
        simulation: The sample, with its truth.
        tolerance: Largest L2 error allowed for (coef, intercept).
        naive_margin: Each estimate's error must also be below this fraction
            of the naive fit's, the model's start, which ignores phi; None
            skips the comparison, as without truncation.
    """
    model.fit(simulation.X, simulation.y)
    naive = simulation.error(model.emp_coef_, model.emp_intercept_)
    for name in [*linear_model.ESTIMATORS, ""]:
        prefix = f"{name}_" if name else ""
        coef = getattr(model, f"{prefix}coef_")
        intercept = getattr(model, f"{prefix}intercept_")
        error = simulation.error(coef, intercept)
        label = f"{prefix}coef_ and {prefix}intercept_"
        assert error <= tolerance, (
            f"{label}: error {error:.4f} above {tolerance} (naive {naive:.4f})"
        )
        if naive_margin is not None:
            assert error <= naive_margin * naive, (
                f"{label}: error {error:.4f} not below {naive_margin} of "
                f"the naive fit's {naive:.4f}"
            )
