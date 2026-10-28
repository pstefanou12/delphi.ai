# Author: pstefanou12@
"""Test harness for the truncated binary classifiers.

The latent ``z = w^T x + b + e`` labels a sample ``y = 1{z >= 0}``, kept
with probability ``phi(z)``. ``simulate`` draws such a sample with known
parameters, and ``assert_recovers`` fits a model to it and checks every
estimate the model returns against the truth.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Optional

import torch as ch

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


@dataclasses.dataclass(frozen=True)
class Simulation:  # pylint: disable=too-many-instance-attributes
    """A truncated classification sample and the parameters that produced it.

    Attributes:
        X: Observed covariates, shaped (n, d).
        y: Observed labels, shaped (n, 1).
        coef: True coefficients, shaped (1, d).
        intercept: True intercept, shaped (1,).
        phi: The truncation the sample passed.
        noise: The law of the latent noise.
        alpha: The smallest per-label mass of any observed sample at the
            truth, the alpha of R_alpha.
    """

    X: ch.Tensor  # pylint: disable=invalid-name
    y: ch.Tensor
    coef: ch.Tensor
    intercept: ch.Tensor
    phi: PiecewiseSurvival
    noise: sampling.LatentNoise
    alpha: float

    def error(self, coef: ch.Tensor, intercept: Optional[ch.Tensor]) -> float:
        """Return the L2 distance of (coef, intercept) from the truth."""
        estimate = coef.flatten()
        truth = self.coef.flatten()
        if intercept is not None:
            estimate = ch.cat([estimate, intercept.flatten()])
            truth = ch.cat([truth, self.intercept])
        return float((estimate - truth).norm())


def simulate(  # pylint: disable=too-many-arguments,too-many-locals
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
    ch.manual_seed(seed)
    weight = ch.tensor(coef)[:, None]
    covariance = ch.eye(weight.size(0)) if covariance is None else covariance
    sampler = ch.distributions.MultivariateNormal(ch.zeros(weight.size(0)), covariance)
    kept_X: list[ch.Tensor] = []  # pylint: disable=invalid-name
    kept_z: list[ch.Tensor] = []
    survived = 0
    while survived < n:
        X = sampler.sample([_BLOCK])  # pylint: disable=invalid-name
        uniform = ch.rand(_BLOCK, 1, dtype=ch.float64).clamp(1e-15, 1.0 - 1e-15)
        z = X @ weight + intercept + noise.icdf(uniform).float()
        kept = phi(z[:, None, :]).reshape(-1)
        kept = ch.bernoulli(kept).bool() if kept.is_floating_point() else kept
        kept_X.append(X[kept])
        kept_z.append(z[kept])
        survived += int(kept.sum())
    X, z = ch.cat(kept_X)[:n], ch.cat(kept_z)[:n]  # pylint: disable=invalid-name
    y = (z >= 0).float()
    latent = (X @ weight + intercept).double()
    return Simulation(
        X=X,
        y=y,
        coef=weight.T,
        intercept=ch.tensor([intercept]),
        phi=phi,
        noise=noise,
        alpha=float(phi.label_mass(latent, noise).min()),
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
