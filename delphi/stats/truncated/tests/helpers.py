# Author: pstefanou12@
"""Shared harness for the truncated linear model tests.

``make_dataset`` draws a truncated regression sample with known parameters,
and ``assert_recovers`` fits a model to it and checks every estimate the
model returns against the truth.
"""

from collections.abc import Callable, Sequence
import dataclasses
import math
from typing import Optional

import torch as ch

from delphi.stats import linear_model

# Draws per block while collecting survivors.
_BLOCK = 100_000


@dataclasses.dataclass(frozen=True)
class TruncatedDataset:  # pylint: disable=too-many-instance-attributes
    """A truncated regression sample and the parameters that produced it.

    Attributes:
        X: Observed covariates, shaped (n, d).
        y: Observed responses, shaped (n, 1).
        coef: True coefficients, shaped (1, d).
        intercept: True intercept, shaped (1,).
        noise_var: True noise variance.
        phi: The truncation oracle the sample passed.
        alpha: Fraction of the draws that survived.
        ols_coef: OLS coefficients on the observed sample.
        ols_intercept: OLS intercept on the observed sample.
    """

    X: ch.Tensor  # pylint: disable=invalid-name
    y: ch.Tensor
    coef: ch.Tensor
    intercept: ch.Tensor
    noise_var: float
    phi: Callable
    alpha: float
    ols_coef: ch.Tensor
    ols_intercept: ch.Tensor

    def error(self, coef: ch.Tensor, intercept: Optional[ch.Tensor]) -> float:
        """Return the L2 distance of (coef, intercept) from the truth."""
        estimate = coef.flatten()
        truth = self.coef.flatten()
        if intercept is not None:
            estimate = ch.cat([estimate, intercept.flatten()])
            truth = ch.cat([truth, self.intercept])
        return float((estimate - truth).norm())


def make_dataset(  # pylint: disable=too-many-arguments,too-many-locals
    coef: Sequence[float],
    intercept: float,
    phi: Callable,
    n: int,
    *,
    noise_var: float = 1.0,
    covariance: Optional[ch.Tensor] = None,
    seed: int = 0,
) -> TruncatedDataset:
    """Draw ``y = coef^T x + intercept + e`` until n samples pass phi.

    Args:
        coef: True coefficients.
        intercept: True intercept.
        phi: Oracle over responses shaped (n, m, 1); a float result is a
            survival probability, applied by a coin flip.
        n: Survivors to collect.
        noise_var: Noise variance.
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
    kept_y: list[ch.Tensor] = []
    survived, drawn = 0, 0
    while survived < n:
        X = sampler.sample([_BLOCK])  # pylint: disable=invalid-name
        y = X @ weight + intercept + math.sqrt(noise_var) * ch.randn(_BLOCK, 1)
        kept = phi(y[:, None, :]).reshape(-1)
        kept = ch.bernoulli(kept).bool() if kept.is_floating_point() else kept
        kept_X.append(X[kept])
        kept_y.append(y[kept])
        survived += int(kept.sum())
        drawn += _BLOCK
    X, y = ch.cat(kept_X)[:n], ch.cat(kept_y)[:n]  # pylint: disable=invalid-name
    design = ch.cat([X, ch.ones(n, 1)], dim=1).double()
    ols = ch.linalg.lstsq(design, y.double()).solution.flatten().float()  # pylint: disable=not-callable
    return TruncatedDataset(
        X=X,
        y=y,
        coef=weight.T,
        intercept=ch.tensor([intercept]),
        noise_var=noise_var,
        phi=phi,
        alpha=survived / drawn,
        ols_coef=ols[:-1][None, :],
        ols_intercept=ols[-1:],
    )


def assert_recovers(
    model,
    dataset: TruncatedDataset,
    tolerance: float,
    *,
    ols_margin: Optional[float] = 0.5,
) -> None:
    """Fit model to dataset and check each estimate it returns.

    Args:
        model: An unfitted truncated linear model.
        dataset: The sample, with its truth.
        tolerance: Largest L2 error allowed for (coef, intercept), and for
            the variance when the model learns it.
        ols_margin: Each estimate's error must also be below this fraction
            of OLS's; None skips the comparison, as without truncation.
    """
    model.fit(dataset.X, dataset.y)
    ols = dataset.error(dataset.ols_coef, dataset.ols_intercept)
    names = [*linear_model.ESTIMATORS, ""]
    for name in names:
        prefix = f"{name}_" if name else ""
        coef = getattr(model, f"{prefix}coef_")
        intercept = getattr(model, f"{prefix}intercept_")
        error = dataset.error(coef, intercept)
        label = f"{prefix}coef_ and {prefix}intercept_"
        assert error <= tolerance, (
            f"{label}: error {error:.4f} above {tolerance} (OLS {ols:.4f})"
        )
        if ols_margin is not None:
            assert error <= ols_margin * ols, (
                f"{label}: error {error:.4f} not below {ols_margin} of OLS {ols:.4f}"
            )
        if getattr(model, "noise_var", 1.0) is None:
            variance = float(getattr(model, f"{prefix}variance_"))
            assert abs(variance - dataset.noise_var) <= tolerance, (
                f"{prefix}variance_: {variance:.4f} against {dataset.noise_var}"
            )
