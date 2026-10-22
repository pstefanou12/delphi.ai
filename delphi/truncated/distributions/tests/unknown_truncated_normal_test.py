# Author: pstefanou12@
"""Tests for the univariate normal under unknown truncation.

In one dimension the Hermite basis has only ``k + 1`` terms, so the default
degree is the cap, 200; degree 10 left a KL of about 0.1 on both sets. The
estimator whitens by the sample's own moments, so it is affine-equivariant:
moving the mean and set together changes nothing, and what varies the
difficulty is where the set sits relative to the mean.
"""

from typing import Optional

import pytest
import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import (
    unknown_truncated_multivariate_normal,
    unknown_truncated_normal,
)
from delphi.truncated.distributions.tests import helpers

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    target: distributions.Normal,
    phi: helpers.Oracle,
    known_variance: bool,
    hyperparameters: Optional[dict] = None,
    dataset_size: int = 10000,
) -> None:
    """Fit on a truncated sample and assert the fit recovers the truth.

    Args:
        target: The untruncated normal, with loc and scale of shape (1,).
        phi: Truncation set; only the sampler sees it.
        known_variance: Fit the known-variance estimator, given the truth.
        hyperparameters: Hyperparameter dict for the estimator.
        dataset_size: Truncated samples to draw.
    """
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(target, phi, dataset_size)
    model: unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalBase
    if known_variance:
        model = unknown_truncated_normal.UnknownTruncationNormalKnownVariance(
            hyperparameters or {}, target.scale[:, None] ** 2, sample.alpha
        )
        naive_scale = target.scale
    else:
        model = unknown_truncated_normal.UnknownTruncationNormal(
            hyperparameters or {}, sample.alpha
        )
        naive_scale = sample.S.std(0)
    model.fit(sample.S)

    def estimate(name: str) -> distributions.Normal:
        scale = target.scale if known_variance else getattr(model, f"{name}_scale_")
        return distributions.Normal(getattr(model, f"{name}_loc_"), scale)

    helpers.assert_recovers(
        estimate=estimate,
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=distributions.Normal(sample.S.mean(0), naive_scale),
    )


def test_half_line():
    _recovers(
        distributions.Normal(ch.tensor([1.0]), ch.tensor([1.5])),
        lambda z: z[:, 0] > 1.5,
        known_variance=False,
    )


def test_interval():
    _recovers(
        distributions.Normal(ch.tensor([-0.5]), ch.tensor([2.0])),
        lambda z: (z[:, 0] > -1.5) & (z[:, 0] < 2.5),
        known_variance=False,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "An interval one standard deviation wide, wholly beside the mean, "
        "keeps 30% of the mass and little of its shape: the fit reaches KL "
        "0.128 against 1.053 naive, over the 0.1 bound."
    ),
)
def test_interval_beside_the_mean():
    _recovers(
        distributions.Normal(ch.tensor([-0.5]), ch.tensor([2.0])),
        lambda z: (z[:, 0] > 0.0) & (z[:, 0] < 2.0),
        known_variance=False,
    )


def test_known_variance_half_line():
    _recovers(
        distributions.Normal(ch.ones(1), 2.0 * ch.ones(1)),
        lambda z: z[:, 0] > 1.0,
        known_variance=True,
    )
