# Author: pstefanou12@
"""Tests for the multivariate normal with known covariance under unknown truncation.

Each recovery test checks both the fitted mean and the recovered set.
"""

from typing import Optional

import pytest
import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import (
    unknown_truncated_multivariate_normal_known_covariance,
)
from delphi.truncated.distributions.tests import (
    helpers,
    truncation_sets,
    unknown_truncation,
)

ch.set_printoptions(precision=4, sci_mode=False)

_ESTIMATOR = unknown_truncated_multivariate_normal_known_covariance.UnknownTruncationMultivariateNormalKnownCovariance


def _recovers(
    loc: ch.Tensor,
    covariance_matrix: ch.Tensor,
    phi: helpers.Oracle,
    tolerances: Optional[dict[str, float]] = None,
) -> None:
    target = distributions.MultivariateNormal(loc, covariance_matrix)
    model = _ESTIMATOR({}, loc.numel(), covariance_matrix)
    unknown_truncation.assert_recovers(
        model,
        target,
        phi,
        estimate=lambda name: distributions.MultivariateNormal(
            getattr(model, f"{name}_loc_"), covariance_matrix
        ),
        baseline=lambda s: distributions.MultivariateNormal(
            s.mean(0), covariance_matrix
        ),
        tolerances=tolerances,
    )


def test_half_space():
    _recovers(
        ch.tensor([0.5, -0.5]), unknown_truncation.CORRELATED, lambda z: z.sum(1) > 0.5
    )


def test_ball_off_the_mean():
    _recovers(
        ch.tensor([1.0, -0.5]),
        unknown_truncation.CORRELATED,
        lambda z: z.norm(dim=-1) < 1.5,
    )


def test_three_dimensional_polytope():
    _recovers(
        ch.tensor([0.5, -0.3, 0.2]),
        unknown_truncation.CORRELATED_3D,
        truncation_sets.random_polytope(0, 3, 3, 0.5),
    )


def test_ten_dimensional_ball():
    _recovers(
        ch.linspace(-0.5, 0.5, 10),
        unknown_truncation.correlated(10, 10),
        lambda z: (z - 0.3).norm(dim=-1) < 3.0,
    )


# Kontonis et al. (2019)'s Figure 1: an isotropic Gaussian truncated to a
# non-convex Sierpinski triangle, about 2% of its mass.
_FIGURE_ONE_MEANS = ((0.1, 0.78), (0.0, 0.0))


@pytest.mark.parametrize("loc", _FIGURE_ONE_MEANS)
def test_sierpinski_triangle(loc):
    # The naive estimates score 0.15 to 0.18, at the suite's loosest bound,
    # and the fits about 0.001, so a bound of 0.02 is what measures the fit.
    _recovers(
        ch.tensor(loc),
        ch.eye(2),
        truncation_sets.sierpinski(3),
        tolerances={"ema": 0.02, "avg": 0.02},
    )


def test_sierpinski_pair_is_told_apart():
    """Figure 1's two Gaussians leave samples whose means nearly coincide."""
    phi = truncation_sets.sierpinski(3)
    targets = tuple(
        distributions.MultivariateNormal(ch.tensor(loc), ch.eye(2))
        for loc in _FIGURE_ONE_MEANS
    )
    fits, naive = [], []
    for target in targets:
        ch.manual_seed(0)
        sample = helpers.generate_truncated_distribution_dataset(target, phi, 20000)
        model = _ESTIMATOR({}, 2, ch.eye(2), sample.alpha)
        model.fit(sample.S)
        fits.append(model)
        naive.append(distributions.MultivariateNormal(sample.S.mean(0), ch.eye(2)))
    for name in helpers.TOLERANCES:
        helpers.assert_distinguishes(
            targets=targets,
            recovered=tuple(
                distributions.MultivariateNormal(
                    getattr(fit, f"{name}_loc_"), ch.eye(2)
                )
                for fit in fits
            ),
            naive=(naive[0], naive[1]),
            divergence=kl.kl_divergence,
        )


def test_rejects_a_covariance_of_the_wrong_shape():
    with pytest.raises(ValueError, match="covariance_matrix"):
        _ESTIMATOR({}, 2, ch.eye(3))
