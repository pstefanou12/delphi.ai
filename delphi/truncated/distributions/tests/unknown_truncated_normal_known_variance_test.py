# Author: pstefanou12@
"""Tests for the univariate normal with known variance under unknown truncation.

Each test checks both the fitted mean and the recovered set.
"""

import torch as ch
from torch import distributions

from delphi.truncated.distributions import unknown_truncated_normal_known_variance
from delphi.truncated.distributions.tests import helpers, unknown_truncation

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(target: distributions.Normal, phi: helpers.Oracle) -> None:
    variance = (target.scale**2)[:, None]
    model = (
        unknown_truncated_normal_known_variance.UnknownTruncationNormalKnownVariance(
            {}, variance
        )
    )
    unknown_truncation.assert_recovers(
        model,
        target,
        phi,
        estimate=lambda name: distributions.Normal(
            getattr(model, f"{name}_loc_"), target.scale
        ),
        baseline=lambda s: distributions.Normal(s.mean(0), target.scale),
    )


def test_half_line():
    _recovers(
        distributions.Normal(ch.ones(1), 2.0 * ch.ones(1)),
        lambda z: z[:, 0] > 1.0,
    )


def test_interval_beside_the_mean():
    _recovers(
        distributions.Normal(ch.tensor([-0.5]), ch.tensor([2.0])),
        lambda z: (z[:, 0] > 0.0) & (z[:, 0] < 2.0),
    )
