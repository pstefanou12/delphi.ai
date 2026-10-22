# Author: pstefanou12@
"""Tests for the univariate normal under unknown truncation.

Each test checks both the fitted normal and the recovered set. The estimator
standardizes by the sample's own moments, so it is affine-equivariant:
moving the mean and set together changes nothing, and what varies the
difficulty is where the set sits relative to the mean.
"""

import torch as ch
from torch import distributions

from delphi.truncated.distributions import unknown_truncated_normal
from delphi.truncated.distributions.tests import helpers, unknown_truncation

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(target: distributions.Normal, phi: helpers.Oracle) -> None:
    model = unknown_truncated_normal.UnknownTruncationNormal({})
    unknown_truncation.assert_recovers(
        model,
        target,
        phi,
        estimate=lambda name: distributions.Normal(
            getattr(model, f"{name}_loc_"), getattr(model, f"{name}_scale_")
        ),
        baseline=lambda s: distributions.Normal(s.mean(0), s.std(0)),
    )


def test_half_line():
    _recovers(
        distributions.Normal(ch.tensor([1.0]), ch.tensor([1.5])),
        lambda z: z[:, 0] > 1.5,
    )


def test_interval():
    _recovers(
        distributions.Normal(ch.tensor([-0.5]), ch.tensor([2.0])),
        lambda z: (z[:, 0] > -1.5) & (z[:, 0] < 2.5),
    )


def test_interval_beside_the_mean():
    # One standard deviation wide and wholly beside the mean, it keeps 30% of
    # the mass and little of its shape: degree 200 left KL 0.155, the
    # default 1200 reaches 0.057, against 1.053 naive.
    _recovers(
        distributions.Normal(ch.tensor([-0.5]), ch.tensor([2.0])),
        lambda z: (z[:, 0] > 0.0) & (z[:, 0] < 2.0),
    )
