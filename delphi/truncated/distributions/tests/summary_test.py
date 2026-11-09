# Author: pstefanou12@
"""Tests for the fit summary."""

import pytest
import torch as ch
import torch.distributions

from delphi.truncated.distributions import truncated_exponential
from delphi.truncated.distributions.tests import helpers
from delphi.utils import constants


def _fitted():
    ch.manual_seed(0)
    target = torch.distributions.Exponential(ch.tensor([1.0, 2.0]))

    def phi(z):
        return z[:, 0] > 0.5

    sample = helpers.generate_truncated_distribution_dataset(target, phi, 2000)
    model = truncated_exponential.TruncatedExponential(
        phi,
        sample.alpha,
        hyperparameters={"iterations": 200, "num_samples": 500},
    )
    return model.fit(sample.S)


def test_reports_every_section():
    report = _fitted().summary()
    for heading in (
        "Model and data",
        "Optimization",
        "Estimates",
        "Truncation and curvature",
    ):
        assert heading in report
    for estimator in constants.Estimator:
        assert f"{estimator}_rate_" in report
    assert "condition number" in report


def test_raises_before_fit():
    model = truncated_exponential.TruncatedExponential(
        lambda z: z[:, 0] > 0.5, hyperparameters={"iterations": 10}
    )
    with pytest.raises(RuntimeError, match="has not been fit"):
        model.summary()
