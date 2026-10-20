# Author: pstefanou12@
"""Tests for the fit summary."""

import pytest
import torch as ch
from torch import distributions

from delphi.truncated.distributions import summary, truncated_exponential
from delphi.truncated.distributions.tests import helpers


def _fitted():
    ch.manual_seed(0)
    target = distributions.Exponential(ch.Tensor([1.0, 2.0]))

    def phi(z):
        return z[:, 0] > 0.5

    sample = helpers.generate_truncated_distribution_dataset(target, phi, 2000)
    model = truncated_exponential.TruncatedExponential(
        {"iterations": 200, "batch_size": 10, "num_samples": 500},
        phi,
        2,
        sample.alpha,
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
    for estimator in summary.ESTIMATORS:
        assert f"{estimator}_rate_" in report
    assert "condition number" in report


def test_raises_before_fit():
    model = truncated_exponential.TruncatedExponential(
        {"iterations": 10}, lambda z: z[:, 0] > 0.5, 2
    )
    with pytest.raises(RuntimeError, match="has not been fit"):
        model.summary()
