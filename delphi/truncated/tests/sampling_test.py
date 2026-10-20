# Author: pstefanou12@
"""Tests for rejection sampling."""

import pytest
import torch as ch
from torch import distributions

from delphi.truncated import sampling


def test_returns_at_least_the_samples_asked_for():
    ch.manual_seed(0)
    samples, mass = sampling.rejection_sample(
        distributions.Normal(0.0, 1.0), lambda z: z > 1.0, 500
    )
    assert samples.size(0) >= 500
    assert bool((samples > 1.0).all())
    # P(z > 1) under a standard normal.
    assert float(mass) == pytest.approx(0.1587, abs=0.02)


def test_gives_up_on_a_set_with_no_mass():
    # Rate 28 puts 1.8e-08 of a Poisson's mass on {3, 4}; this once ran for
    # 5.5e7 draws per accepted sample instead of failing.
    with pytest.raises(sampling.SamplingBudgetExceeded, match="almost no mass"):
        sampling.rejection_sample(
            distributions.Poisson(ch.tensor([28.0])),
            lambda z: ((z >= 3) & (z <= 4)).flatten(),
            100,
        )
