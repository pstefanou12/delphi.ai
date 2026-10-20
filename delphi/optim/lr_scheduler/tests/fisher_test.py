# Author: pstefanou12@
"""Tests for the Fisher learning-rate schedule."""

import pytest
import torch as ch

from delphi.optim.lr_scheduler import fisher


def _optimizer():
    return ch.optim.SGD([ch.nn.Parameter(ch.zeros(1))], lr=1.0)


def _rate(schedule):
    return schedule.optimizer.param_groups[0]["lr"]


def _advance(schedule, steps):
    for _ in range(steps):
        schedule.optimizer.step()
        schedule.step()


def test_decays_as_one_over_l_plus_mu_t():
    schedule = fisher.FisherLR(_optimizer(), mu=2.0, smoothness=50.0)
    _advance(schedule, 7)
    assert _rate(schedule) == pytest.approx(1.0 / (50.0 + 2.0 * 7))


def test_refresh_takes_the_new_constants_and_keeps_t():
    schedule = fisher.FisherLR(
        _optimizer(),
        mu=2.0,
        smoothness=50.0,
        curvature=lambda: (0.5, 4.0),
        refresh_every=10,
    )
    _advance(schedule, 9)
    assert _rate(schedule) == pytest.approx(1.0 / (50.0 + 2.0 * 9))
    _advance(schedule, 1)
    assert (schedule.mu, schedule.smoothness) == (0.5, 4.0)
    assert _rate(schedule) == pytest.approx(1.0 / (4.0 + 0.5 * 10))


def test_failed_refresh_keeps_the_old_constants():
    def unavailable():
        raise ValueError("no mass on the truncation set")

    schedule = fisher.FisherLR(
        _optimizer(),
        mu=2.0,
        smoothness=50.0,
        curvature=unavailable,
        refresh_every=5,
    )
    _advance(schedule, 5)
    assert (schedule.mu, schedule.smoothness) == (2.0, 50.0)
    assert _rate(schedule) == pytest.approx(1.0 / (50.0 + 2.0 * 5))


def test_inconsistent_refresh_is_ignored():
    schedule = fisher.FisherLR(
        _optimizer(),
        mu=2.0,
        smoothness=50.0,
        curvature=lambda: (9.0, 3.0),
        refresh_every=5,
    )
    _advance(schedule, 5)
    assert (schedule.mu, schedule.smoothness) == (2.0, 50.0)


def test_negative_refresh_interval_is_rejected():
    with pytest.raises(ValueError, match="refresh_every"):
        fisher.FisherLR(_optimizer(), mu=1.0, smoothness=2.0, refresh_every=-1)
