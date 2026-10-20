# Author: pstefanou12@
"""Tests for the truncated geometric distribution."""

import functools
from typing import Optional

import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_geometric
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    probs: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_geometric.TruncatedGeometric,
        target=distributions.Geometric(probs),
        phi=phi,
        dims=dims,
        parameter="probs_",
        empirical=lambda s: 1.0 / (1.0 + s.mean(0)),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
    )


def test_one_dimension():
    _recovers(dims=1, probs=ch.Tensor([0.3]), phi=lambda z: z[:, 0] > 2)


def test_two_dimensions():
    probs = ch.Tensor([0.3, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Geometric(probs),
        target_alpha=(0.10, 0.40),
        bounds=(-1.0, 200.0),
    )
    _recovers(dims=2, probs=probs, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    probs = ch.Tensor([0.3, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.Geometric(probs),
        target_alpha=(0.10, 0.40),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, probs=probs, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        probs=ch.Tensor([0.3]),
        phi=truncation_sets.identity,
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    probs = ch.linspace(0.2, 0.6, 10)
    _recovers(
        dims=10,
        probs=probs,
        phi=truncation_sets.identity,
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    probs = ch.Tensor([0.3])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Geometric(probs),
        target_alpha=(0.005, 0.03),
        bounds=(-1.0, 200.0),
    )
    _recovers(dims=1, probs=probs, phi=calibrated.phi)


def test_ten_dimensions_sum_threshold():
    probs = ch.linspace(0.2, 0.6, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Geometric(probs),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 400.0),
    )
    _recovers(dims=10, probs=probs, phi=calibrated.phi)


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=truncated_geometric.TruncatedGeometric,
        targets=(
            distributions.Geometric(ch.Tensor([0.2])),
            distributions.Geometric(ch.Tensor([0.5])),
        ),
        phi=lambda z: (z[:, 0] > 2.5) & (z[:, 0] < 4.5),
        dims=1,
        parameter="probs_",
        empirical=lambda s: 1.0 / (1.0 + s.mean(0)),
    )
