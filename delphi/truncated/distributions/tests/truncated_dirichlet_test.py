# Author: pstefanou12@
"""Tests for the truncated Dirichlet distribution.

``dims`` is the number of categories, and every sample sums to one, so a sum
bound cuts nothing; the ten-category coupling test bounds the norm instead.
"""

import functools
from typing import Optional

import pytest
import torch as ch
from torch import distributions

import delphi.oracle
from delphi.truncated.distributions import truncated_dirichlet
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _recovers(
    concentration: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    categories = concentration.size(0)
    helpers.assert_family_recovers(
        model_class=truncated_dirichlet.TruncatedDirichlet,
        target=distributions.Dirichlet(concentration),
        phi=phi,
        dims=categories,
        parameter="concentration_",
        empirical=helpers.untruncated_mle(
            truncated_dirichlet.TruncatedDirichlet, categories
        ),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
    )


def test_three_categories():
    _recovers(concentration=ch.Tensor([2.0, 3.0, 4.0]), phi=lambda z: z[:, 0] > 0.3)


def test_three_categories_random_half_space():
    concentration = ch.Tensor([2.0, 3.0, 4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 3),
        distributions.Dirichlet(concentration),
        target_alpha=(0.15, 0.25),
        bounds=(-2.0, 2.0),
    )
    _recovers(concentration=concentration, phi=calibrated.phi)


def test_three_categories_triangle():
    _recovers(
        concentration=ch.Tensor([2.0, 3.0, 4.0]),
        phi=truncation_sets.triangle([(0.3, 0.0), (0.7, 0.0), (0.3, 0.4)]),
    )


def test_three_categories_identity():
    _recovers(
        concentration=ch.Tensor([2.0, 3.0, 4.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_categories_identity():
    _recovers(
        concentration=ch.linspace(1.0, 5.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
    )


@pytest.mark.slow
def test_three_categories_small_mass():
    concentration = ch.Tensor([2.0, 3.0, 4.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Dirichlet(concentration),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 1.0),
    )
    # At 1% mass the default 2500 steps end at ema KL 0.171; 5000 pass.
    _recovers(
        concentration=concentration,
        phi=calibrated.phi,
        hyperparameters={"iterations": 5000},
    )


def test_ten_categories_norm_threshold():
    concentration = ch.linspace(1.0, 5.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.norm_threshold,
        distributions.Dirichlet(concentration),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 1.0),
    )
    _recovers(concentration=concentration, phi=calibrated.phi)


def test_indistinguishable_pair():
    # Given x0, the other categories' shares are Dirichlet in the rest of the
    # concentration, so a pair differing only in the first is told apart by
    # x0 alone, which a slab flattens.
    helpers.assert_family_distinguishes(
        model_class=truncated_dirichlet.TruncatedDirichlet,
        targets=(
            distributions.Dirichlet(ch.Tensor([2.0, 3.0, 4.0])),
            distributions.Dirichlet(ch.Tensor([5.0, 3.0, 4.0])),
        ),
        phi=lambda z: (z[:, 0] > 0.25) & (z[:, 0] < 0.35),
        dims=3,
        parameter="concentration_",
        empirical=helpers.untruncated_mle(truncated_dirichlet.TruncatedDirichlet, 3),
    )
