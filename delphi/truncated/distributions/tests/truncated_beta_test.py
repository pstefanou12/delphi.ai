# Author: pstefanou12@
"""Tests for the truncated Beta distribution."""

import functools

import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_beta
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _moments(s: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
    mean, var = s.mean(0), s.var(0)
    common = mean * (1 - mean) / var - 1
    return mean * common, (1 - mean) * common


def _recovers(
    dims: int,
    first: ch.Tensor,
    second: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: dict,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_beta.TruncatedBeta,
        target=distributions.Beta(first, second),
        phi=phi,
        dims=dims,
        parameter=("concentration1_", "concentration0_"),
        empirical=_moments,
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        first=ch.Tensor([2.0]),
        second=ch.Tensor([3.0]),
        phi=lambda z: z[:, 0] > 0.5,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
            "scheduler": "fisher",
        },
    )


def test_two_dimensions_random_half_space():
    first, second = ch.Tensor([2.0, 4.0]), ch.Tensor([3.0, 2.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.Beta(first, second),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=2,
        first=first,
        second=second,
        phi=calibrated.phi,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
            "scheduler": "fisher",
        },
    )


def test_one_dimension_identity():
    _recovers(
        dims=1,
        first=ch.Tensor([2.0]),
        second=ch.Tensor([3.0]),
        phi=truncation_sets.identity,
        hyperparameters={
            "iterations": 1500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        first=ch.linspace(1.5, 3.0, 10),
        second=ch.linspace(2.0, 4.0, 10),
        phi=truncation_sets.identity,
        hyperparameters={
            "iterations": 1500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
        },
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_sum_threshold():
    first, second = ch.linspace(1.5, 3.0, 10), ch.linspace(2.0, 4.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Beta(first, second),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 10.0),
    )
    _recovers(
        dims=10,
        first=first,
        second=second,
        phi=calibrated.phi,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
            "scheduler": "fisher",
        },
    )


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=truncated_beta.TruncatedBeta,
        # Symmetric about the window, so neither slopes across it, and wide
        # enough to carry the curvature a second parameter needs: on
        # (0.45, 0.55) the Cramer-Rao bound on a is 3.35, more than a itself.
        targets=(
            distributions.Beta(ch.Tensor([2.0]), ch.Tensor([2.0])),
            distributions.Beta(ch.Tensor([5.0]), ch.Tensor([5.0])),
        ),
        phi=lambda z: (z[:, 0] > 0.25) & (z[:, 0] < 0.75),
        dims=1,
        parameter=("concentration1_", "concentration0_"),
        empirical=_moments,
        hyperparameters={
            "iterations": 2500,
            "trials": 1,
            "batch_size": 10,
            "num_samples": 1000,
            "optimizer": "sgd",
            "lr": 1e-2,
            "scheduler": "fisher",
        },
    )
