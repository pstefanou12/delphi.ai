# Author: pstefanou12@
"""Tests for the truncated Pareto (known scale) distribution."""

import functools
from typing import Optional

import torch as ch
from torch import distributions

from delphi.truncated.distributions import truncated_pareto
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def pareto_kl_divergence(
    fitted: distributions.Distribution, target: distributions.Distribution
) -> ch.Tensor:
    """Return KL between two Pareto distributions sharing a scale.

    torch registers no Pareto KL. With a common scale the divergence depends
    only on the tail indices: ``log(a1 / a2) + a2 / a1 - 1``.

    Args:
        fitted: First Pareto distribution.
        target: Second Pareto distribution, with the same scale.

    Returns:
        Per-coordinate KL divergence.
    """
    first, second = fitted.alpha, target.alpha  # type: ignore[attr-defined]
    return ch.log(first / second) + second / first - 1


def _recovers(
    dims: int,
    scale: ch.Tensor,
    tail: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    helpers.assert_family_recovers(
        model_class=truncated_pareto.TruncatedPareto,
        target=distributions.Pareto(scale, tail),
        phi=phi,
        dims=dims,
        parameter="alpha_",
        empirical=lambda s: 1.0 / (ch.log(s) - ch.log(scale)).mean(0),
        hyperparameters=hyperparameters,
        pass_alpha=pass_alpha,
        baseline_must_fail=baseline_must_fail,
        model_args=(scale,),
        rebuild=functools.partial(distributions.Pareto, scale),
        divergence=pareto_kl_divergence,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        scale=ch.Tensor([1.0]),
        tail=ch.Tensor([2.5]),
        phi=lambda z: z[:, 0] > 2.0,
    )


def test_two_dimensions():
    scale, tail = ch.Tensor([1.0, 2.0]), ch.Tensor([2.5, 1.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Pareto(scale, tail),
        target_alpha=(0.15, 0.25),
        bounds=(1.0, 500.0),
    )
    _recovers(
        dims=2,
        scale=scale,
        tail=tail,
        phi=calibrated.phi,
    )


def test_two_dimensions_random_half_space():
    scale, tail = ch.Tensor([1.0, 2.0]), ch.Tensor([2.5, 1.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.Pareto(scale, tail),
        bounds=(-25.0, 25.0),
    )
    _recovers(
        dims=2,
        scale=scale,
        tail=tail,
        phi=calibrated.phi,
    )


def test_one_dimension_identity():
    _recovers(
        dims=1,
        scale=ch.Tensor([1.0]),
        tail=ch.Tensor([2.5]),
        phi=truncation_sets.identity,
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        scale=ch.ones(10),
        tail=ch.linspace(1.5, 4.0, 10),
        phi=truncation_sets.identity,
        hyperparameters={"iterations": 1500},
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    scale, tail = ch.Tensor([1.0]), ch.Tensor([2.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.Pareto(scale, tail),
        target_alpha=(0.008, 0.02),
        bounds=(1.0, 500.0),
    )
    _recovers(
        dims=1,
        scale=scale,
        tail=tail,
        phi=calibrated.phi,
    )


def test_ten_dimensions_sum_threshold():
    scale, tail = ch.ones(10), ch.linspace(1.5, 4.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.Pareto(scale, tail),
        target_alpha=(0.15, 0.25),
        bounds=(10.0, 500.0),
    )
    _recovers(
        dims=10,
        scale=scale,
        tail=tail,
        phi=calibrated.phi,
    )


def test_indistinguishable_pair():
    helpers.assert_family_distinguishes(
        model_class=truncated_pareto.TruncatedPareto,
        targets=(
            distributions.Pareto(ch.Tensor([1.0]), ch.Tensor([1.5])),
            distributions.Pareto(ch.Tensor([1.0]), ch.Tensor([4.0])),
        ),
        phi=lambda z: (z[:, 0] > 1.2) & (z[:, 0] < 1.4),
        dims=1,
        parameter="alpha_",
        empirical=lambda s: 1.0 / ch.log(s).mean(0),
        model_args=(ch.Tensor([1.0]),),
        rebuild=functools.partial(distributions.Pareto, ch.Tensor([1.0])),
        divergence=pareto_kl_divergence,
    )
