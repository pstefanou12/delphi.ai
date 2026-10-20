# Author: pstefanou12@
"""Tests for the truncated log-normal distribution."""

import functools
from typing import Optional

import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.truncated.distributions import truncated_log_normal
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _fit(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    target: distributions.LogNormal,
    phi: helpers.Oracle,
    dims: int,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
) -> tuple[distributions.LogNormal, dict[str, distributions.LogNormal]]:
    """Fit on data standardized in log space and map the estimates back.

    Unstandardized, the Hessian's condition number is about 270 and 2500
    steps reach KL 0.15; standardized, it is about 7 and they reach 0.001.

    Args:
        target: Untruncated distribution the data is drawn from.
        phi: Truncation set oracle, in the original coordinates.
        dims: Number of dimensions.
        hyperparameters: Hyperparameter dict for the procedure.
        pass_alpha: Supply the measured survival mass to the model.

    Returns:
        The naive estimate, and each estimator's fit, in the original
        coordinates.
    """
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(target, phi, 10000)
    log_s = ch.log(sample.S)
    mean, std = log_s.mean(0), log_s.std(0)
    args = (sample.alpha,) if pass_alpha else ()
    truncated = truncated_log_normal.TruncatedLogNormal(
        hyperparameters or {},
        lambda z: phi(ch.exp(ch.log(z) * std + mean)),
        dims,
        *args,
    )
    truncated.fit(ch.exp((log_s - mean) / std))
    fits = {
        name: distributions.LogNormal(
            getattr(truncated, f"{name}_loc_") * std + mean,
            getattr(truncated, f"{name}_scale_") * std,
        )
        for name in helpers.TOLERANCES
    }
    return distributions.LogNormal(mean, std), fits


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    loc: ch.Tensor,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
) -> None:
    target = distributions.LogNormal(loc, scale)
    naive, fits = _fit(target, phi, dims, hyperparameters, pass_alpha)
    helpers.assert_recovers(
        estimate=lambda name: fits[name],
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=naive,
        baseline_must_fail=baseline_must_fail,
    )


def test_one_dimension():
    _recovers(
        dims=1,
        loc=ch.Tensor([0.0]),
        scale=ch.Tensor([1.0]),
        phi=lambda z: z[:, 0] > 2.0,
    )


def test_two_dimensions():
    loc, scale = ch.Tensor([0.0, 0.5]), ch.Tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.LogNormal(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi)


def test_two_dimensions_random_half_space():
    loc, scale = ch.Tensor([0.0, 0.5]), ch.Tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        distributions.LogNormal(loc, scale),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi)


def test_one_dimension_identity():
    _recovers(
        dims=1,
        loc=ch.Tensor([0.0]),
        scale=ch.Tensor([1.0]),
        phi=truncation_sets.identity,
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_ten_dimensions_identity():
    _recovers(
        dims=10,
        loc=ch.linspace(-0.5, 0.5, 10),
        scale=ch.linspace(0.5, 1.0, 10),
        phi=truncation_sets.identity,
        pass_alpha=False,
        baseline_must_fail=False,
    )


def test_one_dimension_small_mass():
    loc, scale = ch.Tensor([0.0]), ch.Tensor([1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        distributions.LogNormal(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=1, loc=loc, scale=scale, phi=calibrated.phi)


def test_ten_dimensions_sum_threshold():
    loc, scale = ch.linspace(-0.5, 0.5, 10), ch.linspace(0.5, 1.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        distributions.LogNormal(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 500.0),
    )
    _recovers(dims=10, loc=loc, scale=scale, phi=calibrated.phi)


def test_indistinguishable_pair():
    # Two parameters need a window wide enough to carry curvature: on
    # (1.5, 1.8) the Cramer-Rao bound on the natural parameters is 4.05.
    targets = (
        distributions.LogNormal(ch.Tensor([0.3]), ch.Tensor([0.4])),
        distributions.LogNormal(ch.Tensor([0.0]), ch.Tensor([0.8])),
    )

    def phi(z):
        return (z[:, 0] > 1.0) & (z[:, 0] < 2.0)

    results = [_fit(target, phi, 1) for target in targets]
    for name in helpers.TOLERANCES:
        helpers.assert_distinguishes(
            targets=targets,
            recovered=(results[0][1][name], results[1][1][name]),
            naive=(results[0][0], results[1][0]),
            divergence=kl.kl_divergence,
        )
