# Author: pstefanou12@
"""Tests for the truncated log-normal distribution."""

import functools
from typing import Optional

import pytest
import torch as ch
import torch.distributions
import torch.distributions.kl

import delphi.oracle
from delphi.truncated.distributions import truncated_log_normal
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)


def _fit(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    target: torch.distributions.LogNormal,
    phi: helpers.Oracle,
    dims: int,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    known: helpers.Known = True,
    set_tolerance: float = helpers.SET_TOLERANCE,
) -> tuple[torch.distributions.LogNormal, dict[str, torch.distributions.LogNormal]]:
    """Fit on data standardized in log space and map the estimates back.

    Unstandardized, the Hessian's condition number is about 270 and 2500
    steps reach KL 0.15; standardized, it is about 7 and they reach 0.001.

    Args:
        target: Untruncated distribution the data is drawn from.
        phi: Truncation set oracle, in the original coordinates.
        dims: Number of dimensions.
        pass_alpha: Supply the measured survival mass to the model.
        hyperparameters: Hyperparameter dict for the procedure.
        known: Give the model phi; else it learns the set, scored against phi.
        set_tolerance: Bound on the learned set's error.

    Returns:
        The naive estimate, and each estimator's fit, in the original
        coordinates.
    """
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(target, phi, 10000)
    log_s = ch.log(sample.S)
    mean, std = log_s.mean(0), log_s.std(0)
    args = (sample.alpha,) if pass_alpha else ()
    truncated = helpers.build(
        truncated_log_normal.TruncatedLogNormal,
        lambda z: phi(ch.exp(ch.log(z) * std + mean)),
        known,
        dims,
        *args,
        hyperparameters=hyperparameters,
    )
    truncated.fit(ch.exp((log_s - mean) / std))
    if helpers.learns_set(known):
        helpers.assert_learned_set(
            truncated,
            phi,
            target,
            set_tolerance,
            to_model=lambda z: ch.exp((ch.log(z) - mean) / std),
        )
    fits = {
        name: torch.distributions.LogNormal(
            getattr(truncated, f"{name}_loc_") * std + mean,
            getattr(truncated, f"{name}_scale_") * std,
        )
        for name in helpers.TOLERANCES
    }
    return torch.distributions.LogNormal(mean, std), fits


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    dims: int,
    loc: ch.Tensor,
    scale: ch.Tensor,
    phi: helpers.Oracle,
    hyperparameters: Optional[dict] = None,
    pass_alpha: bool = True,
    baseline_must_fail: bool = True,
    known: helpers.Known = True,
    unknown_tolerances: Optional[dict[str, float]] = None,
    set_tolerance: float = helpers.SET_TOLERANCE,
) -> None:
    target = torch.distributions.LogNormal(loc, scale)
    naive, fits = _fit(
        target, phi, dims, hyperparameters, pass_alpha, known, set_tolerance
    )
    helpers.assert_recovers(
        estimate=lambda name: fits[name],
        divergence=lambda fitted: torch.distributions.kl.kl_divergence(fitted, target),
        baseline=naive,
        baseline_must_fail=baseline_must_fail,
        **helpers.unknown_bounds(known, unknown_tolerances),
    )


@helpers.known_and_unknown(
    ci=True,
    constrained_ci=False,
    constrained_marks=(
        pytest.mark.xfail(
            reason=(
                "constrained learner: the estimate's KL measured 0.08 to "
                "0.14 against 0.1, by thread count"
            )
        ),
    ),
)
def test_one_dimension(known):
    _recovers(
        dims=1,
        loc=ch.tensor([0.0]),
        scale=ch.tensor([1.0]),
        phi=lambda z: z[:, 0] > 2.0,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions(known):
    loc, scale = ch.tensor([0.0, 0.5]), ch.tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.LogNormal(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 100.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi, known=known)


@helpers.KNOWN_AND_UNKNOWN
def test_two_dimensions_random_half_space(known):
    loc, scale = ch.tensor([0.0, 0.5]), ch.tensor([1.0, 0.5])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        torch.distributions.LogNormal(loc, scale),
        bounds=(-50.0, 50.0),
    )
    _recovers(dims=2, loc=loc, scale=scale, phi=calibrated.phi, known=known)


@helpers.known_and_unknown(constrained_ci=True)
def test_one_dimension_identity(known):
    _recovers(
        dims=1,
        loc=ch.tensor([0.0]),
        scale=ch.tensor([1.0]),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_identity(known):
    _recovers(
        dims=10,
        loc=ch.linspace(-0.5, 0.5, 10),
        scale=ch.linspace(0.5, 1.0, 10),
        phi=delphi.oracle.Identity(),
        pass_alpha=False,
        baseline_must_fail=False,
        known=known,
    )


# Unknown truncation, rho 0.3: over seeds 0,1,2 the estimate's KL measured
# 0.003 to 0.164 (ema), 0.009 to 0.206 (avg).
@helpers.KNOWN_AND_UNKNOWN
def test_one_dimension_small_mass(known):
    loc, scale = ch.tensor([0.0]), ch.tensor([1.0])
    calibrated = truncation_sets.calibrate(
        functools.partial(truncation_sets.coordinate_threshold, 0),
        torch.distributions.LogNormal(loc, scale),
        target_alpha=(0.008, 0.02),
        bounds=(0.0, 100.0),
    )
    _recovers(
        dims=1,
        loc=loc,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.33, "avg": 0.42},
        hyperparameters={"truncation_set_rho": 0.3},
    )


# Unknown truncation, degree 3: over seeds 0,1,2 the estimate's KL measured
# 0.213 to 0.242 (ema), 0.214 to 0.243 (avg); the set error 0.101 to 0.112.
@helpers.KNOWN_AND_UNKNOWN
def test_ten_dimensions_sum_threshold(known):
    loc, scale = ch.linspace(-0.5, 0.5, 10), ch.linspace(0.5, 1.0, 10)
    calibrated = truncation_sets.calibrate(
        truncation_sets.sum_threshold,
        torch.distributions.LogNormal(loc, scale),
        target_alpha=(0.15, 0.25),
        bounds=(0.0, 500.0),
    )
    _recovers(
        dims=10,
        loc=loc,
        scale=scale,
        phi=calibrated.phi,
        known=known,
        unknown_tolerances={"ema": 0.49, "avg": 0.49},
        set_tolerance=0.23,
        hyperparameters={"truncation_set_degree": 3},
    )


@helpers.known_and_unknown(
    pytest.mark.xfail(
        strict=True,
        reason="unknown truncation: the learned window is too coarse to separate the pair",
    )
)
def test_indistinguishable_pair(known):
    # Two parameters need a window wide enough to carry curvature: on
    # (1.5, 1.8) the Cramer-Rao bound on the natural parameters is 4.05.
    targets = (
        torch.distributions.LogNormal(ch.tensor([0.3]), ch.tensor([0.4])),
        torch.distributions.LogNormal(ch.tensor([0.0]), ch.tensor([0.8])),
    )

    def phi(z):
        return (z[:, 0] > 1.0) & (z[:, 0] < 2.0)

    results = [_fit(target, phi, 1, known=known) for target in targets]
    for name in helpers.TOLERANCES:
        helpers.assert_distinguishes(
            targets=targets,
            recovered=(results[0][1][name], results[1][1][name]),
            naive=(results[0][0], results[1][0]),
            divergence=torch.distributions.kl.kl_divergence,
        )
