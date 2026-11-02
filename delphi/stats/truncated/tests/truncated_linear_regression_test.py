# Author: pstefanou12@
"""Tests for truncated linear regression.

Each truncation-set case runs twice on the shared harness: with the noise
variance given, and with it learned (``noise_var=None``). Each tolerance is
about twice the error of the exact truncated MLE on the same sample,
measured with the closed-form NLL, over the weight alone when the variance is
known and over (weight, variance) when it is not; every estimate lands within
a few hundredths of that MLE. The dependent-data cases follow, then both
variants' confidence intervals: their covariance against the exact NLL's,
and their coverage over repeated samples, on two cases by default and on
every regression case with ``--coverage-sweep``.
"""

from collections.abc import Callable, Sequence
import math
import os
import shutil
from typing import Optional

import pytest
import torch as ch
import torch.autograd.functional
import torch.nn.functional

from delphi.stats.truncated import truncated_linear_regression
from delphi.stats.truncated import (
    truncated_linear_regression as truncated_linear_regression_module,
)
from delphi.stats.truncated.tests import regression
from delphi.utils import helpers

INF = math.inf
SEED = 69

# The (low, high) intervals whose union is a truncation set.
Bounds = Sequence[tuple[float, float]]

_VARIANTS = pytest.mark.parametrize(
    "variance_known", [True, False], ids=["known_variance", "unknown_variance"]
)


def _intervals(*bounds):
    """Return an oracle admitting the union of the open (a, b) intervals."""

    def phi(z):
        kept = ch.zeros_like(z, dtype=ch.bool)
        for low, high in bounds:
            kept |= (z > low) & (z < high)
        return kept

    return phi


def _fit(  # pylint: disable=too-many-arguments
    simulation: regression.Simulation,
    variance_known: bool,
    known: float,
    unknown: tuple[float, float],
    *,
    alpha_known: bool = True,
    ols_margin: Optional[float] = 0.5,
    fit_intercept: bool = True,
) -> None:
    """Fit regression to a simulation as one variant and check its estimates.

    Args:
        simulation: The sample and its truth.
        variance_known: Whether the model is given the noise variance.
        known: The weight's tolerance when the variance is given.
        unknown: The weight's and the variance's tolerances when it is not.
        alpha_known: Whether the model is given the survival mass.
        ols_margin: As ``regression.assert_recovers``.
        fit_intercept: Whether to fit an intercept.
    """
    model = truncated_linear_regression_module.TruncatedLinearRegression(
        {},
        simulation.phi,
        simulation.alpha if alpha_known else None,
        noise_var=simulation.noise_var if variance_known else None,
        fit_intercept=fit_intercept,
    )
    tolerance, variance_tolerance = (known, None) if variance_known else unknown
    regression.assert_recovers(
        model,
        simulation,
        tolerance,
        variance_tolerance=variance_tolerance,
        ols_margin=ols_margin,
    )


def _case(
    coef: Sequence[float],
    intercept: float,
    phi: Callable[[ch.Tensor], ch.Tensor],
    n: int,
    **options,
) -> Callable[[int], regression.Simulation]:
    """Return a map from a seed to a sample, as ``regression.simulate``'s."""

    def simulate(seed: int) -> regression.Simulation:
        return regression.simulate(coef, intercept, phi, n, seed=seed, **options)

    return simulate


def _ill_conditioned_covariates(seed: int) -> regression.Simulation:
    """Return the 10d sample whose covariance spans three decades."""
    ch.manual_seed(1)
    rotation, _ = ch.linalg.qr(ch.randn(10, 10))  # pylint: disable=not-callable
    spectrum = ch.logspace(0.0, -3.0, 10)
    covariance = rotation @ ch.diag(spectrum) @ rotation.T
    return regression.simulate(
        _ONES_10D, 0.5, _intervals((1.0, INF)), 5000, covariance=covariance, seed=seed
    )


def _coin_flip(z: ch.Tensor) -> ch.Tensor:
    """Return a survival probability: each response is kept by a coin flip."""
    return ch.sigmoid(3.0 * (z - 4.0))


_ONES_10D = ch.ones(10).tolist()
_UNION_2D = [(-INF, -2.0), (0.0, 1.0), (2.5, INF)]
_CORRELATED_2D = ch.tensor([[1.0, 0.5], [0.5, 1.0]])
# The truncation cases of the regression tests, which the opt-in coverage
# sweep reuses.
_CASES: dict[str, Callable[[int], regression.Simulation]] = {
    "1d_no_truncation": _case([2.0], -1.0, _intervals((-INF, INF)), 2000),
    "1d_left_truncation": _case([2.0], -1.0, _intervals((0.0, INF)), 2000),
    "1d_left_truncation_no_intercept": _case([2.0], 0.0, _intervals((0.0, INF)), 2000),
    "1d_right_truncation": _case([2.0], -1.0, _intervals((-INF, -1.0)), 2000),
    "1d_interval": _case([2.0], -1.0, _intervals((-1.0, 1.0)), 2000),
    "1d_union_of_intervals": _case(
        [2.0], -1.0, _intervals((-INF, -3.0), (-1.5, -0.5), (1.0, INF)), 2000
    ),
    "1d_small_mass": _case([2.0], -1.0, _intervals((3.85, INF)), 2000),
    "2d_union_of_intervals": _case(
        [1.0, -1.0], 0.5, _intervals(*_UNION_2D), 2000, covariance=_CORRELATED_2D
    ),
    "2d_two_sided_tails": _case(
        [1.0, -1.0], 0.5, _intervals((-INF, -1.5), (1.5, INF)), 2000
    ),
    "10d_left_truncation": _case(_ONES_10D, 0.5, _intervals((4.0, INF)), 5000),
    "10d_union_of_intervals": _case(
        _ONES_10D, 0.5, _intervals((-2.0, 0.0), (3.0, INF)), 5000
    ),
    "10d_no_truncation": _case(_ONES_10D, 0.5, _intervals((-INF, INF)), 5000),
    "100d_union_of_intervals": _case(
        (ch.ones(100) / math.sqrt(10)).tolist(),
        0.5,
        _intervals((-INF, -8.0), (-3.0, 3.0), (8.0, INF)),
        20000,
    ),
    "10d_randomized_truncation": _case(_ONES_10D, 0.5, _coin_flip, 5000),
    "10d_ill_conditioned_covariates": _ill_conditioned_covariates,
    "10d_rescaled_covariates": _case(
        (1e-3 * ch.ones(10)).tolist(),
        0.5,
        _intervals((4.0, INF)),
        5000,
        covariance=1e6 * ch.eye(10),
    ),
    "1d_left_truncation_noise_var_5": _case(
        [2.0], -1.0, _intervals((0.0, INF)), 2000, noise_var=5.0
    ),
    "1d_left_truncation_noise_var_10": _case(
        [2.0], -1.0, _intervals((0.0, INF)), 2000, noise_var=10.0
    ),
    "10d_left_truncation_noise_var_5": _case(
        _ONES_10D, 0.5, _intervals((4.0, INF)), 5000, noise_var=5.0
    ),
    "10d_left_truncation_noise_var_10": _case(
        _ONES_10D, 0.5, _intervals((4.0, INF)), 5000, noise_var=10.0
    ),
    "10d_union_of_intervals_noise_var_10": _case(
        _ONES_10D, 0.5, _intervals((-2.0, 0.0), (3.0, INF)), 5000, noise_var=10.0
    ),
}


def _simulate(case: str, seed: int = 0) -> regression.Simulation:
    """Return one of ``_CASES``' samples."""
    return _CASES[case](seed)


@_VARIANTS
def test_1d_no_truncation(variance_known):
    _fit(
        _simulate("1d_no_truncation"),
        variance_known,
        0.04,
        (0.04, 0.02),
        alpha_known=False,
        ols_margin=None,
    )


@_VARIANTS
def test_1d_left_truncation(variance_known):
    _fit(_simulate("1d_left_truncation"), variance_known, 0.18, (0.31, 0.10))


@_VARIANTS
def test_1d_left_truncation_no_intercept(variance_known):
    simulation = _simulate("1d_left_truncation_no_intercept")
    _fit(simulation, variance_known, 0.005, (0.02, 0.14), fit_intercept=False)


@_VARIANTS
def test_1d_right_truncation(variance_known):
    _fit(_simulate("1d_right_truncation"), variance_known, 0.06, (0.05, 0.11))


@_VARIANTS
def test_1d_interval(variance_known):
    _fit(_simulate("1d_interval"), variance_known, 0.35, (0.54, 0.10))


@_VARIANTS
def test_1d_union_of_intervals(variance_known):
    _fit(_simulate("1d_union_of_intervals"), variance_known, 0.02, (0.02, 0.05))


@_VARIANTS
def test_1d_small_mass(variance_known):
    # The sampled loss picking best_ cannot resolve this flat valley (Hessian
    # condition number 2,300): best_ lands at 0.39 and variance error 0.08,
    # the others at 0.16 and 0.03.
    _fit(_simulate("1d_small_mass"), variance_known, 0.15, (0.45, 0.09))


@_VARIANTS
def test_2d_union_of_intervals(variance_known):
    _fit(_simulate("2d_union_of_intervals"), variance_known, 0.09, (0.08, 0.03))


@_VARIANTS
def test_2d_two_sided_tails(variance_known):
    _fit(_simulate("2d_two_sided_tails"), variance_known, 0.08, (0.07, 0.05))


@_VARIANTS
def test_10d_left_truncation(variance_known):
    # Daskalakis et al. (2020), Section 6: w = 1, kept only where y > 4.
    _fit(_simulate("10d_left_truncation"), variance_known, 0.16, (0.12, 0.04))


@_VARIANTS
def test_10d_union_of_intervals(variance_known):
    _fit(_simulate("10d_union_of_intervals"), variance_known, 0.11, (0.11, 0.01))


@_VARIANTS
def test_10d_no_truncation(variance_known):
    _fit(
        _simulate("10d_no_truncation"),
        variance_known,
        0.09,
        (0.09, 0.02),
        alpha_known=False,
        ols_margin=None,
    )


@pytest.mark.slow
@_VARIANTS
def test_100d_union_of_intervals(variance_known):
    _fit(_simulate("100d_union_of_intervals"), variance_known, 0.15, (0.15, 0.01))


@_VARIANTS
def test_10d_randomized_truncation(variance_known):
    simulation = _simulate("10d_randomized_truncation")
    _fit(simulation, variance_known, 0.13, (0.15, 0.02))


@_VARIANTS
def test_10d_alpha_unknown(variance_known):
    simulation = _simulate("10d_left_truncation")
    _fit(simulation, variance_known, 0.16, (0.12, 0.04), alpha_known=False)


@_VARIANTS
def test_10d_ill_conditioned_covariates(variance_known):
    simulation = _simulate("10d_ill_conditioned_covariates")
    # The least-variance direction, 1e-3, sets the MLE's error: 0.93.
    _fit(simulation, variance_known, 1.3, (1.3, 0.09), ols_margin=0.8)


@_VARIANTS
def test_10d_rescaled_covariates(variance_known):
    _fit(_simulate("10d_rescaled_covariates"), variance_known, 0.12, (0.05, 0.04))


@_VARIANTS
@pytest.mark.parametrize(
    "seed, unknown", [(1, (0.15, 0.02)), (2, (0.12, 0.02)), (3, (0.29, 0.13))]
)
def test_10d_seeds(variance_known, seed, unknown):
    _fit(_simulate("10d_left_truncation", seed), variance_known, 0.13, unknown)


@_VARIANTS
def test_1d_left_truncation_noise_var_5(variance_known):
    simulation = _simulate("1d_left_truncation_noise_var_5")
    _fit(simulation, variance_known, 0.07, (0.91, 1.49))


@_VARIANTS
def test_1d_left_truncation_noise_var_10(variance_known):
    simulation = _simulate("1d_left_truncation_noise_var_10")
    _fit(simulation, variance_known, 0.33, (0.37, 1.47))


@_VARIANTS
def test_10d_left_truncation_noise_var_5(variance_known):
    simulation = _simulate("10d_left_truncation_noise_var_5")
    _fit(simulation, variance_known, 0.26, (0.26, 0.12))


@_VARIANTS
def test_10d_left_truncation_noise_var_10(variance_known):
    simulation = _simulate("10d_left_truncation_noise_var_10")
    # The EMA, which the Newton steps do not refine, trails at 0.71.
    _fit(simulation, variance_known, 0.47, (0.55, 0.8))


@_VARIANTS
def test_10d_union_of_intervals_noise_var_10(variance_known):
    simulation = _simulate("10d_union_of_intervals_noise_var_10")
    _fit(simulation, variance_known, 0.40, (0.39, 0.22))


@pytest.mark.slow
def test_truncated_dependent_regression():
    ch.manual_seed(SEED)
    D = 10  # number of dimensions for A_{*} matrix
    T = 10000  # uncensored system trajectory length

    spectral_norm = float("inf")
    while spectral_norm > 1.0:
        A = 0.25 * ch.randn((D, D))
        spectral_norm = helpers.calc_spectral_norm(A)

    spectral_norm = helpers.calc_spectral_norm(A)
    print(f"A spectral norm: {helpers.calc_spectral_norm(A)}")

    phi_bound = 4.0

    def phi(x):
        return x.norm(dim=-1, keepdim=True) <= phi_bound

    X, Y = ch.Tensor([]), ch.Tensor([])
    NOISE_VAR = ch.eye(D)
    M = ch.distributions.MultivariateNormal(ch.zeros(D), NOISE_VAR)
    x_t = ch.zeros((1, D))
    for i in range(T):
        noise = M.sample()
        y_t = x_t @ A + noise
        if phi(y_t):  # returns a boolean
            X = ch.cat([X, x_t])
            Y = ch.cat([Y, y_t])
        x_t = y_t

    alpha = X.size(0) / T

    print(f"alpha: {alpha}")

    train_kwargs = {
        "c_eta": 0.5,
        "epochs": 1,
        "trials": 1,
        "batch_size": 1,
        "num_samples": 100,
        "T": X.size(0),
        "c_s": 10.0,
        "alpha": alpha,
        "tol": 1e-1,
        "c_gamma": 2.0,
    }
    trunc_lds = truncated_linear_regression_module.TruncatedLinearRegression(
        train_kwargs,
        phi,
        train_kwargs["alpha"],
        noise_var=NOISE_VAR,
        dependent=True,
        rand_seed=SEED,
    )
    trunc_lds.fit(X, Y)
    # coef_ is (outputs, inputs), and y_t = x_t A, so it estimates A^T.
    A_ = trunc_lds.coef_.T
    A0_ = trunc_lds.emp_coef_.T
    trunc_spec_norm = helpers.calc_spectral_norm(A - A_)
    emp_spec_norm = helpers.calc_spectral_norm(A - A0_)

    print(f"alpha: {alpha}")
    print(f"A spectral norm: {spectral_norm}")
    print(f"truncated spectral norm: {trunc_spec_norm}")
    print(f"ols spectral norm: {emp_spec_norm}")

    assert trunc_spec_norm <= emp_spec_norm, (
        f"truncated spectral norm {trunc_spec_norm}, while OLS spectral norm is: {emp_spec_norm}"
    )


def test_dependent_step_is_preconditioned_by_sigma():
    """SON-SG's step is -lr Sigma^-1 grad, with Sigma updated by the batch."""
    ch.manual_seed(SEED)
    dims, num_samples = 3, 200
    X = ch.randn(num_samples, dims)
    Y = X @ (0.5 * ch.eye(dims)) + ch.randn(num_samples, dims)

    def phi(x):
        return x.norm(dim=-1, keepdim=True) <= 10.0

    model = truncated_linear_regression_module.TruncatedLinearRegression(
        {"iterations": 1, "num_samples": 10},
        phi,
        0.9,
        noise_var=ch.eye(dims),
        dependent=True,
    ).fit(X, Y)
    model.trainer._reset_run_state()  # pylint: disable=protected-access
    design, targets = model._data  # pylint: disable=protected-access
    inp, targ = design[:50], targets[:50]
    start, sigma = model.weight.detach().clone(), model.Sigma.clone()  # pylint: disable=no-member

    ch.manual_seed(1)
    model.trainer.train_step((inp, targ))
    step = model.weight.detach() - start

    sigma = sigma + ch.einsum("ni,nj->ij", inp, inp) / inp.size(0)
    weight = start.clone().requires_grad_()
    ch.manual_seed(1)
    loss = model.required_criterion(inp @ weight, targ, *model.criterion_params)  # pylint: disable=not-callable
    (plain,) = ch.autograd.grad(loss.sum(), weight)
    lr = model.optimizer.param_groups[0]["lr"]
    expected = -lr * ch.linalg.solve(sigma, plain)  # pylint: disable=not-callable
    assert ch.allclose(step, expected, atol=1e-5), (step, expected)
    assert not ch.allclose(step, -lr * plain, atol=1e-5)


def _unknown_variance_set() -> truncated_linear_regression.UnknownVarianceSet:
    residual_var = ch.tensor([1.0, 4.0])
    anchor = ch.cat([ch.zeros(2, 2), (1.0 / residual_var)[None]])
    return truncated_linear_regression.UnknownVarianceSet(
        residual_var, 2.0, anchor, bound=1.5
    )


def test_unknown_variance_set_bounds_the_precision_by_the_survival_mass():
    projection = _unknown_variance_set()
    lower, upper = projection.precision_bounds()
    # alpha = 1 / radius = 1 / 2.
    residual_var = ch.tensor([1.0, 4.0])
    assert ch.allclose(lower, 0.25 / (96.0 * residual_var))
    assert ch.allclose(upper, 8.0 * (5.0 + 2.0 * math.log(2.0)) / residual_var)


def test_unknown_variance_set_projects_to_the_nearest_point():
    projection = _unknown_variance_set()
    lower, upper = projection.precision_bounds()
    ch.manual_seed(0)
    # Feasible points, to check no candidate beats the projection.
    precision = lower + (upper - lower) * ch.rand(20000, 2)
    direction = torch.nn.functional.normalize(ch.randn(20000, 2, 2), dim=1)
    radius = 1.5 * precision * ch.rand(20000, 2)
    feasible = ch.cat([direction * radius[:, None], precision[:, None]], dim=1)
    for scale in [0.1, 3.0, 100.0]:
        point = scale * ch.randn(3, 2)
        projected = projection.project(point)
        assert projection.contains(projected + 1e-6 * (projection.anchor - projected))
        nearest = (feasible - point).norm(dim=1).min(0).values
        assert ((projected - point).norm(dim=0) <= nearest + 1e-5).all()
        assert ch.allclose(projection.project(projected), projected)


def _log_survival_mass(
    mean: ch.Tensor, noise_sd: ch.Tensor, bounds: Bounds
) -> ch.Tensor:
    """Return log P(z in the union of bounds) for z ~ N(mean, noise_sd^2).

    A finite interval above the mean is reflected below it, where the
    difference of normal CDFs keeps its precision.

    Args:
        mean: The untruncated conditional mean ``w^T x`` at each sample.
        noise_sd: The noise's sd: the known one, or ``lambda^(-1/2)``.
        bounds: The intervals whose union is the truncation set.
    """
    terms = []
    for low, high in bounds:
        lower, upper = (low - mean) / noise_sd, (high - mean) / noise_sd
        if low == -INF:
            terms.append(ch.special.log_ndtr(upper))  # pylint: disable=not-callable
        elif high == INF:
            terms.append(ch.special.log_ndtr(-lower))  # pylint: disable=not-callable
        else:
            above = lower > 0.0
            left = ch.where(above, -upper, lower)
            right = ch.special.log_ndtr(ch.where(above, -lower, upper))  # pylint: disable=not-callable
            terms.append(right + ch.log1p(-ch.exp(ch.special.log_ndtr(left) - right)))  # pylint: disable=not-callable
    return ch.logsumexp(ch.stack(terms), 0)


def _truncated_nll_unknown_variance(
    natural_params: ch.Tensor, design: ch.Tensor, y: ch.Tensor, bounds: Bounds
) -> ch.Tensor:
    """Return the mean truncated NLL of regression with unknown noise variance.

    In closed form through normal CDFs, so exact where the model samples.

    Args:
        natural_params: The natural parameters the model trains, ``(v,
            lambda) = (w / sigma^2, 1 / sigma^2)``: v's entries, then the
            noise precision lambda.
        design: The design matrix, intercept column included, shaped (n, d).
        y: The responses, shaped (n,).
        bounds: As ``_log_survival_mass``.
    """
    v, precision = natural_params[:-1], natural_params[-1]
    mean = design @ v / precision
    nll = (
        0.5 * precision * (y - mean).pow(2)
        - 0.5 * ch.log(precision)
        + _log_survival_mass(mean, precision.rsqrt(), bounds)
    )
    return nll.mean()


def _truncated_nll_known_variance(
    weight: ch.Tensor,
    design: ch.Tensor,
    y: ch.Tensor,
    noise_var: float,
    bounds: Bounds,
) -> ch.Tensor:
    """Return the mean truncated NLL of regression with known noise variance.

    Exact, as ``_truncated_nll_unknown_variance``.

    Args:
        weight: The weight w, intercept last, shaped (d,).
        design: As ``_truncated_nll_unknown_variance``.
        y: As ``_truncated_nll_unknown_variance``.
        noise_var: The known noise variance sigma^2.
        bounds: As ``_log_survival_mass``.
    """
    mean = design @ weight
    noise_sd = ch.tensor(math.sqrt(noise_var), dtype=ch.float64)
    nll = (y - mean).pow(2) / (2.0 * noise_var) + _log_survival_mass(
        mean, noise_sd, bounds
    )
    return nll.mean()


def _design_and_responses(
    simulation: regression.Simulation,
) -> tuple[ch.Tensor, ch.Tensor]:
    """Return a sample's design, with an intercept column, and y in float64."""
    ones = ch.ones(simulation.X.size(0), 1)
    design = ch.cat([simulation.X, ones], dim=1).double()
    return design, simulation.y.double().flatten()


def test_unknown_variance_covariance_inverts_the_exact_information():
    simulation = _simulate("2d_union_of_intervals")
    model = truncated_linear_regression_module.TruncatedLinearRegression(
        {}, simulation.phi, simulation.alpha, noise_var=None
    ).fit(simulation.X, simulation.y)
    variance = model.avg_variance_
    weight = ch.cat([model.avg_coef_.flatten(), model.avg_intercept_])
    natural_params = ch.cat([weight / variance, 1.0 / variance])
    design, y = _design_and_responses(simulation)
    hessian = torch.autograd.functional.hessian(
        lambda theta: _truncated_nll_unknown_variance(theta, design, y, _UNION_2D),
        natural_params,
    )
    jacobian = torch.autograd.functional.jacobian(
        lambda theta: ch.cat([theta[:-1], ch.ones(1)]) / theta[-1], natural_params
    )
    n = design.size(0)
    expected = jacobian @ ch.linalg.solve(n * hessian, jacobian.T)  # pylint: disable=not-callable
    error = (model.covariance() - expected).norm() / expected.norm()
    assert error <= 0.02, (model.covariance(), expected)


def test_known_variance_covariance_inverts_the_exact_information():
    simulation = _simulate("2d_union_of_intervals")
    model = truncated_linear_regression_module.TruncatedLinearRegression(
        {}, simulation.phi, simulation.alpha, noise_var=simulation.noise_var
    ).fit(simulation.X, simulation.y)
    weight = ch.cat([model.avg_coef_.flatten(), model.avg_intercept_]).double()
    design, y = _design_and_responses(simulation)
    hessian = torch.autograd.functional.hessian(
        lambda theta: _truncated_nll_known_variance(
            theta, design, y, simulation.noise_var, _UNION_2D
        ),
        weight,
    )
    expected = ch.linalg.inv(design.size(0) * hessian)  # pylint: disable=not-callable
    error = (model.covariance() - expected).norm() / expected.norm()
    assert error <= 0.02, (model.covariance(), expected)
    intervals = model.confidence_intervals()
    assert intervals.intercept is not None and intervals.variance is None


_COVERAGE_CASES = {
    "1d_left_truncation": _case([2.0], -1.0, _intervals((0.0, INF)), 1000),
    "2d_union_of_intervals": _case(
        [1.0, -1.0], 0.5, _intervals(*_UNION_2D), 1000, covariance=_CORRELATED_2D
    ),
}
_REPLICATIONS = 200
_CHUNKS = 8
# These move the learned variance's estimates by about a hundredth of their
# sd, at a fifth of the default's time.
_FAST_UNKNOWN = {"iterations": 2000, "num_samples": 100}
# This keeps the known variance's within 0.04 sd of the exact MLE.
_FAST_KNOWN = {"num_samples": 100}


def _coverage_hits(
    simulate: Callable[[int], regression.Simulation],
    variance_known: bool,
    chunk: int,
    hyperparameters: dict,
) -> ch.Tensor:
    """Return how many of a chunk's samples' 95% intervals cover each truth.

    Args:
        simulate: Maps a seed to a sample.
        variance_known: Whether the model is given the noise variance.
        chunk: Which of the ``_CHUNKS`` slices of the replications' seeds.
        hyperparameters: The model's.

    Returns:
        Each entry's hits: coefficients, intercept, then the learned
        variance.
    """
    size = _REPLICATIONS // _CHUNKS
    covered = []
    for seed in range(chunk * size, (chunk + 1) * size):
        simulation = simulate(seed)
        model = truncated_linear_regression_module.TruncatedLinearRegression(
            hyperparameters,
            simulation.phi,
            simulation.alpha,
            noise_var=simulation.noise_var if variance_known else None,
        ).fit(simulation.X, simulation.y)
        intervals = model.confidence_intervals()
        truths = [
            (intervals.coef, simulation.coef),
            (intervals.intercept, simulation.intercept),
            (intervals.variance, ch.tensor([simulation.noise_var])),
        ]
        covered.append(
            ch.cat(
                [
                    ((interval.lower <= truth) & (truth <= interval.upper)).flatten()
                    for interval, truth in truths
                    if interval is not None
                ]
            )
        )
    return ch.stack(covered).long().sum(0)


def _pool(
    tmp_path_factory: pytest.TempPathFactory, run: str, chunk: int, hits: ch.Tensor
) -> Optional[ch.Tensor]:
    """Record a chunk's hits; return all chunks' summed once each has run.

    xdist workers share no memory, so the hits go to files under the run's
    temporary root, which every worker's basetemp sits in. A chunk writes
    before it lists, so whichever finishes last sees every file; the first
    to claim the folder sums and deletes it.
    """
    root = tmp_path_factory.getbasetemp()
    if "PYTEST_XDIST_WORKER" in os.environ:
        root = root.parent
    folder = root / "coverage" / run
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / f"{chunk}.partial"
    ch.save(hits, partial)
    os.replace(partial, folder / f"{chunk}.pt")
    done = list(folder.glob("*.pt"))
    if len(done) < _CHUNKS:
        return None
    try:
        (folder / "claimed").mkdir()
    except FileExistsError:
        return None
    pooled = sum((ch.load(path) for path in done), ch.zeros_like(hits))
    shutil.rmtree(folder)
    try:
        folder.parent.rmdir()
    except OSError:
        pass  # Another run's chunks are still pooling there.
    return pooled


def _check_coverage(  # pylint: disable=too-many-arguments
    simulate: Callable[[int], regression.Simulation],
    variance_known: bool,
    hyperparameters: dict,
    run: str,
    chunk: int,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Run a chunk; once all have, check the pooled coverage of each entry.

    Its standard error is sqrt(0.95 * 0.05 / 200) = 0.015, and the bounds
    sit three of them from 0.95.
    """
    hits = _coverage_hits(simulate, variance_known, chunk, hyperparameters)
    pooled = _pool(tmp_path_factory, run, chunk, hits)
    if pooled is not None:
        coverage = pooled.double() / _REPLICATIONS
        assert ((coverage >= 0.905) & (coverage <= 0.995)).all(), coverage


# The replications run in chunks that xdist spreads over its workers.
@pytest.mark.slow
@pytest.mark.parametrize("chunk", range(_CHUNKS))
@pytest.mark.parametrize("case", list(_COVERAGE_CASES))
@_VARIANTS
def test_intervals_cover_at_their_level(variance_known, case, chunk, tmp_path_factory):
    _check_coverage(
        _COVERAGE_CASES[case],
        variance_known,
        _FAST_KNOWN if variance_known else _FAST_UNKNOWN,
        f"{variance_known}-{case}",
        chunk,
        tmp_path_factory,
    )


# Every regression case, at the defaults; deselected unless --coverage-sweep.
@pytest.mark.coverage_sweep
@pytest.mark.parametrize("chunk", range(_CHUNKS))
@pytest.mark.parametrize("case", list(_CASES))
@_VARIANTS
def test_intervals_cover_at_their_level_on_every_case(
    variance_known, case, chunk, tmp_path_factory
):
    _check_coverage(
        _CASES[case],
        variance_known,
        {},
        f"sweep-{variance_known}-{case}",
        chunk,
        tmp_path_factory,
    )
