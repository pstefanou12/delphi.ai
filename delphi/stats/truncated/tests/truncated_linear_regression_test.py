# Author: pstefanou12@
"""Tests for truncated linear regression.

Each truncation-set case runs twice on the shared harness: with the noise
variance given, and with it learned (``noise_var=None``). Each tolerance is
about twice the error of the exact truncated MLE on the same sample,
measured with the closed-form NLL, over the weight alone when the variance is
known and over (weight, variance) when it is not; every estimate lands within
a few hundredths of that MLE. The dependent-data cases follow.
"""

import math

import pytest
import torch as ch

from delphi import stats
from delphi.stats.truncated.tests import regression
from delphi.utils import helpers

INF = math.inf
SEED = 69

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
    dataset,
    variance_known,
    known,
    unknown,
    *,
    alpha="known",
    ols_margin=0.5,
    fit_intercept=True,
):
    """Fit regression to dataset as one variant and check its estimates.

    Args:
        dataset: The sample.
        variance_known: Whether the model is given the noise variance.
        known: The weight's tolerance when the variance is given.
        unknown: The weight's and the variance's tolerances when it is not.
        alpha: "known" passes the dataset's survival mass; else this value.
        ols_margin: As ``regression.assert_recovers``.
        fit_intercept: Whether to fit an intercept.
    """
    model = stats.TruncatedLinearRegression(
        {},
        dataset.phi,
        dataset.alpha if alpha == "known" else alpha,
        noise_var=dataset.noise_var if variance_known else None,
        fit_intercept=fit_intercept,
    )
    tolerance, variance_tolerance = (known, None) if variance_known else unknown
    regression.assert_recovers(
        model,
        dataset,
        tolerance,
        variance_tolerance=variance_tolerance,
        ols_margin=ols_margin,
    )


@_VARIANTS
def test_1d_no_truncation(variance_known):
    dataset = regression.make_dataset([2.0], -1.0, _intervals((-INF, INF)), 2000)
    _fit(dataset, variance_known, 0.04, (0.04, 0.02), alpha=None, ols_margin=None)


@_VARIANTS
def test_1d_left_truncation(variance_known):
    dataset = regression.make_dataset([2.0], -1.0, _intervals((0.0, INF)), 2000)
    _fit(dataset, variance_known, 0.18, (0.31, 0.10))


@_VARIANTS
def test_1d_left_truncation_no_intercept(variance_known):
    dataset = regression.make_dataset([2.0], 0.0, _intervals((0.0, INF)), 2000)
    _fit(dataset, variance_known, 0.005, (0.02, 0.14), fit_intercept=False)


@_VARIANTS
def test_1d_right_truncation(variance_known):
    dataset = regression.make_dataset([2.0], -1.0, _intervals((-INF, -1.0)), 2000)
    _fit(dataset, variance_known, 0.06, (0.05, 0.11))


@_VARIANTS
def test_1d_interval(variance_known):
    dataset = regression.make_dataset([2.0], -1.0, _intervals((-1.0, 1.0)), 2000)
    _fit(dataset, variance_known, 0.35, (0.54, 0.10))


@_VARIANTS
def test_1d_union_of_intervals(variance_known):
    phi = _intervals((-INF, -3.0), (-1.5, -0.5), (1.0, INF))
    dataset = regression.make_dataset([2.0], -1.0, phi, 2000)
    _fit(dataset, variance_known, 0.02, (0.02, 0.05))


@_VARIANTS
def test_1d_small_mass(variance_known):
    dataset = regression.make_dataset([2.0], -1.0, _intervals((3.85, INF)), 2000)
    _fit(dataset, variance_known, 0.15, (0.33, 0.06))


@_VARIANTS
def test_2d_union_of_intervals(variance_known):
    phi = _intervals((-INF, -2.0), (0.0, 1.0), (2.5, INF))
    covariance = ch.tensor([[1.0, 0.5], [0.5, 1.0]])
    dataset = regression.make_dataset(
        [1.0, -1.0], 0.5, phi, 2000, covariance=covariance
    )
    _fit(dataset, variance_known, 0.09, (0.08, 0.03))


@_VARIANTS
def test_2d_two_sided_tails(variance_known):
    phi = _intervals((-INF, -1.5), (1.5, INF))
    dataset = regression.make_dataset([1.0, -1.0], 0.5, phi, 2000)
    _fit(dataset, variance_known, 0.08, (0.07, 0.05))


@_VARIANTS
def test_10d_left_truncation(variance_known):
    # Daskalakis et al. (2020), Section 6: w = 1, kept only where y > 4.
    dataset = regression.make_dataset(
        ch.ones(10).tolist(), 0.5, _intervals((4.0, INF)), 5000
    )
    _fit(dataset, variance_known, 0.16, (0.12, 0.04))


@_VARIANTS
def test_10d_union_of_intervals(variance_known):
    phi = _intervals((-2.0, 0.0), (3.0, INF))
    dataset = regression.make_dataset(ch.ones(10).tolist(), 0.5, phi, 5000)
    _fit(dataset, variance_known, 0.11, (0.11, 0.01))


@_VARIANTS
def test_10d_no_truncation(variance_known):
    dataset = regression.make_dataset(
        ch.ones(10).tolist(), 0.5, _intervals((-INF, INF)), 5000
    )
    _fit(dataset, variance_known, 0.09, (0.09, 0.02), alpha=None, ols_margin=None)


@pytest.mark.slow
@_VARIANTS
def test_100d_union_of_intervals(variance_known):
    phi = _intervals((-INF, -8.0), (-3.0, 3.0), (8.0, INF))
    coef = (ch.ones(100) / math.sqrt(10)).tolist()
    dataset = regression.make_dataset(coef, 0.5, phi, 20000)
    _fit(dataset, variance_known, 0.15, (0.15, 0.01))


@_VARIANTS
def test_10d_randomized_truncation(variance_known):
    def phi(z):
        return ch.sigmoid(3.0 * (z - 4.0))

    dataset = regression.make_dataset(ch.ones(10).tolist(), 0.5, phi, 5000)
    _fit(dataset, variance_known, 0.13, (0.15, 0.02))


@_VARIANTS
def test_10d_alpha_unknown(variance_known):
    dataset = regression.make_dataset(
        ch.ones(10).tolist(), 0.5, _intervals((4.0, INF)), 5000
    )
    _fit(dataset, variance_known, 0.16, (0.12, 0.04), alpha=None)


@_VARIANTS
def test_10d_ill_conditioned_covariates(variance_known):
    ch.manual_seed(1)
    rotation, _ = ch.linalg.qr(ch.randn(10, 10))  # pylint: disable=not-callable
    spectrum = ch.logspace(0.0, -3.0, 10)
    covariance = rotation @ ch.diag(spectrum) @ rotation.T
    dataset = regression.make_dataset(
        ch.ones(10).tolist(), 0.5, _intervals((1.0, INF)), 5000, covariance=covariance
    )
    # The least-variance direction, 1e-3, sets the MLE's error: 0.93.
    _fit(dataset, variance_known, 1.3, (1.3, 0.09), ols_margin=0.8)


@_VARIANTS
def test_10d_rescaled_covariates(variance_known):
    covariance = 1e6 * ch.eye(10)
    coef = (1e-3 * ch.ones(10)).tolist()
    dataset = regression.make_dataset(
        coef, 0.5, _intervals((4.0, INF)), 5000, covariance=covariance
    )
    _fit(dataset, variance_known, 0.12, (0.05, 0.04))


@_VARIANTS
@pytest.mark.parametrize(
    "seed, unknown", [(1, (0.15, 0.02)), (2, (0.12, 0.02)), (3, (0.29, 0.13))]
)
def test_10d_seeds(variance_known, seed, unknown):
    dataset = regression.make_dataset(
        ch.ones(10).tolist(), 0.5, _intervals((4.0, INF)), 5000, seed=seed
    )
    _fit(dataset, variance_known, 0.13, unknown)


@_VARIANTS
def test_1d_left_truncation_noise_var_5(variance_known):
    dataset = regression.make_dataset(
        [2.0], -1.0, _intervals((0.0, INF)), 2000, noise_var=5.0
    )
    _fit(dataset, variance_known, 0.07, (0.91, 1.49))


@_VARIANTS
def test_1d_left_truncation_noise_var_10(variance_known):
    dataset = regression.make_dataset(
        [2.0], -1.0, _intervals((0.0, INF)), 2000, noise_var=10.0
    )
    _fit(dataset, variance_known, 0.33, (0.37, 1.47))


@_VARIANTS
def test_10d_left_truncation_noise_var_5(variance_known):
    dataset = regression.make_dataset(
        ch.ones(10).tolist(), 0.5, _intervals((4.0, INF)), 5000, noise_var=5.0
    )
    _fit(dataset, variance_known, 0.26, (0.26, 0.12))


@_VARIANTS
def test_10d_left_truncation_noise_var_10(variance_known):
    dataset = regression.make_dataset(
        ch.ones(10).tolist(), 0.5, _intervals((4.0, INF)), 5000, noise_var=10.0
    )
    _fit(dataset, variance_known, 0.47, (0.55, 0.47))


@_VARIANTS
def test_10d_union_of_intervals_noise_var_10(variance_known):
    phi = _intervals((-2.0, 0.0), (3.0, INF))
    dataset = regression.make_dataset(
        ch.ones(10).tolist(), 0.5, phi, 5000, noise_var=10.0
    )
    _fit(dataset, variance_known, 0.40, (0.39, 0.22))


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
    trunc_lds = stats.TruncatedLinearRegression(
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

    model = stats.TruncatedLinearRegression(
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
