# Author: pstefanou12@
"""Tests for the multivariate normal under unknown truncation.

None of these fits sees the truncation set. The estimator's error at a fixed
Hermite degree k does not vanish with the sample size: the degree-k
expansion of the weighted indicator is biased, by ``O(k ** (-1 / 4))`` in
Kontonis et al. (2019)'s bound. The fits use the default degree, the largest
within the term budget, and the tolerances are the suite's usual ones. The
truths have nonzero means and correlated covariances, and the sets cut away
from the mean, so the truncation shifts and reshapes the sample.
"""

from typing import Optional

import pytest
import torch as ch
from torch import distributions
from torch.distributions import kl

import delphi.oracle
from delphi.truncated.distributions import losses, unknown_truncated_multivariate_normal
from delphi.truncated.distributions.tests import helpers, truncation_sets
from delphi.utils import datasets

ch.set_printoptions(precision=4, sci_mode=False)

CORRELATED = ch.tensor([[2.0, 0.5], [0.5, 1.0]])
CORRELATED_3D = ch.tensor([[1.5, 0.4, -0.3], [0.4, 1.0, 0.2], [-0.3, 0.2, 0.8]])


def _correlated(dims: int, seed: int) -> ch.Tensor:
    """Return a fixed random covariance with correlated coordinates."""
    generator = ch.Generator().manual_seed(seed)
    factor = 0.4 * ch.randn(dims, dims, generator=generator)
    return factor @ factor.T + 0.6 * ch.eye(dims)


def _two_half_spaces(z: ch.Tensor) -> ch.Tensor:
    return (z[:, 0] > -0.5) & (z[:, 1] > -0.5)


def _model(
    known_covariance: bool,
    dims: int,
    covariance_matrix: ch.Tensor,
    alpha: Optional[float] = None,
    hyperparameters: Optional[dict] = None,
) -> unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalBase:
    """Return the known- or unknown-covariance estimator."""
    if known_covariance:
        return unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalKnownCovariance(
            hyperparameters or {}, dims, covariance_matrix, alpha
        )
    return unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        hyperparameters or {}, dims, alpha
    )


def _fitted(
    model: unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalBase,
    name: str,
    covariance_matrix: Optional[ch.Tensor],
) -> distributions.MultivariateNormal:
    """Return one estimator's fit; covariance_matrix is the known one, if any."""
    fitted_covariance = (
        getattr(model, f"{name}_covariance_matrix_")
        if covariance_matrix is None
        else covariance_matrix
    )
    return distributions.MultivariateNormal(
        getattr(model, f"{name}_loc_"), fitted_covariance
    )


def _recovers(
    loc: ch.Tensor,
    covariance_matrix: ch.Tensor,
    phi: helpers.Oracle,
    known_covariance: bool,
    dataset_size: int = 20000,
    tolerances: Optional[dict[str, float]] = None,
) -> None:
    """Fit on a truncated sample and assert the fit recovers the truth.

    Args:
        loc: True mean.
        covariance_matrix: True covariance.
        phi: Truncation set; only the sampler sees it.
        known_covariance: Fit the known-covariance estimator, given the truth.
        dataset_size: Truncated samples to draw.
        tolerances: Per-estimator bounds; the suite's usual ones by default.
    """
    target = distributions.MultivariateNormal(loc, covariance_matrix)
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(target, phi, dataset_size)
    dims = loc.numel()
    model = _model(known_covariance, dims, covariance_matrix, sample.alpha)
    model.fit(sample.S)
    known = covariance_matrix if known_covariance else None
    naive_covariance = known if known is not None else ch.cov(sample.S.T)
    helpers.assert_recovers(
        estimate=lambda name: _fitted(model, name, known),
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=distributions.MultivariateNormal(
            sample.S.mean(0), naive_covariance.reshape(dims, dims)
        ),
        tolerances=tolerances,
    )


def test_known_covariance_half_space():
    _recovers(
        ch.tensor([0.5, -0.5]),
        CORRELATED,
        lambda z: z.sum(1) > 0.5,
        known_covariance=True,
    )


def test_known_covariance_ball_off_the_mean():
    _recovers(
        ch.tensor([1.0, -0.5]),
        CORRELATED,
        lambda z: z.norm(dim=-1) < 1.5,
        known_covariance=True,
    )


def test_known_covariance_three_dimensional_polytope():
    _recovers(
        ch.tensor([0.5, -0.3, 0.2]),
        CORRELATED_3D,
        truncation_sets.random_polytope(0, 3, 3, 0.5),
        known_covariance=True,
    )


def test_known_covariance_ten_dimensional_ball():
    _recovers(
        ch.linspace(-0.5, 0.5, 10),
        _correlated(10, 10),
        lambda z: (z - 0.3).norm(dim=-1) < 3.0,
        known_covariance=True,
    )


def test_half_space():
    _recovers(
        ch.tensor([0.5, -0.5]),
        CORRELATED,
        truncation_sets.coordinate_threshold(0, 0.5),
        known_covariance=False,
    )


def test_intersection_of_half_spaces():
    _recovers(
        ch.tensor([0.3, 0.2]),
        CORRELATED,
        lambda z: (z[:, 0] > 0.0) & (z[:, 1] > 0.0),
        known_covariance=False,
    )


def test_ball_off_the_mean():
    _recovers(
        ch.tensor([0.5, 0.5]),
        CORRELATED,
        lambda z: z.norm(dim=-1) < 1.5,
        known_covariance=False,
    )


def test_three_dimensional_polytope():
    _recovers(
        ch.tensor([0.5, -0.3, 0.2]),
        CORRELATED_3D,
        truncation_sets.random_polytope(0, 3, 3, 0.5),
        known_covariance=False,
    )


@pytest.mark.parametrize("dims", [4, 5])
def test_higher_dimensional_half_space(dims: int):
    _recovers(
        ch.linspace(-0.5, 0.5, dims),
        _correlated(dims, dims),
        lambda z: z.sum(1) > 0.0,
        known_covariance=False,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Hermite bias in four dimensions: the default degree (18) fits a ball "
        "off the mean to KL 0.162 against 0.700 naive, over the 0.1 bound."
    ),
)
def test_four_dimensional_ball_off_the_mean():
    _recovers(
        ch.linspace(-0.5, 0.5, 4),
        _correlated(4, 4),
        lambda z: (z - 0.3).norm(dim=-1) < 2.0,
        known_covariance=False,
    )


# Kontonis et al. (2019)'s Figure 1: an isotropic Gaussian truncated to a
# non-convex Sierpinski triangle, about 2% of its mass.
_FIGURE_ONE_MEANS = ((0.1, 0.78), (0.0, 0.0))


@pytest.mark.parametrize("loc", _FIGURE_ONE_MEANS)
def test_sierpinski_triangle(loc):
    # The naive estimates score 0.15 to 0.18, at the suite's loosest bound,
    # and the fits about 0.001, so a bound of 0.02 is what measures the fit.
    _recovers(
        ch.tensor(loc),
        ch.eye(2),
        truncation_sets.sierpinski(3),
        known_covariance=True,
        tolerances={"ema": 0.02, "avg": 0.02},
    )


def test_sierpinski_pair_is_told_apart():
    """Figure 1's two Gaussians leave samples whose means nearly coincide."""
    phi = truncation_sets.sierpinski(3)
    targets = tuple(
        distributions.MultivariateNormal(ch.tensor(loc), ch.eye(2))
        for loc in _FIGURE_ONE_MEANS
    )
    fits = []
    naive = []
    for target in targets:
        ch.manual_seed(0)
        sample = helpers.generate_truncated_distribution_dataset(target, phi, 20000)
        model = _model(True, 2, ch.eye(2), sample.alpha)
        model.fit(sample.S)
        fits.append(model)
        naive.append(distributions.MultivariateNormal(sample.S.mean(0), ch.eye(2)))
    for name in helpers.TOLERANCES:
        helpers.assert_distinguishes(
            targets=targets,
            recovered=(
                _fitted(fits[0], name, ch.eye(2)),
                _fitted(fits[1], name, ch.eye(2)),
            ),
            naive=(naive[0], naive[1]),
            divergence=kl.kl_divergence,
        )


def test_recovers_the_set():
    """Theorem 6: thresholding psi_k against the fit recovers S."""
    target = distributions.MultivariateNormal(ch.tensor([0.5, -0.5]), CORRELATED)

    def phi(z):
        return z.sum(1) > 0.5

    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(target, phi, 20000)
    model = _model(False, 2, CORRELATED, sample.alpha)
    model.fit(sample.S)
    ch.manual_seed(1)
    population = target.sample([20000])
    inside = phi(population)

    # The naive rule thresholds the same psi_k, but against the Gaussian
    # with the sample's own mean and covariance instead of the fit.
    z = model.frame_.to_frame(sample.S)
    naive = delphi.oracle.RecoveredSet(
        model.expansion_,
        model.frame_.to_frame,
        distributions.MultivariateNormal(z.mean(0), ch.cov(z.T)),
    )
    naive_error = float((naive(population) != inside).float().mean())
    error = float((model.truncation_set()(population) != inside).float().mean())
    print(f"naive rule error: {naive_error:.4f}, fitted rule error: {error:.4f}")

    assert naive_error > 0.1
    assert error <= 0.05


@pytest.mark.parametrize("known_covariance", [False, True])
def test_terms_are_the_papers_objective(known_covariance: bool) -> None:
    """Each row's term is exp(h(u, B; z)) N(0, I; z) psi_k(z), equation (5)."""
    target = distributions.MultivariateNormal(ch.zeros(2), CORRELATED)
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(
        target, _two_half_spaces, 2000
    )
    model: unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalBase
    if known_covariance:
        model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalKnownCovariance(
            {"iterations": 1, "degree": 4}, 2, CORRELATED
        )
    else:
        model = (
            unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
                {"iterations": 1, "degree": 4}, 2
            )
        )
    model.fit(sample.S)
    # The frame centers the sample, and whitens it unless the covariance is
    # known, when B = I makes the second moment drop out of h.
    mean, second = ch.zeros(2).double(), ch.eye(2).double()
    assert model.frame_ is not None and model.expansion_ is not None
    assert model.emp_theta is not None
    z = model.frame_.to_frame(sample.S.double())[:50]
    psi = model.expansion_(z).clamp(min=0.0)
    statistic, offset = model._statistic(z)  # pylint: disable=protected-access
    rows = ch.cat([statistic, offset[:, None], psi[:, None]], dim=1)
    ch.manual_seed(3)
    start = model.emp_theta.double()
    theta = model._project(  # pylint: disable=protected-access
        start + 0.1 * ch.randn(start.numel(), dtype=ch.float64)
    )
    u, precision = model._natural(theta)  # pylint: disable=protected-access

    h = (
        0.5 * ((z @ precision) * z).sum(1)
        - 0.5 * ch.trace((precision - ch.eye(2).double()) @ second)
        - (z - mean) @ u
        + ch.log(ch.tensor(2.0 * ch.pi))
    )
    reference = distributions.MultivariateNormal(
        ch.zeros(2, dtype=z.dtype), ch.eye(2, dtype=z.dtype)
    )
    expected = h.exp() * reference.log_prob(z).exp() * psi

    ch.testing.assert_close(
        losses.UnknownTruncationMultivariateNormalNLL.terms(theta, rows), expected
    )


def test_exact_psi_puts_the_minimizer_at_the_truth():
    """Claim 2: with the exact psi the objective is minimized at the truth.

    Separates the objective, the frame and the parameterization from the
    Hermite approximation, whose bias every recovery test also carries.
    """
    target = distributions.MultivariateNormal(ch.tensor([0.5, -0.5]), CORRELATED)
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(
        target, _two_half_spaces, 20000
    )
    model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        {"iterations": 1, "degree": 2}, 2, sample.alpha
    )
    model.fit(sample.S)
    z = model.frame_.to_frame(sample.S.double())
    scale, center = model.frame_.scale, model.frame_.center
    truth_in_frame = distributions.MultivariateNormal(
        ch.linalg.solve(scale, target.loc.double() - center),  # pylint: disable=not-callable
        ch.linalg.solve(  # pylint: disable=not-callable
            scale,
            ch.linalg.solve(scale, CORRELATED.double()).T,  # pylint: disable=not-callable
        ),
    )
    reference = distributions.MultivariateNormal(
        ch.zeros(2, dtype=z.dtype), ch.eye(2, dtype=z.dtype)
    )
    # The 1 / alpha factor scales the objective and leaves its minimizer.
    psi = (truth_in_frame.log_prob(z) - reference.log_prob(z)).exp()
    statistic, offset = model._statistic(z)  # pylint: disable=protected-access
    rows = ch.cat([statistic, offset[:, None], psi[:, None]], dim=1)

    theta = model.emp_theta.double().clone().requires_grad_(True)
    optimizer = ch.optim.LBFGS(
        [theta], max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-12
    )

    def closure():
        optimizer.zero_grad()
        loss = losses.UnknownTruncationMultivariateNormalNLL.apply(theta, rows)
        loss.backward()
        return loss

    optimizer.step(closure)
    loc, covariance_matrix = model._canonical(theta.detach())  # pylint: disable=protected-access
    divergence = kl.kl_divergence(
        distributions.MultivariateNormal(loc.float(), covariance_matrix.float()),
        target,
    )
    print(f"exact-psi divergence: {float(divergence):.6f}")

    assert float(divergence) < 2e-3


def _descent_rows() -> tuple[ch.Tensor, ch.Tensor]:
    """Return a fitted model's descent rows and a point near its start."""
    target = distributions.MultivariateNormal(ch.zeros(2), ch.eye(2))
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(
        target, _two_half_spaces, 2000
    )
    model = _model(False, 2, ch.eye(2), hyperparameters={"iterations": 1, "degree": 4})
    model.fit(sample.S)
    assert model.train_loader_ is not None and model.emp_theta is not None
    dataset = model.train_loader_.dataset
    assert isinstance(dataset, datasets.UnknownTruncationMultivariateNormalDataset)
    return dataset.data, model.emp_theta + 0.05


def _plain_objective(theta: ch.Tensor, rows: ch.Tensor) -> ch.Tensor:
    return losses.UnknownTruncationMultivariateNormalNLL.terms(theta, rows).mean()


def test_backward_matches_autograd():
    rows, theta = _descent_rows()
    point = theta.clone().requires_grad_(True)
    losses.UnknownTruncationMultivariateNormalNLL.apply(point, rows).backward()
    reference = theta.clone().requires_grad_(True)
    _plain_objective(reference, rows).backward()

    ch.testing.assert_close(point.grad, reference.grad)


def test_hessian_matches_autograd():
    rows, theta = _descent_rows()
    expected = ch.autograd.functional.hessian(
        lambda point: _plain_objective(point, rows), theta
    )

    ch.testing.assert_close(
        losses.UnknownTruncationMultivariateNormalNLL.hessian(theta, rows),
        expected,
        rtol=1e-4,
        atol=1e-5,
    )


def test_projection_bounds_the_precision_and_the_mean():
    model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        {"min_precision": 0.5, "max_precision": 2.0, "loc_radius": 1.0}, 3
    )
    target = distributions.MultivariateNormal(ch.zeros(3), ch.eye(3))
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(
        target, lambda z: z[:, 0] > 0, 500
    )
    model.fit(sample.S.double())
    ch.manual_seed(4)
    theta = 10.0 * ch.randn(model.emp_theta.numel(), dtype=ch.float64)

    projected = model._project(theta)  # pylint: disable=protected-access

    u, precision = model._natural(projected)  # pylint: disable=protected-access
    values = ch.linalg.eigvalsh(precision)  # pylint: disable=not-callable
    assert float(values.min()) >= 0.5 - 1e-9
    assert float(values.max()) <= 2.0 + 1e-9
    assert float(u.norm()) <= 1.0 + 1e-9
    ch.testing.assert_close(model._project(projected), projected)  # pylint: disable=protected-access


def test_rejects_an_alpha_outside_the_unit_interval():
    with pytest.raises(ValueError, match="alpha"):
        unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
            {}, 2, 1.0
        )


def test_rejects_samples_of_the_wrong_shape():
    model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        {}, 2
    )
    with pytest.raises(ValueError, match="shape"):
        model.fit(ch.randn(100, 3))


def test_rejects_a_covariance_of_the_wrong_shape():
    with pytest.raises(ValueError, match="covariance_matrix"):
        unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalKnownCovariance(
            {}, 2, ch.eye(3)
        )


def test_estimates_need_a_fit():
    model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        {}, 2
    )
    with pytest.raises(RuntimeError, match="fit"):
        _ = model.avg_loc_
