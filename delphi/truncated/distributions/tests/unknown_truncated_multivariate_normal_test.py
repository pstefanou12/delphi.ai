# Author: pstefanou12@
"""Tests for the multivariate normal under unknown truncation.

None of these fits sees the truncation set. Each recovery test checks both
the fitted Gaussian and the recovered set. The truths have nonzero means and
correlated covariances, and the sets cut away from the mean.
"""

from typing import Optional

import pytest
import torch as ch
from torch import distributions
from torch.distributions import kl

from delphi.torchnomials import polynomial
from delphi.truncated.distributions import (
    losses,
    unknown_truncated_multivariate_normal,
    unknown_truncated_multivariate_normal_known_covariance,
)
from delphi.truncated.distributions.tests import (
    helpers,
    truncation_sets,
    unknown_truncation,
)
from delphi.utils import datasets

ch.set_printoptions(precision=4, sci_mode=False)


def _model(
    dims: int, alpha: Optional[float] = None, hyperparameters: Optional[dict] = None
) -> unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal:
    return unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        hyperparameters or {}, dims, alpha
    )


def _recovers(
    loc: ch.Tensor,
    covariance_matrix: ch.Tensor,
    phi: helpers.Oracle,
    dataset_size: int = 20000,
) -> None:
    target = distributions.MultivariateNormal(loc, covariance_matrix)
    model = _model(loc.numel())
    unknown_truncation.assert_recovers(
        model,
        target,
        phi,
        estimate=lambda name: distributions.MultivariateNormal(
            getattr(model, f"{name}_loc_"),
            getattr(model, f"{name}_covariance_matrix_"),
        ),
        baseline=lambda s: distributions.MultivariateNormal(s.mean(0), ch.cov(s.T)),
        dataset_size=dataset_size,
    )


def test_half_space():
    _recovers(
        ch.tensor([0.5, -0.5]),
        unknown_truncation.CORRELATED,
        truncation_sets.coordinate_threshold(0, 0.5),
    )


def test_intersection_of_half_spaces():
    _recovers(
        ch.tensor([0.3, 0.2]),
        unknown_truncation.CORRELATED,
        lambda z: (z[:, 0] > 0.0) & (z[:, 1] > 0.0),
    )


def test_ball_off_the_mean():
    _recovers(
        ch.tensor([0.5, 0.5]),
        unknown_truncation.CORRELATED,
        lambda z: z.norm(dim=-1) < 1.5,
    )


def test_three_dimensional_polytope():
    _recovers(
        ch.tensor([0.5, -0.3, 0.2]),
        unknown_truncation.CORRELATED_3D,
        truncation_sets.random_polytope(0, 3, 3, 0.5),
    )


@pytest.mark.parametrize("dims", [4, 5])
def test_higher_dimensional_half_space(dims: int):
    _recovers(
        ch.linspace(-0.5, 0.5, dims),
        unknown_truncation.correlated(dims, dims),
        lambda z: z.sum(1) > 0.0,
    )


@pytest.mark.slow
def test_four_dimensional_ball_off_the_mean():
    # The Hermite bias needs degree here, and degree needs samples: on
    # 20000 the fit reached KL 0.162 at degree 18 and 0.121 at 26; on 80000
    # the default degree, 27, clears the bound. Naive scores 0.69.
    _recovers(
        ch.linspace(-0.5, 0.5, 4),
        unknown_truncation.correlated(4, 4),
        lambda z: (z - 0.3).norm(dim=-1) < 2.0,
        dataset_size=80000,
    )


@pytest.mark.parametrize("known_covariance", [False, True])
def test_terms_are_the_papers_objective(known_covariance: bool) -> None:
    """Each row's term is exp(h(u, B; z)) N(0, I; z) psi_k(z), equation (5)."""
    target = distributions.MultivariateNormal(
        ch.zeros(2), unknown_truncation.CORRELATED
    )
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(
        target, unknown_truncation.two_half_spaces, 2000
    )
    model: unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalBase
    if known_covariance:
        model = unknown_truncated_multivariate_normal_known_covariance.UnknownTruncationMultivariateNormalKnownCovariance(
            {"iterations": 1, "degree": 4}, 2, unknown_truncation.CORRELATED
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
    assert model.standardization_ is not None and model.recovered_set_ is not None
    assert model.emp_theta is not None
    z = model.standardization_.transform(sample.S.double())[:50]
    psi = model.recovered_set_.weighted_indicator(z)
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
    target = distributions.MultivariateNormal(
        ch.tensor([0.5, -0.5]), unknown_truncation.CORRELATED
    )
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(
        target, unknown_truncation.two_half_spaces, 20000
    )
    model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        {"iterations": 1, "degree": 2}, 2, sample.alpha
    )
    model.fit(sample.S)
    z = model.standardization_.transform(sample.S.double())
    scale, center = model.standardization_.scale, model.standardization_.mean
    truth_in_frame = distributions.MultivariateNormal(
        ch.linalg.solve(scale, target.loc.double() - center),  # pylint: disable=not-callable
        ch.linalg.solve(  # pylint: disable=not-callable
            scale,
            ch.linalg.solve(scale, unknown_truncation.CORRELATED.double()).T,  # pylint: disable=not-callable
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
        target, unknown_truncation.two_half_spaces, 2000
    )
    model = _model(2, hyperparameters={"iterations": 1, "degree": 4})
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


def test_estimates_need_a_fit():
    model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        {}, 2
    )
    with pytest.raises(RuntimeError, match="fit"):
        _ = model.avg_loc_


def test_default_degree_follows_the_sample():
    target = distributions.MultivariateNormal(ch.zeros(2), ch.eye(2))
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(
        target, unknown_truncation.two_half_spaces, 4000
    )
    model = _model(2, hyperparameters={"iterations": 1})
    model.fit(sample.S)
    assert model.recovered_set_ is not None
    assert model.recovered_set_.polynomial is not None
    # 10% validation, then half of the rest estimates the coefficients.
    coefficient_samples = 4000 - int(0.1 * 4000)
    coefficient_samples = int(0.5 * coefficient_samples)
    expected = polynomial.largest_degree(2, coefficient_samples, 1200)
    assert model.recovered_set_.polynomial.degree == expected
    assert polynomial.num_terms(expected, 2) <= coefficient_samples


def test_an_explicit_degree_beyond_the_sample_warns():
    target = distributions.MultivariateNormal(ch.zeros(2), ch.eye(2))
    ch.manual_seed(0)
    sample = helpers.generate_truncated_distribution_dataset(
        target, unknown_truncation.two_half_spaces, 400
    )
    model = _model(2, hyperparameters={"iterations": 1, "degree": 40})
    with pytest.warns(UserWarning, match="more than"):
        model.fit(sample.S)
