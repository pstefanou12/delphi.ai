# Author: pstefanou12@
"""Tests for the multivariate normal under unknown truncation.

None of these fits sees the truncation set. The estimator's error at a fixed
Hermite degree k does not vanish with the sample size: the degree-k
expansion of the weighted indicator is biased, by ``O(k ** (-1 / 4))`` in
Kontonis et al. (2019)'s bound. The tests therefore fix the degree, and the
tolerances are the suite's usual ones, which every fit here clears.
"""

from typing import Optional

import pytest
import torch as ch
from torch import distributions
from torch.distributions import kl

import delphi.oracle
from delphi.truncated.distributions import losses, unknown_truncated_multivariate_normal
from delphi.truncated.distributions.tests import helpers, truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)

CORRELATED = ch.tensor([[2.0, 0.5], [0.5, 1.0]])


def _two_half_spaces(z: ch.Tensor) -> ch.Tensor:
    return (z[:, 0] > -0.5) & (z[:, 1] > -0.5)


def _sample(
    target: distributions.MultivariateNormal, phi: helpers.Oracle, size: int
) -> helpers.TruncatedSample:
    ch.manual_seed(0)
    return helpers.generate_truncated_distribution_dataset(target, phi, size)


def _recovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    loc: ch.Tensor,
    covariance_matrix: ch.Tensor,
    phi: helpers.Oracle,
    known_covariance: bool,
    hyperparameters: Optional[dict] = None,
    dataset_size: int = 20000,
) -> None:
    """Fit on a truncated sample and assert the fit recovers the truth.

    Args:
        loc: True mean.
        covariance_matrix: True covariance.
        phi: Truncation set; only the sampler sees it.
        known_covariance: Fit the known-covariance estimator, given the truth.
        hyperparameters: Hyperparameter dict for the estimator.
        dataset_size: Truncated samples to draw.
    """
    target = distributions.MultivariateNormal(loc, covariance_matrix)
    sample = _sample(target, phi, dataset_size)
    dims = loc.numel()
    model: unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalBase
    if known_covariance:
        model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormalKnownCovariance(
            hyperparameters or {}, dims, covariance_matrix, sample.alpha
        )
        naive_covariance = covariance_matrix
    else:
        model = (
            unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
                hyperparameters or {}, dims, sample.alpha
            )
        )
        naive_covariance = ch.cov(sample.S.T).reshape(dims, dims)
    model.fit(sample.S)

    def estimate(name: str) -> distributions.MultivariateNormal:
        fitted_covariance = (
            covariance_matrix
            if known_covariance
            else getattr(model, f"{name}_covariance_matrix_")
        )
        return distributions.MultivariateNormal(
            getattr(model, f"{name}_loc_"), fitted_covariance
        )

    helpers.assert_recovers(
        estimate=estimate,
        divergence=lambda fitted: kl.kl_divergence(fitted, target),
        baseline=distributions.MultivariateNormal(sample.S.mean(0), naive_covariance),
    )


def test_known_covariance_half_space():
    _recovers(
        ch.zeros(2),
        CORRELATED,
        truncation_sets.sum_threshold(0.5),
        known_covariance=True,
    )


def test_known_covariance_ball():
    _recovers(
        ch.ones(2),
        ch.eye(2),
        lambda z: z.norm(dim=-1) < 2.0,
        known_covariance=True,
    )


def test_known_covariance_three_dimensional_polytope():
    _recovers(
        ch.zeros(3),
        ch.eye(3),
        truncation_sets.random_polytope(0, 3, 3, 0.5),
        known_covariance=True,
    )


def test_known_covariance_ten_dimensional_ball():
    # Degree 6 in ten dimensions is 8008 terms; degree 4 leaves KL 0.05.
    _recovers(
        ch.zeros(10),
        ch.eye(10),
        lambda z: (z - 0.5).norm(dim=-1) < 3.0,
        known_covariance=True,
        hyperparameters={"degree": 6},
    )


def test_half_space():
    # The boundary runs through the mode, the hardest place for a
    # polynomial to approximate a jump: degree 10 leaves KL 0.09 to 0.10.
    _recovers(
        ch.zeros(2),
        ch.eye(2),
        truncation_sets.coordinate_threshold(0, 0.0),
        known_covariance=False,
        hyperparameters={"degree": 16},
    )


def test_intersection_of_half_spaces():
    _recovers(ch.zeros(2), ch.eye(2), _two_half_spaces, known_covariance=False)


def test_ball():
    _recovers(
        ch.ones(2),
        ch.eye(2),
        lambda z: z.norm(dim=-1) < 2.0,
        known_covariance=False,
    )


def test_three_dimensional_polytope():
    _recovers(
        ch.zeros(3),
        ch.eye(3),
        truncation_sets.random_polytope(0, 3, 3, 0.5),
        known_covariance=False,
        hyperparameters={"degree": 12},
    )


def test_recovers_the_set():
    """Theorem 6: thresholding psi_k against the fit recovers S."""
    target = distributions.MultivariateNormal(ch.zeros(2), ch.eye(2))
    sample = _sample(target, _two_half_spaces, 20000)
    model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        {}, 2, sample.alpha
    )
    model.fit(sample.S)
    ch.manual_seed(1)
    population = target.sample([100000])
    inside = _two_half_spaces(population)

    # The same rule with the sample's own moments in place of the fit, which
    # isolates what recovering the Gaussian contributes.
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
    assert error <= 0.1


@pytest.mark.parametrize("known_covariance", [False, True])
def test_terms_are_the_papers_objective(known_covariance: bool) -> None:
    """Each row's term is exp(h(u, B; z)) N(0, I; z) psi_k(z), equation (5)."""
    target = distributions.MultivariateNormal(ch.zeros(2), CORRELATED)
    sample = _sample(target, _two_half_spaces, 2000)
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

    ch.testing.assert_close(losses.unknown_truncation_terms(theta, rows), expected)


def test_exact_psi_puts_the_minimizer_at_the_truth():
    """Claim 2: with the exact psi the objective is minimized at the truth.

    Separates the objective, the frame and the parameterization from the
    Hermite approximation, whose bias every recovery test also carries.
    """
    target = distributions.MultivariateNormal(ch.tensor([0.5, -0.5]), CORRELATED)
    sample = _sample(target, _two_half_spaces, 20000)
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
        loss = losses.unknown_truncation_objective(theta, rows)
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


def test_hessian_matches_autograd():
    target = distributions.MultivariateNormal(ch.zeros(2), ch.eye(2))
    sample = _sample(target, _two_half_spaces, 2000)
    model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        {"iterations": 1, "degree": 4}, 2
    )
    model.fit(sample.S)
    rows = model.train_loader_.dataset.tensors[1]
    theta = model.emp_theta + 0.05

    expected = ch.autograd.functional.hessian(
        lambda point: losses.unknown_truncation_objective(point, rows), theta
    )

    ch.testing.assert_close(model.hessian(theta), expected, rtol=1e-4, atol=1e-5)


def test_projection_bounds_the_precision_and_the_mean():
    model = unknown_truncated_multivariate_normal.UnknownTruncationMultivariateNormal(
        {"min_precision": 0.5, "max_precision": 2.0, "loc_radius": 1.0}, 3
    )
    target = distributions.MultivariateNormal(ch.zeros(3), ch.eye(3))
    model.fit(_sample(target, lambda z: z[:, 0] > 0, 500).S.double())
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
