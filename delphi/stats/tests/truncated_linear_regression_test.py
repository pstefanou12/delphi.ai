# Author: pstefanou12@
"""Tests for truncated linear regression with known noise variance."""

from collections.abc import Callable
import dataclasses
import math

import pytest
import torch as ch

from delphi import stats
from delphi.stats import losses, projection_sets


@dataclasses.dataclass(frozen=True)
class Sample:
    """A truncated regression sample and the parameters behind it."""

    X: ch.Tensor  # pylint: disable=invalid-name
    y: ch.Tensor
    alpha: float
    truth: ch.Tensor


def _truncated_sample(  # pylint: disable=too-many-arguments
    weight: ch.Tensor,
    intercept: float,
    phi: Callable,
    *,
    population: int,
    correlation: float = 0.0,
    noise_var: float = 1.0,
    seed: int = 0,
) -> Sample:
    """Draw x ~ N(0, Sigma), y = w^T x + b + e, and keep phi's rows."""
    ch.manual_seed(seed)
    dims = weight.size(0)
    cov = (1 - correlation) * ch.eye(dims) + correlation * ch.ones(dims, dims)
    X = ch.distributions.MultivariateNormal(ch.zeros(dims), cov).sample(  # pylint: disable=invalid-name
        [population]
    )
    y = X @ weight[:, None] + intercept + math.sqrt(noise_var) * ch.randn(population, 1)
    kept = phi(y[:, None, :]).reshape(-1)
    return Sample(
        X[kept],
        y[kept],
        float(kept.float().mean()),
        ch.cat([weight, ch.tensor([intercept])]),
    )


def _errors(model, sample: Sample) -> tuple[float, float]:
    """Return the parameter errors of the averaged iterate and of OLS."""

    def error(coef, intercept):
        return float((ch.cat([coef.flatten(), intercept]) - sample.truth).norm())

    return (
        error(model.avg_coef_, model.avg_intercept_),
        error(model.ols_coef_, model.ols_intercept_),
    )


def _fit(sample: Sample, phi: Callable, hyperparameters=None, **kwargs):
    model = stats.TruncatedLinearRegression(
        hyperparameters or {}, phi, sample.alpha, noise_var=1.0, **kwargs
    )
    return model.fit(sample.X, sample.y)


def _left(threshold: float) -> Callable:
    return lambda z: z > threshold


def test_recovers_intercept_with_correlated_covariates():
    phi = _left(2.0)
    sample = _truncated_sample(
        ch.tensor([1.0, -0.5, 2.0]), 1.5, phi, population=8000, correlation=0.5
    )
    avg, ols = _errors(_fit(sample, phi), sample)
    assert avg < 0.1, f"averaged iterate error {avg:.3f}"
    assert avg < ols / 3, f"averaged {avg:.3f} against OLS {ols:.3f}"


def test_recovers_under_a_union_of_intervals():
    def phi(z):
        return (z < -1.5) | ((z > 0.5) & (z < 1.0)) | (z > 2.5)

    sample = _truncated_sample(ch.tensor([1.0, 1.0]), -0.5, phi, population=8000)
    avg, ols = _errors(_fit(sample, phi), sample)
    assert avg < 0.15, f"averaged iterate error {avg:.3f}"
    assert avg < ols / 2, f"averaged {avg:.3f} against OLS {ols:.3f}"


def test_recovers_ten_dimensions_where_ols_is_biased():
    # Daskalakis et al. (2020), Section 6: w = 1 in ten dimensions, y > 4.
    phi = _left(4.0)
    sample = _truncated_sample(ch.ones(10), 0.5, phi, population=20000)
    avg, ols = _errors(_fit(sample, phi), sample)
    assert ols > 1.5, f"OLS error {ols:.3f}; the set should bias it"
    assert avg < 0.35, f"averaged iterate error {avg:.3f}"


def test_more_passes_tighten_the_estimate():
    phi = _left(4.0)
    sample = _truncated_sample(ch.ones(10), 0.5, phi, population=20000)
    one_pass, _ = _errors(_fit(sample, phi), sample)
    three_passes, _ = _errors(
        _fit(sample, phi, {"epochs": 3, "avg_burn_in": 0.5}), sample
    )
    assert three_passes < one_pass, f"{three_passes:.3f} vs {one_pass:.3f}"


def test_recovers_without_intercept_from_a_projected_zero_start():
    phi = _left(0.0)
    sample = _truncated_sample(ch.tensor([2.0, -1.0]), 0.0, phi, population=6000)
    model = _fit(sample, phi, {"start": "zero"}, fit_intercept=False)
    assert model.avg_intercept_ is None
    error = float((model.avg_coef_.flatten() - sample.truth[:-1]).norm())
    ols = float((model.ols_coef_.flatten() - sample.truth[:-1]).norm())
    assert error < 0.05, f"averaged iterate error {error:.3f}"
    assert error < ols, f"averaged {error:.3f} against OLS {ols:.3f}"


def test_predict_uses_the_averaged_estimate():
    phi = _left(1.0)
    sample = _truncated_sample(ch.tensor([1.0, 1.0]), 0.5, phi, population=3000)
    model = _fit(sample, phi)
    expected = sample.X @ model.avg_coef_.T + model.avg_intercept_
    assert ch.allclose(model.predict(sample.X), expected)


def test_summary_reports_every_section():
    phi = _left(1.0)
    sample = _truncated_sample(ch.tensor([1.0, 1.0]), 0.5, phi, population=3000)
    report = _fit(sample, phi).summary()
    for heading in (
        "Model and data",
        "Optimization",
        "Estimates",
        "Projection set",
        "Curvature",
    ):
        assert heading in report
    for estimator in ("avg", "ema", "final", "best"):
        assert f"{estimator}_coef_" in report


def test_summary_raises_before_fit():
    model = stats.TruncatedLinearRegression({}, _left(0.0), 0.5, noise_var=1.0)
    with pytest.raises(RuntimeError, match="has not been fit"):
        model.summary()


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="alpha"):
        stats.TruncatedLinearRegression({}, _left(0.0), 0.0, noise_var=1.0)
    with pytest.raises(ValueError, match="noise_var"):
        stats.TruncatedLinearRegression({}, _left(0.0), 0.5, noise_var=0.0)
    model = stats.TruncatedLinearRegression({}, _left(0.0), 0.5, noise_var=1.0)
    with pytest.raises(ValueError, match="shaped"):
        model.fit(ch.randn(100, 2), ch.randn(100))


def test_nll_gradient_matches_the_truncated_normal_mean():
    # For z > c, E[e | phi] = phi(a) / (1 - Phi(a)) with a = c - h.
    ch.manual_seed(0)
    pred = ch.tensor([[-0.5], [0.0], [1.0]], requires_grad=True)
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    noise = ch.distributions.Normal(ch.zeros(1), ch.ones(1))
    loss = losses.TruncatedLinearModelNLL.apply(pred, targ, _left(0.0), noise, 200000)
    loss.backward()
    a = -pred.detach()
    mills = ch.exp(noise.log_prob(a)) / (1 - noise.cdf(a))
    expected = (mills - (targ - pred.detach())) / 3
    assert ch.allclose(pred.grad, expected, atol=5e-3)


def test_paper_radius_holds_the_truth():
    phi = _left(2.0)
    sample = _truncated_sample(ch.tensor([1.0, 1.0]), 0.5, phi, population=8000)
    design = ch.cat([sample.X, ch.ones(sample.X.size(0), 1)], dim=1)
    ols = ch.linalg.lstsq(design, sample.y).solution  # pylint: disable=not-callable
    projection = projection_sets.ResidualSpectralSet(
        design,
        sample.y,
        1.0,
        projection_sets.ResidualSpectralSet.paper_radius(sample.alpha),
        ols,
    )
    assert projection.contains(ols)
    assert projection.contains(sample.truth[:, None])
    far = 100.0 * ch.ones(3, 1)
    projected = projection.project(far)
    assert not projection.contains(far)
    assert projection.contains(projected)
    # The retraction stays on the segment from the anchor.
    direction = (far - ols).flatten()
    moved = (projected - ols).flatten()
    assert ch.allclose(moved / moved.norm(), direction / direction.norm(), atol=1e-4)
