# Author: pstefanou12@
"""Tests for the standardization and the pipeline."""

import pytest
import torch as ch

from delphi.transformations import standardization, transformation


def _correlated_sample() -> ch.Tensor:
    ch.manual_seed(0)
    factor = ch.tensor([[1.5, 0.0], [0.6, 0.8]], dtype=ch.float64)
    return ch.randn(5000, 2, dtype=ch.float64) @ factor.T + ch.tensor([1.0, -2.0])


def test_standardized_data_have_zero_mean_and_identity_covariance():
    x = _correlated_sample()
    z = standardization.Standardization().fit(x).transform(x)
    ch.testing.assert_close(
        z.mean(0), ch.zeros(2, dtype=ch.float64), atol=1e-12, rtol=0
    )
    ch.testing.assert_close(ch.cov(z.T), ch.eye(2, dtype=ch.float64))


def test_inverse_undoes_the_transform():
    x = _correlated_sample()
    step = standardization.Standardization().fit(x)
    ch.testing.assert_close(step.inverse_transform(step.transform(x)), x)


def test_a_known_covariance_is_used_as_given():
    x = _correlated_sample()
    known = ch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=ch.float64)
    step = standardization.Standardization(known).fit(x)
    ch.testing.assert_close(step.scale @ step.scale.T, known)


def test_gaussian_parameters_map_back_with_the_data():
    x = _correlated_sample()
    step = standardization.Standardization().fit(x)
    z = step.transform(x)
    loc, covariance_matrix = step.inverse_transform_gaussian(z.mean(0), ch.cov(z.T))
    ch.testing.assert_close(loc, x.mean(0))
    ch.testing.assert_close(covariance_matrix, ch.cov(x.T))


def test_pipeline_chains_and_undoes_in_reverse():
    x = _correlated_sample()
    pipeline = transformation.Pipeline(
        [
            standardization.Standardization(),
            standardization.Standardization(4.0 * ch.eye(2, dtype=ch.float64)),
        ]
    ).fit(x)
    z = pipeline.transform(x)
    ch.testing.assert_close(
        z, pipeline.steps[1].transform(pipeline.steps[0].transform(x))
    )
    ch.testing.assert_close(pipeline.inverse_transform(z), x)


def test_transforming_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        standardization.Standardization().transform(ch.zeros(3, 2))
