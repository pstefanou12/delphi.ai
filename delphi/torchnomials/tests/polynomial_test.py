# Author: pstefanou12@
"""Tests for the family-agnostic parts of the polynomial families."""

import pytest
import torch as ch
import torch.distributions

from delphi.torchnomials import empirical, hermite, laguerre, polynomial


@pytest.mark.parametrize("degree,dims", [(0, 1), (3, 1), (4, 2), (3, 3)])
def test_multi_indices_enumerate_the_degree_simplex(degree, dims):
    indices = polynomial.multi_indices(degree, dims)
    assert indices.shape == (polynomial.num_terms(degree, dims), dims)
    assert len({tuple(row) for row in indices.tolist()}) == indices.size(0)
    totals = indices.sum(1)
    assert int(totals.max()) == degree
    assert bool((totals[1:] >= totals[:-1]).all()), "not ordered by degree"


def test_largest_degree_fills_the_term_budget():
    for dims in (1, 2, 3, 10):
        degree = polynomial.largest_degree(dims, 8000, 200)
        assert polynomial.num_terms(degree, dims) <= 8000
        assert degree == 200 or polynomial.num_terms(degree + 1, dims) > 8000


def test_largest_degree_respects_its_cap_and_floor():
    assert polynomial.largest_degree(1, 8000, 20) == 20
    assert polynomial.largest_degree(50, 10, 200) == 1


def _families() -> list[polynomial.Polynomial]:
    reference = torch.distributions.MultivariateNormal(
        ch.zeros(2, dtype=ch.float64), ch.eye(2, dtype=ch.float64)
    )
    return [
        hermite.Hermite(8, 2, prob=True, norm=True),
        laguerre.Laguerre(8, 2, alpha=0.5, norm=True),
        empirical.EmpiricalOrthonormal(hermite.Hermite(4, 2), reference),
    ]


@pytest.mark.parametrize(
    "family", _families(), ids=["hermite", "laguerre", "empirical"]
)
def test_chunked_evaluation_matches_one_pass(family):
    ch.manual_seed(2)
    family.fit(ch.rand(500, 2, dtype=ch.float64))
    points = ch.rand(family.chunk_size() * 3 + 7, 2, dtype=ch.float64)
    ch.testing.assert_close(family(points), family.basis(points) @ family.coefficients)


def test_evaluating_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        hermite.Hermite(3)(ch.zeros(4, 1))


def test_rejects_a_batch_of_the_wrong_width():
    with pytest.raises(ValueError, match="shape"):
        hermite.Hermite(3, 2).basis(ch.zeros(4, 3))
