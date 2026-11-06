# Author: pstefanou12@
"""Tests for polynomial families in transformed coordinates."""

import torch as ch
import torch.distributions
import torch.distributions.transforms

from delphi.torchnomials import hermite, transformed

_LOG = torch.distributions.transforms.ExpTransform().inv


def test_basis_is_the_base_basis_at_the_transformed_points():
    base = hermite.Hermite(3, 2, prob=True, norm=True)
    points = ch.rand(50, 2, dtype=ch.float64) + 0.1
    family = transformed.Transformed(base, _LOG)
    ch.testing.assert_close(family.basis(points), base.basis(ch.log(points)))
    ch.testing.assert_close(family.squared_norms(), base.squared_norms())


def test_orthonormal_under_the_pulled_back_weight():
    family = transformed.Transformed(hermite.Hermite(3, prob=True, norm=True), _LOG)
    ch.manual_seed(0)
    # The standard normal pulled back through log is the standard log-normal.
    draws = family.weight().sample((200000,)).to(ch.float64)
    values = family.basis(draws)
    gram = values.T @ values / values.size(0)
    # Sampling noise: the worst entry measured 0.008 to 0.029 over seeds 0-4.
    assert (gram - ch.eye(family.num_terms, dtype=ch.float64)).abs().max() < 0.06
