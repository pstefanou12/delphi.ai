# Author: pstefanou12@
"""Tests for the truncated regression losses against closed forms.

With unit noise and the set ``z > 0``, a prediction h has survival mass
``Phi(h)``, and ``z | S`` is a normal truncated at ``a = -h`` with inverse
Mills ratio ``lambda = pdf(a) / (1 - Phi(a))``, mean ``h + lambda`` and
variance ``1 + a lambda - lambda^2``.
"""

import torch as ch
import torch.distributions

from delphi.stats.truncated import losses

_DRAWS = 200_000
_STANDARD = torch.distributions.Normal(0.0, 1.0)


def _positive(z: ch.Tensor) -> ch.Tensor:
    return z > 0.0


def _mills(pred: ch.Tensor) -> ch.Tensor:
    return ch.exp(_STANDARD.log_prob(-pred)) / _STANDARD.cdf(pred)


def test_truncated_mse_matches_the_closed_form_nll():
    ch.manual_seed(0)
    pred = ch.tensor([[-0.5], [0.0], [1.0]])
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    loss = losses.TruncatedMSE.apply(pred, targ, _positive, ch.ones(1), _DRAWS)
    expected = (0.5 * (targ - pred).pow(2) + ch.log(_STANDARD.cdf(pred))).mean()
    assert ch.allclose(loss, expected, atol=5e-3), (loss, expected)


def test_truncated_mse_gradient_is_the_conditional_mean_residual():
    ch.manual_seed(0)
    pred = ch.tensor([[-0.5], [0.0], [1.0]], requires_grad=True)
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    loss = losses.TruncatedMSE.apply(pred, targ, _positive, ch.ones(1), _DRAWS)
    loss.backward()
    conditional_mean = pred.detach() + _mills(pred.detach())
    expected = (conditional_mean - targ) / pred.size(0)
    assert ch.allclose(pred.grad, expected, atol=5e-3), (pred.grad, expected)


def test_truncated_mse_hessian_weights_the_design_by_the_variance():
    ch.manual_seed(0)
    pred = ch.tensor([[-0.5], [0.0], [1.0]])
    design = ch.tensor([[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0]])
    hessian = losses.TruncatedMSE.hessian(pred, design, _positive, ch.ones(1), _DRAWS)
    mills = _mills(pred)
    variance = 1.0 - pred * mills - mills.pow(2)
    expected = (design.T * variance.T) @ design / pred.size(0)
    assert ch.allclose(hessian, expected, atol=1e-2), (hessian, expected)
