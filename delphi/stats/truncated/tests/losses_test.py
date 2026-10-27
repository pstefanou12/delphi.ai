# Author: pstefanou12@
"""Tests for the truncated regression losses against closed forms.

With unit noise and the set ``z > 0``, a prediction h has survival mass
``Phi(h)``, and ``z | S`` is a normal truncated at ``a = -h`` with inverse
Mills ratio ``lambda = pdf(a) / (1 - Phi(a))``, mean ``h + lambda`` and
variance ``1 + a lambda - lambda^2``.
"""

import torch as ch
import torch.autograd.functional
import torch.distributions

from delphi.stats.truncated import losses
from delphi.truncated import sampling

_SAMPLER = sampling.GaussianRejectionSampler(num_samples=200_000)
_STANDARD = torch.distributions.Normal(0.0, 1.0)


def _positive(z: ch.Tensor) -> ch.Tensor:
    return z > 0.0


def _mills(pred: ch.Tensor) -> ch.Tensor:
    return ch.exp(_STANDARD.log_prob(-pred)) / _STANDARD.cdf(pred)


def test_truncated_gaussian_nll_matches_the_closed_form_nll():
    ch.manual_seed(0)
    pred = ch.tensor([[-0.5], [0.0], [1.0]])
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    loss = losses.TruncatedGaussianNLL.apply(
        pred, targ, _positive, ch.ones(1), _SAMPLER
    )
    expected = (0.5 * (targ - pred).pow(2) + ch.log(_STANDARD.cdf(pred))).flatten()
    assert ch.allclose(loss, expected, atol=5e-3), (loss, expected)


def test_truncated_gaussian_nll_gradient_is_the_conditional_mean_residual():
    ch.manual_seed(0)
    pred = ch.tensor([[-0.5], [0.0], [1.0]], requires_grad=True)
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    loss = losses.TruncatedGaussianNLL.apply(
        pred, targ, _positive, ch.ones(1), _SAMPLER
    )
    loss.mean().backward()
    conditional_mean = pred.detach() + _mills(pred.detach())
    expected = (conditional_mean - targ) / pred.size(0)
    assert ch.allclose(pred.grad, expected, atol=5e-3), (pred.grad, expected)


def test_truncated_gaussian_nll_hessian_weights_the_design_by_the_variance():
    ch.manual_seed(0)
    pred = ch.tensor([[-0.5], [0.0], [1.0]])
    design = ch.tensor([[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0]])
    hessian = losses.TruncatedGaussianNLL.hessian(
        pred, design, _positive, ch.ones(1), _SAMPLER
    )
    mills = _mills(pred)
    variance = 1.0 - pred * mills - mills.pow(2)
    expected = ((design.T * variance.T) @ design / pred.size(0)).double()
    assert ch.allclose(hessian, expected, atol=1e-2), (hessian, expected)


class _PositiveHalfLine(sampling.GaussianSampler):
    """Exact mass and mean of unit-variance latents given ``z > 0``."""

    def moments(self, loc, noise_var, phi, second_moment=False):
        return sampling.ConditionalMoments(
            mass=_STANDARD.cdf(loc), mean=(loc + _mills(loc)).double()
        )


def test_truncated_gaussian_nll_takes_any_sampler():
    pred = ch.tensor([[-0.5], [0.0], [1.0]], requires_grad=True)
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    loss = losses.TruncatedGaussianNLL.apply(
        pred, targ, _positive, ch.ones(1), _PositiveHalfLine()
    )
    loss.mean().backward()
    expected = (pred.detach() + _mills(pred.detach()) - targ) / pred.size(0)
    assert ch.allclose(pred.grad, expected, atol=1e-6), (pred.grad, expected)


def _natural_nll(eta: ch.Tensor, lambda_: ch.Tensor, targ: ch.Tensor) -> ch.Tensor:
    """Return the closed-form NLL in ``(eta, lambda)`` given ``z > 0``."""
    pred = eta / lambda_
    return (
        0.5 * lambda_ * (targ - pred).pow(2)
        - 0.5 * ch.log(lambda_)
        + ch.log(_STANDARD.cdf(pred * lambda_.sqrt()))
    ).flatten()


def test_truncated_gaussian_nll_unknown_variance_matches_the_closed_form_nll():
    ch.manual_seed(0)
    eta, lambda_ = ch.tensor([[-1.0], [0.0], [2.0]]), ch.tensor([2.0])
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    loss = losses.TruncatedGaussianNLLUnknownVariance.apply(
        eta, lambda_, targ, _positive, _SAMPLER
    )
    expected = _natural_nll(eta, lambda_, targ)
    assert ch.allclose(loss, expected, atol=1e-2), (loss, expected)


def test_truncated_gaussian_nll_unknown_variance_gradient_matches_autograd():
    ch.manual_seed(0)
    eta = ch.tensor([[-1.0], [0.0], [2.0]], requires_grad=True)
    lambda_ = ch.tensor([2.0], requires_grad=True)
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    losses.TruncatedGaussianNLLUnknownVariance.apply(
        eta, lambda_, targ, _positive, _SAMPLER
    ).mean().backward()
    exact = ch.autograd.grad(_natural_nll(eta, lambda_, targ).mean(), [eta, lambda_])
    assert ch.allclose(eta.grad, exact[0], atol=5e-3), (eta.grad, exact[0])
    assert ch.allclose(lambda_.grad, exact[1], atol=5e-3), (lambda_.grad, exact[1])


def test_truncated_gaussian_nll_unknown_variance_hessian_matches_autograd():
    ch.manual_seed(0)
    design = ch.tensor([[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0]])
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    natural = ch.tensor([[0.5], [0.5], [2.0]])

    def mean_nll(flat):
        return _natural_nll(design @ flat[:-1, None], flat[-1:], targ).mean()

    expected = torch.autograd.functional.hessian(mean_nll, natural.flatten())
    hessian = losses.TruncatedGaussianNLLUnknownVariance.hessian(
        design @ natural[:-1], natural[-1], design, _positive, _SAMPLER
    )
    assert ch.allclose(hessian, expected.double(), atol=2e-2), (hessian, expected)
