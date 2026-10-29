# Author: pstefanou12@
"""Tests for the truncated regression and classification losses.

With unit noise and the set ``z > 0``, a prediction h has survival mass
``Phi(h)``, and ``z | S`` is a normal truncated at ``a = -h`` with inverse
Mills ratio ``lambda = pdf(a) / (1 - Phi(a))``, mean ``h + lambda`` and
variance ``1 + a lambda - lambda^2``. The classification losses are checked
against the closed-form label masses, and their derivatives, of an interval
of the binary latent and of a half-line of one multinomial latent.
"""

import math

import pytest
import torch as ch
import torch.autograd.functional
import torch.distributions
import torch.nn.functional

from delphi.stats.truncated import losses
from delphi.stats.truncated.tests import classification
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


def _all(z: ch.Tensor) -> ch.Tensor:
    return ch.ones_like(z, dtype=ch.bool)


def test_truncated_gaussian_nll_without_truncation_is_the_gaussian_nll():
    pred = ch.tensor([[-0.5], [0.0], [1.0]])
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    noise_var = ch.tensor([2.0])
    loss = losses.TruncatedGaussianNLL.apply(pred, targ, _all, noise_var, _SAMPLER, 0.0)
    expected = torch.nn.functional.gaussian_nll_loss(
        pred, targ, noise_var.expand_as(pred), reduction="none"
    ).flatten()
    assert ch.allclose(loss, expected, atol=1e-6), (loss, expected)


def _natural_nll(eta: ch.Tensor, lambda_: ch.Tensor, targ: ch.Tensor) -> ch.Tensor:
    """Return the closed-form NLL in ``(v^T x, lambda)`` given ``z > 0``."""
    pred = eta / lambda_
    return (
        0.5 * lambda_ * (targ - pred).pow(2)
        - 0.5 * ch.log(lambda_)
        + ch.log(_STANDARD.cdf(pred * lambda_.sqrt()))
    ).flatten()


def test_truncated_gaussian_nll_with_learned_variance_matches_the_closed_form():
    ch.manual_seed(0)
    eta, lambda_ = ch.tensor([[-1.0], [0.0], [2.0]]), ch.tensor([2.0])
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    loss = losses.TruncatedGaussianNLL.apply(
        eta / lambda_, targ, _positive, 1.0 / lambda_, _SAMPLER
    )
    expected = _natural_nll(eta, lambda_, targ)
    assert ch.allclose(loss, expected, atol=1e-2), (loss, expected)


def test_truncated_gaussian_nll_natural_gradient_matches_autograd():
    ch.manual_seed(0)
    eta = ch.tensor([[-1.0], [0.0], [2.0]], requires_grad=True)
    lambda_ = ch.tensor([2.0], requires_grad=True)
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    losses.TruncatedGaussianNLL.apply(
        eta / lambda_, targ, _positive, 1.0 / lambda_, _SAMPLER
    ).mean().backward()
    exact = ch.autograd.grad(_natural_nll(eta, lambda_, targ).mean(), [eta, lambda_])
    assert ch.allclose(eta.grad, exact[0], atol=5e-3), (eta.grad, exact[0])
    assert ch.allclose(lambda_.grad, exact[1], atol=5e-3), (lambda_.grad, exact[1])


def test_truncated_gaussian_nll_natural_hessian_matches_autograd():
    ch.manual_seed(0)
    design = ch.tensor([[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0]])
    targ = ch.tensor([[0.5], [1.0], [2.0]])
    natural = ch.tensor([[0.5], [0.5], [2.0]])

    def mean_nll(flat):
        return _natural_nll(design @ flat[:-1, None], flat[-1:], targ).mean()

    expected = torch.autograd.functional.hessian(mean_nll, natural.flatten())
    hessian = losses.TruncatedGaussianNLL.natural_hessian(
        design @ natural[:-1] / natural[-1],
        design,
        _positive,
        1.0 / natural[-1],
        _SAMPLER,
    )
    assert ch.allclose(hessian, expected.double(), atol=2e-2), (hessian, expected)


# The classifiers' latent is kept where z > -1; label 1 is z >= 0.
_ABOVE_MINUS_ONE = classification.PiecewiseSurvival((-1.0, math.inf, 1.0))
_LATENT_SAMPLER = sampling.StratifiedLatentSampler(num_samples=100_000)
_LABEL_LOSSES = [
    (losses.TruncatedProbitNLL, sampling.GAUSSIAN_NOISE),
    (losses.TruncatedBCE, sampling.LOGISTIC_NOISE),
]


def _closed_form_nll(pred: ch.Tensor, targ: ch.Tensor, noise) -> ch.Tensor:
    """Return each sample's truncated label NLL from the closed-form masses."""
    mass = _ABOVE_MINUS_ONE.label_mass(pred.double(), noise)
    return ch.log(mass.sum(1)) - ch.log(mass.gather(1, targ.long()).flatten())


@pytest.mark.parametrize("loss, noise", _LABEL_LOSSES)
def test_truncated_label_nll_matches_the_closed_form_nll(loss, noise):
    ch.manual_seed(0)
    pred = ch.tensor([[-1.0], [0.0], [1.5]])
    targ = ch.tensor([[0.0], [1.0], [0.0]])
    value = loss.apply(pred, targ, _ABOVE_MINUS_ONE, _LATENT_SAMPLER)
    expected = _closed_form_nll(pred, targ, noise).float()
    assert ch.allclose(value, expected, atol=5e-3), (value, expected)


@pytest.mark.parametrize("loss, noise", _LABEL_LOSSES)
def test_truncated_label_nll_gradient_matches_the_closed_form_gradient(loss, noise):
    ch.manual_seed(0)
    pred = ch.tensor([[-1.0], [0.0], [1.5]], requires_grad=True)
    targ = ch.tensor([[0.0], [1.0], [0.0]])
    loss.apply(pred, targ, _ABOVE_MINUS_ONE, _LATENT_SAMPLER).sum().backward()
    exact = pred.detach().double().requires_grad_()
    _closed_form_nll(exact, targ, noise).sum().backward()
    assert ch.allclose(pred.grad, exact.grad.float(), atol=5e-3), (
        pred.grad,
        exact.grad,
    )


@pytest.mark.parametrize("loss, noise", _LABEL_LOSSES)
def test_truncated_label_nll_hessian_is_the_fisher_information(loss, noise):
    ch.manual_seed(0)
    pred = ch.tensor([[-1.0], [0.0], [1.5]])
    design = ch.tensor([[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0]])
    hessian = loss.hessian(pred, design, _ABOVE_MINUS_ONE, _LATENT_SAMPLER)
    information = []
    for label in (0.0, 1.0):
        exact = pred.double().requires_grad_()
        nll = _closed_form_nll(exact, ch.full_like(pred, label), noise)
        (score,) = ch.autograd.grad(nll.sum(), exact)
        information.append(ch.exp(-nll.detach()) * score.flatten().pow(2))
    weight = sum(information)
    expected = (design.double().T * weight) @ design.double() / pred.size(0)
    assert ch.allclose(hessian, expected, atol=5e-3), (hessian, expected)


class _UntruncatedLogistic(sampling.LatentSampler):
    """Exact per-label moments of a logistic latent with nothing truncated.

    ``2 sigma(e) - 1`` given ``e >= -h`` averages to ``sigma(-h)``, and
    given ``e < -h`` to ``-sigma(h)``.
    """

    def label_moments(self, loc, noise, phi):
        loc = loc.double()
        return sampling.LabelMoments(
            mass=ch.cat([ch.sigmoid(-loc), ch.sigmoid(loc)], dim=1),
            score=ch.stack([-ch.sigmoid(loc), ch.sigmoid(-loc)], dim=1),
        )


def test_truncated_bce_takes_any_sampler():
    pred = ch.tensor([[-1.0], [0.0], [1.5]], requires_grad=True)
    targ = ch.tensor([[0.0], [1.0], [0.0]])
    losses.TruncatedBCE.apply(pred, targ, None, _UntruncatedLogistic()).sum().backward()
    expected = ch.sigmoid(pred.detach()) - targ
    assert ch.allclose(pred.grad, expected, atol=1e-6), (pred.grad, expected)


# Kept where the latent of class 1 exceeds -1.
_CLASS_ONE_ABOVE_MINUS_ONE = classification.CoordinateSurvival(1, (-1.0, math.inf, 1.0))
_GUMBEL_SAMPLER = sampling.StratifiedGumbelSampler(num_samples=100_000)
_LOGITS = ch.tensor([[0.0, -1.0, 0.5], [0.0, 0.5, 2.0], [0.0, 1.5, -1.0]])
_LABELS = ch.tensor([[0], [1], [2]])


def _closed_form_ce(pred: ch.Tensor, targ: ch.Tensor) -> ch.Tensor:
    """Return each sample's truncated CE from the closed-form label masses."""
    mass = _CLASS_ONE_ABOVE_MINUS_ONE.label_mass(pred.double())
    return ch.log(mass.sum(1)) - ch.log(mass.gather(1, targ).flatten())


def _everything(z: ch.Tensor) -> ch.Tensor:
    return ch.ones(z.shape[:-1], dtype=ch.bool)


def test_truncated_ce_without_truncation_is_the_cross_entropy():
    ch.manual_seed(0)
    pred = _LOGITS.clone().requires_grad_()
    loss = losses.TruncatedCE.apply(pred, _LABELS, _everything, _GUMBEL_SAMPLER)
    loss.sum().backward()
    expected = torch.nn.functional.cross_entropy(
        _LOGITS, _LABELS.flatten(), reduction="none"
    )
    assert ch.allclose(loss, expected, atol=1e-6), (loss, expected)
    onehot = torch.nn.functional.one_hot(_LABELS.flatten(), 3)  # pylint: disable=not-callable
    gradient = ch.softmax(_LOGITS, dim=1) - onehot
    assert ch.allclose(pred.grad, gradient, atol=5e-3), (pred.grad, gradient)


def test_truncated_ce_matches_the_closed_form_nll():
    ch.manual_seed(0)
    value = losses.TruncatedCE.apply(
        _LOGITS, _LABELS, _CLASS_ONE_ABOVE_MINUS_ONE, _GUMBEL_SAMPLER
    )
    expected = _closed_form_ce(_LOGITS, _LABELS).float()
    assert ch.allclose(value, expected, atol=5e-3), (value, expected)


def test_truncated_ce_gradient_matches_the_closed_form_gradient():
    ch.manual_seed(0)
    pred = _LOGITS.clone().requires_grad_()
    losses.TruncatedCE.apply(
        pred, _LABELS, _CLASS_ONE_ABOVE_MINUS_ONE, _GUMBEL_SAMPLER
    ).sum().backward()
    exact = _LOGITS.double().requires_grad_()
    _closed_form_ce(exact, _LABELS).sum().backward()
    assert ch.allclose(pred.grad, exact.grad.float(), atol=5e-3), (
        pred.grad,
        exact.grad,
    )


def test_truncated_ce_hessian_is_the_fisher_information():
    ch.manual_seed(0)
    design = ch.tensor([[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0]])
    hessian = losses.TruncatedCE.hessian(
        _LOGITS, design, _CLASS_ONE_ABOVE_MINUS_ONE, _GUMBEL_SAMPLER
    )
    information = ch.zeros(3, 3, 3, dtype=ch.float64)
    for label in range(3):
        exact = _LOGITS.double().requires_grad_()
        nll = _closed_form_ce(exact, ch.full_like(_LABELS, label))
        (score,) = ch.autograd.grad(nll.sum(), exact)
        information += ch.exp(-nll.detach())[:, None, None] * (
            score[:, :, None] * score[:, None, :]
        )
    design = design.double()
    expected = ch.einsum("ni,nj,nab->iajb", design, design, information) / 3
    assert ch.allclose(hessian, expected.reshape(6, 6), atol=5e-3), (
        hessian,
        expected,
    )


class _UntruncatedGumbel(sampling.GumbelSampler):
    """Exact per-label moments of Gumbel latents with nothing truncated.

    Given label c the score averages to ``onehot(c) - softmax(h)``.
    """

    def label_moments(self, loc, phi):
        prob = ch.softmax(loc.double(), dim=1)
        eye = ch.eye(loc.size(1), dtype=ch.float64)
        return sampling.LabelMoments(mass=prob, score=eye - prob[:, None, :])


def test_truncated_ce_takes_any_sampler():
    pred = _LOGITS.clone().requires_grad_()
    losses.TruncatedCE.apply(pred, _LABELS, None, _UntruncatedGumbel()).sum().backward()
    onehot = torch.nn.functional.one_hot(_LABELS.flatten(), 3)  # pylint: disable=not-callable
    expected = ch.softmax(_LOGITS, dim=1) - onehot
    assert ch.allclose(pred.grad, expected, atol=1e-6), (pred.grad, expected)
