# Author: pstefanou12@
"""Autograd loss functions for truncated regression and classification models."""

from collections.abc import Callable
from typing import Any, Optional, Union, cast

import torch as ch
import torch.distributions

from delphi.truncated import sampling


class TruncatedGaussianNLL(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated NLL of regression with Gaussian noise.

    Per sample, ``(y - h)^2 / (2 sigma^2) + log(sigma^2) / 2 +
    log P(h + e in S)`` with ``e ~ N(0, sigma^2)``; without truncation it is
    ``torch.nn.functional.gaussian_nll_loss``. Its gradients are
    ``(E[z | S] - y) / sigma^2`` in h and
    ``(E[(z - h)^2 | S] - (y - h)^2) / (2 sigma^4)`` in sigma^2, for
    ``z ~ N(h, sigma^2)``, whose moments the sampler estimates. Returns one
    loss per sample; the caller reduces.
    """

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx: Any,
        pred: ch.Tensor,
        targ: ch.Tensor,
        phi: Callable,
        noise_var: Union[float, ch.Tensor],
        sampler: sampling.GaussianSampler,
        eps: float = 1e-5,
    ) -> ch.Tensor:
        """Compute each sample's truncated NLL, summed over its outputs.

        Args:
            pred: Predictions h, shaped (n, k).
            targ: Observed responses, shaped (n, k).
            phi: Truncation oracle over latents shaped (n, m, k).
            noise_var: Noise variance, broadcastable to (n, k); a learned
                one receives a gradient.
            sampler: Estimates the conditional moments.
            eps: Floor on the survival mass, so a sample phi rejects
                entirely contributes a finite loss.

        Returns:
            The losses, shaped (n,).
        """
        noise_var = ch.as_tensor(noise_var, dtype=pred.dtype)
        learned = ctx.needs_input_grad[3]
        moments = sampler.moments(pred, noise_var, phi, second_moment=learned)
        mean = moments.mean.to(pred.dtype)
        residual = (targ - pred).pow(2)
        var_grad = None
        if learned:
            spread = cast(ch.Tensor, moments.second_moment).to(pred.dtype) - (
                2.0 * pred * mean - pred.pow(2)
            )
            var_grad = (spread - residual) / (2.0 * noise_var.pow(2))
        ctx.save_for_backward((mean - targ) / noise_var, var_grad, noise_var)
        log_mass = ch.log(moments.mass + eps)
        nll = 0.5 * residual / noise_var + 0.5 * ch.log(noise_var) + log_mass
        return nll.sum(-1)

    @staticmethod
    def backward(  # pylint: disable=arguments-differ
        ctx: Any, grad_output: ch.Tensor
    ) -> tuple[Optional[ch.Tensor], ...]:
        """Return the gradients in pred and, when learned, noise_var."""
        pred_grad, var_grad, noise_var = ctx.saved_tensors
        weights = grad_output[:, None]
        if var_grad is not None:
            var_grad = (weights * var_grad).sum_to_size(noise_var.shape)
        return weights * pred_grad, None, None, var_grad, None, None

    @staticmethod
    def hessian(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        pred: ch.Tensor,
        X: ch.Tensor,  # pylint: disable=invalid-name
        phi: Callable,
        noise_var: Union[float, ch.Tensor],
        sampler: sampling.GaussianSampler,
        natural: bool = False,
    ) -> ch.Tensor:
        """Return the batch-mean Hessian of the NLL, per output.

        In the weight w it is ``mean_i Var[z_i | S] / sigma^4 x_i x_i^T``.
        With ``natural``, it is in ``(v, lambda) = (w / sigma^2, 1 / sigma^2)``,
        in which the NLL is convex: ``mean_i Cov[(z x_i, -z^2 / 2) | S]``,
        whose v block is w's times sigma^4.

        Args:
            pred: Predictions h, shaped (n, k).
            X: Design matrix, shaped (n, d).
            phi: As ``forward``.
            noise_var: As ``forward``.
            sampler: As ``forward``.
            natural: Differentiate in ``(v, lambda)`` rather than in w.

        Returns:
            The Hessian for the weight of shape (d, k), or for ``natural``
            the parameter of shape (d + 1, k), v's rows then lambda's,
            flattened row-major. It is in float64: scaled covariates leave its
            smallest eigenvalue below float32's resolution of its largest.
        """
        with ch.no_grad():
            noise_var = ch.as_tensor(noise_var, dtype=pred.dtype)
            moments = sampler.moments(
                pred, noise_var, phi, second_moment=True, fourth_moment=natural
            )
            mean = moments.mean
            second = cast(ch.Tensor, moments.second_moment)
            n, k = pred.shape
            features = X.size(1)
            width = features + 1 if natural else features
            design = X.double()
            var = second - mean.pow(2)
            if not natural:
                var = var / noise_var.double().pow(2)
            block = ch.zeros(k, width, width, dtype=ch.float64)
            block[:, :features, :features] = ch.einsum(
                "ni,nj,na->aij", design, design, var
            )
            if natural:
                third = cast(ch.Tensor, moments.third_moment)
                fourth = cast(ch.Tensor, moments.fourth_moment)
                block[:, :features, features] = -ch.einsum(
                    "ni,na->ai", design, 0.5 * (third - mean * second)
                )
                block[:, features, :features] = block[:, :features, features]
                block[:, features, features] = (0.25 * (fourth - second.pow(2))).sum(0)
            hess = ch.einsum("aij,ab->iajb", block, ch.eye(k, dtype=ch.float64))
            size = width * k
            return hess.reshape(size, size) / n


class SwitchGrad(ch.autograd.Function):  # pylint: disable=abstract-method
    """Switching gradient of the truncated NLL of unit-variance responses.

    Per sample, the sampler draws latents ``z ~ N(h, I)`` and keeps those phi
    admits. When at least ``2 gamma`` of the draws are kept, the gradient in
    h is the censor-aware ``mean(kept z) - y``, an unbiased estimate of the
    truncated NLL's ``E[z | S] - y``; otherwise it is the censor-oblivious
    ``h - y``. The loss is the matching NLL, ``||y - h||^2 / 2`` plus, when
    censor-aware, the log of the kept fraction. Returns one loss per sample,
    which the caller reduces, and which samples were censor-aware.
    """

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx: Any,
        pred: ch.Tensor,
        targ: ch.Tensor,
        phi: Callable,
        gamma: float,
        sampler: sampling.GaussianSampler,
    ) -> tuple[ch.Tensor, ch.Tensor]:
        """Compute each sample's loss and its switched gradient.

        Args:
            pred: Predictions h, shaped (n, k).
            targ: Observed responses, shaped (n, k).
            phi: Truncation oracle over latents shaped (n, m, k).
            gamma: Half the kept fraction above which the gradient is
                censor-aware.
            sampler: Draws the latents; its kept fraction is the test's
                estimate of the survival mass, so it must draw a fixed
                number per sample.

        Returns:
            The losses, shaped (n,), and a mask of the censor-aware samples,
            shaped (n,).
        """
        moments = sampler.moments(pred, ch.ones(pred.size(1)), phi)
        aware = moments.mass >= 2.0 * gamma
        latent = ch.where(aware, moments.mean.to(pred.dtype), pred)
        ctx.save_for_backward(latent - targ)
        log_mass = ch.where(aware, ch.log(moments.mass), 0.0).to(pred.dtype)
        loss = 0.5 * (targ - pred).pow(2).sum(-1) + log_mass[:, 0]
        ctx.mark_non_differentiable(aware)
        return loss, aware[:, 0]

    @staticmethod
    def backward(  # pylint: disable=arguments-differ
        ctx: Any, grad_output: ch.Tensor, _grad_aware: Optional[ch.Tensor]
    ) -> tuple[Optional[ch.Tensor], ...]:
        """Return the switched gradient in pred."""
        (grad,) = ctx.saved_tensors
        return grad_output[:, None] * grad, None, None, None, None


def _label_nll(
    moments: sampling.LabelMoments, pred: ch.Tensor, targ: ch.Tensor, eps: float
) -> tuple[ch.Tensor, ch.Tensor]:
    """Return each sample's truncated label NLL and its gradient in pred."""
    label = targ.long()
    label_prob = moments.mass / moments.mass.sum(1, keepdim=True).clamp_min(1e-300)
    mean_score = (label_prob[..., None] * moments.score).sum(1)
    loss = -ch.log(label_prob.gather(1, label).clamp_min(eps))
    label_score = moments.score.gather(
        1, label[..., None].expand(-1, 1, moments.score.size(2))
    )[:, 0]
    grad = mean_score - label_score
    return loss.flatten().to(pred.dtype), grad.to(pred.dtype)


def _label_fisher(moments: sampling.LabelMoments, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
    """Return the batch-mean Fisher information of the label NLL in the weight.

    For the weight of shape (d, q) flattened row-major, shaped (d q, d q).
    """
    label_prob = moments.mass / moments.mass.sum(1, keepdim=True).clamp_min(1e-300)
    centered = moments.score - (label_prob[..., None] * moments.score).sum(
        1, keepdim=True
    )
    info = ch.einsum("nc,nca,ncb->nab", label_prob, centered, centered)
    design = X.double()
    hess = ch.einsum("ni,nj,nab->iajb", design, design, info)
    size = X.size(1) * info.size(1)
    return hess.reshape(size, size) / X.size(0)


class _TruncatedLabelNLL(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated NLL of a label of the latent ``z = h + e`` seen through phi.

    Per sample, ``-log P(label y | z in S)``; its gradient in h is
    ``E[s(e) | S] - E[s(e) | S, y]``, for the noise's score
    ``s = -grad log f``. A subclass fixes the noise and the labeling.
    Returns one loss per sample; the caller reduces.
    """

    @staticmethod
    def backward(  # pylint: disable=arguments-differ
        ctx: Any, grad_output: ch.Tensor
    ) -> tuple[Optional[ch.Tensor], ...]:
        """Return the gradient in pred."""
        (grad,) = ctx.saved_tensors
        return (grad_output[:, None] * grad,) + (None,) * 4


class TruncatedProbitNLL(_TruncatedLabelNLL):  # pylint: disable=abstract-method
    """Truncated probit NLL: ``e ~ N(0, 1)``, score ``s(e) = e``."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx: Any,
        pred: ch.Tensor,
        targ: ch.Tensor,
        phi: Callable,
        sampler: sampling.LatentSampler,
        eps: float = 1e-5,
    ) -> ch.Tensor:
        """Compute each sample's truncated NLL.

        Args:
            pred: Latent means h, shaped (n, 1).
            targ: Labels in {0, 1}, shaped (n, 1).
            phi: Truncation oracle over latents shaped (n, m, 1).
            sampler: Estimates the per-label moments.
            eps: Floor on ``P(label y | S)``, so a sample the iterate deems
                impossible contributes a finite loss.

        Returns:
            The losses, shaped (n,).
        """
        moments = sampler.label_moments(pred, sampling.GAUSSIAN_NOISE, phi)
        loss, grad = _label_nll(moments, pred, targ, eps)
        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def hessian(  # pylint: disable=invalid-name
        pred: ch.Tensor, X: ch.Tensor, phi: Callable, sampler: sampling.LatentSampler
    ) -> ch.Tensor:
        """Return the batch-mean Fisher information in the weight.

        The expected Hessian over the label, which unlike the observed one
        is positive semi-definite everywhere.

        Args:
            pred: Latent means h, shaped (n, 1).
            X: Design matrix, shaped (n, d).
            phi: As ``forward``.
            sampler: As ``forward``.

        Returns:
            The (d, d) information in float64.
        """
        with ch.no_grad():
            moments = sampler.label_moments(pred, sampling.GAUSSIAN_NOISE, phi)
            return _label_fisher(moments, X)


class TruncatedBCE(_TruncatedLabelNLL):  # pylint: disable=abstract-method
    """Truncated logistic NLL: ``e ~ Logistic(0, 1)``, ``s(e) = 2 sigma(e) - 1``."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx: Any,
        pred: ch.Tensor,
        targ: ch.Tensor,
        phi: Callable,
        sampler: sampling.LatentSampler,
        eps: float = 1e-5,
    ) -> ch.Tensor:
        """Compute each sample's truncated NLL; as ``TruncatedProbitNLL``."""
        moments = sampler.label_moments(pred, sampling.LOGISTIC_NOISE, phi)
        loss, grad = _label_nll(moments, pred, targ, eps)
        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def hessian(  # pylint: disable=invalid-name
        pred: ch.Tensor, X: ch.Tensor, phi: Callable, sampler: sampling.LatentSampler
    ) -> ch.Tensor:
        """Return the Fisher information; as ``TruncatedProbitNLL.hessian``."""
        with ch.no_grad():
            moments = sampler.label_moments(pred, sampling.LOGISTIC_NOISE, phi)
            return _label_fisher(moments, X)


class TruncatedCE(_TruncatedLabelNLL):  # pylint: disable=abstract-method
    """Truncated softmax cross-entropy: K logits, i.i.d. Gumbel noise.

    The label is ``argmax_k (h_k + g_k)``; the score is
    ``s(g) = 1 - exp(-g)``, per coordinate. Without truncation the loss is
    ``torch.nn.functional.cross_entropy`` and its gradient ``softmax(h) -
    onehot(y)``.
    """

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx: Any,
        pred: ch.Tensor,
        targ: ch.Tensor,
        phi: Callable,
        sampler: sampling.GumbelSampler,
        eps: float = 1e-5,
    ) -> ch.Tensor:
        """Compute each sample's truncated NLL.

        Args:
            pred: Logits h, shaped (n, K).
            targ: Labels in {0, ..., K - 1}, shaped (n, 1).
            phi: Truncation oracle over latents shaped (n, m, K).
            sampler: Estimates the per-label moments.
            eps: Floor on ``P(label y | S)``, so a sample the iterate deems
                impossible contributes a finite loss.

        Returns:
            The losses, shaped (n,).
        """
        loss, grad = _label_nll(sampler.label_moments(pred, phi), pred, targ, eps)
        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def hessian(  # pylint: disable=invalid-name
        pred: ch.Tensor, X: ch.Tensor, phi: Callable, sampler: sampling.GumbelSampler
    ) -> ch.Tensor:
        """Return the batch-mean Fisher information in the weight.

        Args:
            pred: Logits h, shaped (n, K).
            X: Design matrix, shaped (n, d).
            phi: As ``forward``.
            sampler: As ``forward``.

        Returns:
            The information for the weight of shape (d, K) flattened
            row-major, shaped (d K, d K), in float64.
        """
        with ch.no_grad():
            return _label_fisher(sampler.label_moments(pred, phi), X)
