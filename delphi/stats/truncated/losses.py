# Author: pstefanou12@
"""Autograd loss functions for truncated regression and classification models."""

import math
from typing import cast

import torch as ch
import torch.distributions
import torch.nn

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
        ctx,
        pred,
        targ,
        phi,
        noise_var,
        sampler: sampling.GaussianSampler,
        eps=1e-5,
    ):
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
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Return the gradients in pred and, when learned, noise_var."""
        pred_grad, var_grad, noise_var = ctx.saved_tensors
        weights = grad_output[:, None]
        if var_grad is not None:
            var_grad = (weights * var_grad).sum_to_size(noise_var.shape)
        return weights * pred_grad, None, None, var_grad, None, None

    @staticmethod
    def hessian(pred, X, phi, noise_var, sampler: sampling.GaussianSampler):  # pylint: disable=invalid-name
        """Return the batch-mean Hessian of the NLL in the weight.

        ``mean_i Var[z_i | S] / sigma^4 x_i x_i^T``, per output.

        Args:
            pred: Predictions h, shaped (n, k).
            X: Design matrix, shaped (n, d).
            phi: As ``forward``.
            noise_var: As ``forward``.
            sampler: As ``forward``.

        Returns:
            The Hessian for the weight of shape (d, k) flattened row-major,
            shaped (d * k, d * k), in float64: scaled covariates leave its
            smallest eigenvalue below float32's resolution of its largest.
        """
        with ch.no_grad():
            noise_var = ch.as_tensor(noise_var, dtype=pred.dtype)
            moments = sampler.moments(pred, noise_var, phi, second_moment=True)
            second = cast(ch.Tensor, moments.second_moment)
            var = (second - moments.mean.pow(2)) / noise_var.double().pow(2)
            design = X.double()
            hess = ch.einsum(
                "ni,nj,na,ab->iajb",
                design,
                design,
                var,
                ch.eye(pred.size(1), dtype=ch.float64),
            )
            size = X.size(1) * pred.size(1)
            return hess.reshape(size, size) / pred.size(0)

    @staticmethod
    def natural_hessian(pred, X, phi, noise_var, sampler: sampling.GaussianSampler):  # pylint: disable=invalid-name
        """Return the batch-mean Hessian of the NLL in ``(v, lambda)``.

        For ``(v, lambda) = (w / sigma^2, 1 / sigma^2)``, in which the NLL is
        convex, it is ``mean_i Cov[(z x_i, -z^2 / 2) | S]`` per output.

        Args:
            pred: As ``hessian``.
            X: As ``hessian``.
            phi: As ``forward``.
            noise_var: The noise variance, shaped (k,).
            sampler: As ``forward``.

        Returns:
            The Hessian for the parameter of shape (d + 1, k), v's rows then
            lambda's, flattened row-major, shaped ((d + 1) k, (d + 1) k), in
            float64 as ``hessian``.
        """
        with ch.no_grad():
            moments = sampler.moments(pred, noise_var, phi, fourth_moment=True)
            mean = moments.mean
            second = cast(ch.Tensor, moments.second_moment)
            third = cast(ch.Tensor, moments.third_moment)
            fourth = cast(ch.Tensor, moments.fourth_moment)
            n, k = pred.shape
            design = X.double()
            block = ch.zeros(k, X.size(1) + 1, X.size(1) + 1, dtype=ch.float64)
            block[:, :-1, :-1] = ch.einsum(
                "ni,nj,na->aij", design, design, second - mean.pow(2)
            )
            block[:, :-1, -1] = -ch.einsum(
                "ni,na->ai", design, 0.5 * (third - mean * second)
            )
            block[:, -1, :-1] = block[:, :-1, -1]
            block[:, -1, -1] = (0.25 * (fourth - second.pow(2))).sum(0)
            hess = ch.einsum("aij,ab->iajb", block, ch.eye(k, dtype=ch.float64))
            size = (X.size(1) + 1) * k
            return hess.reshape(size, size) / n


class SwitchGrad(ch.autograd.Function):  # pylint: disable=abstract-method
    """Gradient of the negative population log likelihood for truncated regression.

    Uses known noise variance and switches between censor-aware and
    censor-oblivious gradient depending on the acceptance rate.
    """

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,arguments-differ
        ctx,
        pred,
        targ,
        phi,
        c_gamma,
        alpha,
        T,  # pylint: disable=invalid-name
        noise_var,
        num_samples=10,
        eps=1e-5,
    ):
        """Compute switch loss.

        Args:
            pred: Size (batch_size, d) regression model predictions.
            targ: Size (batch_size, d) regression target values.
            phi: Dependent variable membership oracle.
            c_gamma: Large constant >= 0.
            alpha: Survival probability.
            T: Number of samples within dataset.
            noise_var: Noise distribution variance parameter.
            num_samples: Number of samples per batch element in rejection
                sampling.
            eps: Denominator error constant to avoid divide by zero.
        """
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        M = ch.distributions.MultivariateNormal(  # pylint: disable=invalid-name
            ch.zeros(pred[0].size(0)), noise_var
        )
        result = SwitchGrad.survival_test(pred, phi, c_gamma, alpha, T)

        noised = stacked + M.sample(stacked.size()[:-1])

        filtered = phi(noised)
        z_ = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + eps)

        z = result.float() * z_ + (~result).float() * pred

        ctx.save_for_backward(pred, targ, z)
        loss = -0.5 * (targ - pred).norm(p=2, keepdim=True, dim=-1).pow(2) + 0.5 * (
            z - pred
        ).norm(p=2, keepdim=True, dim=-1).pow(2)
        return loss.mean(0)

    @staticmethod
    def survival_test(mu, phi, c_gamma, alpha, T):  # pylint: disable=invalid-name
        """Return whether each prediction keeps enough mass for the censor-aware gradient.

        Args:
            mu: Current conditional mean for LDS.
            phi: Oracle.
            c_gamma: Constant.
            alpha: Survival probability.
            T: Number of timesteps in dataset.
        """
        M = ch.distributions.MultivariateNormal(  # pylint: disable=invalid-name
            ch.zeros(mu[0].size(0)), ch.eye(mu[0].size(0))
        )

        # Threshold constant.
        gamma = (alpha / 2) ** c_gamma

        # Number of samples.
        k = int((4 / gamma) * math.log(T))
        stacked = mu.repeat(k, 1, 1)
        noise = M.sample(stacked.size()[:-1])
        ci = stacked + noise
        p = phi(ci).float().mean(0)
        # Check whether the probability that a sample falls within the
        # truncation set is greater than the survival probability.
        return p >= (2 * gamma)

    @staticmethod
    def backward(ctx, _grad_output):  # pylint: disable=arguments-differ,invalid-name
        """Compute gradient of switch loss."""
        pred, targ, z = ctx.saved_tensors
        return (
            (z - targ) / pred.size(0),
            targ / pred.size(0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class GumbelCE(ch.autograd.Function):  # pylint: disable=abstract-method
    """Gumbel cross-entropy gradient for multiclass classification."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx,
        pred,
        targ,
        num_samples=1000,
        eps=1e-5,  # pylint: disable=invalid-name
    ):
        """Compute cross-entropy loss."""
        ctx.save_for_backward(pred, targ)
        ce_loss = ch.nn.CrossEntropyLoss()
        ctx.num_samples = num_samples
        ctx.eps = eps
        return ce_loss(pred, targ)

    @staticmethod
    def backward(ctx, _grad_output):  # pylint: disable=arguments-differ,invalid-name
        """Compute gradient of Gumbel CE w.r.t. pred."""
        pred, targ = ctx.saved_tensors
        gumbel_dist = torch.distributions.Gumbel(0, 1)  # pylint: disable=invalid-name
        stacked = pred[None, ...].repeat(ctx.num_samples, 1, 1)
        rand_noise = gumbel_dist.sample(stacked.size())
        noised = stacked + rand_noise
        noised_labs = noised.argmax(-1)
        mask = noised_labs.eq(targ)[..., None]
        inner_exp = 1 - ch.exp(-rand_noise)
        avg = (inner_exp * mask).sum(0) / (mask.sum(0) + ctx.eps) / pred.size(0)
        return -avg, None, None, None


def _label_nll(
    pred: ch.Tensor,
    targ: ch.Tensor,
    phi,
    noise: sampling.LatentNoise,
    sampler: sampling.LatentSampler,
    eps: float,
) -> tuple[ch.Tensor, ch.Tensor]:
    """Return each sample's truncated label NLL and its gradient in pred."""
    moments = sampler.label_moments(pred, noise, phi)
    label = targ.long()
    label_prob = moments.mass / moments.mass.sum(1, keepdim=True).clamp_min(1e-300)
    mean_score = (label_prob * moments.score).sum(1, keepdim=True)
    loss = -ch.log(label_prob.gather(1, label).clamp_min(eps))
    grad = mean_score - moments.score.gather(1, label)
    return loss.flatten().to(pred.dtype), grad.to(pred.dtype)


def _label_fisher(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    pred: ch.Tensor,
    X: ch.Tensor,  # pylint: disable=invalid-name
    phi,
    noise: sampling.LatentNoise,
    sampler: sampling.LatentSampler,
) -> ch.Tensor:
    """Return the batch-mean Fisher information of the label NLL in the weight."""
    with ch.no_grad():
        moments = sampler.label_moments(pred, noise, phi)
        label_prob = moments.mass / moments.mass.sum(1, keepdim=True).clamp_min(1e-300)
        mean_score = (label_prob * moments.score).sum(1, keepdim=True)
        info = (label_prob * (moments.score - mean_score).pow(2)).sum(1)
        design = X.double()
        return (design.T * info) @ design / pred.size(0)


class _TruncatedLabelNLL(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated NLL of a binary label ``1{h + e >= 0}`` seen through phi.

    Per sample, ``-log P(label y | z in S)`` for ``z = h + e``; its gradient
    in h is ``E[s(e) | S] - E[s(e) | S, y]``, for the noise's score s. A
    subclass fixes the noise. Returns one loss per sample; the caller
    reduces.
    """

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Return the gradient in pred."""
        (grad,) = ctx.saved_tensors
        return (grad_output[:, None] * grad,) + (None,) * 4


class TruncatedProbitNLL(_TruncatedLabelNLL):  # pylint: disable=abstract-method
    """Truncated probit NLL: ``e ~ N(0, 1)``, score ``s(e) = e``."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx,
        pred,
        targ,
        phi,
        sampler: sampling.LatentSampler,
        eps=1e-5,
    ):
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
        loss, grad = _label_nll(pred, targ, phi, sampling.GAUSSIAN_NOISE, sampler, eps)
        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def hessian(pred, X, phi, sampler: sampling.LatentSampler):  # pylint: disable=invalid-name
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
        return _label_fisher(pred, X, phi, sampling.GAUSSIAN_NOISE, sampler)


class TruncatedBCE(_TruncatedLabelNLL):  # pylint: disable=abstract-method
    """Truncated logistic NLL: ``e ~ Logistic(0, 1)``, ``s(e) = 2 sigma(e) - 1``."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx,
        pred,
        targ,
        phi,
        sampler: sampling.LatentSampler,
        eps=1e-5,
    ):
        """Compute each sample's truncated NLL; as ``TruncatedProbitNLL``."""
        loss, grad = _label_nll(pred, targ, phi, sampling.LOGISTIC_NOISE, sampler, eps)
        ctx.save_for_backward(grad)
        return loss

    @staticmethod
    def hessian(pred, X, phi, sampler: sampling.LatentSampler):  # pylint: disable=invalid-name
        """Return the Fisher information; as ``TruncatedProbitNLL.hessian``."""
        return _label_fisher(pred, X, phi, sampling.LOGISTIC_NOISE, sampler)
