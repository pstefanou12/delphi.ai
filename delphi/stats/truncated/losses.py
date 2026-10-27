# Author: pstefanou12@
"""Autograd loss functions for truncated regression and classification models."""

import math
from typing import cast

import torch as ch
from torch import sigmoid as sig  # pylint: disable=no-name-in-module
import torch.distributions
import torch.nn

from delphi.truncated import sampling
from delphi.utils import helpers

gumbel = torch.distributions.Gumbel(0, 1)


class TruncatedGaussianNLL(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated NLL of regression with known Gaussian noise variance.

    Per sample, ``(y - h)^2 / (2 sigma^2) + log P(h + e in S)`` with
    ``e ~ N(0, sigma^2)``; its gradient in h is ``(E[z | S] - y) / sigma^2``
    for ``z ~ N(h, sigma^2)``. The sampler estimates ``P`` and ``E[z | S]``
    at each sample's own h. Returns one loss per sample; the caller reduces.
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
            noise_var: Known noise variance, broadcastable to (k,).
            sampler: Estimates the conditional moments.
            eps: Floor on the survival mass, so a sample phi rejects
                entirely contributes a finite loss.

        Returns:
            The losses, shaped (n,).
        """
        noise_var = ch.as_tensor(noise_var, dtype=pred.dtype)
        moments = sampler.moments(pred, noise_var, phi)
        conditional_mean = moments.mean.to(pred.dtype)
        ctx.save_for_backward((conditional_mean - targ) / noise_var)
        quadratic = 0.5 * (targ - pred).pow(2) / noise_var
        log_mass = ch.log(moments.mass + eps)
        return (quadratic + log_mass).sum(-1)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Return the gradient in pred."""
        (grad,) = ctx.saved_tensors
        return (grad_output[:, None] * grad,) + (None,) * 5

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


class TruncatedGaussianNLLUnknownVariance(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated NLL of Gaussian regression in the natural parameters.

    With ``eta = v^T x`` and ``lambda = 1 / sigma^2``, so ``h = eta / lambda``,
    each sample's NLL is
    ``lambda (y - h)^2 / 2 - log(lambda) / 2 + log P(h + e in S)``, convex in
    ``(v, lambda)``; Daskalakis, Stefanou, Yao and Zampetakis (2022), (4.1).
    Its gradients are ``E[z | S] - y`` in eta and ``(y^2 - E[z^2 | S]) / 2``
    in lambda, for ``z ~ N(h, sigma^2)``, whose moments the sampler
    estimates. Returns one loss per sample; the caller reduces.
    """

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx,
        eta,
        lambda_,
        targ,
        phi,
        sampler: sampling.GaussianSampler,
        eps=1e-5,
    ):
        """Compute each sample's truncated NLL, summed over its outputs.

        Args:
            eta: Natural locations ``v^T x``, shaped (n, k).
            lambda_: Precisions ``1 / sigma^2``, shaped (k,).
            targ: Observed responses, shaped (n, k).
            phi: Truncation oracle over latents shaped (n, m, k).
            sampler: Estimates the conditional moments.
            eps: Floor on the survival mass, as ``TruncatedGaussianNLL``.

        Returns:
            The losses, shaped (n,).
        """
        pred = eta / lambda_
        moments = sampler.moments(pred, 1.0 / lambda_, phi, second_moment=True)
        second = cast(ch.Tensor, moments.second_moment)
        ctx.save_for_backward(
            moments.mean.to(eta.dtype) - targ,
            0.5 * (targ.pow(2) - second.to(eta.dtype)),
        )
        quadratic = 0.5 * lambda_ * (targ - pred).pow(2)
        log_mass = ch.log(moments.mass + eps)
        return (quadratic - 0.5 * ch.log(lambda_) + log_mass).sum(-1)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Return the gradients in eta and lambda."""
        eta_grad, lambda_grad = ctx.saved_tensors
        weights = grad_output[:, None]
        return (weights * eta_grad, (weights * lambda_grad).sum(0)) + (None,) * 4

    @staticmethod
    def hessian(
        eta,
        lambda_,
        X,  # pylint: disable=invalid-name
        phi,
        sampler: sampling.GaussianSampler,
    ):
        """Return the batch-mean Hessian of the NLL in ``(v, lambda)``.

        Per output, ``mean_i Cov[(z x_i, -z^2 / 2) | S]``; Daskalakis,
        Stefanou, Yao and Zampetakis (2022), (4.3).

        Args:
            eta: As ``forward``.
            lambda_: As ``forward``.
            X: Design matrix, shaped (n, d).
            phi: As ``forward``.
            sampler: As ``forward``.

        Returns:
            The Hessian for the parameter of shape (d + 1, k), v's rows then
            lambda's, flattened row-major, shaped ((d + 1) k, (d + 1) k), in
            float64 as ``TruncatedGaussianNLL.hessian``.
        """
        with ch.no_grad():
            moments = sampler.moments(
                eta / lambda_, 1.0 / lambda_, phi, fourth_moment=True
            )
            mean = moments.mean
            second = cast(ch.Tensor, moments.second_moment)
            third = cast(ch.Tensor, moments.third_moment)
            fourth = cast(ch.Tensor, moments.fourth_moment)
            var = second - mean.pow(2)
            cov = 0.5 * (third - mean * second)
            var_sq = 0.25 * (fourth - second.pow(2))
            n, k = eta.shape
            design = X.double()
            block = ch.zeros(k, X.size(1) + 1, X.size(1) + 1, dtype=ch.float64)
            block[:, :-1, :-1] = ch.einsum("ni,nj,na->aij", design, design, var)
            block[:, :-1, -1] = -ch.einsum("ni,na->ai", design, cov)
            block[:, -1, :-1] = block[:, :-1, -1]
            block[:, -1, -1] = var_sq.sum(0)
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


class TruncatedBCE(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated binary cross entropy gradient for truncated binary classification."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx,
        pred,
        targ,
        phi,
        num_samples=1000,
        eps=1e-5,  # pylint: disable=invalid-name
    ):
        """Compute truncated BCE loss.

        Args:
            pred: Size (batch_size, 1) regression model predictions.
            targ: Size (batch_size, 1) regression target values.
            phi: Dependent variable membership oracle.
            num_samples: Number of samples per batch element in rejection
                sampling.
            eps: Denominator error constant to avoid divide by zero.
        """
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        rand_noise = helpers.logistic.sample(stacked.size())
        noised = stacked + rand_noise
        noised_labs = noised >= 0
        filtered = phi(noised)
        mask = (noised_labs).eq(targ)
        filtered = filtered.float()
        ctx.save_for_backward(mask, filtered, rand_noise)
        ctx.eps = eps
        prob_est = (mask * filtered + eps).sum(0) / (filtered.sum(0) + ctx.eps)
        return -ch.log(prob_est) / pred.size(0)

    @staticmethod
    def backward(ctx, _grad_output):  # pylint: disable=arguments-differ,invalid-name
        """Compute gradient of truncated BCE w.r.t. pred."""
        mask, filtered, rand_noise = ctx.saved_tensors

        avg = (
            2
            * (sig(rand_noise) * mask * filtered).sum(0)
            / ((mask * filtered).sum(0) + ctx.eps)
        )
        norm_const = (2 * sig(rand_noise) * filtered).sum(0) / (
            filtered.sum(0) + ctx.eps
        )
        return -(avg - norm_const) / rand_noise.size(1), None, None, None, None


class TruncatedProbitMLE(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated probit MLE gradient for binary classification."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx,
        pred,
        targ,
        phi,
        num_samples=1000,
        eps=1e-5,  # pylint: disable=invalid-name
    ):
        """Compute truncated probit MLE loss.

        Args:
            pred: Size (batch_size, 1) regression model predictions.
            targ: Size (batch_size, 1) regression target values.
            phi: Dependent variable membership oracle.
            num_samples: Number of samples per batch element in rejection
                sampling.
            eps: Denominator error constant to avoid divide by zero.
        """
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        rand_noise = ch.randn(stacked.size())
        noised = stacked + rand_noise
        noised_labs = noised >= 0
        mask = noised_labs.eq(targ)
        filtered = phi(noised)

        mle = (filtered * mask).sum(0) + eps
        trunc_const = (filtered).sum(0)

        ctx.save_for_backward(rand_noise, filtered, mask)
        ctx.eps = eps

        return -ch.log(mle / (trunc_const + eps)) / pred.size(0)

    @staticmethod
    def backward(ctx, _grad_output):  # pylint: disable=arguments-differ,invalid-name
        """Compute gradient of truncated probit MLE w.r.t. pred."""
        rand_noise, filtered, mask = ctx.saved_tensors
        nll = (mask * filtered * rand_noise).sum(dim=0) / (
            (mask * filtered).sum(dim=0) + ctx.eps
        )
        const = (rand_noise * filtered).sum(dim=0) / (filtered.sum(dim=0) + ctx.eps)
        return -(nll - const) / rand_noise.size(1), None, None, None, None


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


class TruncatedCE(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated cross-entropy gradient for truncated multiclass classification."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx,
        pred,
        targ,
        phi,
        num_samples=5000,
        eps=1e-5,  # pylint: disable=invalid-name
    ):
        """Compute truncated cross-entropy loss.

        Args:
            pred: Size (batch_size, 1) regression model predictions.
            targ: Size (batch_size, 1) regression target values.
            phi: Dependent variable membership oracle.
            num_samples: Number of samples per batch element in rejection
                sampling.
            eps: Denominator error constant to avoid divide by zero.
        """
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        rand_noise = gumbel.sample(stacked.size())
        noised = stacked + rand_noise
        noised_labs = noised.argmax(-1, keepdim=True)
        filtered = phi(noised).float()
        mask = (noised_labs).eq(targ)
        ctx.save_for_backward(mask, filtered, rand_noise, pred)
        ctx.eps = eps
        prob_est = (mask * filtered + eps).sum(0) / (filtered.sum(0) + eps)
        return -ch.log(prob_est) / pred.size(0)

    @staticmethod
    def backward(ctx, _grad_output):  # pylint: disable=arguments-differ,invalid-name
        """Compute gradient of truncated CE w.r.t. pred."""
        mask, filtered, rand_noise, pred = ctx.saved_tensors
        inner_exp = 1 - ch.exp(-rand_noise)
        nll = (inner_exp * mask * filtered).sum(0) / (
            (mask * filtered).sum(0) + ctx.eps
        )
        const = (inner_exp * filtered).sum(0) / (filtered.sum(0) + ctx.eps)

        return (-nll + const) / pred.size(0), None, None, None, None


class TruncatedCELabels(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated cross-entropy gradient using label filtering."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx,
        pred,
        targ,
        phi,
        num_samples=5000,
        eps=1e-5,  # pylint: disable=invalid-name
    ):
        """Compute truncated cross-entropy loss with label filtering.

        Args:
            pred: Size (batch_size, 1) regression model predictions.
            targ: Size (batch_size, 1) regression target values.
            phi: Dependent variable membership oracle.
            num_samples: Number of samples per batch element in rejection
                sampling.
            eps: Denominator error constant to avoid divide by zero.
        """
        stacked = pred[None, ...].repeat(num_samples, 1, 1)
        rand_noise = gumbel.sample(stacked.size())
        noised = stacked + rand_noise
        noised_labs = noised.argmax(-1, keepdim=True)
        filtered = phi(noised, targ).float()
        mask = (noised_labs).eq(targ)
        ctx.save_for_backward(mask, filtered, rand_noise, pred)
        ctx.eps = eps
        prob_est = (mask * filtered + eps).sum(0) / (filtered.sum(0) + eps)
        return -ch.log(prob_est) / pred.size(0)

    @staticmethod
    def backward(ctx, _grad_output):  # pylint: disable=arguments-differ,invalid-name
        """Compute gradient of truncated CE with labels w.r.t. pred."""
        mask, filtered, rand_noise, pred = ctx.saved_tensors
        inner_exp = 1 - ch.exp(-rand_noise)
        nll = (inner_exp * mask * filtered).sum(0) / (
            (mask * filtered).sum(0) + ctx.eps
        )
        const = (inner_exp * filtered).sum(0) / (filtered.sum(0) + ctx.eps)
        return (-nll + const) / pred.size(0), None, None, None, None
