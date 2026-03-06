# Author: pstefanou12@
"""Autograd loss functions for truncated regression and classification models."""

import math

import torch as ch
import torch.distributions
import torch.nn
from torch import sigmoid as sig  # pylint: disable=no-name-in-module

from delphi.utils import helpers

softmax = torch.nn.Softmax(dim=1)
gumbel = torch.distributions.Gumbel(0, 1)


def Test(mu, phi, c_gamma, alpha, T):  # pylint: disable=invalid-name
    """Check which gradient to take at timestep t.

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


class TruncatedMSE(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated mean squared error loss for regression with known noise variance."""

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx,
        pred,
        targ,
        phi,
        noise_var,
        num_samples=1000,
        eps=1e-10,  # pylint: disable=invalid-name
    ):
        """Compute truncated MSE loss."""
        stacked = pred.unsqueeze(1).repeat(1, num_samples, 1)
        noise = (noise_var**0.5) * ch.randn_like(stacked)
        noised = stacked + noise

        mask = phi(noised).float()
        z = (mask * noised).sum(dim=1) / (mask.sum(dim=1) + eps)
        P_hat = mask.mean(dim=1).clamp_min(eps)  # pylint: disable=invalid-name

        ctx.save_for_backward(targ, z, noise_var)
        quadratic_loss = -0.5 * (pred - targ).pow(2) / noise_var
        trunc_const = ch.log(P_hat + eps)

        return -(quadratic_loss - trunc_const) / pred.size(0)

    @staticmethod
    def backward(ctx, _grad_output):  # pylint: disable=arguments-differ,invalid-name
        """Compute gradient of truncated MSE w.r.t. pred."""
        targ, z, noise_var = ctx.saved_tensors

        grad_pred = (targ - z) / noise_var
        return -grad_pred / targ.size(0), None, None, None, None, None


class TruncatedUnknownVarianceMSE(ch.autograd.Function):  # pylint: disable=abstract-method
    """MLE for truncated Gaussian regression with unknown noise variance.

    Optimization variables: mu (pred) and lambda_ (1/sigma^2).
    """

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ,too-many-locals
        ctx,
        pred,
        targ,
        lambda_,
        phi,  # pylint: disable=invalid-name
        num_samples=1000,
        eps=1e-10,
        noise=None,
    ):
        """Compute truncated MSE loss with unknown noise variance."""
        noise_var = 1.0 / lambda_
        sigma = ch.sqrt(noise_var)

        stacked = pred[..., None].repeat(1, num_samples, 1)
        noise = sigma * ch.randn_like(stacked)
        noised = stacked + noise

        mask = phi(noised).float()
        filtered = mask * noised
        z = filtered.sum(dim=1) / (mask.sum(dim=1) + eps)
        z_2 = filtered.pow(2).sum(dim=1) / (mask.sum(dim=1) + eps)
        P_hat = mask.mean(dim=1).clamp_min(eps)  # pylint: disable=invalid-name

        quadratic_loss = -0.5 * lambda_ * (targ - pred).pow(2)
        log_lambda_ = -0.5 * ch.log(lambda_)
        trunc_const = ch.log(P_hat)
        ctx.save_for_backward(pred, targ, lambda_, z, z_2)
        return -(quadratic_loss - log_lambda_ - trunc_const) / pred.size(0)

    @staticmethod
    def backward(ctx, _grad_output):  # pylint: disable=arguments-differ,invalid-name
        """Compute gradients of truncated MSE w.r.t. pred and lambda."""
        pred, targ, lambda_, z, z_2 = ctx.saved_tensors

        mu_grad = lambda_ * (z - targ)
        lambda_grad = 0.5 * (targ.pow(2).mean(0) - z_2.mean(0))[..., None]

        return mu_grad / pred.size(0), None, lambda_grad, None, None, None, None


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
        result = Test(pred, phi, c_gamma, alpha, T)

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
