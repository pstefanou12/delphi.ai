# Author: pstefanou12@
"""Autograd loss functions for truncated regression and classification models."""

import math

import torch as ch
from torch import sigmoid as sig  # pylint: disable=no-name-in-module
import torch.distributions
import torch.nn

from delphi.utils import helpers

gumbel = torch.distributions.Gumbel(0, 1)

# Draws per sample a TruncatedMSE estimate may take to collect min_accepted;
# a sample whose survival mass is below min_accepted / MAX_DRAWS_PER_SAMPLE
# keeps what it has.
MAX_DRAWS_PER_SAMPLE = 1_000_000
# Caps one round's draws across the batch, bounding its memory.
MAX_DRAWS_PER_ROUND = 10_000_000


class TruncatedMSE(ch.autograd.Function):  # pylint: disable=abstract-method
    """Truncated NLL of regression with known noise variance.

    Per sample, ``(y - h)^2 / (2 sigma^2) + log P(h + e in S)`` with
    ``e ~ N(0, sigma^2)``; its gradient in h is ``(E[z | S] - y) / sigma^2``
    for ``z ~ N(h, sigma^2)``, estimated from rejection-sampled draws at the
    sample's own h. ``phi`` may return a survival probability rather than a
    mask, in which case each draw is kept with that probability.
    """

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ
        ctx,
        pred,
        targ,
        phi,
        noise_var,
        num_samples=1000,
        eps=1e-5,
        min_accepted=10,
    ):
        """Compute the batch-mean truncated NLL.

        Args:
            pred: Predictions h, shaped (n, k).
            targ: Observed responses, shaped (n, k).
            phi: Truncation oracle over latents shaped (n, m, k).
            noise_var: Known noise variance, broadcastable to (k,).
            num_samples: Draws per sample in the first round.
            eps: Floor on the accepted fraction, so a sample phi rejects
                entirely contributes a finite loss.
            min_accepted: Kept draws each sample collects; see ``sample``.
        """
        noise_var = ch.as_tensor(noise_var, dtype=pred.dtype)
        accepted, drawn, total, _ = TruncatedMSE.sample(
            pred, phi, noise_var, num_samples, min_accepted
        )
        conditional_mean = total / (accepted + eps)
        ctx.save_for_backward((conditional_mean - targ) / noise_var)
        quadratic = 0.5 * (targ - pred).pow(2) / noise_var
        log_mass = ch.log(accepted / drawn + eps)
        return (quadratic + log_mass).sum(-1).mean()

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Return the gradient in pred."""
        (grad,) = ctx.saved_tensors
        return (grad_output * grad / grad.size(0),) + (None,) * 6

    @staticmethod
    def sample(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        pred, phi, noise_var, num_samples, min_accepted, squares=False
    ):
        """Rejection-sample N(pred, sigma^2) until phi keeps enough draws.

        A fixed number of draws leaves a sample phi rarely admits with no
        kept draw, whose conditional mean then reads as zero and biases the
        gradient. Samples short of min_accepted draw again, each round sized
        from the slowest one's acceptance rate.

        Args:
            pred: Predictions h, shaped (n, k).
            phi: As ``forward``.
            noise_var: Known noise variance, broadcastable to (k,).
            num_samples: Draws per sample in the first round.
            min_accepted: Kept draws each sample collects, up to
                ``MAX_DRAWS_PER_SAMPLE`` draws.
            squares: Also accumulate the kept draws' squares.

        Returns:
            Per sample: kept draws (n, 1), draws (n, 1), and the sums of the
            kept latents and, if requested, their squares (n, k), the latter
            accumulated in float64.
        """
        n = pred.size(0)
        accepted, drawn = ch.zeros(n, 1), ch.zeros(n, 1)
        total = ch.zeros_like(pred, dtype=ch.float64)
        total_sq = ch.zeros_like(total) if squares else None
        rows, block = ch.arange(n), num_samples
        while rows.numel() > 0:
            z = pred[rows, None, :] + noise_var.sqrt() * ch.randn(
                rows.numel(), block, pred.size(1)
            )
            kept = phi(z).reshape(rows.numel(), block, 1)
            kept = ch.bernoulli(kept) if kept.is_floating_point() else kept.float()
            accepted[rows] += kept.sum(1)
            drawn[rows] += block
            total[rows] += (kept * z).sum(1).double()
            if total_sq is not None:
                total_sq[rows] += (kept * z.pow(2)).sum(1).double()
            short = (accepted[rows, 0] < min_accepted) & (
                drawn[rows, 0] < MAX_DRAWS_PER_SAMPLE
            )
            rows = rows[short]
            if rows.numel() > 0:
                rate = float((accepted[rows] / drawn[rows]).min().clamp_min(1e-4))
                needed = float((min_accepted - accepted[rows]).max()) / rate
                block = int(
                    min(
                        needed + 1,
                        max(MAX_DRAWS_PER_ROUND // rows.numel(), 1),
                        MAX_DRAWS_PER_SAMPLE - float(drawn[rows].max()),
                    )
                )
        return accepted, drawn, total.float(), total_sq

    @staticmethod
    def hessian(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        pred, X, phi, noise_var, num_samples=1000, eps=1e-5, min_accepted=10
    ):  # pylint: disable=invalid-name
        """Return the batch-mean Hessian of the NLL in the weight.

        ``mean_i Var[z_i | S] / sigma^4 x_i x_i^T``, per output.

        Args:
            pred: Predictions h, shaped (n, k).
            X: Design matrix, shaped (n, d).
            phi: As ``forward``.
            noise_var: As ``forward``.
            num_samples: As ``forward``.
            eps: As ``forward``.
            min_accepted: As ``forward``.

        Returns:
            The Hessian for the weight of shape (d, k) flattened row-major,
            shaped (d * k, d * k).
        """
        with ch.no_grad():
            noise_var = ch.as_tensor(noise_var, dtype=pred.dtype)
            accepted, _, total, total_sq = TruncatedMSE.sample(
                pred, phi, noise_var, num_samples, min_accepted, squares=True
            )
            count = accepted.double() + eps
            mean = total.double() / count
            var = (total_sq / count - mean.pow(2)) / noise_var.double().pow(2)
            design = X.double()
            hess = ch.einsum(
                "ni,nj,na,ab->iajb",
                design,
                design,
                var,
                ch.eye(pred.size(1), dtype=ch.float64),
            )
            size = X.size(1) * pred.size(1)
            return (hess.reshape(size, size) / pred.size(0)).float()


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
