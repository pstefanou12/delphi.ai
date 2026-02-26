# Author: pstefanou12@
"""Truncated NLL autograd functions for exponential family distributions."""

import torch as ch


class TruncatedExponentialFamilyDistributionNLL(  # pylint: disable=abstract-method
    ch.autograd.Function
):
    """Truncated negative log likelihood for exponential family distributions.

    Calculates the truncated NLL in forward and gradients w.r.t. theta in
    backward.  Samples batch_size * num_samples samples, filters those in the
    truncation set, and retains up to num_samples filtered samples.  If fewer
    than num_samples remain, uses zeros and computes the untruncated log
    likelihood.
    """

    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments,arguments-differ,too-many-locals
        ctx,
        theta,
        data,
        phi,
        dims,
        dist,
        calc_suff_stat,  # pylint: disable=invalid-name
        num_samples=1000,
        eps=1e-12,
    ):
        """Compute truncated NLL."""
        S = data[:, :dims]  # pylint: disable=invalid-name
        S_suff_stat = data[:, dims:]  # pylint: disable=invalid-name

        D = dist(theta, dims)  # pylint: disable=invalid-name
        log_prob = D.log_prob(S)

        z = []
        num_sampled = 0
        num_accepted = 0

        while num_accepted < num_samples:
            s = D.sample([10 * num_samples])
            mask = phi(s)
            accepted = s[mask.nonzero()[:, 0]]
            z.append(accepted)
            num_accepted += accepted.size(0)
            num_sampled += 10 * num_samples
        z = ch.cat(z)

        p_hat = ch.Tensor([num_accepted / num_sampled])
        if p_hat < 0.01:
            print(f"acceptance rate: {p_hat.item()}")

        trunc_const = ch.log(p_hat + eps)
        ll = log_prob - trunc_const
        ctx.save_for_backward(z, S_suff_stat)
        ctx.calc_suff_stat = calc_suff_stat
        return -ll.mean()

    @staticmethod
    def backward(ctx, _grad_output):  # pylint: disable=arguments-differ,invalid-name
        """Compute gradients of truncated NLL w.r.t. theta."""
        s, S_suff_stat = ctx.saved_tensors  # pylint: disable=invalid-name
        trunc_const_suff_stat = ctx.calc_suff_stat(s).mean(0)
        # Average over the batch dimension so the returned gradient has the
        # same shape as theta (the first input), regardless of batch size.
        grad = (-S_suff_stat + trunc_const_suff_stat).mean(0)
        return (
            grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class UnknownTruncationMultivariateNormalNLL(  # pylint: disable=abstract-method
    ch.autograd.Function
):
    """Negative population log likelihood for multivariate normal with unknown truncation.

    Calculates the population log-likelihood for the current batch in forward,
    then calculates its gradient in backward.
    """

    @staticmethod
    def forward(ctx, theta, data, phi, exp_h, dims):  # pylint: disable=invalid-name,arguments-differ,too-many-arguments,too-many-positional-arguments
        """Compute population log-likelihood.

        Args:
            theta: Current reparameterized mean and covariance matrix estimates
                concatenated.
            data: Precomputed gradient values for both the mean and covariance
                matrix.
            phi: Oracle object for learning truncation set.
            exp_h: Helper class for calculating exponential in the gradient.
            dims: The dimension number.
        """
        T = theta[: dims**2].view(dims, dims)  # pylint: disable=invalid-name
        v = theta[dims**2 :]
        x = data[:, :dims].view(data.size(0), dims)
        pdf = data[:, dims][..., None]
        loc_grad = data[:, dims + 1 : dims + dims + 1].view(data.size(0), dims)
        cov_grad = data[:, dims + dims + 1 :].view(data.size(0), dims, dims)
        exp = exp_h(v, T, x)
        psi = phi.psi_k(x)
        loss = exp * pdf * psi
        print(f"psi: {psi}")
        print(f"exp: {exp}")
        print(f"pdf: {pdf}")
        ctx.save_for_backward(loss, loc_grad, cov_grad)
        ctx.dims = dims
        return loss / data.size(0)

    @staticmethod
    def backward(ctx, _grad_output):  # pylint: disable=arguments-differ,invalid-name
        """Compute gradients of NLL w.r.t. theta."""
        loss, loc_grad, cov_grad = ctx.saved_tensors
        term_one = loc_grad * loss
        term_two = cov_grad.flatten(1) * loss
        print(f"loc grad: {term_one.mean(0)}")
        print(f"cov grad: {term_two.mean(0)}")
        return (
            ch.cat([term_two, term_one], dim=1) / cov_grad.size(0),
            None,
            None,
            None,
            None,
            None,
        )
