# Author: pstefanou12@
"""Truncated NLL autograd functions for exponential family distributions."""

import torch as ch

from delphi.truncated import sampling


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

        z, p_hat = sampling.rejection_sample(D, phi, num_samples)
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

    @staticmethod
    def hessian(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        theta,
        phi,  # pylint: disable=invalid-name
        dims,
        dist,
        calc_suff_stat,  # pylint: disable=invalid-name
        num_samples=1000,
    ):
        """Compute the Hessian of the truncated NLL at theta.

        Args:
            theta: Natural parameters to evaluate at.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            dist: Callable building the distribution from theta and dims.
            calc_suff_stat: Sufficient statistic of the family.
            num_samples: Rejection samples backing the estimate.

        Returns:
            ``Cov[T(z)]`` over the truncated distribution, shaped (k, k) for
            k sufficient statistics. Product families leave the off-diagonal
            entries at sampling noise, so callers wanting per-coordinate
            curvature should take the diagonal.
        """
        # Differentiating backward() cannot produce this: its gradient is a
        # constant built from saved samples, so theta's influence -- which
        # lives in the sampling distribution -- is invisible to autograd.
        # The identity below is exact for an exponential family, where the
        # log-partition function's second derivative is Cov[T].
        with ch.no_grad():
            z, _ = sampling.rejection_sample(dist(theta, dims), phi, num_samples)
            return calc_suff_stat(z).T.cov()


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
