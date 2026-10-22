# Author: pstefanou12@
"""Losses for the truncated distributions.

The truncated NLL of an exponential family, its untruncated counterpart for
the warm start, and the objective of the unknown-truncation normal.
"""

from collections.abc import Callable
import logging

import torch as ch

from delphi.truncated import sampling

logger = logging.getLogger(__name__)


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
            logger.warning(f"low acceptance rate: {p_hat.item()}")

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
        # Autograd cannot produce this: backward() returns a constant built
        # from saved samples, hiding theta's effect on where they came from.
        with ch.no_grad():
            z, _ = sampling.rejection_sample(dist(theta, dims), phi, num_samples)
            # torch.cov returns a 0-d scalar for a single statistic, not a
            # 1 x 1 matrix.
            return ch.atleast_2d(calc_suff_stat(z).T.cov())


class ExponentialFamilyDistributionNLL:
    """Untruncated negative log likelihood for exponential family distributions.

    The warm start's criterion. Its log-partition is in closed form in the
    family's ``log_prob``, so autograd differentiates it exactly and it needs
    no hand-written backward, unlike the truncated NLL, whose normalizer is a
    Monte Carlo estimate. ``apply`` and ``hessian`` mirror that class.
    """

    @staticmethod
    def apply(
        theta: ch.Tensor, data: ch.Tensor, dims: int, dist: Callable
    ) -> ch.Tensor:
        """Return each sample's term of the batch's mean NLL.

        Each term carries the 1 / batch size, as the other criteria's do, so
        the trainer's sum over the batch is the mean.

        Args:
            theta: Natural parameters.
            data: Batch whose first ``dims`` columns are the samples.
            dims: Number of dimensions.
            dist: Maps ``(theta, dims)`` to the family's distribution.

        Returns:
            One term per sample, shaped (batch,).
        """
        return -dist(theta, dims).log_prob(data[:, :dims]) / data.size(0)

    @staticmethod
    def hessian(
        theta: ch.Tensor, data: ch.Tensor, dims: int, dist: Callable
    ) -> ch.Tensor:
        """Return the Hessian of the mean NLL, ``Cov[T(z)]`` in natural parameters.

        Args:
            theta: Point to evaluate at.
            data: Samples, shaped (n, dims) or wider.
            dims: Number of dimensions.
            dist: Maps ``(theta, dims)`` to the family's distribution.

        Returns:
            The Hessian, shaped (k, k).
        """
        return ch.autograd.functional.hessian(
            lambda point: ExponentialFamilyDistributionNLL.apply(
                point, data, dims, dist
            ).sum(),
            theta.detach(),
        )


class UnknownTruncationMultivariateNormalNLL(  # pylint: disable=abstract-method
    ch.autograd.Function
):
    """Objective of the normal under unknown truncation, with its gradient.

    Kontonis et al. (2019)'s equation (5), the convex surrogate whose
    minimizer recovers the Gaussian; named like its siblings. Each row of
    ``data`` is ``[T(z), c(z), psi_k(z)]`` for one whitened sample, and its
    term is ``psi_k(z) exp(<theta, T(z)> + c(z))``, which is
    ``exp(h(u, B; z)) N(0, I; z) psi_k(z)`` written to be affine in theta
    inside the exponential.
    """

    @staticmethod
    def terms(theta: ch.Tensor, data: ch.Tensor) -> ch.Tensor:
        """Return each row's term of the objective.

        Args:
            theta: Free natural parameters, shaped (p,).
            data: Rows shaped (n, p + 2).

        Returns:
            Shaped (n,).
        """
        num_params = theta.numel()
        statistic, offset, psi = (
            data[:, :num_params],
            data[:, num_params],
            data[:, num_params + 1],
        )
        return psi * ch.exp(statistic @ theta + offset)

    @staticmethod
    def forward(ctx, theta, data):  # pylint: disable=arguments-differ
        """Return the mean term over the batch."""
        terms = UnknownTruncationMultivariateNormalNLL.terms(theta, data)
        ctx.save_for_backward(terms, data[:, : theta.numel()])
        return terms.mean()

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Return the gradient ``E[T(z) term(z)]``, the step of Algorithm 1."""
        terms, statistic = ctx.saved_tensors
        return grad_output * (statistic * terms[:, None]).mean(0), None

    @staticmethod
    def hessian(theta: ch.Tensor, data: ch.Tensor) -> ch.Tensor:
        """Return the Hessian ``E[term(z) T(z) T(z)^T]``, PSD as Claim 1 says.

        Args:
            theta: Point to evaluate at, shaped (p,).
            data: Rows shaped (n, p + 2).

        Returns:
            Shaped (p, p).
        """
        statistic = data[:, : theta.numel()]
        terms = UnknownTruncationMultivariateNormalNLL.terms(theta.detach(), data)
        return statistic.T @ (terms[:, None] * statistic) / data.size(0)
