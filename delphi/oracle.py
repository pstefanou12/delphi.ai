# Author: pstefanou12@
"""Membership oracle for unknown truncation.

A known truncation set is a predicate the caller supplies as a plain
callable. This oracle is the exception: it is not given but estimated from
the data with a Hermite expansion, so it carries state.
"""

from abc import ABC

import torch as ch
from torch.distributions.multivariate_normal import MultivariateNormal

from delphi.polynomials import NormalizedProbabilistHermitePolynomial
from delphi.utils.helpers import cov


class oracle(ABC):  # pylint: disable=invalid-name,too-few-public-methods
    """
    Oracle for data sets.
    """

    def __call__(self, x):
        """
        Membership oracle.
        Args:
            x: samples to check membership
        """


class UnknownGaussian(oracle):  # pylint: disable=too-many-instance-attributes
    """
    Oracle for an unknown Gaussian distribution estimated via Hermite polynomial expansion.
    """

    def __init__(self, k, s):  # pylint: disable=invalid-name
        """
        Args:
            k: maximum Hermite polynomial degree
            s (torch.Tensor): whitened sample set of shape (n, d), assumed N(0, I)
        """
        self.emp_loc = s.mean(0)
        self.emp_cov = cov(s)

        # verify that the s is whitened to N(0, I)
        if (
            ch.norm(self.emp_loc - ch.zeros(s.size(1))) >= 1e-3
            or ch.norm(self.emp_cov - ch.eye(s.size(1))) >= 1e-3
        ):
            raise ValueError(
                f"input dataset must be whitened (eg. N(O, I)). \n"
                f" dataset mean: {self.emp_loc}, covariance matrix: {self.emp_cov}"
            )

        self._emp_dist = MultivariateNormal(self.emp_loc, self.emp_cov)
        self._d = int(s.shape[1])
        self._k = k

        self.herm_poly = NormalizedProbabilistHermitePolynomial(self._k, self._d)
        self._c_v = self._compute_hermite_coefficients(s)  # pylint: disable=invalid-name
        self._dist = None

    def _compute_hermite_coefficients(self, s):  # pylint: disable=invalid-name
        """
        Compute c_v = E_D[H_V(z)] where H_V evaluated at standardized samples from s.
        s should be samples *from the truncated set* (i.e., only samples x in s).
        """
        h_vals = self.herm_poly.h_v(s)  # shape (n, num_indices)
        return h_vals.mean(dim=0)

    def psi_k(self, x):
        """
        Evaluate psi approximation on x (tensor shape (n,d)).
        Returns (n, 1) tensor with values ~[0,1].
        """
        h_vals = self.herm_poly.h_v(x)  # (n, num_indices)
        expansion = (self._c_v * h_vals).sum(dim=1)  # (n,)
        return ch.clamp(expansion, min=0)[..., None]

    def __call__(self, x):
        """Return membership indicator using learned distribution ratio and psi approximation."""
        if self._dist is None:
            raise ValueError("must learn underlying distribution for membership oracle")

        ratio = ch.exp(self._emp_dist.log_prob(x) - self._dist.log_prob(x))[:, None]
        return ((ratio * self.psi_k(x)) > 0.5).float()

    @property
    def emp_dist(self):
        """Return the empirical distribution."""
        return self._emp_dist

    @property
    def dist(self):
        """Return the learned distribution."""
        return self._dist

    @dist.setter
    def dist(self, dist_):
        """Set the learned distribution."""
        self._dist = dist_

    @property
    def c_v(self):  # pylint: disable=invalid-name
        """Return the Hermite coefficients."""
        return self._c_v

    @property
    def d(self):
        """Return the dimensionality."""
        return self._d

    def __str__(self):
        """Return string representation."""
        return "unknown gaussian"
