"""Normalized probabilist Hermite polynomial utilities for multivariate Gaussian estimation."""

import math
from itertools import product

import torch as ch


class NormalizedProbabilistHermitePolynomial:  # pylint: disable=too-few-public-methods
    """
    Normalized probabilist hermite polynomial. Assumes that the input samples are whitened N(0, I).
    """

    def __init__(self, k, d):
        """
        Args:
            k: maximum polynomial degree
            d: dimensionality of the input samples
        """
        self._k = k
        self._d = d

        self._factorials = ch.tensor(
            [math.factorial(i) for i in range(k + 1)], dtype=ch.float32
        )
        self.multi_indices = self._generate_multi_indices(self._k)
        self._norm_factors = self._precompute_norm_factors()

    def _generate_multi_indices(self, k):
        """Generate all multi-indices with total degree <= k."""
        indices = []
        for combo in product(range(k + 1), repeat=self._d):
            if sum(combo) <= k:
                indices.append(list(combo))
        return ch.tensor(indices, dtype=ch.long)

    def _precompute_norm_factors(self):
        """Precompute 1/sqrt(prod(v_i!)) for each multi-index v."""
        # for each multi-index, compute product of sqrt(factorial(v_i))
        norm_factors = ch.ones(len(self.multi_indices), dtype=ch.float32)
        for idx, v in enumerate(self.multi_indices):  # pylint: disable=invalid-name
            # product of sqrt(v_i!) over all dimensions
            factorial_prod = ch.prod(ch.sqrt(self._factorials[v]))
            norm_factors[idx] = 1.0 / factorial_prod if factorial_prod > 0 else 1.0
        return norm_factors

    def _hermite_polynomial_batch(self, z, max_degree):
        """
        Compute He_0(z), He_1(z), ..., He_max_degree(z) for all z simultaneously.

        Args:
            z: tensor of shape (n,) or (n, d)
            max_degree: maximum degree to compute

        Returns:
            tensor of shape (n, max_degree+1) containing He_i(z) for i=0..max_degree
        """
        if z.dim() == 1:
            z = z.unsqueeze(-1)  # (n,) -> (n, 1)

        n = z.shape[0]
        d = z.shape[1]

        # initialize result tensor: (n, d_z, max_degree+1)
        he = ch.zeros(n, d, max_degree + 1, dtype=z.dtype, device=z.device)  # pylint: disable=invalid-name

        # base cases
        he[:, :, 0] = 1.0  # He_0(z) = 1
        if max_degree >= 1:
            he[:, :, 1] = z  # He_1(z) = z

        # recurrence: He_n(z) = z * He_{n-1}(z) - (n-1) * He_{n-2}(z)
        for deg in range(2, max_degree + 1):
            he[:, :, deg] = z * he[:, :, deg - 1] - (deg - 1) * he[:, :, deg - 2]

        return he

    def h_v(self, x):  # pylint: disable=invalid-name
        """
        Evaluate multivariate normalized probabilist Hermite polynomials.

        Args:
            x: tensor of shape (n, d) or (n,) for univariate case

        Returns:
            tensor of shape (n, num_indices)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Handle univariate case: (n,) -> (n, 1)

        n = x.shape[0]
        num_indices = len(self.multi_indices)

        # compute all Hermite polynomials up to degree k for all dimensions at once
        # he shape: (n, d, k+1) where he[i, j, deg] = He_deg(x[i, j])
        he = self._hermite_polynomial_batch(x, self._k)  # pylint: disable=invalid-name

        # build result by gathering appropriate Hermite values for each multi-index
        result = ch.ones(n, num_indices, dtype=x.dtype, device=x.device)

        for idx, v in enumerate(self.multi_indices):  # pylint: disable=invalid-name
            for dim in range(self._d):
                degree = v[dim].item()
                result[:, idx] *= he[:, dim, degree]

            result[:, idx] *= self._norm_factors[idx]

        return result
