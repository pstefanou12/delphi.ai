# Author: pstefanou12@
"""Multivariate normal estimation from samples truncated to an unknown set.

Kontonis, Tzamos and Zampetakis (2019), "Efficient Truncated Statistics with
Unknown Truncation". The samples come from ``N(mu*, Sigma*)`` conditioned on a
set ``S`` that cannot be queried. In a frame where the sample is whitened, the
weighted indicator

    psi(x) = 1_S(x) N(mu*, Sigma*; x) / (alpha* N(0, I; x))

entangles the set with the Gaussian, and its Hermite coefficients are the
sample means ``E_{N_S}[H_V]`` (Theorem 4). The objective

    M(u, B) = E_{x ~ N_S}[exp(h(u, B; x)) N(0, I; x) psi_k(x)]

is convex in the natural parameters ``(u, B) = (Sigma^-1 mu, Sigma^-1)``
(Claim 1), and with the exact ``psi`` its minimizer is the truth (Claim 2).
Algorithm 1 minimizes it by projected stochastic gradient descent from the
empirical moments and averages the iterates; Theorem 6 then recovers the set
as ``{N(0, I) psi_k / N(mu_hat, Sigma_hat) > 1 / 2}``.

``h`` is affine in ``(u, B)``, so every per-sample term is
``psi_k(x) exp(<theta, T(x)> + c(x))`` with ``T`` and ``c`` fixed once the frame
is: the descent precomputes them and trains on the rows ``[T, c, psi_k]``.
"""

import abc
import math
from typing import NamedTuple, Optional, cast

import torch as ch
from torch import distributions as torch_distributions
from torch import nn
from torch.utils import data

from delphi import configs, trainer
import delphi.oracle
from delphi.torchnomials import hermite
from delphi.truncated.distributions import distributions, losses

# Sizes the projection set when the caller gives no survival mass. A smaller
# mass only widens the set, so this errs towards not cutting off the truth.
DEFAULT_ALPHA = 0.01


class Frame(NamedTuple):
    """An affine change of coordinates ``x = scale @ z + center``.

    Attributes:
        center: Origin of the frame, shaped (dims,).
        scale: Lower-triangular factor, shaped (dims, dims).
    """

    center: ch.Tensor
    scale: ch.Tensor

    def to_frame(self, x: ch.Tensor) -> ch.Tensor:
        """Return x in frame coordinates.

        Args:
            x: Batch of shape (n, dims).

        Returns:
            ``scale^-1 (x - center)`` row by row, shaped (n, dims).
        """
        centered = (x.to(self.center.dtype) - self.center).T
        return ch.linalg.solve_triangular(self.scale, centered, upper=False).T  # pylint: disable=not-callable

    def from_frame(
        self, loc: ch.Tensor, covariance_matrix: ch.Tensor
    ) -> tuple[ch.Tensor, ch.Tensor]:
        """Map a Gaussian's parameters out of frame coordinates.

        Args:
            loc: Mean in the frame.
            covariance_matrix: Covariance in the frame.

        Returns:
            The mean and covariance in the original coordinates.
        """
        return (
            self.scale @ loc + self.center,
            self.scale @ covariance_matrix @ self.scale.T,
        )


class UnknownTruncationMultivariateNormalBase(  # pylint: disable=too-many-instance-attributes
    distributions.distributions[configs.UnknownTruncationNormalConfig], abc.ABC
):
    """Shared procedure of the unknown-truncation normal estimators.

    A subclass fixes which natural parameters are free: their statistic
    ``T``, the offset ``c``, the start point and the projection set.

    Attributes:
        dims: Number of dimensions.
        alpha: Survival mass of the set, if known; it only sizes the
            projection set.
        frame_: The whitening frame, set during fit.
        expansion_: The degree-k Hermite expansion ``psi_k``, set during fit.
        emp_theta: The start point of Algorithm 1, in the frame.
        best_params: Iterate at the best validation loss.
        final_params: Last iterate.
        ema_params: Exponential moving average of the iterates.
        avg_params: Average of the iterates after the burn-in.
    """

    defaults = configs.UnknownTruncationNormalConfig
    estimate_suffixes: tuple[str, ...] = ("loc_", "covariance_matrix_")
    # Read by FisherLR.spectrum: the coordinates of (u, B) are correlated.
    independent_coordinates = False

    def __init__(self, hyperparameters: dict, dims: int, alpha: Optional[float] = None):
        """Initialize the estimator.

        Args:
            hyperparameters: Hyperparameter dict.
            dims: Number of dimensions.
            alpha: Survival mass of the set in (0, 1), if known.

        Raises:
            ValueError: If alpha lies outside (0, 1).
        """
        super().__init__(hyperparameters)
        if alpha is not None and not 0.0 < alpha < 1.0:
            raise ValueError(
                f"alpha is the survival mass and must lie in (0, 1); got {alpha}."
            )
        self.dims = dims
        self.alpha = alpha
        self.criterion = losses.unknown_truncation_objective
        self.theta = nn.Parameter(ch.zeros(self._num_params()))

        self.frame_: Optional[Frame] = None
        self.expansion_: Optional[hermite.HermiteExpansion] = None
        self.emp_theta: Optional[ch.Tensor] = None
        self.train_loader_: Optional[data.DataLoader] = None
        self.val_loader_: Optional[data.DataLoader] = None
        self.trainer: Optional[trainer.Trainer] = None
        self.procedure: Optional[trainer.ProcedureResult] = None
        self.best_params: Optional[ch.Tensor] = None
        self.final_params: Optional[ch.Tensor] = None
        self.ema_params: Optional[ch.Tensor] = None
        self.avg_params: Optional[ch.Tensor] = None
        # The descent's rows, kept for the Hessian the Fisher schedule reads.
        self._rows: Optional[ch.Tensor] = None
        self._min_precision, self._max_precision, self._loc_radius = 0.0, 0.0, 0.0

    @abc.abstractmethod
    def _num_params(self) -> int:
        """Return the number of free natural parameters."""

    @abc.abstractmethod
    def _make_frame(self, sample: ch.Tensor) -> Frame:
        """Return the frame the expansion and the descent run in."""

    @abc.abstractmethod
    def _statistic(self, z: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        """Return ``T(z)`` and ``c(z)``, with ``h + log N(0, I) = <theta, T> + c``."""

    @abc.abstractmethod
    def _natural(self, theta: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        """Return ``(u, B)`` in the frame from the free parameters."""

    @abc.abstractmethod
    def _project(self, theta: ch.Tensor) -> ch.Tensor:
        """Return the Euclidean projection of theta onto the set D."""

    @abc.abstractmethod
    def _initial_theta(self) -> ch.Tensor:
        """Return Algorithm 1's start: the empirical moments, in the frame."""

    def _largest_precision(self) -> float:
        """Return the largest eigenvalue B may take, which bounds u."""
        return self._max_precision

    def fit(self, S: ch.Tensor):  # pylint: disable=invalid-name
        """Fit the estimator to samples from the truncated distribution.

        Args:
            S: Observed samples of shape (num_samples, dims).

        Returns:
            self.

        Raises:
            TypeError: If S is not a tensor.
            ValueError: If S is not shaped (num_samples, dims) with more
                samples than dimensions.
        """
        if not isinstance(S, ch.Tensor):
            raise TypeError(f"S is type: {type(S)}. expected type torch.Tensor.")
        if S.dim() != 2 or S.size(1) != self.dims or S.size(0) <= self.dims:
            raise ValueError(
                f"expected samples of shape (num_samples, {self.dims}) with more "
                f"samples than dimensions; got {tuple(S.shape)}."
            )
        hermite_part, descent_part, val_part = self._split(S)
        frame = self._make_frame(ch.cat([hermite_part, descent_part]))
        basis = hermite.NormalizedProbabilistHermitePolynomial(
            self.hyperparameters.degree, self.dims
        )
        self.frame_ = frame
        self.expansion_ = hermite.HermiteExpansion.from_samples(
            basis, frame.to_frame(hermite_part)
        )
        self._resolve_projection_set()

        rows = self._make_rows(descent_part, S.dtype)
        self._rows = rows
        self.train_loader_ = data.DataLoader(
            data.TensorDataset(ch.empty(len(rows), 0), rows),
            batch_size=self.hyperparameters.batch_size,
            shuffle=self.hyperparameters.shuffle,
        )
        self.val_loader_ = None
        if len(val_part) > 0:
            val_rows = self._make_rows(val_part, S.dtype)
            self.val_loader_ = data.DataLoader(
                data.TensorDataset(ch.empty(len(val_rows), 0), val_rows),
                batch_size=len(val_rows),
            )

        emp_theta = self._initial_theta().to(S.dtype)
        self.emp_theta = emp_theta
        self.theta.data = emp_theta.clone()
        self.trainer = trainer.Trainer(self, self.hyperparameters)
        self.procedure = self.trainer.train_model(self.train_loader_, self.val_loader_)
        final = self.procedure.final
        self.best_params, self.final_params = final.best_params, final.final_params
        self.ema_params, self.avg_params = final.ema_params, final.avg_params
        return self

    def _split(self, S: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor, ch.Tensor]:  # pylint: disable=invalid-name
        """Shuffle S into the expansion's, the descent's and the validation part.

        Algorithm 1 draws its samples independently of the ones the expansion
        was estimated from; reusing them would bias ``psi_k`` upwards exactly
        at the points the descent evaluates it.
        """
        shuffled = S[ch.randperm(S.size(0))]
        num_val = int(self.hyperparameters.val * S.size(0))
        train = shuffled[num_val:]
        num_hermite = int(self.hyperparameters.hermite_fraction * train.size(0))
        return train[:num_hermite], train[num_hermite:], shuffled[:num_val]

    def _make_rows(self, x: ch.Tensor, dtype: ch.dtype) -> ch.Tensor:
        """Return the rows ``[T(z), c(z), psi_k(z)]`` for samples x.

        ``psi_k`` is clipped at zero as Theorem 4 defines it, which also keeps
        every term of the objective non-negative and so the objective convex.
        """
        frame, expansion = self._fitted()
        z = frame.to_frame(x)
        statistic, offset = self._statistic(z)
        psi = expansion(z).clamp(min=0.0)
        return ch.cat([statistic, offset[:, None], psi[:, None]], dim=1).to(dtype)

    def _resolve_projection_set(self) -> None:
        """Size the projection set D from the config, else from alpha.

        Lemma 18 puts the whitened truth's covariance within ``O(1 / alpha^2)``
        and its mean within ``O(sqrt(log(1 / alpha)))`` standard deviations of
        the origin. The constants here are explicit choices with slack: the
        set only has to contain the truth.
        """
        alpha = self.alpha if self.alpha is not None else DEFAULT_ALPHA
        log_inverse = math.log(1.0 / alpha)
        hp = self.hyperparameters
        self._min_precision = hp.min_precision or alpha**2 / 4.0
        self._max_precision = hp.max_precision or 2.0 * log_inverse + 2.0
        self._loc_radius = hp.loc_radius or math.sqrt(self._largest_precision()) * (
            2.0 * math.sqrt(2.0 * log_inverse) + 1.0
        )

    def forward(self, x):  # pylint: disable=unused-argument
        """Return the free natural parameters; the input is unused."""
        return self.theta

    def hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the Hessian of the objective over the descent's samples.

        ``h`` is affine in theta, so the Hessian is ``E[m T T^T]`` with
        ``m`` the per-sample term, positive semidefinite as Claim 1 says.

        Args:
            theta: Point to evaluate at; the start point by default.

        Returns:
            The Hessian, shaped (num_params, num_params).

        Raises:
            RuntimeError: If called before fit.
        """
        theta = self.emp_theta if theta is None else theta
        rows = self._rows
        if theta is None or rows is None:
            raise RuntimeError("the estimator has not been fit.")
        statistic = rows[:, : theta.numel()]
        terms = losses.unknown_truncation_terms(theta.detach(), rows)
        return statistic.T @ (terms[:, None] * statistic) / rows.size(0)

    def step_post_hook(self, optimizer, args, kwargs) -> None:
        """Project the iterate onto D, as Algorithm 1 does after every step."""
        with ch.no_grad():
            self.theta.copy_(self._project(self.theta))

    def _project_loc(self, u: ch.Tensor) -> ch.Tensor:
        """Return u projected onto the ball of radius ``loc_radius``."""
        norm = u.norm()
        if norm > self._loc_radius:
            return u * (self._loc_radius / norm)
        return u

    def _fitted(self) -> tuple[Frame, hermite.HermiteExpansion]:
        """Return the frame and the expansion.

        Raises:
            RuntimeError: If called before fit.
        """
        if self.frame_ is None or self.expansion_ is None:
            raise RuntimeError("the estimator has not been fit.")
        return self.frame_, self.expansion_

    def _in_frame(self, theta: Optional[ch.Tensor]) -> tuple[ch.Tensor, ch.Tensor]:
        """Return the mean and covariance theta describes, in the frame.

        Raises:
            RuntimeError: If called before fit.
        """
        if theta is None:
            raise RuntimeError("the estimator has not been fit.")
        u, precision = self._natural(theta.to(ch.float64))
        covariance_matrix = ch.linalg.inv(precision)  # pylint: disable=not-callable
        return covariance_matrix @ u, covariance_matrix

    def _canonical(self, theta: Optional[ch.Tensor]) -> tuple[ch.Tensor, ch.Tensor]:
        """Return the mean and covariance theta describes, in original coordinates.

        Raises:
            RuntimeError: If called before fit.
        """
        frame, _ = self._fitted()
        loc, covariance_matrix = frame.from_frame(*self._in_frame(theta))
        dtype = cast(ch.Tensor, theta).dtype
        symmetric = 0.5 * (covariance_matrix + covariance_matrix.T)
        return loc.to(dtype), symmetric.to(dtype)

    def truncation_set(
        self, theta: Optional[ch.Tensor] = None
    ) -> delphi.oracle.RecoveredSet:
        """Return the set estimate of Theorem 6.

        Args:
            theta: Estimate the set is built around; the averaged iterate by
                default.

        Returns:
            Membership oracle for the recovered set, in original coordinates.

        Raises:
            RuntimeError: If called before fit.
        """
        frame, expansion = self._fitted()
        loc, covariance_matrix = self._in_frame(
            self.avg_params if theta is None else theta
        )
        fitted = torch_distributions.MultivariateNormal(loc, covariance_matrix)
        return delphi.oracle.RecoveredSet(expansion, frame.to_frame, fitted)

    @property
    def best_loc_(self) -> ch.Tensor:
        """Mean estimate at the best validation loss."""
        return self._canonical(self.best_params)[0]

    @property
    def final_loc_(self) -> ch.Tensor:
        """Mean estimate at the last iterate."""
        return self._canonical(self.final_params)[0]

    @property
    def ema_loc_(self) -> ch.Tensor:
        """Mean estimate at the moving average of the iterates."""
        return self._canonical(self.ema_params)[0]

    @property
    def avg_loc_(self) -> ch.Tensor:
        """Mean estimate at the average of the iterates, Algorithm 1's output."""
        return self._canonical(self.avg_params)[0]


class UnknownTruncationMultivariateNormal(UnknownTruncationMultivariateNormalBase):
    """Mean and covariance of a normal truncated to an unknown set.

    The frame whitens the sample by its own mean and covariance, the
    estimates of Lemma 17, and the descent learns ``u`` and the upper
    triangle of the symmetric ``B``. Theorem 2 guarantees the result when the
    whitened truth is near-isotropic.
    """

    def _num_params(self) -> int:
        return self.dims * (self.dims + 1) // 2 + self.dims

    def _make_frame(self, sample: ch.Tensor) -> Frame:
        sample = sample.to(ch.float64)
        covariance_matrix = ch.cov(sample.T).reshape(self.dims, self.dims)
        return Frame(sample.mean(0), ch.linalg.cholesky(covariance_matrix))  # pylint: disable=not-callable

    def _statistic(self, z: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        # In the frame the empirical moments are (0, I), so h reduces to
        # <B, z z^T - I> / 2 - <u, z> + const, and an off-diagonal entry of
        # the upper triangle stands for both of its symmetric positions.
        rows, cols = ch.triu_indices(self.dims, self.dims)
        outer = z[:, rows] * z[:, cols] - (rows == cols).to(z.dtype)
        weight = ch.where(rows == cols, 0.5, 1.0).to(z.dtype)
        statistic = ch.cat([outer * weight, -z], dim=1)
        offset = 0.5 * (self.dims - (z**2).sum(1))
        return statistic, offset

    def _natural(self, theta: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        rows, cols = ch.triu_indices(self.dims, self.dims)
        upper = theta.new_zeros(self.dims, self.dims)
        upper[rows, cols] = theta[: rows.numel()]
        precision = upper + upper.T - ch.diag(upper.diagonal())
        return theta[rows.numel() :], precision

    def _project(self, theta: ch.Tensor) -> ch.Tensor:
        u, precision = self._natural(theta)
        values, vectors = ch.linalg.eigh(precision)  # pylint: disable=not-callable
        clipped = values.clamp(self._min_precision, self._max_precision)
        precision = vectors @ ch.diag(clipped) @ vectors.T
        rows, cols = ch.triu_indices(self.dims, self.dims)
        return ch.cat([precision[rows, cols], self._project_loc(u)])

    def _initial_theta(self) -> ch.Tensor:
        rows, cols = ch.triu_indices(self.dims, self.dims)
        return ch.cat([(rows == cols).double(), ch.zeros(self.dims, dtype=ch.float64)])

    @property
    def best_covariance_matrix_(self) -> ch.Tensor:
        """Covariance estimate at the best validation loss."""
        return self._canonical(self.best_params)[1]

    @property
    def final_covariance_matrix_(self) -> ch.Tensor:
        """Covariance estimate at the last iterate."""
        return self._canonical(self.final_params)[1]

    @property
    def ema_covariance_matrix_(self) -> ch.Tensor:
        """Covariance estimate at the moving average of the iterates."""
        return self._canonical(self.ema_params)[1]

    @property
    def avg_covariance_matrix_(self) -> ch.Tensor:
        """Covariance estimate at the average of the iterates."""
        return self._canonical(self.avg_params)[1]

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "multivariate normal distribution under unknown truncation"


class UnknownTruncationMultivariateNormalKnownCovariance(
    UnknownTruncationMultivariateNormalBase
):
    """Mean of a normal with known covariance truncated to an unknown set.

    The frame centers the sample at its mean and whitens it by the known
    covariance, so the whitened truth is ``N(mu, I)``, exactly isotropic as
    Theorem 2 asks, and ``B`` stays fixed at the identity: only ``u`` is
    learned.

    Attributes:
        covariance_matrix: The known covariance.
    """

    estimate_suffixes = ("loc_",)

    def __init__(
        self,
        hyperparameters: dict,
        dims: int,
        covariance_matrix: ch.Tensor,
        alpha: Optional[float] = None,
    ):
        """Initialize the estimator.

        Args:
            hyperparameters: Hyperparameter dict.
            dims: Number of dimensions.
            covariance_matrix: Known covariance, shaped (dims, dims).
            alpha: Survival mass of the set in (0, 1), if known.

        Raises:
            ValueError: As the base class, or if covariance_matrix is not
                shaped (dims, dims).
        """
        super().__init__(hyperparameters, dims, alpha)
        if covariance_matrix.shape != (dims, dims):
            raise ValueError(
                f"covariance_matrix must be shaped ({dims}, {dims}); got "
                f"{tuple(covariance_matrix.shape)}."
            )
        self.covariance_matrix = covariance_matrix

    def _num_params(self) -> int:
        return self.dims

    def _make_frame(self, sample: ch.Tensor) -> Frame:
        sample = sample.to(ch.float64)
        scale = ch.linalg.cholesky(self.covariance_matrix.to(ch.float64))  # pylint: disable=not-callable
        return Frame(sample.mean(0), scale)

    def _statistic(self, z: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        # With B = I, h + log N(0, I) = -<u, z>: the frame centers the sample.
        return -z, z.new_zeros(z.size(0))

    def _natural(self, theta: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
        return theta, ch.eye(self.dims, dtype=theta.dtype)

    def _largest_precision(self) -> float:
        return 1.0

    def _project(self, theta: ch.Tensor) -> ch.Tensor:
        return self._project_loc(theta)

    def _initial_theta(self) -> ch.Tensor:
        return ch.zeros(self.dims, dtype=ch.float64)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return (
            "multivariate normal distribution with known covariance under "
            "unknown truncation"
        )
