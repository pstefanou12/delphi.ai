# Author: pstefanou12@
"""Truncated lasso regression, for sparse coefficients in high dimensions.

Daskalakis, Rohatgi and Zampetakis, "Truncated Linear Regression in High
Dimensions" (NeurIPS 2020): with the noise variance known, minimize the
truncated NLL plus ``l1 ||w||_1``, for ``l1 = c sqrt(log(d) / n)``, by
projected SGD (their Algorithm 1). The iterates stay in
``E_r = {w : ||X w - y||_2 <= r sqrt(n)}``, start at the smallest-L1 point
of E_r, take the steps ``x_j (z - y_j) + l1 sign(w)`` for a sample j and
``z ~ N(w^T x_j, 1)`` conditioned on the truncation set, of constant size
``1 / sqrt(d N)`` for N steps, and are averaged. It needs only
``n = O(k log d)`` samples for k nonzero coefficients, so n may be below d.
"""

from collections.abc import Callable
import math
from typing import Optional, Union, cast

import torch as ch

from delphi.configs.stats.truncated import regression
from delphi.learning import lasso
import delphi.projection_sets
from delphi.stats.truncated import losses
from delphi.stats.truncated.truncated_linear_model import TruncatedLinearModel
from delphi.truncated import sampling


class TruncatedLassoRegression(
    TruncatedLinearModel[regression.TruncatedLassoRegressionConfig]
):
    """Truncated lasso regression ``y = w^T x + b + e``, ``e ~ N(0, sigma^2)``.

    Observed only where ``phi(y)`` admits it, with sigma^2 known. The
    intercept is not penalized.

    Attributes:
        noise_var: The known noise variance.
        l1: The penalty, on the coefficients of ``y / sqrt(noise_var)``;
            None until fit when left to the default.
        intercept_column: The design matrix's intercept column, the rows'
            root mean square norm rather than 1, so the one step size suits
            the intercept as well as the coefficients.
    """

    defaults = regression.TruncatedLassoRegressionConfig

    def __init__(  # pylint: disable=too-many-arguments
        self,
        phi: Callable,
        alpha: Optional[float] = None,
        *,
        l1: Optional[float] = None,
        noise_var: Union[float, ch.Tensor] = 1.0,
        sampler: Optional[sampling.GaussianSampler] = None,
        fit_intercept: bool = True,
        emp_weight: Optional[ch.Tensor] = None,
        rand_seed: int = 0,
        hyperparameters: Optional[dict] = None,
    ):
        """Initialize TruncatedLassoRegression.

        Args:
            phi: Truncation oracle over responses shaped (n, m, 1).
            alpha: Survival probability in (0, 1], or None if unknown.
            l1: The penalty on the coefficients of ``y / sqrt(noise_var)``;
                ``l1_constant sqrt(log(d) / n)`` if None.
            noise_var: The known noise variance.
            sampler: Estimates the loss's conditional moments; rejection
                sampling at ``num_samples`` and ``min_accepted`` by default.
            fit_intercept: Whether to fit an unpenalized intercept.
            emp_weight: Start in place of the smallest-L1 point of E_r; see
                ``TruncatedLinearModel``.
            rand_seed: Seed applied at the start of every run.
            hyperparameters: Hyperparameter dict.

        Raises:
            ValueError: If l1 or noise_var is not positive.
        """
        if l1 is not None and l1 <= 0.0:
            raise ValueError(f"l1 must be positive; got {l1}.")
        if float(noise_var) <= 0.0:
            raise ValueError(f"noise_var must be positive; got {noise_var}.")
        super().__init__(
            phi,
            alpha,
            fit_intercept,
            emp_weight,
            rand_seed,
            hyperparameters=hyperparameters,
        )
        self.noise_var = float(noise_var)
        self.l1 = l1
        self._l1_given = l1 is not None
        self.intercept_column = 1.0
        self._lr_given = "lr" in self.hyperparameters.model_fields_set
        self.sampler = sampler or sampling.GaussianRejectionSampler(
            self.hyperparameters.num_samples, self.hyperparameters.min_accepted
        )
        self.criterion = losses.TruncatedGaussianNLL.apply

    def criterion_parameters(self, X: ch.Tensor, y: ch.Tensor) -> list:  # pylint: disable=invalid-name
        return [self.latent_phi, 1.0, self.sampler, self.hyperparameters.eps]

    def fit(self, X: ch.Tensor, y: ch.Tensor):  # pylint: disable=invalid-name
        """Fit the model to y divided by the noise's standard deviation.

        Args:
            X: Covariates, shaped (n, d); n may be below d.
            y: Responses, shaped (n, 1).

        Returns:
            The fitted model.
        """
        self._check_inputs(X, y)
        n, d = X.shape
        if not self._l1_given:
            self.l1 = self.hyperparameters.l1_constant * math.sqrt(math.log(d) / n)
        self.response_scale = math.sqrt(self.noise_var)
        self.intercept_column = float(X.pow(2).sum(1).mean().sqrt())
        return super().fit(X, y / self.response_scale)

    def _default_step(self, design: ch.Tensor) -> float:
        """Return ``step_constant / (mean_i ||x_i||^2 sqrt(N / p))``.

        Args:
            design: The design matrix in the descent's coordinates.
        """
        hp = self.hyperparameters
        steps = hp.iterations
        if steps is None:
            n = design.size(0)
            batch = n if hp.batch_size == -1 else hp.batch_size
            steps = cast(int, hp.epochs) * math.ceil(n / batch)
        energy = float(design.pow(2).sum(1).mean())
        return hp.step_constant / (energy * math.sqrt(steps / design.size(1)))

    @staticmethod
    def _check_inputs(X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Raise unless X and y are tensors of paired samples, one response each."""
        if not isinstance(X, ch.Tensor) or not isinstance(y, ch.Tensor):
            raise TypeError(f"X and y must be tensors; got {type(X)} and {type(y)}.")
        if X.dim() != 2 or y.dim() != 2 or X.size(0) != y.size(0) or y.size(1) != 1:
            raise ValueError(
                "expected X shaped (n, d) and y shaped (n, 1); got "
                f"{tuple(X.size())} and {tuple(y.size())}."
            )

    def latent_phi(self, z: ch.Tensor) -> ch.Tensor:
        """Return phi at latents in the descent's units, ``z * response_scale``."""
        return self.phi(z * self.response_scale)

    def _calc_emp_model(self, X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Set emp_weight to the smallest-L1 point of E_r, and lr, unless given.

        Args:
            X: Design matrix in the descent's coordinates.
            y: Targets, shaped (n, 1).
        """
        if not self._lr_given:
            self.hyperparameters.lr = self._default_step(X)
        if self.emp_weight is None:
            hp = self.hyperparameters
            radius = self.default_radius() if hp.radius is None else hp.radius
            covariates = X[:, :-1] if self.fit_intercept else X
            start = lasso.Lasso(fit_intercept=self.fit_intercept).fit_constrained(
                covariates, y[:, 0], radius * math.sqrt(X.size(0))
            )
            weight = cast(ch.Tensor, start.coef_)[:, None] / self.covariate_scale
            if start.intercept_ is not None:
                weight = ch.cat([weight, start.intercept_.reshape(1, 1)])
            self.emp_weight = weight * self.response_scale
        self.register_weight(self.emp_start)

    def design_matrix(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return the scaled covariates and, if fit, the intercept column."""
        if self.fit_intercept:
            column = ch.full((X.size(0), 1), self.intercept_column, dtype=X.dtype)
            X = ch.cat([X, column], dim=1)  # pylint: disable=invalid-name
        return X / self.covariate_scale

    def coefficients(self, params):
        coef, intercept = super().coefficients(params)
        if intercept is None:
            return coef, intercept
        return coef, intercept * self.intercept_column

    @property
    def emp_start(self) -> ch.Tensor:
        """emp_weight in the descent's coordinates, shaped (d + 1, 1)."""
        start = super().emp_start
        if not self.fit_intercept:
            return start
        return ch.cat([start[:-1], start[-1:] / self.intercept_column])

    @property
    def emp_intercept_(self) -> Optional[ch.Tensor]:
        """Intercept of emp_weight, the descent's start."""
        return self.coefficients(self.emp_start.flatten())[1]

    def default_radius(self) -> float:
        """Return ``1 + 2 / sqrt(alpha)``, taking alpha = 1 when it is unknown.

        The truth's residual norm is below ``2 sqrt(n / alpha)`` (their Lemma
        B.2), and the optimum lies within O(1) of it.
        """
        alpha = 1.0 if self.alpha is None else self.alpha
        return 1.0 + 2.0 / math.sqrt(alpha)

    def build_projection_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        """Return E_r around the training data."""
        return ResidualBall(X, y, radius, self.emp_start)

    def compute_loss(self, batch, criterion=None) -> ch.Tensor:
        """Return the batch's mean NLL plus the penalty on the coefficients."""
        nll = super().compute_loss(batch, criterion).mean()
        coef = self.weight[:-1] if self.fit_intercept else self.weight
        return nll + cast(float, self.l1) * coef.abs().sum() / self.covariate_scale

    def _hessian(self, X: ch.Tensor, weight: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return the NLL's mean Hessian over X's rows; the penalty has none."""
        return losses.TruncatedGaussianNLL.hessian(
            X @ weight, X, self.latent_phi, 1.0, self.sampler
        )

    def _summary_procedure(self) -> list[tuple[str, str]]:
        return [("l1 penalty", f"{cast(float, self.l1):.4f}")] + (
            super()._summary_procedure()
        )

    def __str__(self):
        """Return a human-readable name for this model."""
        return "truncated lasso regression, known variance"


class ResidualBall(delphi.projection_sets.ProjectionSet):
    """The set ``E_r = {w : ||X w - y||_2 <= r sqrt(n)}``.

    A projection returns the Euclidean nearest point, ``(I + mu X^T X)^-1
    (w + mu X^T y)`` for the mu that lands it on the boundary: with
    ``X = U diag(s) V^T``, the residual there is
    ``||(s V^T w - U^T y) / (1 + mu s^2)||^2`` plus the part of y outside
    U's span, which is decreasing in mu, so mu is bisected.

    Attributes:
        radius: The set's r.
    """

    # Slack on the boundary, so float32 rounding of a projected point does
    # not leave it outside.
    _SLACK = 1e-5

    def __init__(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        radius: float,
        anchor: ch.Tensor,
        bisection_steps: int = 100,
    ):
        """Build E_r from the training data.

        Args:
            X: Design matrix, shaped (n, p).
            y: Responses, shaped (n, 1).
            radius: The set's r.
            anchor: A weight in the set, shaped (p, 1).
            bisection_steps: Halvings of the interval mu is searched in.
        """
        super().__init__(anchor, radius, bisection_steps)
        self._X, self._y = X.double(), y.double()  # pylint: disable=invalid-name
        u, self._spectrum, self._basis = ch.linalg.svd(self._X, full_matrices=False)  # pylint: disable=not-callable
        kept = self._spectrum > self._spectrum.max() * 1e-12
        u, self._spectrum = u[:, kept], self._spectrum[kept]
        self._basis = self._basis[kept]
        self._projected_y = u.T @ self._y
        self._outside = float((self._y.pow(2).sum() - self._projected_y.pow(2).sum()))

    def _bound(self) -> float:
        return cast(float, self.radius) * math.sqrt(self._X.size(0))

    def contains(self, weight: ch.Tensor) -> bool:
        residual = float((self._X @ weight.detach().double() - self._y).norm())
        return residual <= self._bound() * (1.0 + self._SLACK)

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        """Return the Euclidean nearest point of the set to weight.

        Raises:
            ValueError: If the set is empty, as when y lies further than
                the bound from X's column span.
        """
        if self.contains(weight):
            return weight
        bound = self._bound()
        if self._outside > bound**2:
            raise ValueError(
                f"E_r is empty: no weight has a residual norm within {bound:.4f}."
            )
        point = weight.detach().double()
        offset = self._spectrum[:, None] * (self._basis @ point) - self._projected_y

        def residual(mu: float) -> float:
            shrunk = offset / (1.0 + mu * self._spectrum[:, None].pow(2))
            return float(shrunk.pow(2).sum()) + max(self._outside, 0.0)

        low, high = 0.0, 1.0
        while residual(high) > bound**2:
            high *= 2.0
        for _ in range(self.bisection_steps):
            middle = (low + high) / 2.0
            if residual(middle) > bound**2:
                low = middle
            else:
                high = middle
        spectrum = self._spectrum[:, None]
        step = -high * spectrum * offset / (1.0 + high * spectrum.pow(2))
        return (point + self._basis.T @ step).to(weight.dtype)
