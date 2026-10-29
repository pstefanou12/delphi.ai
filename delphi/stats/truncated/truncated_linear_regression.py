# Author: pstefanou12@
"""Truncated linear regression: known variance, unknown variance, dependent data.

- ``noise_var`` given: Daskalakis, Gouleakis, Tzamos and Zampetakis (2020),
  projected SGD without replacement on the truncated NLL in D_{r,B}, then
  Newton steps to the empirical MLE.
- ``noise_var=None``: Daskalakis, Stefanou, Yao and Zampetakis (2022),
  projected SGD on the truncated NLL in the natural parameters
  ``(v, lambda) = (w / sigma^2, 1 / sigma^2)``, kept in their set D, then
  Newton steps to the empirical MLE.
- ``dependent=True``: a linear dynamical system, by the switching gradient
  with ``Sigma^-1`` preconditioning.
"""

from collections.abc import Callable
import math
from typing import Optional, Union, cast

import torch as ch

from delphi import configs
import delphi.projection_sets
from delphi.stats import linear_model
from delphi.stats.truncated import losses
from delphi.stats.truncated.truncated_linear_model import TruncatedLinearModel
from delphi.truncated import sampling
from delphi.truncated.distributions import summary
from delphi.utils import helpers


class TruncatedLinearRegression(  # pylint: disable=too-many-instance-attributes
    TruncatedLinearModel[configs.TruncatedRegressionConfig]
):
    """Truncated linear regression ``y = w^T x + b + e``, ``e ~ N(0, sigma^2)``.

    Observed only where ``phi(y)`` admits it. The variant follows the
    arguments, as an OLS package's would: a known ``noise_var``, an unknown
    one, or dependent data.

    Attributes:
        noise_var: The known noise variance, or None when it is learned.
        dependent: Whether the samples come from one trajectory.
        survival_threshold: The threshold s of dependent data; 1 otherwise.
    """

    # Created by register_buffer during fit.
    Sigma: ch.Tensor  # pylint: disable=invalid-name

    defaults = configs.TruncatedRegressionConfig

    def __init__(  # pylint: disable=too-many-arguments
        self,
        hyperparameters: dict,
        phi: Callable,
        alpha: Optional[float] = None,
        *,
        noise_var: Optional[Union[float, ch.Tensor]] = None,
        dependent: bool = False,
        sampler: Optional[sampling.GaussianSampler] = None,
        fit_intercept: bool = True,
        emp_weight: Optional[ch.Tensor] = None,
        rand_seed: int = 0,
    ):
        """Initialize TruncatedLinearRegression.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation oracle over responses shaped (n, m, k).
            alpha: Survival probability in (0, 1]; None if unknown, which
                dependent data does not support.
            noise_var: The known noise variance; None learns it, training
                ``weight`` as ``(v, lambda)``: v's rows, then lambda's.
            dependent: Whether the samples form one trajectory, x_{t+1} =
                A x_t + e; needs noise_var, its noise covariance.
            sampler: Estimates the loss's conditional moments; rejection
                sampling at ``num_samples`` and ``min_accepted`` by default.
            fit_intercept: Whether to fit an intercept.
            emp_weight: Start in place of OLS; see ``TruncatedLinearModel``.
            rand_seed: Seed applied at the start of every run.

        Raises:
            ValueError: If dependent data lacks noise_var or alpha.
        """
        if dependent and (noise_var is None or alpha is None):
            raise ValueError("dependent data needs noise_var and alpha.")
        if dependent:
            self.defaults = configs.TruncatedLinearDynamicalSystemConfig
        elif noise_var is None:
            self.defaults = configs.TruncatedUnknownVarianceRegressionConfig
        super().__init__(
            hyperparameters, phi, alpha, fit_intercept, emp_weight, rand_seed
        )
        self.noise_var = None if noise_var is None else ch.as_tensor(noise_var)
        self.dependent = dependent
        self._preconditioner: Optional[ch.Tensor] = None
        self._refining = False
        self.survival_threshold = 1.0
        self.sampler = sampler or sampling.GaussianRejectionSampler(
            self.hyperparameters.num_samples, self.hyperparameters.min_accepted
        )
        if dependent:
            lds = cast(
                configs.TruncatedLinearDynamicalSystemConfig, self.hyperparameters
            )
            known = cast(float, alpha)
            self.hyperparameters.lr = (2 / known) ** lds.c_eta
            self.survival_threshold = lds.c_s * (math.sqrt(math.log(1 / known)) + 1)
            self.criterion = losses.SwitchGrad.apply
        else:
            self.criterion = losses.TruncatedGaussianNLL.apply

    def criterion_parameters(self, X: ch.Tensor, y: ch.Tensor) -> list:  # pylint: disable=invalid-name
        hp = self.hyperparameters
        if self.dependent:
            lds = cast(configs.TruncatedLinearDynamicalSystemConfig, hp)
            return [
                self.phi,
                lds.c_gamma,
                self.alpha,
                X.size(0),
                self.noise_var,
                hp.num_samples,
                hp.eps,
            ]
        return [self.latent_phi, self.noise_var, self.sampler, hp.eps]

    def fit(self, X: ch.Tensor, y: ch.Tensor):  # pylint: disable=invalid-name
        """Fit the model, with y divided by OLS's residual sd when learning it.

        That puts the noise variance near 1 whatever its units, so lambda
        and v share one scale. The estimates are reported in y's units.
        """
        self.response_scale = 1.0
        if self.noise_var is None and not self.dependent:
            self._check_inputs(X, y)
            design = ch.cat([X, ch.ones(X.size(0), 1)], dim=1)
            residual = y - design @ ch.linalg.lstsq(design, y).solution  # pylint: disable=not-callable
            self.response_scale = residual.pow(2).mean(0).sqrt()
        return super().fit(X, y / self.response_scale)

    def latent_phi(self, z: ch.Tensor) -> ch.Tensor:
        """Return phi at latents in the descent's units, ``z * response_scale``."""
        return self.phi(z * self.response_scale)

    def register_weight(self, weight: ch.Tensor) -> None:
        if self.noise_var is None:
            precision = 1.0 / self._residual_var(weight)
            super().register_weight(ch.cat([weight * precision, precision[None]]))
            self.weight.register_hook(self.precondition)
            return
        super().register_weight(weight)
        if self.dependent:
            X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
            sigma_0 = X.T @ X / (self.survival_threshold * X.size(0))
            self.register_buffer("Sigma", sigma_0)
            # A tensor hook runs after backward, inside optimizer.step's
            # closure; a step pre-hook runs before the closure's zero_grad
            # and backward, which discard what it writes.
            self.weight.register_hook(self.precondition)

    def _residual_var(self, weight: ch.Tensor) -> ch.Tensor:
        """Return each output's mean squared residual at weight, shaped (k,)."""
        X, y = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        return (X @ weight - y).pow(2).mean(0)

    def precondition(self, grad: ch.Tensor) -> ch.Tensor:
        """Return the descent's step direction for a gradient.

        ``Sigma^-1 grad``, SON-SG's, for dependent data; for a learned
        variance, grad times the inverse of the untruncated Hessian at the
        phase's start, which the natural parameters' condition number, about
        10^3, calls for. It bounds the truncated Hessian above, so the steps
        it scales cannot overshoot, and it needs no sampling.
        """
        if self.dependent:
            return ch.linalg.solve(self.Sigma, grad)  # pylint: disable=not-callable,no-member
        if self._preconditioner is None:
            return grad
        flat = self._preconditioner @ grad.reshape(-1).to(self._preconditioner.dtype)
        return flat.reshape(grad.shape).to(grad.dtype)

    def make_optimizer_and_schedule(
        self,
        checkpoint: Optional[dict] = None,
        hessian: Optional[Callable[..., ch.Tensor]] = None,
    ):
        """Build the optimizer and schedule, preconditioned for learned variance.

        The Fisher schedule then reads the untruncated Hessian in the
        whitened coordinates, the identity at the start, so no step exceeds
        the Newton step: the truncated Hessian's smallest eigenvalue there
        shrinks with the mass phi keeps, and steps scaled by it overshoot.
        """
        self._preconditioner = None
        if self.noise_var is None and hessian is None and not self._refining:
            factor = ch.linalg.cholesky(self._untruncated_hessian())  # pylint: disable=not-callable
            self._preconditioner = ch.cholesky_inverse(factor)
            hessian = helpers.whitened_hessian(self.hessian, factor)
        return super().make_optimizer_and_schedule(checkpoint, hessian)

    def _untruncated_hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the NLL's Hessian in ``(v, lambda)`` without truncation."""
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        natural = (
            self.weight.detach() if theta is None else theta.reshape(self.weight.shape)
        )
        return losses.TruncatedGaussianNLL.natural_hessian(
            X @ natural[:-1] / natural[-1],
            X,
            self.latent_phi,
            1.0 / natural[-1],
            sampling.UntruncatedGaussianMoments(),
        )

    def _hessian_spectrum(
        self,
        current: bool = False,
        hessian: Optional[Callable[..., ch.Tensor]] = None,
    ) -> tuple[float, float]:
        """Return FisherLR's constants, the smoothness a sample's for learned variance.

        In the whitened coordinates the mean's Hessian is near the identity,
        but a sample's has trace ``weight.numel()`` on average; a first step
        of rate 1 overshoots on a batch that stiff.
        """
        mu, smoothness = super()._hessian_spectrum(current, hessian)
        if self._preconditioner is not None:
            smoothness = max(smoothness, float(self.weight.numel()))
        return mu, smoothness

    def refine(self, train_loader, rand_seed: int = 0):
        """Take the Newton steps, which carry their own curvature, unpreconditioned."""
        self._refining = True
        try:
            return super().refine(train_loader, rand_seed)
        finally:
            self._refining = False

    def compute_loss(self, batch, criterion=None) -> ch.Tensor:
        """Average the NLL, which is per sample, over the batch."""
        if self.noise_var is not None or criterion is not None:
            loss = super().compute_loss(batch, criterion)
            return loss if self.dependent else loss.mean()
        X, y = batch  # pylint: disable=invalid-name
        noise_var = 1.0 / self.weight[-1]
        return losses.TruncatedGaussianNLL.apply(
            self(X),
            y,
            self.latent_phi,
            noise_var,
            self.sampler,
            self.hyperparameters.eps,
        ).mean()

    def forward(self, X: ch.Tensor):  # pylint: disable=invalid-name
        """Return ``h_W(x)``; ``v^T x / lambda`` when lambda is learned."""
        if self.noise_var is None:
            return X @ self.weight[:-1] / self.weight[-1]
        if self.dependent:
            self.Sigma += X.T @ X / X.size(0)  # pylint: disable=invalid-name,no-member
        return X @ self.weight

    def default_radius(self) -> float:
        """Return D_{r,B}'s ``4 log(2 / alpha) + 7`` for a known variance.

        A learned variance keeps the base's ``1 / alpha``, the radius of
        ``UnknownVarianceSet``.
        """
        if self.noise_var is None or self.dependent:
            return super().default_radius()
        alpha = 1.0 if self.alpha is None else self.alpha
        return 4.0 * math.log(2.0 / alpha) + 7.0

    def build_projection_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        """Return the set the descent is kept in, anchored at emp_weight.

        D_{r,B} with its r grown until it holds emp_weight, or for learned
        variance D around emp_weight's residual variance.

        Raises:
            ValueError: If emp_weight lies outside the ball of radius
                weight_bound.
        """
        if self.dependent:
            return super().build_projection_set(X, y, radius)
        hp = self.hyperparameters
        anchor = self.emp_start
        if hp.weight_bound is not None and float(anchor.norm()) > hp.weight_bound:
            raise ValueError(
                f"emp_weight has norm {float(anchor.norm()):.4f} in the scaled "
                f"coordinates, outside weight_bound {hp.weight_bound}."
            )
        if self.noise_var is None:
            residual_var = self._residual_var(anchor)
            return UnknownVarianceSet(
                residual_var,
                radius,
                ch.cat([anchor / residual_var, 1.0 / residual_var[None]]),
                hp.weight_bound,
            )
        variance = float(linear_model.require_fitted(self.noise_var, "noise_var"))
        projection = ResidualSpectralSet(
            X,
            y,
            variance,
            radius,
            anchor,
            hp.weight_bound,
            bisection_steps=hp.bisection_steps,
            max_quadratic_entries=hp.max_quadratic_entries,
        )
        while not projection.contains(anchor):
            projection.radius = cast(float, projection.radius) * hp.rate
            self.logger.warning(
                f"emp_weight lies outside D_{{r,B}}; r -> {projection.radius:.4f}."
            )
        return projection

    def hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the Hessian of the NLL, by default at the current weight."""
        if self.dependent:
            return super().hessian(theta)
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        weight = self.weight.detach() if theta is None else theta
        weight = weight.reshape(-1, linear_model.require_fitted(self.k, "k"))
        if self.noise_var is None:
            return losses.TruncatedGaussianNLL.natural_hessian(
                X @ weight[:-1] / weight[-1],
                X,
                self.latent_phi,
                1.0 / weight[-1],
                self.sampler,
            )
        return losses.TruncatedGaussianNLL.hessian(
            X @ weight,
            X,
            self.latent_phi,
            self.noise_var,
            self.sampler,
        )

    def coefficients(self, params):
        """Split a flat iterate; for learned variance, ``(v, lambda)``."""
        if params is not None and self.noise_var is None:
            natural = params.reshape(-1, linear_model.require_fitted(self.k, "k"))
            params = (natural[:-1] / natural[-1]).flatten()
        return super().coefficients(params)

    def _variance(self, name: str) -> Optional[ch.Tensor]:
        """Return the noise variance at one of the estimates."""
        if self.noise_var is not None:
            return self.noise_var
        params = self._estimate(name)
        if params is None:
            return None
        precision = params.reshape(-1, linear_model.require_fitted(self.k, "k"))[-1]
        return ch.as_tensor(self.response_scale).pow(2) / precision

    @property
    def avg_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the averaged iterate."""
        return self._variance("avg")

    @property
    def ema_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the moving average of the iterates."""
        return self._variance("ema")

    @property
    def final_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the last iterate."""
        return self._variance("final")

    @property
    def best_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the iterate with the lowest validation loss."""
        return self._variance("best")

    @property
    def variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the averaged iterate."""
        return self.avg_variance_

    def _summary_estimates(self) -> list[tuple[str, str]]:
        rows = super()._summary_estimates()
        if self.noise_var is None:
            rows += [
                (f"{name}_variance_", summary.tensor(self._variance(name)))
                for name in linear_model.ESTIMATORS
            ]
        return rows

    def __str__(self):
        """Return a human-readable name for this model."""
        if self.dependent:
            return "truncated linear dynamical system"
        if self.noise_var is None:
            return "truncated linear regression, unknown variance"
        return "truncated linear regression, known variance"


class ResidualSpectralSet(delphi.projection_sets.ProjectionSet):
    """The projection set D_{r,B} of a known noise variance.

    ``{w : sum_i ((y_i - w^T x_i) / sigma)^2 x_i x_i^T <= r sum_i x_i x_i^T,
    ||w|| <= B}``, in the Loewner order. In coordinates where
    ``sum_i x_i x_i^T`` is the identity the first constraint bounds the
    largest eigenvalue of ``sum_i rho_i^2 u_i u_i^T`` by r. ``rho_i`` is
    linear in ``v = (-w, 1)``, so that matrix is ``sum_pq v_p v_q Q_pq`` for
    a tensor Q summed over the data once, and a membership test costs
    O(d^4) instead of O(n d^2) whenever d^2 is below n and Q fits in memory.

    Attributes:
        radius: The set's r, bounding the residual-weighted covariance
            relative to the covariance.
        bound: Bound B on the weight's norm, or None for no ball.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        noise_var: float,
        radius: float,
        anchor: ch.Tensor,
        bound: Optional[float] = None,
        bisection_steps: int = 40,
        max_quadratic_entries: int = 2**24,
    ):
        """Build D_{r,B} from the training data.

        Args:
            X: Design matrix, shaped (n, d).
            y: Responses, shaped (n, 1).
            noise_var: Known noise variance sigma^2.
            radius: The set's r.
            anchor: Weight of shape (d, 1) to retract towards.
            bound: The set's B, or None for no ball.
            bisection_steps: As ``ProjectionSet``.
            max_quadratic_entries: Largest precomputed quadratic form, in
                float64 entries; 2^24 is 128 MB. A larger one is skipped for
                the O(n d^2) test.
        """
        super().__init__(anchor, radius, bisection_steps)
        self.bound = bound
        scale = math.sqrt(noise_var)
        chol = ch.linalg.cholesky(X.T @ X)  # pylint: disable=not-callable
        whitened = ch.linalg.solve_triangular(chol, X.T, upper=False).T  # pylint: disable=not-callable
        joint = ch.cat([-X, y], dim=1) / scale
        self._direct: Optional[tuple[ch.Tensor, ch.Tensor]] = None
        self._quadratic: Optional[ch.Tensor] = None
        entries = ((X.size(1) + 1) * X.size(1)) ** 2
        if (X.size(1) + 1) ** 2 <= X.size(0) and entries <= max_quadratic_entries:
            whitened, joint = whitened.double(), joint.double()
            self._quadratic = ch.einsum(
                "np,nq,na,nb->pqab", joint, joint, whitened, whitened
            )
        else:
            self._direct = (joint, whitened)

    def contains(self, weight: ch.Tensor) -> bool:
        if self.bound is not None and float(weight.norm()) > self.bound:
            return False
        v = ch.cat([weight.flatten(), ch.ones(1)])
        if self._quadratic is not None:
            v = v.double()
            weighted = ch.einsum("p,q,pqab->ab", v, v, self._quadratic)
        else:
            joint, whitened = cast(tuple[ch.Tensor, ch.Tensor], self._direct)
            weighted = (whitened * (joint @ v)[:, None].pow(2)).T @ whitened
        return float(ch.linalg.eigvalsh(weighted)[-1]) <= cast(float, self.radius)  # pylint: disable=not-callable


class UnknownVarianceSet(delphi.projection_sets.ProjectionSet):
    """The projection set D of a learned noise variance.

    It bounds the noise variance relative to the start's residual variance
    ``sigma_0^2``, and the weight by B, which in ``(v, lambda) =
    (w / sigma^2, 1 / sigma^2)`` is the convex set
    ``{a^2 / (96 sigma_0^2) <= lambda <= 8 (5 + 2 log(1 / a)) / sigma_0^2,
    ||v|| <= B lambda}``. Its radius is ``1 / a``, for a the survival mass.
    Each output's column ``(v, lambda)`` lies in its own copy, and the
    projection is Euclidean.

    Attributes:
        residual_var: Each output's ``sigma_0^2``, shaped (k,).
        radius: The set's ``1 / a``, at least 1.
        bound: Bound B on the weight's norm, or None for no ball.
    """

    def __init__(
        self,
        residual_var: ch.Tensor,
        radius: float,
        anchor: ch.Tensor,
        bound: Optional[float] = None,
    ):
        """Build D around the start's residual variance.

        Args:
            residual_var: Each output's sigma_0^2, shaped (k,).
            radius: The set's 1 / a.
            anchor: The start ``(v, lambda)``, shaped (d + 1, k) with lambda
                as the last row.
            bound: The set's B, or None for no ball.

        Raises:
            ValueError: If radius is below 1.
        """
        if radius < 1.0:
            raise ValueError(f"radius is 1 / alpha, at least 1; got {radius}.")
        super().__init__(anchor, radius)
        self.residual_var = residual_var
        self.bound = bound

    def precision_bounds(self) -> tuple[ch.Tensor, ch.Tensor]:
        """Return the lower and upper bounds on lambda, each shaped (k,)."""
        radius = cast(float, self.radius)
        lower = 1.0 / (96.0 * radius**2 * self.residual_var)
        upper = 8.0 * (5.0 + 2.0 * math.log(radius)) / self.residual_var
        return lower, upper

    def contains(self, weight: ch.Tensor) -> bool:
        lower, upper = self.precision_bounds()
        precision = weight[-1]
        if bool((precision < lower).any() or (precision > upper).any()):
            return False
        if self.bound is None:
            return True
        return bool((weight[:-1].norm(dim=0) <= self.bound * precision).all())

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        """Return the nearest point of the set to weight."""
        if self.contains(weight):
            return weight
        lower, upper = self.precision_bounds()
        precision = ch.minimum(ch.maximum(weight[-1], lower), upper)
        if self.bound is None:
            return ch.cat([weight[:-1], precision[None]])
        projected = [
            self._project_column(weight[:, a], float(lower[a]), float(upper[a]))
            for a in range(weight.size(1))
        ]
        return ch.stack(projected, dim=1)

    def _project_column(
        self, column: ch.Tensor, lower: float, upper: float
    ) -> ch.Tensor:
        """Return the nearest point to one output's ``(v, lambda)``.

        The nearest point keeps v's direction, so the problem is planar in
        ``(||v||, lambda)``, over the trapezoid with corners ``(0, lower)``,
        ``(B lower, lower)``, ``(B upper, upper)`` and ``(0, upper)``; a
        point outside it is nearest to one of its edges.
        """
        bound = cast(float, self.bound)
        norm, precision = float(column[:-1].norm()), float(column[-1])
        if lower <= precision <= upper and norm <= bound * precision:
            return column
        point = ch.tensor([norm, precision], dtype=ch.float64)
        corners = ch.tensor(
            [
                [0.0, lower],
                [bound * lower, lower],
                [bound * upper, upper],
                [0.0, upper],
            ],
            dtype=ch.float64,
        )
        candidates = []
        for start, end in zip(corners, corners.roll(-1, 0)):
            edge = end - start
            share = float((point - start) @ edge / (edge @ edge))
            candidates.append(start + min(max(share, 0.0), 1.0) * edge)
        nearest = min(
            candidates, key=lambda candidate: float((candidate - point).norm())
        )
        direction = column[:-1] / norm if norm > 0.0 else ch.zeros_like(column[:-1])
        return ch.cat(
            [float(nearest[0]) * direction, column.new_tensor([float(nearest[1])])]
        )
