# Author: pstefanou12@
"""Truncated linear regression: known variance, unknown variance, dependent data.

- ``noise_var`` given: Daskalakis, Gouleakis, Tzamos and Zampetakis (2020),
  projected SGD without replacement on the truncated NLL in D_{r,B}, then
  Newton steps to the empirical MLE. ``confidence_intervals`` are Wald
  intervals at the inverse of the NLL's Hessian in w,
  ``mean_i Var[z_i | S] x_i x_i^T / sigma^4``.
- ``noise_var=None``: Daskalakis, Stefanou, Yao and Zampetakis (2022),
  projected SGD on the truncated NLL in the natural parameters
  ``(v, lambda) = (w / sigma^2, 1 / sigma^2)``, kept in their set D, whose
  tail average converges to the empirical MLE. ``confidence_intervals`` are
  Wald intervals at the inverse of their Hessian (4.3), mapped to
  ``(w, sigma^2)`` by the delta method of their Section 5. Their Theorem 5.1
  covariance, a Lyapunov solution, is single-pass SGD's last iterate's at
  steps ``1 / (zeta t)``; the averaged iterate here sits at the empirical
  MLE, whose covariance is the inverse information.
- ``dependent=True``: Plevrakis, "Learning from Censored and Dependent
  Data: The case of Linear Dynamics" (arXiv:2104.05087), a linear dynamical
  system ``x_{t+1} = A x_t + w_t``, with a drift b as the intercept, seen
  through the pairs of consecutive states both observed, in temporal
  order. Algorithm 4: least squares on the leading pairs, a quarter by
  default where the paper takes half (the ``warm_start_fraction``
  hyperparameter), gives the start ``A_0`` and the confidence ellipsoid
  ``||A - A_0||_{Sigma_0} <= 1``, for ``Sigma_0`` their covariance divided
  by s. SON-SG (Algorithm 1) then takes one step per remaining pair,
  ``A - eta g Sigma_i^-1`` with ``Sigma_i = Sigma_{i-1} + x x^T``,
  projected onto the ellipsoid in the ``Sigma_i`` norm; g is SwitchGrad's
  (Algorithms 2 and 3). With an intercept, x carries a trailing 1 and A
  the column b. The paper's noise is N(0, I); a known covariance
  ``L L^T`` is reduced to it by fitting ``L^-1 x_{t+1}``.
"""

from collections.abc import Callable
import dataclasses
import math
from typing import Optional, Union, cast

import torch as ch
import torch.autograd.functional

from delphi.configs.stats.truncated import regression
import delphi.projection_sets
from delphi.stats import linear_model
from delphi.stats.truncated import losses
from delphi.stats.truncated.truncated_linear_model import TruncatedLinearModel
from delphi.truncated import sampling
from delphi.truncated.distributions import summary
from delphi.utils import constants, helpers


@dataclasses.dataclass(frozen=True)
class Interval:
    """Elementwise bounds on an estimate.

    Attributes:
        lower: Lower bounds, shaped as the estimate.
        upper: Upper bounds, shaped as the estimate.
    """

    lower: ch.Tensor
    upper: ch.Tensor


@dataclasses.dataclass(frozen=True)
class ConfidenceIntervals:
    """Wald intervals for a truncated regression's estimates.

    Attributes:
        coef: For ``coef_``, shaped (k, d).
        intercept: For ``intercept_``, shaped (k,); None without one.
        variance: For ``variance_``, shaped (k,); None when it is known.
    """

    coef: Interval
    intercept: Optional[Interval] = None
    variance: Optional[Interval] = None

    def items(self) -> list[tuple[str, Interval]]:
        """Return each present interval with its field's name."""
        named = [
            (field.name, getattr(self, field.name))
            for field in dataclasses.fields(self)
        ]
        return [(name, interval) for name, interval in named if interval is not None]


class TruncatedLinearRegression(  # pylint: disable=too-many-instance-attributes
    TruncatedLinearModel[regression.TruncatedRegressionConfig]
):
    """Truncated linear regression ``y = w^T x + b + e``, ``e ~ N(0, sigma^2)``.

    Observed only where ``phi(y)`` admits it. The variant follows the
    arguments, as an OLS package's would: a known ``noise_var``, an unknown
    one, or dependent data.

    Attributes:
        noise_var: The known noise variance, or None when it is learned.
        dependent: Whether the samples come from one trajectory.
        survival_threshold: The s dividing dependent data's ``Sigma_0``; 1
            otherwise.
    """

    # Created by register_buffer during fit; SON-SG's preconditioner.
    Sigma: ch.Tensor  # pylint: disable=invalid-name

    defaults = regression.TruncatedRegressionConfig

    def __init__(  # pylint: disable=too-many-arguments
        self,
        phi: Callable,
        alpha: Optional[float] = None,
        *,
        hyperparameters: Optional[dict] = None,
        noise_var: Optional[Union[float, ch.Tensor]] = None,
        dependent: bool = False,
        sampler: Optional[sampling.GaussianSampler] = None,
        fit_intercept: bool = True,
        emp_weight: Optional[ch.Tensor] = None,
        rand_seed: int = 0,
    ):
        """Initialize TruncatedLinearRegression.

        Args:
            phi: Truncation oracle over responses shaped (n, m, k).
            alpha: Survival probability in (0, 1]; None if unknown, which
                dependent data does not support.
            noise_var: The known noise variance; None learns it, training
                ``weight`` as ``(v, lambda)``: v's rows, then lambda's.
            dependent: Whether each row of X is a state x_t and its row of y
                the next state x_{t+1}, both observed, the rows in temporal
                order; needs noise_var, the noise covariance, and alpha, a
                lower bound on ``P(x_{t+1} in S | x_t)``.
            sampler: Estimates the loss's conditional moments; rejection
                sampling at ``num_samples`` and ``min_accepted`` by default.
                Unused for dependent data, whose test draws a fixed number.
            fit_intercept: Whether to fit an intercept.
            emp_weight: Start in place of OLS; see ``TruncatedLinearModel``.
            rand_seed: Seed applied at the start of every run.
            hyperparameters: Hyperparameter dict.

        Raises:
            ValueError: If dependent data lacks noise_var or alpha.
        """
        if dependent and (noise_var is None or alpha is None):
            raise ValueError("dependent data needs noise_var and alpha.")
        if dependent:
            self.defaults = regression.TruncatedLinearDynamicalSystemConfig
        elif noise_var is None:
            self.defaults = regression.TruncatedUnknownVarianceRegressionConfig
        super().__init__(
            phi,
            alpha,
            fit_intercept,
            emp_weight,
            rand_seed,
            hyperparameters=hyperparameters,
        )
        self.noise_var = None if noise_var is None else ch.as_tensor(noise_var)
        self.dependent = dependent
        self._preconditioner: Optional[ch.Tensor] = None
        self._refining = False
        self.survival_threshold = 1.0
        self._noise_factor: Optional[ch.Tensor] = None
        self._warmup_size = 0
        self._aware_pairs = 0
        self._oblivious_pairs = 0
        self.sampler = sampler or sampling.GaussianRejectionSampler(
            self.hyperparameters.num_samples, self.hyperparameters.min_accepted
        )
        self._untruncated_moments = sampling.UntruncatedGaussianMoments()
        draws = self.hyperparameters.information_samples
        self._information_sampler = sampling.GaussianRejectionSampler(draws, draws)
        if dependent:
            lds = cast(
                regression.TruncatedLinearDynamicalSystemConfig, self.hyperparameters
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
            lds = cast(regression.TruncatedLinearDynamicalSystemConfig, hp)
            gamma = (cast(float, self.alpha) / 2.0) ** lds.c_gamma
            # Enough draws that each of the n tests errs with probability 1 / n^2.
            draws = max(math.ceil(4.0 / gamma * math.log(X.size(0))), 1)
            test = sampling.GaussianRejectionSampler(draws, min_accepted=0)
            return [self.latent_phi, gamma, test]
        return [self.latent_phi, self.noise_var, self.sampler, hp.eps]

    def fit(self, X: ch.Tensor, y: ch.Tensor):  # pylint: disable=invalid-name
        """Fit the model, with y divided by OLS's residual sd when learning it.

        That puts the noise variance near 1 whatever its units, so lambda
        and v share one scale. The estimates are reported on the scale of
        the y given. A learned variance runs in float64: with covariates on
        scales far apart, float32 rounds away the late steps of the scaled
        intercept. Dependent data fits the responses whitened by the noise
        covariance instead.

        Raises:
            ValueError: If dependent data's warm-up has no more pairs
                than the design matrix has columns.
        """
        self.response_scale = 1.0
        if self.dependent:
            self._check_inputs(X, y)
            lds = cast(
                regression.TruncatedLinearDynamicalSystemConfig, self.hyperparameters
            )
            self._warmup_size = int(lds.warm_start_fraction * X.size(0))
            if self._warmup_size <= X.size(1) + int(self.fit_intercept):
                raise ValueError(
                    f"the warm-up's {self._warmup_size} pairs cannot fit "
                    f"{X.size(1)} covariates; pass more pairs."
                )
            self._noise_factor = self._cholesky_noise(y.size(1))
            whitened = ch.linalg.solve_triangular(  # pylint: disable=not-callable
                self._noise_factor, y.T, upper=False
            ).T
            return super().fit(X, whitened)
        if self.noise_var is None:
            self._check_inputs(X, y)
            X, y = X.double(), y.double()  # pylint: disable=invalid-name
            design = ch.cat([X, ch.ones(X.size(0), 1)], dim=1)
            residual = y - design @ ch.linalg.lstsq(design, y).solution  # pylint: disable=not-callable
            self.response_scale = residual.pow(2).mean(0).sqrt()
        return super().fit(X, y / self.response_scale)

    def _cholesky_noise(self, k: int) -> ch.Tensor:
        """Return L, shaped (k, k), with ``L L^T`` the noise covariance."""
        variance = ch.as_tensor(
            linear_model.require_fitted(self.noise_var, "noise_var"),
            dtype=ch.float32,
        )
        if variance.dim() < 2:
            return ch.diag(variance.sqrt().expand(k))
        return ch.linalg.cholesky(variance)  # pylint: disable=not-callable

    def latent_phi(self, z: ch.Tensor) -> ch.Tensor:
        """Return phi at latents in the descent's units.

        ``z * response_scale``, or ``L z`` for dependent data.
        """
        if self._noise_factor is not None:
            return self.phi(z @ self._noise_factor.T)
        return self.phi(z * self.response_scale)

    def _calc_emp_model(self, X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Set emp_weight to least squares on dependent data's warm-up pairs."""
        if self.dependent and self.emp_weight is None:
            warmup = self._warmup_size
            ols = ch.linalg.lstsq(X[:warmup], y[:warmup]).solution  # pylint: disable=not-callable
            factor = linear_model.require_fitted(self._noise_factor, "noise")
            self.emp_weight = ols @ factor.T / self.covariate_scale
        super()._calc_emp_model(X, y)

    def _loaders(self, X: ch.Tensor, y: ch.Tensor, config):  # pylint: disable=invalid-name
        """Return the loaders; dependent data's descent skips the warm-up."""
        if self.dependent:
            X, y = X[self._warmup_size :], y[self._warmup_size :]  # pylint: disable=invalid-name
        return super()._loaders(X, y, config)

    @property
    def emp_start(self) -> ch.Tensor:
        """emp_weight in the descent's coordinates, shaped (d + 1, k).

        With a known noise covariance ``L L^T``, dependent data fits the
        whitened ``L^-1 y``, so the start is ``emp_weight L^-T``.
        """
        start = super().emp_start
        if self._noise_factor is None:
            return start
        return ch.linalg.solve_triangular(  # pylint: disable=not-callable
            self._noise_factor, start.T, upper=False
        ).T

    @property
    def emp_coef_(self) -> Optional[ch.Tensor]:
        """Coefficients of emp_weight; dependent data's undo the whitening."""
        if not self.dependent:
            return super().emp_coef_
        return self.coefficients(self.emp_start.flatten())[0]

    @property
    def emp_intercept_(self) -> Optional[ch.Tensor]:
        """Intercept of emp_weight, the descent's start."""
        if not self.dependent:
            return super().emp_intercept_
        return self.coefficients(self.emp_start.flatten())[1]

    def register_weight(self, weight: ch.Tensor) -> None:
        if self.noise_var is None:
            precision = 1.0 / self._residual_var(weight)
            super().register_weight(ch.cat([weight * precision, precision[None]]))
            self.weight.register_hook(self.precondition)
            return
        super().register_weight(weight)
        if self.dependent:
            self.register_buffer("Sigma", self._sigma_0())
            # A tensor hook runs after backward, inside optimizer.step's
            # closure; a step pre-hook runs before the closure's zero_grad
            # and backward, which discard what it writes.
            self.weight.register_hook(self.precondition)

    def _sigma_0(self) -> ch.Tensor:
        """Return dependent data's ``Sigma_0``, the warm-up's covariance over s."""
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        warmup = X[: self._warmup_size]
        return warmup.T @ warmup / (self.survival_threshold * warmup.size(0))

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
        it scales cannot overshoot, and it needs no sampling. Lambda's row
        is then scaled by ``variance_lr_scale``, its own learning rate.
        """
        if self.dependent:
            return ch.linalg.solve(self.Sigma, grad)  # pylint: disable=not-callable,no-member
        if self._preconditioner is None:
            return grad
        flat = self._preconditioner @ grad.reshape(-1).to(self._preconditioner.dtype)
        step = flat.reshape(grad.shape).to(grad.dtype)
        hp = cast(
            regression.TruncatedUnknownVarianceRegressionConfig, self.hyperparameters
        )
        scale = ch.ones_like(step)
        scale[-1] = hp.variance_lr_scale
        return step * scale

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
        return self._hessian(X, natural, self._untruncated_moments)

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
        """Reduce the per-sample NLL: its mean, or its sum for dependent data."""
        if self.dependent:
            return self._switching_loss(batch)
        if self.noise_var is not None or criterion is not None:
            return super().compute_loss(batch, criterion).mean()
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
        return X @ self.weight

    def _switching_loss(self, batch) -> ch.Tensor:
        """Return SwitchGrad's summed loss, counting a training batch's branches."""
        X, y = batch  # pylint: disable=invalid-name
        loss, aware = self.required_criterion(self(X), y, *self.criterion_params)  # pylint: disable=not-callable
        if self.training:
            self._aware_pairs += int(aware.sum())
            self._oblivious_pairs += int((~aware).sum())
        # Sigma grows by the batch's sum of x x^T, so the gradient sums too.
        return loss.sum()

    def pre_phase_hook(self) -> None:
        """Build or resize the projection set, and zero the branch counts.

        The counts are per phase: the trainer copies ``run_metrics`` into each
        phase's result when it ends, and ``summary`` sums them.
        """
        super().pre_phase_hook()
        self._aware_pairs, self._oblivious_pairs = 0, 0

    def run_metrics(self) -> dict:
        """Add dependent data's censor-aware and censor-oblivious pair counts."""
        metrics = super().run_metrics()
        if self.dependent:
            metrics["censor_aware"] = self._aware_pairs
            metrics["censor_oblivious"] = self._oblivious_pairs
        return metrics

    def train_batch(self, batch) -> Optional[dict]:
        """Grow dependent data's Sigma by the batch, then let the trainer step.

        Here rather than in forward, which evaluation also calls.
        """
        if self.dependent:
            X, _ = batch  # pylint: disable=invalid-name
            with ch.no_grad():
                self.Sigma += X.T @ X  # pylint: disable=no-member
        return None

    def evaluate(self) -> dict:
        """Return the loss over the pairs the descent steps on.

        Dependent data skips the warm-up, and takes the pairs in
        chunks, since each draws as many latents as the test.
        """
        if not self.dependent:
            return super().evaluate()
        X, y = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        draws = self.criterion_params[-1].num_samples
        rows = max(sampling.MAX_DRAWS_PER_ROUND // (draws * y.size(1)), 1)
        pairs = zip(
            X[self._warmup_size :].split(rows),
            y[self._warmup_size :].split(rows),
            strict=True,
        )
        with ch.no_grad(), ch.random.fork_rng():
            ch.manual_seed(self.rand_seed)
            return {"loss": sum(self.compute_loss(pair) for pair in pairs)}

    def default_radius(self) -> float:
        """Return D_{r,B}'s ``4 log(2 / alpha) + 7`` for a known variance.

        A learned variance keeps the base's ``1 / alpha``, the radius of
        ``UnknownVarianceSet``; dependent data's ellipsoid has radius 1.
        """
        if self.dependent:
            return 1.0
        if self.noise_var is None:
            return super().default_radius()
        alpha = 1.0 if self.alpha is None else self.alpha
        return 4.0 * math.log(2.0 / alpha) + 7.0

    def _constrained_projection_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        radius: float,
    ) -> delphi.projection_sets.ProjectionSet:
        """Return the set the descent is kept in, anchored at emp_weight.

        D_{r,B} with its r grown until it holds emp_weight, for learned
        variance D around emp_weight's residual variance, and for dependent
        data the confidence ellipsoid around it.

        Raises:
            ValueError: If emp_weight lies outside the ball of radius
                weight_bound.
        """
        if self.dependent:
            metric = self.Sigma  # pylint: disable=no-member
            return ConfidenceEllipsoid(self.emp_start, self._sigma_0(), metric, radius)
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

    def _hessian(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        weight: ch.Tensor,
        sampler: Optional[sampling.GaussianSampler] = None,
    ) -> ch.Tensor:
        """Return the NLL's mean Hessian over X's rows at a weight.

        In w for a known variance; in ``(v, lambda)`` for a learned one.

        Args:
            X: Design matrix rows, shaped (n, d).
            weight: The trained weight, shaped (d, k), or (d + 1, k) learned.
            sampler: Estimates the conditional moments; ``sampler`` if None.

        Raises:
            NotImplementedError: For dependent data.
        """
        if self.dependent:
            raise NotImplementedError(f"{self} has no hessian().")
        sampler = sampler or self.sampler
        if self.noise_var is None:
            return losses.TruncatedGaussianNLL.hessian(
                X @ weight[:-1] / weight[-1],
                X,
                self.latent_phi,
                1.0 / weight[-1],
                sampler,
                natural=True,
            )
        return losses.TruncatedGaussianNLL.hessian(
            X @ weight, X, self.latent_phi, self.noise_var, sampler
        )

    def coefficients(self, params):
        """Split a flat iterate; for learned variance, ``(v, lambda)``.

        Dependent data's iterate fits ``L^-1 y``, so its estimates are
        multiplied by L.
        """
        if params is not None and self.noise_var is None:
            natural = params.reshape(-1, linear_model.require_fitted(self.k, "k"))
            params = (natural[:-1] / natural[-1]).flatten()
        coef, intercept = super().coefficients(params)
        if coef is None or self._noise_factor is None:
            return coef, intercept
        factor = self._noise_factor
        return factor @ coef, None if intercept is None else factor @ intercept

    def variance(self, params: Optional[ch.Tensor]) -> Optional[ch.Tensor]:
        """Return the noise variance at a flat iterate, shaped (k,).

        The given ``noise_var`` when known. Otherwise ``1 / lambda`` at a
        ``(v, lambda)`` iterate, times ``response_scale ** 2`` to undo
        ``fit``'s scaling of y; None when params is.
        """
        if self.noise_var is not None:
            return self.noise_var
        if params is None:
            return None
        precision = params.reshape(-1, linear_model.require_fitted(self.k, "k"))[-1]
        return self.response_scale**2 / precision

    def _flat_estimates(self, params: ch.Tensor) -> ch.Tensor:
        """Return ``coefficients`` and, when learned, ``variance`` at params.

        A function of the trained weight rather than a property, so that
        ``covariance`` can differentiate it for the delta method. Flattened
        in order: the coefficients shaped (k, d) row-major, the k
        intercepts when fit, then the k variances when learned.
        """
        coef, intercept = self.coefficients(params)
        variance = self.variance(params) if self.noise_var is None else None
        parts = [coef, intercept, variance]
        return ch.cat([part.flatten() for part in parts if part is not None])

    def _information(self, params: ch.Tensor) -> ch.Tensor:
        """Return the batch-mean Fisher information in the trained weight.

        The NLL's Hessian, which does not depend on y in w or in
        ``(v, lambda)``, so the observed information is the expected one.
        Rows go to the sampler in chunks of ``MAX_DRAWS_PER_ROUND`` draws,
        and the same seed every call gives every interval the same draws.
        """
        draws = self._information_sampler.num_samples
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        weight = params.reshape(-1, linear_model.require_fitted(self.k, "k"))
        total = params.new_zeros(weight.numel(), weight.numel())
        chunk = max(sampling.MAX_DRAWS_PER_ROUND // draws, 1)
        with ch.random.fork_rng():
            ch.manual_seed(self.rand_seed)
            for rows in X.to(params.dtype).split(chunk):
                hessian = self._hessian(rows, weight, self._information_sampler)
                total += rows.size(0) * hessian.to(params.dtype)
        return total / X.size(0)

    def covariance(
        self, estimate: constants.Estimator = constants.Estimator.AVG
    ) -> ch.Tensor:
        """Return the asymptotic covariance of an estimate.

        The inverse Fisher information of the n samples at the estimate, in
        the trained weight's scaled coordinates, carried to the
        coefficients, intercept and, when learned, the variance on the scale
        of the data passed to ``fit`` by the delta method.

        Args:
            estimate: The iterate whose estimates to cover.

        Returns:
            The covariance in the dtype the model was fit in, shaped (p, p),
            over the coefficients shaped (k, d) row-major, the k intercepts
            when fit, then the k variances when learned.

        Raises:
            ValueError: If the data is dependent, or estimate is not a
                ``constants.Estimator``.
            RuntimeError: If called before fit.
        """
        params = self._trained_weight(estimate)
        X, _ = linear_model.require_fitted(self._data, "data")  # pylint: disable=invalid-name
        information = X.size(0) * self._information(params)
        jacobian = torch.autograd.functional.jacobian(self._flat_estimates, params)
        covariance = jacobian @ ch.linalg.solve(information, jacobian.T)  # pylint: disable=not-callable
        return (covariance + covariance.T) / 2

    def _trained_weight(self, estimate: constants.Estimator) -> ch.Tensor:
        """Return an estimate's flat trained weight, detached.

        Raises:
            ValueError: As ``covariance``.
            RuntimeError: If called before fit.
        """
        if self.dependent:
            raise ValueError(
                "covariance and confidence intervals need independent data."
            )
        name = constants.Estimator(estimate)
        params = linear_model.require_fitted(self._estimate(name), name)
        return params.detach().flatten()

    def confidence_intervals(
        self,
        level: float = 0.95,
        estimate: constants.Estimator = constants.Estimator.AVG,
    ) -> ConfidenceIntervals:
        """Return Wald intervals for the coefficients, intercept and variance.

        Each entry's ``estimate +- z sd``, for sd from ``covariance`` and z
        the normal's ``(1 + level) / 2`` quantile; each interval covers its
        entry alone, not all jointly.

        Args:
            level: Coverage of each interval, in (0, 1).
            estimate: The iterate whose estimates to cover.

        Returns:
            The intervals, on the scale of the data passed to ``fit``.

        Raises:
            ValueError: If level lies outside (0, 1), or as ``covariance``.
            RuntimeError: If called before fit.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1); got {level}.")
        point = self._flat_estimates(self._trained_weight(estimate))
        quantile = ch.special.ndtri(point.new_tensor(0.5 + level / 2))  # pylint: disable=not-callable
        half_width = quantile * self.covariance(estimate).diagonal().sqrt()
        lower = self._unflatten_estimates(point - half_width)
        upper = self._unflatten_estimates(point + half_width)
        optional = [
            None if low is None or high is None else Interval(low, high)
            for low, high in zip(lower[1:], upper[1:], strict=True)
        ]
        return ConfidenceIntervals(Interval(lower[0], upper[0]), *optional)

    def _unflatten_estimates(
        self, flat: ch.Tensor
    ) -> tuple[ch.Tensor, Optional[ch.Tensor], Optional[ch.Tensor]]:
        """Invert ``_flat_estimates``: return coef, intercept and variance."""
        k = linear_model.require_fitted(self.k, "k")
        intercept: Optional[ch.Tensor] = None
        variance: Optional[ch.Tensor] = None
        if self.noise_var is None:
            flat, variance = flat[:-k], flat[-k:]
        if self.fit_intercept:
            flat, intercept = flat[:-k], flat[-k:]
        return flat.reshape(k, -1), intercept, variance

    @property
    def avg_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the averaged iterate."""
        return self.variance(self._estimate(constants.Estimator.AVG))

    @property
    def ema_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the moving average of the iterates."""
        return self.variance(self._estimate(constants.Estimator.EMA))

    @property
    def final_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the last iterate."""
        return self.variance(self._estimate(constants.Estimator.FINAL))

    @property
    def best_variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the iterate with the lowest validation loss."""
        return self.variance(self._estimate(constants.Estimator.BEST))

    @property
    def variance_(self) -> Optional[ch.Tensor]:
        """Noise variance at the averaged iterate."""
        return self.avg_variance_

    def _summary_procedure(self) -> list[tuple[str, str]]:
        rows = super()._summary_procedure()
        if self.dependent:
            procedure = linear_model.require_fitted(self.procedure, "procedure")
            oblivious = sum(procedure.series("censor_oblivious"))
            total = oblivious + sum(procedure.series("censor_aware"))
            rows.append(("censor-oblivious pairs", f"{oblivious} of {total}"))
        return rows

    def _summary_estimates(self) -> list[tuple[str, str]]:
        rows = super()._summary_estimates()
        if self.noise_var is None:
            rows += [
                (
                    f"{name}_variance_",
                    summary.tensor(getattr(self, f"{name}_variance_")),
                )
                for name in constants.Estimator
            ]
        if not self.dependent:
            rows += [
                (
                    f"{name}_ 95% interval",
                    f"{summary.tensor(interval.lower)} to "
                    f"{summary.tensor(interval.upper)}",
                )
                for name, interval in self.confidence_intervals().items()
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


class ConfidenceEllipsoid(delphi.projection_sets.ProjectionSet):
    """The ellipsoid ``{W : tr((W - W_0)^T Sigma_0 (W - W_0)) <= r^2}``.

    A projection returns the nearest point in the norm
    ``tr(D^T Sigma D)`` of a metric Sigma, not the Euclidean one: with
    ``Sigma_0 = C C^T`` and ``C^-1 Sigma C^-T = Q diag(e) Q^T``, the
    coordinates ``Q^T C^T D`` turn the set into a ball and the metric into
    ``diag(e)``, and the nearest point shrinks coordinate j by
    ``e_j / (e_j + mu)``, for the mu that lands it on the sphere.

    Attributes:
        sigma_0: The ellipsoid's shape, shaped (p, p).
        metric: Sigma, shaped (p, p); its owner may grow it in place.
        radius: The set's r.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        anchor: ch.Tensor,
        sigma_0: ch.Tensor,
        metric: ch.Tensor,
        radius: float = 1.0,
        bisection_steps: int = 60,
    ):
        """Build the ellipsoid around anchor.

        Args:
            anchor: Its center W_0, shaped (p, k).
            sigma_0: Its shape, positive definite, shaped (p, p).
            metric: The projection's Sigma, positive definite, shaped (p, p).
            radius: Its r.
            bisection_steps: Halvings of the interval mu is searched in.
        """
        super().__init__(anchor, radius, bisection_steps)
        self.sigma_0 = sigma_0
        self.metric = metric

    def contains(self, weight: ch.Tensor) -> bool:
        offset = (weight.detach() - self.anchor).double()
        spread = float((offset * (self.sigma_0.double() @ offset)).sum())
        return spread <= cast(float, self.radius) ** 2

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        """Return the nearest point of the set to weight in the metric's norm."""
        if self.contains(weight):
            return weight
        chol = ch.linalg.cholesky(self.sigma_0.double())  # pylint: disable=not-callable
        half = ch.linalg.solve_triangular(chol, self.metric.double(), upper=False)  # pylint: disable=not-callable
        whitened = ch.linalg.solve_triangular(chol, half.T, upper=False)  # pylint: disable=not-callable
        spectrum, basis = ch.linalg.eigh((whitened + whitened.T) / 2.0)  # pylint: disable=not-callable
        target = basis.T @ chol.T @ (weight.detach() - self.anchor).double()
        mass = (spectrum[:, None] * target).pow(2).sum(1)
        radius = cast(float, self.radius)
        # At mu = high every coordinate already lies within the sphere.
        low, high = 0.0, float(mass.sum().sqrt()) / radius
        for _ in range(self.bisection_steps):
            middle = (low + high) / 2.0
            if float((mass / (spectrum + middle).pow(2)).sum()) > radius**2:
                low = middle
            else:
                high = middle
        shrunk = (spectrum / (spectrum + high))[:, None] * target
        offset = ch.linalg.solve_triangular(chol.T, basis @ shrunk, upper=True)  # pylint: disable=not-callable
        return (self.anchor + offset).to(weight.dtype)


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
        for start, end in zip(corners, corners.roll(-1, 0), strict=True):
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
