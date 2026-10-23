# Author: pstefanou12@
"""Parent class for truncated exponential distribution model classes."""

import abc
from collections.abc import Callable
import functools
import logging
import math
from typing import ClassVar, Generic, Optional

import torch as ch
from torch import distributions as torch_distributions
import torch.utils.data

from delphi import configs, trainer
from delphi.truncated.distributions import distributions, losses, summary
from delphi.utils import datasets

# Called as dist(theta, dims) in losses.py on every step.
DistributionFactory = Callable[[ch.Tensor, int], torch_distributions.Distribution]

SufficientStatistic = Callable[[ch.Tensor], ch.Tensor]


# Draws backing mass_on_set.
MASS_DRAWS = 20000


class TruncatedExponentialFamilyDistribution(  # pylint: disable=too-many-instance-attributes
    distributions.distributions[configs._ConfigT], Generic[configs._ConfigT], abc.ABC
):
    """Base class for truncated exponential family distribution models.

    Concrete subclasses either set the ``dist`` class attribute to their
    exponential family class, or override ``_build_distribution`` when the
    family takes an argument fixed at construction.

    Attributes:
        phi (Callable): Truncation set oracle.
        alpha (float): Survival probability lower bound.
        dims (int): Number of dimensions.
        dist: Exponential family class or factory, resolved during ``fit``.
        calc_suff_stat (Callable): Sufficient-statistic function for ``dist``,
            resolved alongside it.
        criterion: NLL loss function applied during training.
        criterion_params (list): Extra parameters passed to criterion.
        best_params: Best natural parameters found across all training phases.
        best_loss: Loss value at best parameters.
        final_params: Natural parameters at the end of the last phase.
        final_loss: Loss value at final parameters.
        ema_params: Exponential moving-average natural parameters.
        avg_params: Running-average natural parameters.
        train_loader_: Training data loader, set during fit.
        val_loader_: Validation data loader, set during fit.
        nll_init: Non-truncated NLL at the empirical initialization theta_0.
        radius (float): Current NLL budget above nll_init for the sublevel-set
            projection (Karatapanis et al., Algorithm 2).
    """

    # Cov[T(z)] is diagonal up to sampling noise for a product family, so
    # FisherLR.spectrum reads the diagonal rather than eigenvalues.
    independent_coordinates: bool = True

    # A type, not a Callable: mypy reads a callable class attribute as a
    # method and drops the first argument.
    dist: ClassVar[Optional[type[torch_distributions.Distribution]]] = None

    # Suffixes of the canonical estimates, so that f"avg_{suffix}" names
    # one. Read by summary().
    estimate_suffixes: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        hyperparameters: dict,
        phi: Callable,
        dims: int,
        alpha: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedExponentialFamilyDistribution.

        Args:
            hyperparameters: Hyperparameter dict.
            phi: Truncation set oracle.
            dims: Number of dimensions.
            alpha: Survival probability in (0, 1]; estimated from phi at the
                empirical initialization when omitted.

        Raises:
            ValueError: If alpha is given and lies outside (0, 1).
        """
        super().__init__(hyperparameters)
        self.logger = logging.getLogger(type(self).__module__)
        self.phi = phi
        if alpha is not None and not 0.0 < alpha < 1.0:
            raise ValueError(
                "alpha is the survival probability and must lie in (0, 1); got "
                f"{alpha}. alpha=1 claims nothing was truncated, which leaves "
                "the procedure nothing to correct for: fit the family directly, "
                "or omit alpha to fall back on min_radius."
            )
        self.alpha = alpha
        self.dims = dims
        self.criterion = losses.TruncatedExponentialFamilyDistributionNLL.apply

        self.best_params: Optional[ch.Tensor] = None
        self.best_loss: Optional[ch.Tensor] = None
        self.final_params: Optional[ch.Tensor] = None
        self.final_loss: Optional[ch.Tensor] = None
        self.ema_params: Optional[ch.Tensor] = None
        self.avg_params: Optional[ch.Tensor] = None

        # Attributes initialized during fit.
        self.train_loader_: Optional[torch.utils.data.DataLoader] = None
        self.val_loader_: Optional[torch.utils.data.DataLoader] = None
        self.radius: Optional[float] = None
        self.trainer: Optional[trainer.Trainer] = None
        self.procedure: Optional[trainer.ProcedureResult] = None
        self.n_samples_: Optional[int] = None
        self.emp_theta: Optional[ch.Tensor] = None
        # The projection's anchor, set once the start point is final.
        self.nll_init: Optional[float] = None
        # Running totals, and their values when the current run began, so
        # run_metrics reports each run's own.
        self._projections, self._projection_distance = 0, 0.0
        self._run_start = (0, 0.0)
        self._previous_phase_loss: Optional[float] = None
        # The inverse curvature each step multiplies the gradient by, when
        # the config asks for preconditioning; set per run.
        self._preconditioner: Optional[ch.Tensor] = None
        self._gradient_hook: Optional[ch.utils.hooks.RemovableHandle] = None

    def fit(self, S: ch.Tensor):  # pylint: disable=invalid-name
        """Fit the model to the observed (truncated) samples S.

        Args:
            S: Observed samples of shape (num_samples, dims).
        """
        if not isinstance(S, ch.Tensor):
            raise TypeError(f"S is type: {type(S)}. expected type torch.Tensor.")
        if S.size(0) <= S.size(1):
            raise ValueError(
                "input expected to be shape num samples by dimensions, "
                f"current input is size {S.size()}."
            )
        if self.hyperparameters.batch_size > self.hyperparameters.num_samples:
            raise ValueError(
                f"batch size ({self.hyperparameters.batch_size}) must be smaller than "
                f"or equal to the number of samples ({self.hyperparameters.num_samples})."
            )

        self.dist, self.calc_suff_stat = self._build_distribution()
        self.criterion_params = [
            self.phi,
            self.dims,
            self.dist,
            self.calc_suff_stat,
            self.hyperparameters.num_samples,
            self.hyperparameters.eps,
        ]
        self.warm_start_criterion = functools.partial(
            losses.ExponentialFamilyDistributionNLL.apply,
            dims=self.dims,
            dist=self.dist,
        )
        self.warm_start_hessian = self._untruncated_hessian

        self.train_loader_, self.val_loader_ = datasets.make_train_and_val_distr(
            self.hyperparameters,
            S,
            datasets.TruncatedExponentialDistributionDataset,
            {"calc_suff_stat": self.calc_suff_stat},
        )

        self.n_samples_ = S.size(0)
        self._projections, self._projection_distance = 0, 0.0
        self.nll_init, self.radius, self._previous_phase_loss = None, None, None
        # _calc_emp_model registers a fresh theta, which needs its own hook.
        self._gradient_hook = None
        self._calc_emp_model()
        self.trainer = trainer.Trainer(self, self.hyperparameters)
        self.procedure = self.trainer.train_model(self.train_loader_, self.val_loader_)
        self._adopt_results(self.procedure)
        return self

    def post_warm_start_hook(self, result: trainer.RunResult) -> None:
        """Start the procedure from the warm start's averaged iterate."""
        # The config requires record_params_every >= 1, so there are iterates.
        self.emp_theta = self._constraints(result.avg_params.detach().clone())
        with ch.no_grad():
            self._write_theta(self.emp_theta)

    def pre_phase_hook(self) -> None:
        """Start the projection before the first phase; mark each phase's counts."""
        if self.radius is None:
            self._start_projection()
        self._run_start = (self._projections, self._projection_distance)

    def _start_projection(self) -> None:
        """Anchor the projection at the start point, with its first budget."""
        self.nll_init = self._compute_nll(self.emp_theta)
        self.radius = self._initial_radius()

    def run_metrics(self) -> dict:
        """Return the budget the run used and how often it projected."""
        return {
            "radius": self.radius,
            "projections": self._projections - self._run_start[0],
            "projection_distance": self._projection_distance - self._run_start[1],
        }

    def post_phase_hook(self, result: trainer.RunResult) -> tuple[bool, Optional[str]]:
        """Widen the projection budget, or say why the procedure is done.

        With alpha known the budget is log(1 / alpha), whose sublevel set
        holds the truth (Karatapanis et al., 2025), so one phase suffices.
        Without it the budget is a guess, widened by ``rate`` up to
        ``max_radius``. A budget the phase never projected onto constrained
        nothing, so widening it would only rerun the same fit, and widening
        stops once it no longer improves the loss.
        """
        if self.alpha is not None:
            return True, "alpha_known"
        if result.metrics["projections"] == 0:
            return True, "budget_not_binding"
        stop, reason = self._phase_loss_converged(result)
        if stop:
            return stop, reason
        if self.radius >= self.hyperparameters.max_radius:
            return True, "max_radius"
        previous = self.radius
        self.radius = min(
            self.radius * self.hyperparameters.rate, self.hyperparameters.max_radius
        )
        self.logger.info(
            f"widening the projection budget {previous:.4f} -> {self.radius:.4f}"
        )
        return False, None

    def _phase_loss_converged(
        self, result: trainer.RunResult
    ) -> tuple[bool, Optional[str]]:
        """Compare this phase's best loss with the previous phase's."""
        previous, self._previous_phase_loss = (
            self._previous_phase_loss,
            None if result.best_loss is None else float(result.best_loss),
        )
        current = self._previous_phase_loss
        if previous is None or current is None:
            return False, None
        hp = self.hyperparameters
        improvement = previous - current
        if hp.phase_loss_tol is not None and abs(improvement) < hp.phase_loss_tol:
            return True, f"phase_loss_converged ({improvement:.3e})"
        relative = abs(improvement) / (abs(previous) + 1e-10)
        if (
            hp.phase_relative_loss_tol is not None
            and relative < hp.phase_relative_loss_tol
        ):
            return True, f"phase_relative_loss_converged ({relative:.3e})"
        if (
            hp.phase_loss_increase_tol is not None
            and -improvement > hp.phase_loss_increase_tol
        ):
            return True, f"phase_loss_increasing ({-improvement:.3e})"
        return False, None

    def _adopt_results(self, result: trainer.ProcedureResult) -> None:
        """Take the estimates from the last phase."""
        final = result.final
        self.best_params, self.best_loss = final.best_params, final.best_loss
        self.final_params, self.final_loss = final.final_params, final.final_loss
        self.ema_params, self.avg_params = final.ema_params, final.avg_params

    def _build_distribution(self) -> tuple[DistributionFactory, SufficientStatistic]:
        """Return the distribution factory and its sufficient statistic.

        Resolved when ``fit`` runs rather than during ``__init__``, so a
        subclass sets whatever it needs after ``super().__init__`` like any
        other attribute.

        Returns:
            The callable building the distribution from theta, and the
            sufficient statistic of the family.

        Raises:
            NotImplementedError: If the subclass declares no ``dist``.
        """
        if getattr(self, "dist", None) is None:
            raise NotImplementedError(
                f"{type(self).__name__} must declare a dist class attribute or "
                "override _build_distribution."
            )
        return self.dist, self.dist.calc_suff_stat

    def hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the Hessian of the truncated NLL, for deriving step sizes.

        Args:
            theta: Point to evaluate at; defaults to the empirical
                initialization, which is where step sizes are chosen.

        Returns:
            ``Cov[T(z)]`` over the truncated distribution, shaped (k, k).

        Raises:
            sampling.SamplingBudgetExceeded: If theta puts almost no mass on
                the truncation set.
        """
        theta = self.emp_theta if theta is None else theta
        return losses.TruncatedExponentialFamilyDistributionNLL.hessian(
            theta,
            self.phi,
            self.dims,
            self.dist,
            self.calc_suff_stat,
            self.hyperparameters.num_samples,
        )

    def make_optimizer_and_schedule(
        self,
        checkpoint: Optional[dict] = None,
        hessian: Optional[Callable[..., ch.Tensor]] = None,
    ):
        """Build the optimizer and schedule, preconditioned if configured.

        With ``precondition`` set, every step of a phase moves by
        ``C^{-1} g`` rather than ``g``, for ``C`` the covariance of ``T(x)``
        over the observed sample, which estimates the truncated NLL's Hessian
        at the truth without sampling. That is plain SGD after the change of
        variables ``theta = L^{-T} psi`` for ``C = L L^T``, so the Fisher
        schedule reads the spectrum of ``L^{-1} H L^{-T}``, near the identity
        where ``C`` is accurate. A warm start, which passes its own
        ``hessian``, is not preconditioned: its curvature at a rough start
        can be far from the optimum's, and a step scaled by it left the
        Wishart's domain at once.

        Args:
            checkpoint: As the base class.
            hessian: As the base class.

        Returns:
            Tuple of (optimizer, scheduler).
        """
        self._preconditioner = None
        if getattr(self.hyperparameters, "precondition", False) and hessian is None:
            factor = ch.linalg.cholesky(self._empirical_hessian())  # pylint: disable=not-callable
            self._preconditioner = ch.cholesky_inverse(factor)
            hessian = self._whitened(self.hessian, factor)
            # A tensor hook rather than step_pre_hook, which runs before
            # optimizer.step's closure recomputes the gradient.
            if self._gradient_hook is None:
                self._gradient_hook = self.theta.register_hook(self._precondition)
        return super().make_optimizer_and_schedule(checkpoint, hessian)

    def _empirical_hessian(self) -> ch.Tensor:
        """Return the covariance of ``T(x)`` over the training sample."""
        return ch.atleast_2d(self.train_loader_.dataset.S_grad.T.cov())

    @staticmethod
    def _whitened(
        hessian: Callable[..., ch.Tensor], factor: ch.Tensor
    ) -> Callable[..., ch.Tensor]:
        """Return ``hessian`` in the coordinates ``factor`` whitens."""

        def whitened(theta: Optional[ch.Tensor] = None) -> ch.Tensor:
            left = ch.linalg.solve_triangular(factor, hessian(theta), upper=False)  # pylint: disable=not-callable
            return ch.linalg.solve_triangular(factor, left.T, upper=False)  # pylint: disable=not-callable

        return whitened

    def _precondition(self, grad: ch.Tensor) -> ch.Tensor:
        """Return the gradient times the run's preconditioner, if it has one."""
        return grad if self._preconditioner is None else self._preconditioner @ grad

    def _untruncated_hessian(self, theta: Optional[ch.Tensor] = None) -> ch.Tensor:
        """Return the Hessian of the warm-start criterion, as ``hessian`` does."""
        theta = self.emp_theta if theta is None else theta
        return losses.ExponentialFamilyDistributionNLL.hessian(
            theta, self.train_loader_.dataset.S, self.dims, self.dist
        )

    def mass_on_set(self, theta: ch.Tensor) -> float:
        """Return the mass theta's distribution puts on the truncation set.

        Plain Monte Carlo rather than the rejection sampler, so a theta with
        almost no mass there returns a small number instead of raising.

        Args:
            theta: Natural parameter to evaluate.

        Returns:
            Estimated probability of the truncation set.
        """
        with ch.no_grad():
            samples = self.dist(theta, self.dims).sample([MASS_DRAWS])
            return float(self.phi(samples.view(MASS_DRAWS, -1)).float().mean())

    def _initial_radius(self) -> float:
        """Return the NLL budget above the empirical initialization.

        Returns:
            An explicit ``min_radius``, else ``log(1 / alpha)`` when alpha is
            known, else the ``min_radius`` default.
        """
        if "min_radius" in self.hyperparameters.model_fields_set:
            return self.hyperparameters.min_radius
        if self.alpha is not None:
            return math.log(1.0 / self.alpha)
        return self.hyperparameters.min_radius

    @property
    def nll_threshold(self) -> float:
        """NLL budget: non-truncated NLL upper bound for the projection set."""
        return self.nll_init + self.radius

    def _calc_emp_model(self):
        """Initialize theta from the training data.

        Raises:
            NotImplementedError: Always. The empirical mean of ``T(x)`` is the
                mean parameter, and mapping it back to the natural one inverts
                the gradient of the log-partition, which is family-specific.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _calc_emp_model to convert "
            "the empirical mean of T(x) to a natural parameter."
        )

    def _compute_nll(self, theta: ch.Tensor) -> float:
        """Compute non-truncated NLL of training samples under theta.

        Args:
            theta: Natural parameters to evaluate.

        Returns:
            Mean negative log-likelihood over training samples.
        """
        with ch.no_grad():
            S = self.train_loader_.dataset.S  # pylint: disable=invalid-name
            D = self.dist(theta, self.dims)  # pylint: disable=invalid-name
            return -D.log_prob(S).mean().item()

    def _project_onto_sublevel_set(self, theta: ch.Tensor) -> ch.Tensor:
        """Project theta onto {θ : L(θ) ≤ nll_threshold} via bisection.

        Performs binary search on the interpolation weight λ ∈ [0, 1] along
        the segment θ(λ) = (1−λ)·emp_theta + λ·theta.  At λ=0 the point is
        emp_theta, which always satisfies the constraint.  At λ=1 the point is
        theta, which may violate it.  The search finds the largest feasible λ,
        giving the boundary point closest to theta along this segment.

        Args:
            theta: Current parameter vector to project.

        Returns:
            Projected parameter vector satisfying the NLL constraint.
        """
        if self._compute_nll(theta) <= self.nll_threshold:
            return theta

        emp = self.emp_theta.detach()
        lambda_lo, lambda_hi = 0.0, 1.0
        for _ in range(50):  # 50 iterations → ~2^{-50} precision on λ
            lambda_mid = (lambda_lo + lambda_hi) / 2.0
            theta_mid = (1.0 - lambda_mid) * emp + lambda_mid * theta.detach()
            if self._compute_nll(theta_mid) <= self.nll_threshold:
                lambda_lo = lambda_mid
            else:
                lambda_hi = lambda_mid
        projected = (1.0 - lambda_lo) * emp + lambda_lo * theta.detach()
        self._projections += 1
        self._projection_distance += float((projected - theta.detach()).norm())
        return projected

    def summary(self) -> str:
        """Return a plain-text report on how the fit went.

        Covers the data, the optimization, the estimates, the fitted model's
        own mass on the truncation set and the curvature at the estimate.
        There are no standard errors or p-values: neither analysis this
        follows gives an asymptotic distribution for the estimate.

        Returns:
            The report, ready to print.

        Raises:
            RuntimeError: If called before fit.
        """
        return summary.render(self)

    def forward(self, x):  # pylint: disable=unused-argument
        """Return the current theta parameter (input is unused).

        Args:
            x: Input data (unused).
        """
        return self.theta

    def _constraints(self, theta):
        """Apply parameter constraints. Override in subclasses if needed."""
        return theta

    def step_post_hook(self, optimizer, args, kwargs) -> None:
        """Keep theta in its domain, and optionally on the NLL sublevel set.

        The constraints always apply: a step can leave the parameter's domain
        whether or not the projection is on. With ``project`` set, theta is
        then projected onto ``{θ : L(θ) ≤ nll_threshold}``, per Karatapanis et
        al. (2025), Algorithm 2.
        """
        with ch.no_grad():
            self._write_theta(self._constraints(self.theta))
            # Before nll_init is set, as during a warm start, there is no
            # sublevel set to project onto.
            if self.hyperparameters.project and self.nll_init is not None:
                proj_theta = self._project_onto_sublevel_set(self.theta)
                self._write_theta(self._constraints(proj_theta))

    def _write_theta(self, value: ch.Tensor) -> None:
        """Write a projected theta value back to the parameter storage."""
        self.theta.copy_(value)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated exponential family distribution"
