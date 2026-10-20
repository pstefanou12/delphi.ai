# Author: pstefanou12@
"""Parent class for truncated exponential distribution model classes."""

import abc
from collections.abc import Callable
import logging
import math
import time
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
        self.prev_best_loss: Optional[ch.Tensor] = None
        self.radius_history: Optional[list[float]] = None
        self.loss_history: Optional[list[ch.Tensor]] = None
        self.radius: Optional[float] = None
        self.trainer: Optional[trainer.Trainer] = None
        self.prev_theta: Optional[ch.Tensor] = None
        self.prev_loss: Optional[ch.Tensor] = None
        self.n_samples_: Optional[int] = None
        self.steps_: int = 0
        self.projections_: int = 0
        self.projection_distance_: float = 0.0
        self.fit_seconds_: Optional[float] = None
        self.emp_theta: Optional[ch.Tensor] = None
        self.nll_init: Optional[ch.Tensor] = None

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

        self.train_loader_, self.val_loader_ = datasets.make_train_and_val_distr(
            self.hyperparameters,
            S,
            datasets.TruncatedExponentialDistributionDataset,
            {"calc_suff_stat": self.calc_suff_stat},
        )

        self.prev_best_loss = None
        self.n_samples_ = S.size(0)
        self.steps_, self.projections_, self.projection_distance_ = 0, 0, 0.0
        started = time.perf_counter()
        self.radius_history = []
        self.loss_history = []
        self._calc_emp_model()
        self.nll_init = self._compute_nll(self.emp_theta)
        self.radius = self._initial_radius()

        phase = 0
        while phase < self.hyperparameters.max_phases:
            phase += 1
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"phase {phase}: training with radius={self.radius:.4f}")
            self.logger.info(f"\n{'=' * 60}")

            self.trainer = trainer.Trainer(self, self.hyperparameters)
            self.trainer.train_model(self.train_loader_, self.val_loader_)
            self.steps_ += self.trainer.iterations

            current_loss = self.trainer.best_loss
            self.radius_history.append(self.radius)
            self.loss_history.append(current_loss)

            self.best_params, self.best_loss = (
                self.trainer.best_params,
                self.trainer.best_loss,
            )
            self.prev_theta, self.prev_loss = self.best_params.clone(), self.best_loss
            self.final_params, self.final_loss = (
                self.trainer.final_params,
                self.trainer.final_loss,
            )
            self.ema_params = self.trainer.ema_params
            self.avg_params = self.trainer.avg_params

            should_stop, reason = self._check_convergence()

            if should_stop:
                self.logger.info(f"\n{'=' * 60}")
                self.logger.info(f"procedure converged: {reason}")
                self.logger.info(
                    f"final radius: {self.radius:.4f}, final loss: {current_loss:.6f}"
                )
                self.logger.info(f"total phases: {phase}")
                self.logger.info(f"\n{'=' * 60}")
                break

            # Expand radius for next phase
            self.prev_best_loss = current_loss
            old_radius = self.radius
            self.radius = min(
                self.radius * self.hyperparameters.rate, self.hyperparameters.max_radius
            )

            self.logger.info(
                f"expanding radius: {old_radius:.4f} -> {self.radius:.4f}, "
                f"loss improved by: {self.prev_best_loss - current_loss:.6e}"
            )

        self.fit_seconds_ = time.perf_counter() - started
        return self

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

    def _check_convergence(self):
        """Determine if the entire training procedure should stop.

        Returns:
            Tuple of (should_stop, reason) where should_stop is a bool and
            reason is a string label for the convergence criterion, or None
            if convergence has not been reached.
        """
        current_loss = self.trainer.best_loss

        # Criterion 1: Reached maximum radius
        if self.radius >= self.hyperparameters.max_radius:
            return True, "max_radius_reached"

        # Criterion 2: Loss improvement is negligible
        if self.prev_best_loss is not None:
            loss_improvement = self.prev_best_loss - current_loss

            if abs(loss_improvement) < self.hyperparameters.loss_convergence_tol:
                return True, f"loss_improvement_small (Δloss={loss_improvement:.6e})"

            # Relative threshold
            relative_improvement = abs(loss_improvement) / (
                abs(self.prev_best_loss) + 1e-10
            )
            if relative_improvement < self.hyperparameters.relative_loss_tol:
                return True, f"relative_improvement_small ({relative_improvement:.6e})"

        # Criterion 3: Loss is increasing (suggests we've passed optimum)
        if self.prev_best_loss is not None and current_loss > self.prev_best_loss:
            if (
                current_loss - self.prev_best_loss
                > self.hyperparameters.loss_increase_tol
            ):
                return True, "loss_increasing"

        return False, None

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
        self.projections_ += 1
        self.projection_distance_ += float((projected - theta.detach()).norm())
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
        """Optionally project theta onto the NLL sublevel set after each update.

        When args.project is True, implements the per-step projection from
        Karatapanis et al. (2025), Algorithm 2:
            θ ← Π_D(θ),  D = {θ : L(θ) ≤ nll_threshold}.
        When args.project is False, the step is a no-op.
        """
        if self.hyperparameters.project:
            with ch.no_grad():
                # Enforce constraints before projection so that _compute_nll
                # inside _project_onto_sublevel_set always receives a theta in
                # the valid domain (e.g. strictly negative for exponential /
                # Weibull).  Constraints are applied again after projection to
                # keep the final value valid as well.
                self._write_theta(self._constraints(self.theta))
                proj_theta = self._project_onto_sublevel_set(self.theta)
                self._write_theta(self._constraints(proj_theta))

    def _write_theta(self, value: ch.Tensor) -> None:
        """Write a projected theta value back to the parameter storage."""
        self.theta.copy_(value)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated exponential family distribution"
