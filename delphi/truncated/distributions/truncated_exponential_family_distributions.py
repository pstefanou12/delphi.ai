# Author: pstefanou12@
"""Parent class for truncated exponential distribution model classes."""

# pylint: disable=duplicate-code

import functools
from collections.abc import Callable

import torch as ch
from torch import nn
from torch.distributions import exp_family

from delphi import delphi_logger, trainer
from delphi.truncated.distributions import distributions, losses
from delphi.utils import configs, datasets


class TruncatedExponentialFamilyDistribution(distributions.distributions):  # pylint: disable=too-many-instance-attributes
    """Base class for truncated exponential family distribution models.

    Attributes:
        phi (Callable): Truncation set oracle.
        alpha (float): Survival probability lower bound.
        dims (int): Number of dimensions.
        dist (ExponentialFamily): Exponential family distribution class.
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

    def __init__(
        self,
        args: configs.TruncatedExponentialFamilyDistributionConfig,
        phi: Callable,
        alpha: float,
        dims: int,
        dist: exp_family.ExponentialFamily,
        logger: delphi_logger.delphiLogger,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Initialize TruncatedExponentialFamilyDistribution.

        Args:
            args: Validated configuration object.
            phi: Truncation set oracle.
            alpha: Survival probability lower bound.
            dims: Number of dimensions.
            dist: Exponential family distribution class.
            logger: Logger instance.
        """
        super().__init__(args, logger)
        self.phi = phi
        self.alpha = alpha
        self.dims = dims
        self.dist = dist

        dist_cls = dist.func if isinstance(dist, functools.partial) else dist
        self.criterion = losses.TruncatedExponentialFamilyDistributionNLL.apply
        self.criterion_params = [
            self.phi,
            self.dims,
            self.dist,
            dist_cls.calc_suff_stat,
            self.args.num_samples,
            self.args.eps,
        ]

        self.best_params = None
        self.best_loss = None
        self.final_params = None
        self.final_loss = None
        self.ema_params = None
        self.avg_params = None

        # Attributes initialized during fit.
        self.train_loader_ = None
        self.val_loader_ = None
        self.prev_best_loss = None
        self.radius_history = None
        self.loss_history = None
        self.radius = None
        self.trainer = None
        self.prev_theta = None
        self.prev_loss = None
        self.emp_theta = None
        self.nll_init = None

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
        if self.args.batch_size > self.args.num_samples:
            raise ValueError(
                f"batch size ({self.args.batch_size}) must be smaller than "
                f"or equal to the number of samples ({self.args.num_samples})."
            )

        dist_cls = (
            self.dist.func if isinstance(self.dist, functools.partial) else self.dist
        )
        self.train_loader_, self.val_loader_ = datasets.make_train_and_val_distr(
            self.args,
            S,
            datasets.TruncatedExponentialDistributionDataset,
            {"calc_suff_stat": dist_cls.calc_suff_stat},
        )

        self.prev_best_loss = None
        self.radius_history = []
        self.loss_history = []
        self._calc_emp_model()
        self.radius = self.args.min_radius

        phase = 0
        while phase < self.args.max_phases:
            phase += 1
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"phase {phase}: training with radius={self.radius:.4f}")
            self.logger.info(f"\n{'=' * 60}")

            self.trainer = trainer.Trainer(self, self.args, self.logger)
            self.trainer.train_model(self.train_loader_, self.val_loader_)

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
            self.radius = min(self.radius * self.args.rate, self.args.max_radius)

            self.logger.info(
                f"expanding radius: {old_radius:.4f} -> {self.radius:.4f}, "
                f"loss improved by: {self.prev_best_loss - current_loss:.6e}"
            )

        return self

    def _check_convergence(self):
        """Determine if the entire training procedure should stop.

        Returns:
            Tuple of (should_stop, reason) where should_stop is a bool and
            reason is a string label for the convergence criterion, or None
            if convergence has not been reached.
        """
        current_loss = self.trainer.best_loss

        # Criterion 1: Reached maximum radius
        if self.radius >= self.args.max_radius:
            return True, "max_radius_reached"

        # Criterion 2: Loss improvement is negligible
        if self.prev_best_loss is not None:
            loss_improvement = self.prev_best_loss - current_loss

            if abs(loss_improvement) < self.args.loss_convergence_tol:
                return True, f"loss_improvement_small (Δloss={loss_improvement:.6e})"

            # Relative threshold
            relative_improvement = abs(loss_improvement) / (
                abs(self.prev_best_loss) + 1e-10
            )
            if relative_improvement < self.args.relative_loss_tol:
                return True, f"relative_improvement_small ({relative_improvement:.6e})"

        # Criterion 3: Loss is increasing (suggests we've passed optimum)
        if self.prev_best_loss is not None and current_loss > self.prev_best_loss:
            if current_loss - self.prev_best_loss > self.args.loss_increase_tol:
                return True, "loss_increasing"

        return False, None

    @property
    def nll_threshold(self) -> float:
        """NLL budget: non-truncated NLL upper bound for the projection set."""
        return self.nll_init + self.radius

    def _calc_emp_model(self):
        """Calculate empirical natural parameters from training data."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        dist_cls = (
            self.dist.func if isinstance(self.dist, functools.partial) else self.dist
        )
        self.emp_theta = dist_cls.calc_suff_stat(dataset_s).mean(0)
        self.register_parameter("theta", nn.Parameter(self.emp_theta.clone()))
        with ch.no_grad():
            self.nll_init = self._compute_nll(self.emp_theta)

    def _compute_nll(self, theta: ch.Tensor) -> float:
        """Compute non-truncated NLL of training samples under theta.

        Args:
            theta: Natural parameters to evaluate.

        Returns:
            Mean negative log-likelihood over training samples.
        """
        S = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        D = self.dist(theta.detach(), self.dims)  # pylint: disable=invalid-name
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
        return (1.0 - lambda_lo) * emp + lambda_lo * theta.detach()

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
        if self.args.project:
            with ch.no_grad():
                proj_theta = self._project_onto_sublevel_set(self.theta)
                self._write_theta(self._constraints(proj_theta))

    def _write_theta(self, value: ch.Tensor) -> None:
        """Write a projected theta value back to the parameter storage."""
        self.theta.copy_(value)

    def __str__(self):
        """Return a human-readable name for this distribution."""
        return "truncated exponential family distribution"
