# Author: pstefanou12@
"""
Parent class for truncated exponential distribution model classes.
"""

# pylint: disable=duplicate-code

from typing import Callable

import torch as ch
from torch import Tensor
from torch.distributions.exp_family import ExponentialFamily
from torch import nn

from delphi.distributions.distributions import distributions
from delphi.delphi_logger import delphiLogger
from delphi.utils.datasets import (
    TruncatedExponentialDistributionDataset,
    make_train_and_val_distr,
)
from delphi.grad import TruncatedExponentialFamilyDistributionNLL
from delphi.trainer import Trainer
from delphi.utils.helpers import Parameters


class TruncatedExponentialFamilyDistribution(distributions):  # pylint: disable=too-many-instance-attributes
    """Base class for truncated exponential family distribution models.

    Attributes:
        phi (Callable): Truncation set oracle.
        alpha (float): Survival probability lower bound.
        dims (int): Number of dimensions.
        dist (ExponentialFamily): Exponential family distribution class.
        calc_suff_stat (Callable): Sufficient statistic calculator.
        criterion: NLL loss function applied during training.
        criterion_params (list): Extra parameters passed to criterion.
        best_params: Best canonical parameters found across all training phases.
        best_loss: Loss value at best parameters.
        final_params: Canonical parameters at the end of the last phase.
        final_loss: Loss value at final parameters.
        ema_params: Exponential moving-average canonical parameters.
        avg_params: Running-average canonical parameters.
        train_loader_: Training data loader, set during fit.
        val_loader_: Validation data loader, set during fit.
        radius (float): Current projection ball radius.
    """

    def __init__(
        self,
        args: Parameters,
        phi: Callable,
        alpha: float,
        dims: int,
        dist: ExponentialFamily,
        calc_suff_stat: Callable,
        logger: delphiLogger,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Args:
            args (Parameters): parameter object holding hyperparameters
            phi (Callable): truncation set oracle
            alpha (float): survival probability lower bound
            dims (int): number of dimensions
            dist (ExponentialFamily): exponential family distribution class
            calc_suff_stat (Callable): sufficient statistic calculator
            logger (delphiLogger): logger instance
        """
        super().__init__(args, logger)
        self.phi = phi
        self.alpha = alpha
        self.dims = dims
        self.dist = dist
        self.calc_suff_stat = calc_suff_stat
        self.criterion = TruncatedExponentialFamilyDistributionNLL.apply
        self.criterion_params = [
            self.phi,
            self.dims,
            self.dist,
            self.calc_suff_stat,
            self.args.num_samples,
            self.args.eps,
        ]

        self.best_params, self.best_loss = None, None
        self.final_params, self.final_loss = None, None
        self.ema_params = None
        self.avg_params = None

        # Attributes initialized during fit
        self.train_loader_ = None
        self.val_loader_ = None
        self.prev_best_loss = None
        self.radius_history = None
        self.loss_history = None
        self.radius = None
        self.trainer = None
        self.prev_theta = None
        self.prev_loss = None
        self.emp_canon_params = None
        self.emp_theta = None

    def fit(self, S: Tensor):  # pylint: disable=invalid-name
        """
        Fit the model to the observed (truncated) samples S.

        Args:
            S (Tensor): observed samples of shape (num_samples, dims)
        """
        assert isinstance(S, Tensor), (
            f"S is type: {type(S)}. expected type torch.Tensor."
        )
        assert S.size(0) > S.size(1), (
            "input expected to be shape num samples by dimenions, "
            f"current input is size {S.size()}."
        )
        assert self.args.batch_size <= self.args.num_samples, (
            "batch size must be smaller than or equal to the number of samples being sampled"
        )

        self.train_loader_, self.val_loader_ = make_train_and_val_distr(
            self.args,
            S,
            TruncatedExponentialDistributionDataset,
            {"calc_suff_stat": self.calc_suff_stat},
        )
        # Initialize tracking
        self.prev_best_loss = None
        self.radius_history = []
        self.loss_history = []
        # Initialize radius and parameters
        self._calc_emp_model()
        self.radius = self.args.min_radius

        phase = 0
        while phase < self.args.max_phases:
            phase += 1
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"phase {phase}: training with radius={self.radius:.4f}")
            self.logger.info(f"\n{'=' * 60}")

            self.trainer = Trainer(self, self.args, self.logger)
            self.trainer.train_model(self.train_loader_, self.val_loader_)

            # Update tracking
            current_loss = self.trainer.best_loss
            self.radius_history.append(self.radius)
            self.loss_history.append(current_loss)

            self.best_params, self.best_loss = (
                self._reparameterize_canon_form(self.trainer.best_params),
                self.trainer.best_loss,
            )
            self.prev_theta, self.prev_loss = self.best_params.clone(), self.best_loss
            self.final_params, self.final_loss = (
                self._reparameterize_canon_form(self.trainer.final_params),
                self.trainer.final_loss,
            )
            self.ema_params = self._reparameterize_canon_form(self.trainer.ema_params)
            self.avg_params = self._reparameterize_canon_form(self.trainer.avg_params)

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
        """
        Determine if the entire training procedure should stop.

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

    def _calc_emp_model(self):
        """Calculate empirical model parameters from training data."""
        dataset_s = self.train_loader_.dataset.S  # pylint: disable=invalid-name
        self.emp_canon_params = self.calc_suff_stat(dataset_s).mean(0)
        self.emp_theta = self._reparameterize_nat_form(self.emp_canon_params)
        self.register_parameter("theta", nn.Parameter(self.emp_theta))

    def forward(self, x):  # pylint: disable=unused-argument
        """Return the current theta parameter (input is unused).

        Args:
            x: input data (unused)
        """
        return self.theta

    def _constraints(self, theta):
        """Apply parameter constraints. Override in subclasses if needed."""
        return theta

    def step_post_hook(self, optimizer, args, kwargs) -> None:
        """Project theta back onto the L2 ball after each training update."""
        with ch.no_grad():
            proj_theta = self.emp_theta
            theta_diff = (self.theta - self.emp_theta)[..., None].norm()
            if theta_diff > self.radius:
                theta_diff = theta_diff.renorm(
                    p=2, dim=0, maxnorm=self.radius
                ).flatten()
                proj_theta = self.emp_theta + theta_diff

            self.theta.copy_(self._constraints(proj_theta))

    def _reparameterize_nat_form(self, theta):
        """Convert canonical parameters to natural form. Override in subclasses."""
        return theta

    def _reparameterize_canon_form(self, theta):
        """Convert natural parameters to canonical form. Override in subclasses."""
        return theta

    def __str__(self):
        return "truncated exponential family distribution"
