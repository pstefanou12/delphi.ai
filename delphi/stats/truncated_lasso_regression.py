"""
Truncated Lasso Regression.
"""

import warnings
from typing import Callable

import numpy as np
import torch as ch
from torch import nn
from sklearn.linear_model import LassoCV
from torch import Tensor

from ..delphi_logger import delphiLogger
from ..grad import TruncatedMSE
from ..trainer import Trainer
from ..utils.datasets import make_train_and_val
from ..utils.defaults import TRUNC_LASSO_DEFAULTS, check_and_fill_args
from ..utils.helpers import Parameters
from .linear_model import LinearModel


class TruncatedLassoRegression(LinearModel):  # pylint: disable=too-many-instance-attributes
    """
    Truncated LASSO regression class. Supports truncated LASSO regression
    with known noise, unknown noise, and confidence intervals. Module uses
    delphi.trainer.Trainer to train truncated linear regression by performing
    projected stochastic gradient descent on the truncated population log likelihood.
    Module requires the user to specify an oracle from the delphi.oracle.oracle class,
    and the survival probability.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        args: Parameters,
        phi: Callable,
        alpha: float,
        l1: float = 1.0,
        fit_intercept: bool = True,
        noise_var: float = 1.0,
        emp_weight: ch.Tensor = None,
        rand_seed: int = 0,
    ):
        """
        Initialize TruncatedLassoRegression.

        Args:
            phi (delphi.oracle.oracle) : oracle object for truncated regression model
            alpha (float) : survival probability for truncated regression model
            fit_intercept (bool) : boolean indicating whether to fit a intercept or not
            steps (int) : number of gradient steps to take
            val (int) : number of samples to use for validation set
            tol (float) : gradient tolerance threshold
            workers (int) : number of workers to spawn
            r (float) : size for projection set radius
            rate (float): rate at which to increase the size of the projection set
            num_samples (int) : number of samples to sample in gradient
            batch_size (int) : batch size
            lr (float) : initial learning rate for regression parameters
            var_lr (float) : initial learning rate for variance parameter
            step_lr (int) : number of gradient steps before decaying learning rate
            custom_lr_multiplier (str) : "cosine", "adam" - different lr schedulers
            lr_interpolation (str) : "linear" linear interpolation
            step_lr_gamma (float) : amount to decay learning rate for step lr
            momentum (float) : momentum for SGD optimizer
            l1 (float) : weight decay for SGD optimizer
            eps (float) : epsilon value for gradient to prevent zero in denominator
            store (cox.store.Store) : cox store object for logging
        """
        logger = delphiLogger()
        args = check_and_fill_args(args, TRUNC_LASSO_DEFAULTS)
        super().__init__(args, False, logger, emp_weight=emp_weight)
        self.phi = phi
        self.alpha = alpha
        self.l1 = l1
        self.fit_intercept = fit_intercept
        self.noise_var = noise_var
        self.rand_seed = rand_seed

        del self.criterion

        self.criterion = TruncatedMSE.apply
        self.criterion_params = [
            self.phi,
            self.noise_var,
            self.args.num_samples,
            self.args.eps,
        ]

    def fit(  # pylint: disable=attribute-defined-outside-init
        self,
        X: Tensor,
        y: Tensor,  # pylint: disable=invalid-name
    ):
        """
        Train truncated lasso regression model by running PSGD on the truncated negative
        population log likelihood.

        Args:
            X (torch.Tensor): input feature covariates num_samples by dims
            y (torch.Tensor): dependent variable predictions num_samples by 1
        """
        assert isinstance(X, Tensor), (
            f"X is type: {type(X)}. expected type torch.Tensor."
        )
        assert isinstance(y, Tensor), (
            f"y is type: {type(y)}. expected type torch.Tensor."
        )
        assert X.size(0) > X.size(1), (
            "number of dimensions, larger than number of samples. "
            "procedure expects matrix with size num samples by num feature dimensions."
        )
        assert y.dim() == 2 and y.size(1) == 1, (
            f"y is size: {y.size()}. expecting y tensor with size num_samples by 1."
        )
        # add one feature to x when fitting intercept
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)  # pylint: disable=invalid-name

        if self.fit_intercept:
            k = X.size(1) - 1
        else:
            k = X.size(1)

        # Normalization factor: B * sqrt(k)
        # Compute B = maximum L-inf norm across all samples
        B = X.norm(dim=1, p=float("inf")).max()  # pylint: disable=invalid-name
        self.beta = B * (k**0.5)

        # Normalize all features except intercept column
        if self.fit_intercept:
            X_normalized = X[:, :-1] / self.beta  # pylint: disable=invalid-name
            X = ch.cat([X_normalized, X[:, -1:]], dim=1)  # pylint: disable=invalid-name
        else:
            X = X / self.beta  # pylint: disable=invalid-name

        self.train_loader, self.val_loader = make_train_and_val(self.args, X, y)

        self.trainer = Trainer(self, self.args, self.logger)
        self.trainer.train_model(
            self.train_loader, self.val_loader, rand_seed=self.rand_seed
        )
        return self

    def pretrain_hook(self):
        """Set up empirical model and projection set before training."""
        self.calc_emp_model()
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * self.base_radius
        self.register_parameter("weight", nn.Parameter(self.emp_weight.clone()))

    def calc_emp_model(self) -> None:
        """Calculate empirical lasso regression estimates using sklearn."""
        X, y = self.train_loader.dataset.tensors  # pylint: disable=invalid-name
        emp_lasso = LassoCV(fit_intercept=self.fit_intercept, alphas=[self.l1])
        emp_lasso.fit(X, y)
        if self.args.fit_intercept:
            lasso_coef_ = ch.from_numpy(
                np.concatenate([emp_lasso.coef_.flatten(), emp_lasso.intercept_])
            )
        else:
            lasso_coef_ = ch.from_numpy(emp_lasso.coef_).float()[..., None]
        self.register_buffer("emp_weight", lasso_coef_)

    def forward(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Forward pass through the model."""
        return X @ self.weight

    def pre_step_hook(self, inp: ch.Tensor) -> None:
        """Pre-step hook to apply L1 regularization gradient."""
        # only regularize the weight coefficients and not intercept
        if self.fit_intercept:
            self.weight.grad[:-1] += (self.l1 * ch.sign(inp[:, :-1])).mean(0)[..., None]
        else:
            self.weight.grad += (self.l1 * ch.sign(inp)).mean(0)[..., None]

    def post_training_hook(self):  # pylint: disable=attribute-defined-outside-init
        """Process and store results after training completes."""
        best_params = self.trainer.best_params
        final_params = self.trainer.final_params
        if self.args.r is not None:
            self.args.r *= self.args.rate
        if self.fit_intercept:
            self.best_coef = best_params[:, :-1] / self.beta
            self.best_intercept = best_params[:, -1]
            self.final_coef = final_params[:, :-1] / self.beta
            self.final_intercept = final_params[:, -1]
            self.emp_weight /= self.beta
        else:
            self.best_coef = best_params[:] / self.beta
            self.final_coef = final_params[:] / self.beta
            self.emp_weight /= self.beta

    def predict(self, X: Tensor):  # pylint: disable=invalid-name
        """Make predictions with regression estimates."""
        assert self.coef is not None, "must fit model before using predict method"
        if self.fit_intercept:
            return X @ self.best_coef + self.best_intercept
        return X @ self.best_coef

    @property
    def best_coef_(self):
        """Best coefficient from training."""
        return self.best_coef

    @property
    def best_intercept_(self):
        """Best intercept from training."""
        if self.fit_intercept:
            return self.best_intercept
        warnings.warn("intercept not fit, check inputs.")
        return None

    @property
    def final_coef_(self):
        """Final coefficient from training."""
        return self.final_coef

    @property
    def final_intercept_(self):
        """Final intercept from training."""
        if self.fit_intercept:
            return self.final_intercept
        warnings.warn("intercept not fit, check inputs.")
        return None

    @property
    def coef_(self):
        """Regression coefficient weights."""
        return self.best_coef.clone()

    @property
    def intercept_(self):
        """Regression intercept."""
        if self.best_intercept:
            return self.best_intercept.clone()
        warnings.warn("intercept not fit, check args input.")
        return None
