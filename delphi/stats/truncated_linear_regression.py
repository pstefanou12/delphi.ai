"""
Truncated Linear Regression.
"""

import math
import warnings
from typing import Callable

import torch as ch
from torch import nn
from scipy.linalg import lstsq
from torch import Tensor

from ..delphi_logger import delphiLogger
from ..grad import SwitchGrad, TruncatedMSE, TruncatedUnknownVarianceMSE
from ..trainer import Trainer
from ..utils.datasets import make_train_and_val
from ..utils.defaults import (
    TRUNC_LDS_DEFAULTS,
    TRUNC_REG_DEFAULTS,
    check_and_fill_args,
)
from ..utils.helpers import Bounds, Parameters
from .linear_model import LinearModel


class TruncatedLinearRegression(  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    LinearModel
):
    """
    Truncated linear regression class. Supports truncated linear regression
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
        fit_intercept: bool = True,
        noise_var: ch.Tensor = None,
        dependent: bool = False,
        emp_weight: ch.Tensor = None,
        rand_seed=0,
    ):
        """
        Initialize TruncatedLinearRegression.

        Args:
            phi (delphi.oracle.oracle) : oracle object for truncated regression model
            alpha (float) : survival probability for truncated regression model
            fit_intercept (bool) : boolean indicating whether to fit a intercept or not
            val (int) : number of samples to use for validation set
            tol (float) : gradient tolerance threshold
            workers (int) : number of workers to spawn
            r (float) : size for projection set radius
            rate (float): rate at which to increase the size of the projection set
            num_samples (int) : number of samples to sample in gradient
            batch_size (int) : batch size
            lr (float) : initial learning rate for regression weight parameters
            var_lr (float) : initial learning rate for variance parameter
            step_lr (int) : number of gradient steps before decaying learning rate
            custom_lr_multiplier (str) : "cosine", "adam" - different lr schedulers
            lr_interpolation (str) : "linear" linear interpolation
            step_lr_gamma (float) : amount to decay learning rate for step lr
            momentum (float) : momentum for SGD optimizer
            eps (float) : epsilon value for gradient to prevent zero in denominator
            dependent (bool) : whether dataset is dependent (run SwitchGrad)
            store (cox.store.Store) : cox store object for logging
        """
        logger = delphiLogger()
        if dependent:
            args = check_and_fill_args(args, TRUNC_LDS_DEFAULTS)
            super().__init__(args, dependent, logger, emp_weight=emp_weight)
            setattr(self.args, "lr", (2 / alpha) ** self.args.c_eta)
        else:
            args = check_and_fill_args(args, TRUNC_REG_DEFAULTS)
            super().__init__(args, dependent, logger, emp_weight=emp_weight)
        self.phi = phi
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.noise_var = noise_var
        self.rand_seed = rand_seed
        if self.dependent:
            assert self.noise_var is not None, (
                "if linear dynamical system, noise variance must be known"
            )

        del self.criterion
        del self.criterion_params
        if self.dependent:
            self.criterion = SwitchGrad.apply
        elif self.noise_var is None:
            self.criterion = TruncatedUnknownVarianceMSE.apply
        else:
            self.criterion = TruncatedMSE.apply
            self.criterion_params = [
                self.phi,
                self.noise_var,
                self.args.num_samples,
                self.args.eps,
            ]

        # property instance variables
        self.coef, self.intercept = None, None

    def fit(  # pylint: disable=attribute-defined-outside-init
        self,
        X: Tensor,
        y: Tensor,  # pylint: disable=invalid-name
    ):
        """
        Train truncated linear regression model by running PSGD on the truncated negative
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
        assert y.dim() == 2 and y.size(1) <= X.size(1), (
            f"y is size: {y.size()}. expecting y tensor to have y.size(1) < X.size(1)."
        )
        if self.noise_var is not None:
            assert self.noise_var.size(0) == y.size(1), (
                f"noise var size is: {self.noise_var.size(0)}. "
                f"y size is: {y.size(1)}. "
                "expecting noise_var.size(0) == y.size(1)"
            )

        # add number of samples to args
        setattr(self.args, "T", X.size(0))
        if self.dependent:
            self.criterion_params = [
                self.phi,
                self.args.c_gamma,
                self.alpha,
                self.args.T,
                self.noise_var,
                self.args.num_samples,
                self.args.eps,
            ]

        k = X.size(1)
        # Normalization factor: B * sqrt(k)
        # Compute B = maximum L-inf norm across all samples
        B = X.norm(dim=1, p=float("inf")).max()  # pylint: disable=invalid-name
        self.beta = B * (k**0.5)
        X = X / self.beta  # pylint: disable=invalid-name

        # add one feature to x when fitting intercept
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], dim=1)  # pylint: disable=invalid-name

        self.train_loader, self.val_loader = make_train_and_val(self.args, X, y)
        self.trainer = Trainer(self, self.args, self.logger)
        self.trainer.train_model(
            self.train_loader, self.val_loader, rand_seed=self.rand_seed
        )
        return self

    def pretrain_hook(self):  # pylint: disable=attribute-defined-outside-init
        """Set up empirical model and projection set before training."""
        self.calc_emp_model()
        # use OLS as empirical estimate to define projection set
        self.radius = self.args.r * self.base_radius
        # empirical estimates for projection set
        # generate noise variance radius bounds if unknown
        if self.noise_var is None:
            lower_bound = self.emp_noise_var / (8 * (5 + 2 * math.log(1 / self.alpha)))
            self.var_bounds = Bounds(lower_bound, self.emp_noise_var + self.radius)

            lambda_ = self.emp_noise_var.clone().inverse()
            v = self.emp_weight * lambda_
            self.register_parameter("v", nn.Parameter(v))
            self.register_parameter("lambda_", nn.Parameter(lambda_))

            self.criterion_params = [
                self.lambda_,
                self.phi,
                self.args.num_samples,
                self.args.eps,
            ]
        else:
            self.register_parameter("weight", nn.Parameter(self.emp_weight.clone()))

    def calc_emp_model(self) -> None:  # pylint: disable=attribute-defined-outside-init
        """
        Calculate empirical estimates for a truncated linear model. Assigns
        estimates to a Linear layer. By default calculates OLS for truncated linear
        regression.
        """
        X, y = self.train_loader.dataset.tensors  # pylint: disable=invalid-name
        coef_, _, self.rank_, self.singular_ = lstsq(X, y)
        self.ols_coef_ = Tensor(coef_)
        self.emp_noise_var = ch.var(Tensor(X @ coef_) - y, dim=0)[..., None]

        if self.emp_weight is None:
            self.emp_weight = self.ols_coef_

        if self.dependent:

            def calc_sigma_0(mat):  # pylint: disable=invalid-name
                """Calculate the sum of outer products."""
                return ch.bmm(
                    mat.view(mat.size(0), mat.size(1), 1),
                    mat.view(mat.size(0), 1, mat.size(1)),
                ).sum(0)

            XXT_sum = calc_sigma_0(X)  # pylint: disable=invalid-name
            Sigma_0 = (1 / (self.s * len(X))) * XXT_sum  # pylint: disable=invalid-name
            self.register_buffer("Sigma_0", Sigma_0)
            assert ch.det(self.Sigma_0) != 0, "Sigma_0 is singular and non-invertible"  # pylint: disable=no-member
            self.register_buffer("Sigma", self.Sigma_0.clone())  # pylint: disable=no-member

    def post_training_hook(self):  # pylint: disable=too-many-branches,attribute-defined-outside-init
        """Process and store results after training completes."""
        if self.args.r is not None:
            self.args.r *= self.args.rate
        best_params = self.trainer.best_params
        final_params = self.trainer.final_params
        if self.args.r is not None:
            self.args.r *= self.args.rate
        # remove model from computation graph
        if self.noise_var is None:
            v = self.v.data
            lambda_ = self.lambda_.data

            self.final_variance = 1.0 / lambda_
            self.final_weight = v * self.final_variance

            if self.fit_intercept:
                best_lambda_ = best_params[-1]
                self.best_variance = 1 / best_lambda_
                self.best_coef = (best_params[:-2] * self.best_variance) / self.beta
                self.best_intercept = best_params[-2] * self.best_variance
                final_lambda_ = final_params[-1]
                self.final_variance = 1 / final_lambda_
                self.final_coef = (final_params[:-2] * self.final_variance) / self.beta
                self.final_intercept = final_params[-2] * self.final_variance
                self.emp_weight /= self.beta
            else:
                best_lambda_ = best_params[-1]
                self.best_variance = 1 / best_lambda_
                self.best_coef = (best_params[:-1] * self.best_variance) / self.beta
                final_lambda_ = final_params[-1]
                self.final_variance = 1 / final_lambda_
                self.final_coef = (final_params[:-1] * self.final_variance) / self.beta
                self.emp_weight /= self.beta

        # assign results from procedure to instance variables
        else:
            if self.fit_intercept:
                self.best_coef = best_params[:-1] / self.beta
                self.best_intercept = best_params[-1]
                self.final_coef = final_params[:-1] / self.beta
                self.final_intercept = final_params[-1]
                self.emp_weight /= self.beta
            else:
                self.best_coef = best_params[:] / self.beta
                self.final_coef = final_params[:] / self.beta
                self.emp_weight /= self.beta

    def predict(self, X: Tensor):  # pylint: disable=invalid-name
        """Make predictions with regression estimates."""
        assert self.coef is not None, "must fit model before using predict method"
        if self.fit_intercept:
            return X @ self.coef + self.intercept
        return X @ self.coef

    def loss(self, X: Tensor, y: Tensor) -> Tensor:  # pylint: disable=invalid-name
        """Compute loss for given inputs."""
        with ch.no_grad():
            return self.criterion(self.predict(X), y, *self.criterion_params)

    def emp_nll(self, X: Tensor, y: Tensor) -> Tensor:  # pylint: disable=invalid-name
        """Compute empirical negative log-likelihood."""
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)  # pylint: disable=invalid-name
        with ch.no_grad():
            return self.criterion(X @ self.emp_weight, y, *self.criterion_params)

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

    @property
    def best_variance_(self):
        """
        Noise variance prediction for linear regression with
        unknown noise variance algorithm.
        """
        if self.noise_var is None:
            return self.best_variance
        warnings.warn(
            "no variance prediction because regression with known variance was run"
        )
        return None

    @property
    def final_variance_(self):
        """
        Noise variance prediction for linear regression with
        unknown noise variance algorithm.
        """
        if self.noise_var is None:
            return self.final_variance
        warnings.warn(
            "no variance prediction because regression with known variance was run"
        )
        return None

    @property
    def ols_coef_(self):
        """OLS empirical estimates for coefficients."""
        return self._ols_coef_

    @ols_coef_.setter
    def ols_coef_(self, value):
        """Set OLS empirical estimates for coefficients."""
        self._ols_coef_ = value

    @property
    def ols_intercept_(self):
        """OLS empirical estimates for intercept."""
        return self.trunc_reg.emp_bias.clone()

    @property
    def ols_variance_(self):
        """OLS empirical estimates for noise variance."""
        return self.trunc_reg.emp_var.clone()

    def forward(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Forward pass through the model."""
        if self.noise_var is None:
            return X @ self.v * 1.0 / self.lambda_

        if self.dependent:
            self.Sigma += ch.bmm(  # pylint: disable=no-member,invalid-name
                X.view(X.size(0), X.size(1), 1), X.view(X.size(0), 1, X.size(1))
            ).mean(0)

        return X @ self.weight

    def step_pre_hook(self, optimizer, args, kwargs):
        """Pre-step hook for gradient modification."""
        if self.dependent:
            self.weight.grad = self.Sigma.inverse() @ self.weight.grad  # pylint: disable=no-member

        return args, kwargs

    def step_post_hook(self, optimizer, args, kwargs):
        """Post-step hook for parameter projection."""
        if self.noise_var is None:
            # project model parameters back to domain
            var = 1.0 / self.lambda_
            self.lambda_.data = 1.0 / ch.clamp(
                var, self.var_bounds.lower, self.var_bounds.upper
            )
        return args, kwargs

    def parameters_(self):
        """Return parameter groups for optimizer."""
        if self.noise_var is None and self.args.var_lr is not None:
            return [
                {"params": self.v, "lr": self.args.lr},
                {"params": self.lambda_, "lr": self.args.var_lr},
            ]
        return super().parameters(recurse=True)
