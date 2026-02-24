"""
Truncated Probit Regression.
"""

import math
import warnings
from typing import Callable

import torch as ch
from statsmodels.discrete.discrete_model import Probit
from torch import Tensor

from delphi.delphi_logger import delphiLogger
from delphi.grad import TruncatedProbitMLE
from delphi.trainer import Trainer
from delphi.utils.datasets import make_train_and_val
from delphi.utils.defaults import TRUNC_PROB_REG_DEFAULTS, check_and_fill_args
from delphi.utils.helpers import Parameters
from delphi.stats.linear_model import LinearModel


class TruncatedProbitRegression(LinearModel):  # pylint: disable=too-many-instance-attributes
    """
    Truncated Probit Regression supports binary classification when the noise
    distribution in the latent variable model is N(0, 1).
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        args: Parameters,
        phi: Callable,
        alpha: float,
        fit_intercept: bool = True,
        emp_weight: ch.Tensor = None,
        rand_seed: int = 0,
    ):
        """
        Args:
            phi (delphi.oracle.oracle) : oracle object for truncated regression model
            alpha (float) : survival probability for truncated regression model
            fit_intercept (bool) : whether to fit an intercept
            emp_weight: empirical weight initialization
            rand_seed: random seed
        """
        logger = delphiLogger()
        args = check_and_fill_args(args, TRUNC_PROB_REG_DEFAULTS)
        super().__init__(args, False, logger, emp_weight=emp_weight)
        self.phi = phi
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.rand_seed = rand_seed

        del self.criterion
        del self.criterion_params
        self.criterion = TruncatedProbitMLE.apply
        self.criterion_params = [self.phi, self.args.num_samples, self.args.eps]

    def fit(  # pylint: disable=attribute-defined-outside-init,invalid-name
        self, X: Tensor, y: Tensor
    ):
        """
        Train truncated probit regression model by running PSGD on the truncated negative
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
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)

        self.train_loader, self.val_loader = make_train_and_val(self.args, X, y)

        self.trainer = Trainer(self, self.args, self.logger)
        # run PGD for parameter estimation
        self.trainer.train_model(self.train_loader, self.val_loader)

        return self

    def calc_emp_model(self):  # pylint: disable=attribute-defined-outside-init
        """
        Calculate empirical probit regression estimates using statsmodels module.
        Probit MLE.
        """
        if self.emp_weight is None:
            X, y = self.train_loader.dataset.tensors  # pylint: disable=invalid-name

            # empirical estimates for probit regression
            self.emp_prob_reg = Probit(y.numpy(), X.numpy()).fit()

            self.emp_weight = ch.nn.Parameter(
                ch.from_numpy(self.emp_prob_reg.params)[..., None].float()
            )
        else:
            self.emp_weight = ch.nn.Parameter(self.emp_weight)
        self.register_parameter("weight", self.emp_weight)

    def pretrain_hook(self):  # pylint: disable=attribute-defined-outside-init
        """Set up empirical model and projection set before training."""
        self.calc_emp_model()
        # projection set radius
        self.radius = self.args.r * (math.sqrt(math.log(1.0 / self.alpha)))

    def forward(self, X):  # pylint: disable=invalid-name
        """
        Forward pass through the model.

        Args:
            X: input features
        """
        return X @ self.weight

    def post_step_hook(self, i, loop_type, loss, batch):  # pylint: disable=unused-argument
        """
        Iteration hook for defined model. Method is called after each training update.

        Args:
            i (int) : gradient step or epoch number
            loop_type (str) : 'train' or 'val'; indicating type of loop
            loss (ch.Tensor) : loss for that iteration
            batch: batch data
        """

    def post_training_hook(self):  # pylint: disable=attribute-defined-outside-init
        """Process and store results after training completes."""
        self.args.r *= self.args.rate
        best_params = self.trainer.best_params
        final_params = self.trainer.final_params
        if self.fit_intercept:
            self.best_coef = best_params[:, :-1]
            self.best_intercept = best_params[:, -1]

            self.final_coef = final_params[:, :-1]
            self.final_intercept = final_params[:, -1]

        else:
            self.best_coef = best_params
            self.final_coef = final_params

    def predict(self, X: Tensor):  # pylint: disable=invalid-name
        """Make class predictions with regression estimates."""
        if self.fit_intercept:
            logits = ch.cat([X, ch.ones(X.size(0), 1)], axis=1) @ self.weight
        else:
            logits = X @ self.weight
        return 0.5 * (1 + ch.erf(logits / ch.sqrt(ch.Tensor([2.0])))) > 0.5

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
        """Regression weight."""
        return self.best_coef

    @property
    def intercept_(self):
        """Regression intercept."""
        if self.fit_intercept:
            return self.best_intercept
        warnings.warn("intercept not fit, check args input.")
        return None
