# Author: pstefanou12@
"""
Truncated Probit Regression.
"""

import math
import warnings
from collections.abc import Callable

from statsmodels.discrete import discrete_model
import torch as ch

from delphi import delphi_logger, trainer
from delphi.stats import linear_model, losses
from delphi.utils import datasets, defaults, helpers


class TruncatedProbitRegression(linear_model.LinearModel):  # pylint: disable=too-many-instance-attributes
    """Truncated probit regression for binary classification with N(0,1) latent noise."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        args: helpers.Parameters,
        phi: Callable,
        alpha: float,
        fit_intercept: bool = True,
        emp_weight: ch.Tensor = None,
        rand_seed: int = 0,
    ):
        """Initialize TruncatedProbitRegression.

        Args:
            args (Parameters): hyperparameter object
            phi (Callable): oracle object for the truncated regression model
            alpha (float): survival probability for the truncated regression model
            fit_intercept (bool): whether to fit an intercept term
            emp_weight (Tensor): optional empirical weight initialization
            rand_seed (int): random seed for reproducibility
        """
        logger = delphi_logger.delphiLogger()
        args = defaults.check_and_fill_args(args, defaults.TRUNC_PROB_REG_DEFAULTS)
        super().__init__(args, False, logger, emp_weight=emp_weight)
        self.phi = phi
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.rand_seed = rand_seed

        del self.criterion
        del self.criterion_params
        self.criterion = losses.TruncatedProbitMLE.apply
        self.criterion_params = [self.phi, self.args.num_samples, self.args.eps]

    def fit(  # pylint: disable=attribute-defined-outside-init,invalid-name
        self, X: ch.Tensor, y: ch.Tensor
    ):
        """
        Train truncated probit regression model by running PSGD on the truncated negative
        population log likelihood.

        Args:
            X (torch.Tensor): input feature covariates num_samples by dims
            y (torch.Tensor): dependent variable predictions num_samples by 1
        """
        assert isinstance(X, ch.Tensor), (
            f"X is type: {type(X)}. expected type torch.Tensor."
        )
        assert isinstance(y, ch.Tensor), (
            f"y is type: {type(y)}. expected type torch.Tensor."
        )
        assert X.size(0) > X.size(1), (
            "number of dimensions, larger than number of samples. "
            "procedure expects matrix with size num samples by num feature dimensions."
        )
        assert y.dim() == 2 and y.size(1) == 1, (
            f"y is size: {y.size()}. expecting y tensor with size num_samples by 1."
        )
        # Add one feature column to X when fitting an intercept.
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)

        self.train_loader, self.val_loader = datasets.make_train_and_val(
            self.args, X, y
        )

        self.trainer = trainer.Trainer(self, self.args, self.logger)
        # Run PGD for parameter estimation.
        self.trainer.train_model(self.train_loader, self.val_loader)

        return self

    def calc_emp_model(self):  # pylint: disable=attribute-defined-outside-init
        """
        Calculate empirical probit regression estimates using statsmodels module.
        Probit MLE.
        """
        if self.emp_weight is None:
            X, y = self.train_loader.dataset.tensors  # pylint: disable=invalid-name

            # Empirical estimates for probit regression.
            self.emp_prob_reg = discrete_model.Probit(y.numpy(), X.numpy()).fit()

            self.emp_weight = ch.nn.Parameter(
                ch.from_numpy(self.emp_prob_reg.params)[..., None].float()
            )
        else:
            self.emp_weight = ch.nn.Parameter(self.emp_weight)
        self.register_parameter("weight", self.emp_weight)

    def pretrain_hook(self):  # pylint: disable=attribute-defined-outside-init
        """Set up empirical model and projection set before training."""
        self.calc_emp_model()
        # Projection set radius.
        self.radius = self.args.r * (math.sqrt(math.log(1.0 / self.alpha)))

    def forward(self, X):  # pylint: disable=invalid-name
        """Compute linear predictions X @ weight."""
        return X @ self.weight

    def post_step_hook(self, i, loop_type, loss, batch):  # pylint: disable=unused-argument
        """No-op post-step hook; override in subclasses if needed."""

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

    def predict(self, X: ch.Tensor):  # pylint: disable=invalid-name
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
