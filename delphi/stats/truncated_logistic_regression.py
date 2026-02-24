# Author: pstefanou12@
"""
Truncated Logistic Regression.
"""

import warnings
from typing import Callable

import torch as ch
from sklearn.linear_model import LogisticRegression
from torch import Tensor
from torch.nn import Sigmoid, Softmax

from delphi.delphi_logger import delphiLogger
from delphi.grad import TruncatedBCE, TruncatedCE
from delphi.trainer import Trainer
from delphi.utils.datasets import make_train_and_val
from delphi.utils.defaults import TRUNC_LOG_REG_DEFAULTS, check_and_fill_args
from delphi.utils.helpers import Parameters
from delphi.stats.linear_model import LinearModel


# Module-level constants.
softmax = Softmax(dim=-1)
sig = Sigmoid()
OVR = "ovr"
MULTI = "multinomial"
CLASSIFICATION_PROCEDURES = [OVR, MULTI]


class TruncatedLogisticRegression(LinearModel):  # pylint: disable=too-many-instance-attributes
    """Truncated logistic regression supporting binary and multinomial classification."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        args: Parameters,
        phi: Callable,
        alpha: float,
        fit_intercept: bool = True,
        multi_class: str = "ovr",
        emp_weight: ch.Tensor = None,
        rand_seed: int = 0,
    ):
        """Initialize TruncatedLogisticRegression.

        Args:
            args (Parameters): hyperparameter object
            phi (Callable): oracle object for the truncated regression model
            alpha (float): survival probability for the truncated regression model
            fit_intercept (bool): whether to fit an intercept term
            multi_class (str): ``"ovr"`` for binary or ``"multinomial"`` for
                multi-class classification
            emp_weight (Tensor): optional empirical weight initialization
            rand_seed (int): random seed for reproducibility
        """
        logger = delphiLogger()
        args = check_and_fill_args(args, TRUNC_LOG_REG_DEFAULTS)
        super().__init__(args, False, logger, emp_weight=emp_weight)
        self.phi = phi
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        assert multi_class in CLASSIFICATION_PROCEDURES, (
            f"{multi_class} not in {CLASSIFICATION_PROCEDURES}"
        )
        self.multi_class = multi_class
        self.rand_seed = rand_seed

        del self.criterion
        del self.criterion_params
        if self.multi_class == OVR:
            self.criterion = TruncatedBCE.apply
        else:
            self.criterion = TruncatedCE.apply
        self.criterion_params = [self.phi, self.args.num_samples, self.args.eps]

    def fit(self, X: Tensor, y: Tensor):  # pylint: disable=invalid-name
        """
        Train truncated logistic regression model by running PSGD on the truncated negative
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
        assert X.size(0) == y.size(0), (
            f"number of samples in X and y is unequal. "
            f"X has {X.size(0)} samples, and y has {y.size(0)} samples"
        )
        assert y.dim() == 2 and y.size(1), (
            f"y is size: {y.size()}. expecting y tensor with size num_samples by 1."
        )

        unique_classes = len(ch.unique(y))
        assert unique_classes > 1, (
            "y contains only 1 unique class. "
            "2+ uniques classes are required for classification procedures"
        )
        if self.multi_class == OVR:
            self.K = 1  # pylint: disable=invalid-name,attribute-defined-outside-init
        elif self.multi_class == MULTI:
            self.K = unique_classes  # pylint: disable=invalid-name,attribute-defined-outside-init

        # Add one feature column to X when fitting an intercept.
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)  # pylint: disable=invalid-name
        self.D = X.size(1)  # pylint: disable=invalid-name,attribute-defined-outside-init

        self.train_loader, self.val_loader = make_train_and_val(self.args, X, y)  # pylint: disable=attribute-defined-outside-init

        self.trainer = Trainer(self, self.args, self.logger)  # pylint: disable=attribute-defined-outside-init
        self.trainer.train_model(self.train_loader, self.val_loader)

        return self

    def _calc_emp_model(self):
        """Calculate empirical logistic regression estimates using SKlearn module."""
        X, y = self.train_loader.dataset.tensors  # pylint: disable=invalid-name
        if self._emp_weight is None and self.multi_class == "ovr":  # pylint: disable=access-member-before-definition
            log_reg = LogisticRegression(
                penalty=None, fit_intercept=False, multi_class=self.multi_class
            )
            log_reg.fit(X, y.flatten())
            self._emp_weight = ch.from_numpy(log_reg.coef_).float()  # pylint: disable=attribute-defined-outside-init
        elif self._emp_weight is None:
            self._emp_weight = ch.randn(self.K, self.D)  # pylint: disable=attribute-defined-outside-init
        self.register_parameter("weight", ch.nn.Parameter(self._emp_weight.clone()))

    def pretrain_hook(self):
        """
        Set up empirical model and projection set before training.

        SkLearn sets up multinomial classification differently. So when doing
        multinomial classification, we initialize with random estimates.
        """
        # Calculate empirical estimates for the truncated linear model.
        self._calc_emp_model()
        self.radius = self.args.r * self.base_radius  # pylint: disable=attribute-defined-outside-init

    def forward(self, X):  # pylint: disable=invalid-name
        """Compute logit predictions for input features X."""
        return X @ self.weight.T

    def post_step_hook(self, i, loop_type, loss, batch):
        """No-op post-step hook; override in subclasses if needed."""

    def post_training_hook(self):  # pylint: disable=attribute-defined-outside-init
        """Process and store results after training completes."""
        self.args.r *= self.args.rate
        best_params = self.trainer.best_params.reshape(self.weight.size())
        final_params = self.trainer.final_params.reshape(self.weight.size())
        if self.fit_intercept:
            self.best_coef = best_params[:, :-1]
            self.best_intercept = best_params[:, -1]

            self.final_coef = final_params[:, :-1]
            self.final_intercept = final_params[:, -1]

        else:
            self.best_coef = best_params
            self.final_coef = final_params

    def predict_proba(self, X: Tensor):  # pylint: disable=invalid-name
        """Probability predictions for input features."""
        if self.fit_intercept:
            logits = ch.cat([X, ch.ones(X.size(0), 1)], axis=1) @ self.best_coef.T
        else:
            logits = X @ self.best_coef.T
        if self.multi_class == MULTI:
            return softmax(logits)
        return sig(logits)

    def predict(self, X: Tensor):  # pylint: disable=invalid-name
        """Class predictions for input features."""
        prob_predictions = self.predict_proba(X)
        if prob_predictions.size(-1) > 1:
            return prob_predictions.argmax(-1)
        return prob_predictions > 0.5

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
