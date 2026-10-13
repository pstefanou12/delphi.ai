# Author: pstefanou12@
"""
Truncated Ridge Regression.
"""

import warnings

import cox
from sklearn.linear_model import Ridge
import torch as ch
from torch import Tensor

from delphi import configs
from delphi.delphi_logger import delphiLogger
from delphi.stats.stats import stats
from delphi.stats.truncated_linear_regression import (
    TruncatedLinearRegression,  # pylint: disable=unused-import
)
from delphi.trainer import Trainer
from delphi.utils.datasets import make_train_and_val


class TruncatedRidgeRegression(stats):  # pylint: disable=too-many-instance-attributes
    """Truncated ridge regression via projected SGD on the truncated log-likelihood."""

    def __init__(self, hyperparameters: dict, store: cox.store.Store = None):
        """Initialize TruncatedRidgeRegression.

        Args:
            hyperparameters: hyperparameter dict
            store (cox.store.Store): optional cox store for logging
        """
        super().__init__()
        # Instance variables.
        assert store is None or isinstance(store, cox.store.Store), (
            f"store is type: {type(store)}. expecting cox.store.Store."
        )
        self.store = store
        self.trunc_ridge = None
        # Algorithm hyperparameters.
        self.hyperparameters = configs.TruncatedRidgeRegressionConfig(**hyperparameters)
        if self.hyperparameters.weight_decay <= 0:
            raise ValueError(
                "weight_decay must be nonzero for this estimator; got "
                f"{self.hyperparameters.weight_decay}."
            )

    def fit(  # pylint: disable=attribute-defined-outside-init
        self,
        X: Tensor,
        y: Tensor,  # pylint: disable=invalid-name
    ):
        """
        Train truncated ridge regression model by running PSGD on the truncated negative
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

        self.train_loader_, self.val_loader_ = make_train_and_val(
            self.hyperparameters, X, y
        )

        if self.hyperparameters.noise_var is None:
            self.trunc_ridge = RidgeUnknownVariance(
                self.hyperparameters, self.train_loader_
            )
        else:
            self.trunc_ridge = RidgeKnownVariance(
                self.hyperparameters, self.train_loader_
            )

        # Run PGD for parameter estimation.
        trainer = Trainer(
            self.trunc_ridge, self.hyperparameters, delphiLogger(), store=self.store
        )
        trainer.train_model(self.train_loader_, self.val_loader_)

        # Assign results from the procedure to instance variables.
        self.coef = self.trunc_ridge.model.weight.clone()  # pylint: disable=not-callable
        if self.hyperparameters.fit_intercept:
            self.intercept = self.trunc_ridge.model.bias.clone()  # pylint: disable=not-callable
        if self.hyperparameters.noise_var is None:
            self.variance = self.trunc_ridge.lambda_.clone().inverse()
            self.coef *= self.variance
            if self.hyperparameters.fit_intercept:
                self.intercept *= self.variance.flatten()
        return self

    def predict(self, x: Tensor):
        """Make predictions with regression estimates."""
        return self.trunc_ridge.model(x)  # pylint: disable=not-callable

    @property
    def coef_(self):
        """Regression weight."""
        return self.coef

    @property
    def intercept_(self):
        """Regression intercept."""
        if self.hyperparameters.fit_intercept:
            return self.intercept
        warnings.warn("intercept not fit, check args input.")
        return None


class RidgeKnownVariance(TruncatedLinearRegression):  # pylint: disable=too-few-public-methods
    """Truncated ridge regression with known noise variance model."""

    def __init__(self, args, train_loader):
        """Initialize RidgeKnownVariance.

        Args:
            args: validated hyperparameter config
            train_loader: training data loader
        """
        super().__init__(args, train_loader)  # pylint: disable=no-value-for-parameter

    def calc_emp_model(self):  # pylint: disable=attribute-defined-outside-init,no-member
        """Calculate empirical ridge regression estimates."""
        self.emp_model = Ridge(
            fit_intercept=self.hyperparameters.fit_intercept,
            alpha=self.hyperparameters.weight_decay,
        ).fit(self.X, self.y.flatten())  # pylint: disable=invalid-name
        self.emp_weight = Tensor(self.emp_model.coef_)[None, ...]
        if self.hyperparameters.fit_intercept:
            self.emp_bias = Tensor([self.emp_model.intercept_])
        self.emp_var = ch.var(
            Tensor(self.emp_model.predict(self.X))[..., None] - self.y, dim=0
        )[..., None]


class RidgeUnknownVariance(TruncatedLinearRegression):  # pylint: disable=too-few-public-methods
    """Truncated ridge regression with unknown noise variance model."""

    def __init__(self, args, train_loader):
        """Initialize RidgeUnknownVariance.

        Args:
            args: validated hyperparameter config
            train_loader: training data loader
        """
        super().__init__(args, train_loader)  # pylint: disable=no-value-for-parameter

    def calc_emp_model(self):  # pylint: disable=attribute-defined-outside-init,no-member
        """Calculate empirical ridge regression estimates."""
        self.emp_model = Ridge(
            fit_intercept=self.hyperparameters.fit_intercept,
            alpha=self.hyperparameters.weight_decay,
        ).fit(self.X, self.y.flatten())  # pylint: disable=invalid-name
        self.emp_weight = Tensor(self.emp_model.coef_)[None, ...]
        if self.hyperparameters.fit_intercept:
            self.emp_bias = Tensor([self.emp_model.intercept_])
        self.emp_var = ch.var(
            Tensor(self.emp_model.predict(self.X))[..., None] - self.y, dim=0
        )[..., None]
