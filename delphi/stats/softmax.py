# Author: pstefanou12@
"""
Multinomial logistic regression that uses softmax loss function.
"""

import warnings

import torch as ch
from torch import Tensor
from torch.nn import Softmax, CrossEntropyLoss

from delphi.trainer import Trainer
from delphi.utils.helpers import Parameters
from delphi.utils.datasets import make_train_and_val
from delphi.stats.linear_model import LinearModel

# Module-level constants.
softmax = Softmax(dim=1)
ce = CrossEntropyLoss()


class SoftmaxRegression(LinearModel):  # pylint: disable=too-many-instance-attributes,abstract-method
    """Softmax regression using cross-entropy loss, for the trainer framework.

    Attributes:
        fit_intercept (bool): Whether to fit an intercept term.
        criterion: Cross-entropy loss function.
    """

    def __init__(self, args: Parameters, fit_intercept: bool = True):
        super().__init__(args, dependent=False)  # pylint: disable=no-value-for-parameter
        self.fit_intercept = fit_intercept
        del self.criterion

        self.criterion = ce

        self.d, self.k = None, None  # pylint: disable=invalid-name

    def fit(self, X, y):  # pylint: disable=invalid-name,attribute-defined-outside-init
        """Train the softmax regression model on labeled data.

        Args:
            X (Tensor): input feature matrix of shape (n, d)
            y (Tensor): target class labels of shape (n, 1)
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

        y = y.flatten()
        unique_classes = len(ch.unique(y))
        assert unique_classes > 1, (
            "y contains only 1 unique class. 2+ uniques classes are required for classification procedures"
        )

        # Add one feature column to X when fitting an intercept.
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], axis=1)

        self.d, self.k = X.size(1), unique_classes
        self.train_loader, self.val_loader = make_train_and_val(self.args, X, y)

        self.trainer = Trainer(self, self.args, self.logger)  # pylint: disable=no-value-for-parameter
        self.trainer.train_model(self.train_loader, self.val_loader)

        return self

    def pretrain_hook(self):  # pylint: disable=attribute-defined-outside-init
        """Set up model weight parameter before training."""
        weight = ch.nn.Parameter(ch.randn(self.k, self.d))
        self.register_parameter("weight", weight)

    def predict(self, x):
        """Make class predictions using trained model."""
        with ch.no_grad():
            return softmax(x @ self.best_coef.T).argmax(dim=-1)

    def __call__(self, X, y):
        """
        Compute logit predictions for a batch.

        Args:
            X (Tensor): input features
            y (Tensor): target class labels (unused; present for API consistency)

        Returns:
            Logit predictions of shape (num_samples, k).
        """
        return X @ self.weight.T

    def post_training_hook(self):  # pylint: disable=attribute-defined-outside-init
        """Process and store results after training completes."""
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
