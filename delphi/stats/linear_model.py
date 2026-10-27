# Author: pstefanou12@
"""Linear model class for delphi.

A linear model predicts ``h_W(x) = W x + b`` with ``W`` of shape (k, d): one
output for regression and binary classification, k logits for multiclass,
and k = d for a linear dynamical system.
"""

import logging
from typing import Generic, Optional, TypeVar

import torch as ch
import torch.nn

from delphi import configs
from delphi.statistical_model import StatisticalModel

_LinearConfigT = TypeVar("_LinearConfigT", bound=configs.LinearModelConfig)
_T = TypeVar("_T")

ESTIMATORS = ("avg", "ema", "final", "best")


def require_fitted(value: Optional[_T], name: str) -> _T:
    """Return value, which fit sets.

    Args:
        value: The attribute's value.
        name: The attribute's name, for the error.

    Raises:
        RuntimeError: If value is None, as before fit.
    """
    if value is None:
        raise RuntimeError(f"{name} is set by fit; fit the model first.")
    return value


class LinearModel(  # pylint: disable=abstract-method,too-many-instance-attributes
    StatisticalModel[_LinearConfigT], Generic[_LinearConfigT]
):
    """Abstract base of the delphi linear models.

    Concrete subclasses must define a ``defaults`` class attribute pointing to
    their config class, or shadow it on the instance before calling
    ``super().__init__`` when the config depends on a constructor argument.

    A model fit by the trainer trains ``weight``, of shape (d + 1, k) with an
    intercept, on the design matrix ``design_matrix(X)``. Its estimates are
    read from ``final_run``'s iterates as ``{avg,ema,final,best}_coef_`` and
    ``_intercept_``; ``coef_`` and ``intercept_`` are the averaged ones.

    Attributes:
        emp_weight: Empirical weight initialization, or None.
        fit_intercept: Whether the design matrix carries an intercept column.
        d: Input dimension, set during fitting.
        k: Output dimension, set during fitting.
        base_radius: Base projection set radius.
        covariate_scale: The design matrix, intercept column included, is
            divided by this before the descent, so the estimates are divided
            by it after.
    """

    # Created by register_buffer below, which pylint and mypy cannot see.
    emp_weight: Optional[ch.Tensor]
    # Created by register_parameter during fit, which mypy cannot see.
    weight: torch.nn.Parameter

    def __init__(
        self,
        hyperparameters: dict,
        emp_weight=None,
        fit_intercept: bool = True,
    ):
        """Initialize the linear model.

        Args:
            hyperparameters: Hyperparameter dict, validated and stored as
                ``self.hyperparameters`` via the subclass ``defaults`` config.
            emp_weight: Optional empirical weight initialization.
            fit_intercept: Whether to fit an intercept.
        """
        super().__init__(hyperparameters)
        self.logger = logging.getLogger(type(self).__module__)
        self.register_buffer("emp_weight", emp_weight)
        self.fit_intercept = fit_intercept

        self.d: Optional[int] = None  # pylint: disable=invalid-name
        self.k: Optional[int] = None  # pylint: disable=invalid-name
        self.base_radius = 2.0
        self.covariate_scale = 1.0

    @property
    def required_emp_weight(self) -> ch.Tensor:
        """Return the empirical weight, which _calc_emp_model must have set.

        Returns:
            The empirical weight estimate.

        Raises:
            RuntimeError: If no empirical weight has been computed yet.
        """
        if self.emp_weight is None:
            raise RuntimeError("emp_weight has not been computed on this model.")
        return self.emp_weight

    def design_matrix(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return the covariates, with an intercept column if fit, scaled.

        Args:
            X: Covariates, shaped (n, d).

        Returns:
            The design matrix, shaped (n, d + 1) with an intercept.
        """
        if self.fit_intercept:
            X = ch.cat([X, ch.ones(X.size(0), 1)], dim=1)  # pylint: disable=invalid-name
        return X / self.covariate_scale

    def forward(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return ``h_W(x)`` for each row of the design matrix."""
        return X @ self.weight

    def coefficients(
        self, params: Optional[ch.Tensor]
    ) -> tuple[Optional[ch.Tensor], Optional[ch.Tensor]]:
        """Split a flat iterate into coefficients and intercept.

        Args:
            params: A flattened ``weight``, or None.

        Returns:
            The coefficients, shaped (k, d), and the intercept, shaped (k,)
            or None without one; both None when params is.
        """
        if params is None:
            return None, None
        k = require_fitted(self.k, "k")
        weight = params.reshape(-1, k) / self.covariate_scale
        if not self.fit_intercept:
            return weight.T, None
        return weight[:-1].T, weight[-1]

    def _estimate(self, name: str) -> Optional[ch.Tensor]:
        """Return one of final_run's iterate estimates.

        Raises:
            RuntimeError: If the model has not been fit.
        """
        return getattr(self.final_run, f"{name}_params")

    @property
    def avg_coef_(self) -> Optional[ch.Tensor]:
        """Coefficients at the averaged iterate."""
        return self.coefficients(self._estimate("avg"))[0]

    @property
    def avg_intercept_(self) -> Optional[ch.Tensor]:
        """Intercept at the averaged iterate."""
        return self.coefficients(self._estimate("avg"))[1]

    @property
    def ema_coef_(self) -> Optional[ch.Tensor]:
        """Coefficients at the moving average of the iterates."""
        return self.coefficients(self._estimate("ema"))[0]

    @property
    def ema_intercept_(self) -> Optional[ch.Tensor]:
        """Intercept at the moving average of the iterates."""
        return self.coefficients(self._estimate("ema"))[1]

    @property
    def final_coef_(self) -> Optional[ch.Tensor]:
        """Coefficients at the last iterate."""
        return self.coefficients(self._estimate("final"))[0]

    @property
    def final_intercept_(self) -> Optional[ch.Tensor]:
        """Intercept at the last iterate."""
        return self.coefficients(self._estimate("final"))[1]

    @property
    def best_coef_(self) -> Optional[ch.Tensor]:
        """Coefficients at the iterate with the lowest validation loss."""
        return self.coefficients(self._estimate("best"))[0]

    @property
    def best_intercept_(self) -> Optional[ch.Tensor]:
        """Intercept at the iterate with the lowest validation loss."""
        return self.coefficients(self._estimate("best"))[1]

    @property
    def coef_(self) -> Optional[ch.Tensor]:
        """Coefficients at the averaged iterate."""
        return self.avg_coef_

    @property
    def intercept_(self) -> Optional[ch.Tensor]:
        """Intercept at the averaged iterate."""
        return self.avg_intercept_

    def predict(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return ``h_W(x)`` at ``coef_`` and ``intercept_``, shaped (n, k).

        Args:
            X: Covariates, shaped (n, d).
        """
        pred = X @ require_fitted(self.coef_, "coef_").T
        intercept = self.intercept_
        return pred if intercept is None else pred + intercept
