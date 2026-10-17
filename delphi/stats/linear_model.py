# Author: pstefanou12@
"""Linear model class for delphi."""

import logging
from typing import Generic, Optional, TypeVar

import torch as ch

from delphi import configs
from delphi.delphi import delphi

_LinearConfigT = TypeVar("_LinearConfigT", bound=configs.LinearModelConfig)


class LinearModel(  # pylint: disable=abstract-method
    delphi[_LinearConfigT], Generic[_LinearConfigT]
):
    """Truncated linear model parent class.

    Concrete subclasses must define a ``defaults`` class attribute pointing to
    their config class, or shadow it on the instance before calling
    ``super().__init__`` when the config depends on a constructor argument.

    Attributes:
        emp_weight: Empirical weight initialization, or None.
        d: Input dimension, set during fitting.
        k: Output dimension, set during fitting.
        base_radius: Base projection set radius.
    """

    # Created by register_buffer below, which pylint and mypy cannot see.
    emp_weight: Optional[ch.Tensor]

    def __init__(
        self,
        hyperparameters: dict,
        emp_weight=None,
    ):
        """Initialize the linear model.

        Args:
            hyperparameters: Hyperparameter dict, validated and stored as
                ``self.hyperparameters`` via the subclass ``defaults`` config.
            emp_weight: Optional empirical weight initialization.
        """
        super().__init__(hyperparameters)
        self.logger = logging.getLogger(type(self).__module__)
        self.register_buffer("emp_weight", emp_weight)

        self.d: Optional[int] = None  # pylint: disable=invalid-name
        self.k: Optional[int] = None  # pylint: disable=invalid-name
        self.base_radius = 2.0

    @property
    def required_emp_weight(self) -> ch.Tensor:
        """Return the empirical weight, which calc_emp_model must have set.

        Returns:
            The empirical weight estimate.

        Raises:
            RuntimeError: If no empirical weight has been computed yet.
        """
        if self.emp_weight is None:
            raise RuntimeError("emp_weight has not been computed on this model.")
        return self.emp_weight
