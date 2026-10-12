# Author: pstefanou12@
"""Linear model class for delphi."""

from typing import Generic, Optional, TypeVar

import torch as ch
from torch import Tensor

from delphi.delphi import delphi
from delphi.delphi_logger import delphiLogger
from delphi.utils import configs

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
        dependent: Whether the dataset has temporal dependence.
        s: Survival threshold used when dependent is True.
    """

    def __init__(
        self,
        hyperparameters: dict,
        dependent: bool,
        logger: delphiLogger,
        emp_weight=None,
    ):
        """Initialize the linear model.

        Args:
            hyperparameters: Hyperparameter dict, validated and stored as
                ``self.hyperparameters`` via the subclass ``defaults`` config.
            dependent: Whether the dataset has temporal dependence.
            logger: Logger instance for training diagnostics.
            emp_weight: Optional empirical weight initialization.
        """
        super().__init__(hyperparameters)
        self.logger = logger
        self.emp_weight = emp_weight

        self.d: Optional[int] = None  # pylint: disable=invalid-name
        self.k: Optional[int] = None  # pylint: disable=invalid-name
        self.base_radius = 2.0
        self.dependent = dependent
        self.s: Optional[ch.Tensor] = None  # pylint: disable=invalid-name
        if self.dependent:
            self.s = self.hyperparameters.c_s * (
                ch.sqrt(ch.log(Tensor([1 / self.hyperparameters.alpha]))) + 1
            )

    @property
    def required_survival_threshold(self) -> ch.Tensor:
        """Return the survival threshold s, which only dependent data has.

        Returns:
            The survival threshold computed in ``__init__``.

        Raises:
            RuntimeError: If this model was not constructed with
                ``dependent=True``.
        """
        if self.s is None:
            raise RuntimeError(
                "s is only computed for dependent data; "
                "construct the model with dependent=True."
            )
        return self.s
