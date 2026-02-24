# Author: pstefanou12@
"""
Linear model class for delphi.
"""

import torch as ch
from torch import Tensor

from delphi.delphi import delphi
from delphi.delphi_logger import delphiLogger
from delphi.utils.helpers import Parameters


class LinearModel(delphi):  # pylint: disable=abstract-method
    """Truncated linear model parent class.

    Attributes:
        emp_weight: Empirical weight initialization, or None.
        d: Input dimension, set during fitting.
        k: Output dimension, set during fitting.
        base_radius (float): Base projection set radius.
        dependent (bool): Whether the dataset has temporal dependence.
        s: Survival threshold used when dependent is True.
    """

    def __init__(
        self, args: Parameters, dependent: bool, logger: delphiLogger, emp_weight=None
    ):
        super().__init__(args, logger)
        self.emp_weight = emp_weight

        self.d, self.k = None, None  # pylint: disable=invalid-name
        self.base_radius = 2.0
        self.dependent = dependent
        if self.dependent:
            self.s = self.args.c_s * (  # pylint: disable=invalid-name
                ch.sqrt(ch.log(Tensor([1 / self.args.alpha]))) + 1
            )
