# Author: pstefanou12@
"""Configuration for the least-absolute-deviations solver."""

from typing import Optional

import pydantic

from delphi.configs import base


class L1RegressionConfig(base.DelphiConfig):
    """Configuration for ``L1Regression``.

    Each iteration is one Newton step of the interior point method, which
    converges in a few tens of them, so ``iterations`` only guards against a
    stall.

    Attributes:
        tolerance: Bound on the relative duality gap and the relative primal
            and dual residuals at which the solver stops.
    """

    iterations: Optional[int] = pydantic.Field(default=100, ge=1)
    tolerance: float = pydantic.Field(default=1e-8, gt=0.0)
