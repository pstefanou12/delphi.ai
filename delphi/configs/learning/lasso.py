# Author: pstefanou12@
"""Configuration for the lasso solver."""

from typing import Optional

import pydantic

from delphi.configs import base


class LassoConfig(base.DelphiConfig):
    """Configuration for ``Lasso``.

    Each iteration is one accelerated proximal gradient step, so
    ``iterations`` bounds the steps, not the passes over the data.

    Attributes:
        tolerance: Bound on the duality gap, relative to the objective at
            zero, at which the solver stops.
        bisection_steps: Halvings of the log-penalty interval that
            ``fit_constrained`` searches.
    """

    iterations: Optional[int] = pydantic.Field(default=20000, ge=1)
    tolerance: float = pydantic.Field(default=1e-8, gt=0.0)
    bisection_steps: int = pydantic.Field(default=20, ge=1)
