# Author: pstefanou12@
"""Configuration for the least-absolute-deviations solver."""

from typing import Literal, Optional

import pydantic

from delphi.configs import base


class L1RegressionConfig(base.DelphiConfig):
    """Configuration for ``L1Regression``.

    Each iteration is one Newton step of the interior point method, which
    converges in a few tens of them, so ``iterations`` only guards against a
    stall.

    Attributes:
        dtype: float64 only.
        tolerance: Bound on the relative duality gap and the relative primal
            and dual residuals at which the solver stops.
    """

    iterations: Optional[int] = pydantic.Field(default=100, ge=1)
    dtype: Literal["float32", "float64"] = "float64"
    tolerance: float = pydantic.Field(default=1e-8, gt=0.0)

    @pydantic.field_validator("dtype")
    @classmethod
    def check_dtype(cls, dtype: str) -> str:
        """Reject float32, in which the Newton systems lose the iterate."""
        if dtype != "float64":
            raise ValueError(
                "L1Regression needs dtype float64: in float32 its interior "
                f"point can return NaN; got {dtype}."
            )
        return dtype
