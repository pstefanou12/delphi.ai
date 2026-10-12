# Author: pstefanou12@
"""Configuration model for truncated linear quadratic regulator estimation."""

from typing import Optional

from collections.abc import Callable

import pydantic

from delphi.configs.linear import TruncatedLinearModelConfig


class TruncatedLQRConfig(TruncatedLinearModelConfig):
    """Configuration for truncated linear quadratic regulator estimation.

    Transcribed from TRUNC_LQR_DEFAULTS. R, U_A, U_B, delta and gamma were
    marked REQ there and so are required here.

    Attributes:
        phi: Truncation set oracle.
        target_thickness: Target thickness of the collected trajectories.
        num_traj_phase_one: Trajectory budget for phase one.
        num_traj_phase_two: Trajectory budget for phase two.
        num_traj_gen_samples_A: Trajectory budget for generating A samples.
        num_traj_gen_samples_B: Trajectory budget for generating B samples.
        T_phase_one: Step budget for phase one.
        T_phase_two: Step budget for phase two.
        T_gen_samples_A: Step budget for generating A samples.
        T_gen_samples_B: Step budget for generating B samples.
        R: Radius of the truncation ball.
        U_A: Spectral norm bound on A.
        U_B: Spectral norm bound on B.
        delta: Failure probability.
        eps1: Phase one accuracy target.
        eps2: Phase two accuracy target.
        repeat: Number of repetitions; None runs once.
        gamma: Thickness parameter.
    """

    phi: Callable
    target_thickness: float = float("inf")
    num_traj_phase_one: int = pydantic.Field(default=int(1e100), ge=1)
    num_traj_phase_two: int = pydantic.Field(default=int(1e100), ge=1)
    num_traj_gen_samples_A: int = pydantic.Field(default=int(1e100), ge=1)
    num_traj_gen_samples_B: int = pydantic.Field(default=int(1e100), ge=1)
    T_phase_one: int = pydantic.Field(default=int(1e100), ge=1)
    T_phase_two: int = pydantic.Field(default=int(1e100), ge=1)
    T_gen_samples_A: int = pydantic.Field(default=int(1e100), ge=1)
    T_gen_samples_B: int = pydantic.Field(default=int(1e100), ge=1)
    R: float = pydantic.Field(gt=0.0)
    U_A: float = pydantic.Field(gt=0.0)
    U_B: float = pydantic.Field(gt=0.0)
    delta: float = pydantic.Field(gt=0.0, lt=1.0)
    eps1: float = pydantic.Field(default=0.9, gt=0.0)
    eps2: float = pydantic.Field(default=0.9, gt=0.0)
    repeat: Optional[int] = pydantic.Field(default=None, ge=1)
    gamma: float = pydantic.Field(gt=0.0)
