# Author: pstefanou12@
"""Learning-rate scheduler configuration.

One section per scheduler, combined into ``OptimizerConfig``. Every field a
section owns is named ``<section>_<torch name>`` and declares
``serialization_alias`` carrying torch's own parameter name, so
``delphi.delphi`` can derive each scheduler's keyword arguments from the
section rather than restating them. The prefix is required rather than
stylistic: torch reuses ``gamma`` across StepLR, MultiStepLR, ExponentialLR
and CyclicLR with different defaults, ``mode`` across CyclicLR and
ReduceLROnPlateau, and ``eps`` across Adam and ReduceLROnPlateau as well as
the estimators' own truncation constant, while the hyperparameter dict callers
pass is flat.

Defaults are transcribed from the ``getattr`` call sites that previously lived
in ``delphi.delphi``, which are the operative values: the ``*_LR_DEFAULTS``
mappings in ``delphi.utils.defaults`` are referenced only by tests and are
never applied to a live config.
"""

from typing import Literal, Optional

import pydantic


class StepLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.StepLR.

    Attributes:
        step_lr_step_size: Epochs between learning-rate decays.
        step_lr_gamma: Multiplicative decay factor.
    """

    step_lr_step_size: int = pydantic.Field(
        default=100, ge=1, serialization_alias="step_size"
    )
    step_lr_gamma: float = pydantic.Field(
        default=0.1, ge=0.0, le=1.0, serialization_alias="gamma"
    )


class MultiStepLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.MultiStepLR.

    Attributes:
        multi_step_milestones: Epoch indices at which to decay.
        multi_step_gamma: Multiplicative decay factor.
    """

    multi_step_milestones: list[int] = pydantic.Field(
        default_factory=lambda: [30, 60, 90], serialization_alias="milestones"
    )
    multi_step_gamma: float = pydantic.Field(
        default=0.1, ge=0.0, le=1.0, serialization_alias="gamma"
    )


class ExponentialLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.ExponentialLR.

    Attributes:
        exponential_gamma: Multiplicative decay factor applied every epoch.
    """

    exponential_gamma: float = pydantic.Field(
        default=0.95, ge=0.0, le=1.0, serialization_alias="gamma"
    )


class CosineLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.CosineAnnealingLR.

    T_max is supplied by delphi from the trainer's epochs, so it is not a
    field here.

    Attributes:
        cosine_eta_min: Minimum learning rate.
    """

    cosine_eta_min: float = pydantic.Field(
        default=0.0, ge=0.0, serialization_alias="eta_min"
    )


class LinearLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.LinearLR.

    total_iters is supplied by delphi from the trainer's epochs, so it is not
    a field here.

    Attributes:
        linear_start_factor: Starting multiplier for the learning rate.
        linear_end_factor: Final multiplier for the learning rate.
    """

    linear_start_factor: float = pydantic.Field(
        default=1.0, ge=0.0, serialization_alias="start_factor"
    )
    linear_end_factor: float = pydantic.Field(
        default=0.0, ge=0.0, serialization_alias="end_factor"
    )


class CyclicLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.CyclicLR.

    Attributes:
        cyclic_base_lr: Lower learning-rate bound.
        cyclic_max_lr: Upper learning-rate bound; None makes delphi use the
            optimizer's current learning rate, which is only known once the
            optimizer exists.
        cyclic_step_size_up: Steps in the ascending half-cycle.
        cyclic_step_size_down: Steps in the descending half-cycle; None
            mirrors the ascending half.
        cyclic_mode: Cycle shape.
        cyclic_gamma: Decay factor for the 'exp_range' mode.
    """

    cyclic_base_lr: float = pydantic.Field(
        default=0.0, ge=0.0, serialization_alias="base_lr"
    )
    cyclic_max_lr: Optional[float] = pydantic.Field(
        default=None, gt=0.0, serialization_alias="max_lr"
    )
    cyclic_step_size_up: int = pydantic.Field(
        default=2000, ge=1, serialization_alias="step_size_up"
    )
    cyclic_step_size_down: Optional[int] = pydantic.Field(
        default=None, ge=1, serialization_alias="step_size_down"
    )
    cyclic_mode: Literal["triangular", "triangular2", "exp_range"] = pydantic.Field(
        default="triangular2", serialization_alias="mode"
    )
    cyclic_gamma: float = pydantic.Field(
        default=1.0, ge=0.0, serialization_alias="gamma"
    )


class PlateauLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.ReduceLROnPlateau.

    Attributes:
        plateau_mode: Whether the tracked metric is minimized or maximized.
        plateau_factor: Multiplicative decay factor.
        plateau_patience: Epochs without improvement before decaying.
        plateau_threshold: Minimum change counted as an improvement.
        plateau_threshold_mode: Whether the threshold is relative or absolute.
        plateau_cooldown: Epochs to wait after a decay before resuming.
        plateau_min_lr: Learning-rate floor.
        plateau_eps: Minimum decay; smaller updates are ignored.
    """

    plateau_mode: Literal["min", "max"] = pydantic.Field(
        default="min", serialization_alias="mode"
    )
    plateau_factor: float = pydantic.Field(
        default=0.1, ge=0.0, le=1.0, serialization_alias="factor"
    )
    plateau_patience: int = pydantic.Field(
        default=10, ge=1, serialization_alias="patience"
    )
    plateau_threshold: float = pydantic.Field(
        default=1e-4, ge=0.0, serialization_alias="threshold"
    )
    plateau_threshold_mode: Literal["rel", "abs"] = pydantic.Field(
        default="rel", serialization_alias="threshold_mode"
    )
    plateau_cooldown: int = pydantic.Field(
        default=0, ge=0, serialization_alias="cooldown"
    )
    plateau_min_lr: float = pydantic.Field(
        default=0.0, ge=0.0, serialization_alias="min_lr"
    )
    plateau_eps: float = pydantic.Field(default=1e-8, ge=0.0, serialization_alias="eps")


class WarmupConfig(pydantic.BaseModel):
    """Configuration for the optional linear warmup prepended to a scheduler.

    This has no torch counterpart; delphi implements the warmup by wrapping
    the chosen scheduler in a SequentialLR.

    Attributes:
        warmup_steps: Steps of linear warmup; 0 disables it.
    """

    warmup_steps: int = pydantic.Field(default=0, ge=0)
