# Author: pstefanou12@
"""Learning-rate scheduler configuration.

One mixin per scheduler, combined into ``OptimizerConfig``. Every field here
was previously read by ``delphi._create_*_scheduler`` through
``getattr(self.hyperparameters, name, default)`` without being declared
anywhere, so nothing validated it and a misspelled key silently took the
default. Defaults are transcribed from those call sites, which are the
operative values: the ``*_LR_DEFAULTS`` mappings in ``delphi.utils.defaults``
are referenced only by tests and are never applied to a live config.
"""

from typing import Literal, Optional

import pydantic


class StepLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.StepLR.

    Attributes:
        step_lr: Epochs between learning-rate decays.
        step_lr_gamma: Multiplicative decay factor.
    """

    step_lr: int = pydantic.Field(default=100, ge=1)
    step_lr_gamma: float = pydantic.Field(default=0.1, ge=0.0, le=1.0)


class MultiStepLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.MultiStepLR.

    Attributes:
        milestones: Epoch indices at which to decay the learning rate.
        multi_step_gamma: Multiplicative decay factor. Named distinctly from
            exponential_gamma because delphi read a single ``gamma`` key with
            two different fallbacks, and because TruncatedLQRConfig uses
            ``gamma`` for the thickness parameter.
    """

    milestones: list[int] = pydantic.Field(default_factory=lambda: [30, 60, 90])
    multi_step_gamma: float = pydantic.Field(default=0.1, ge=0.0, le=1.0)


class ExponentialLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.ExponentialLR.

    Attributes:
        exponential_gamma: Multiplicative decay factor applied every epoch.
    """

    exponential_gamma: float = pydantic.Field(default=0.95, ge=0.0, le=1.0)


class CosineLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.CosineAnnealingLR.

    Attributes:
        min_lr: Minimum learning rate (eta_min).
    """

    min_lr: float = pydantic.Field(default=0.0, ge=0.0)


class LinearLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.LinearLR.

    Attributes:
        linear_start_factor: Starting multiplier for the learning rate.
        linear_end_factor: Final multiplier for the learning rate.
    """

    linear_start_factor: float = pydantic.Field(default=1.0, ge=0.0)
    linear_end_factor: float = pydantic.Field(default=0.0, ge=0.0)


class CyclicLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.CyclicLR.

    Attributes:
        cyclic_base_lr: Lower learning-rate bound.
        cyclic_max_lr: Upper learning-rate bound; None uses the optimizer's
            current learning rate, resolved when the scheduler is built.
        cyclic_step_size_up: Steps in the ascending half-cycle.
        cyclic_step_size_down: Steps in the descending half-cycle; None
            mirrors cyclic_step_size_up.
        cyclic_mode: Cycle shape.
        cyclic_gamma: Decay factor for the 'exp_range' mode.
    """

    cyclic_base_lr: float = pydantic.Field(default=0.0, ge=0.0)
    cyclic_max_lr: Optional[float] = pydantic.Field(default=None, gt=0.0)
    cyclic_step_size_up: int = pydantic.Field(default=2000, ge=1)
    cyclic_step_size_down: Optional[int] = pydantic.Field(default=None, ge=1)
    cyclic_mode: Literal["triangular", "triangular2", "exp_range"] = "triangular2"
    cyclic_gamma: float = pydantic.Field(default=1.0, ge=0.0)


class PlateauLRConfig(pydantic.BaseModel):
    """Configuration for torch.optim.lr_scheduler.ReduceLROnPlateau.

    Attributes:
        plateau_mode: Whether the tracked metric should be minimized or
            maximized.
        plateau_factor: Multiplicative decay factor.
        plateau_patience: Epochs without improvement before decaying.
        plateau_threshold: Minimum change counted as an improvement.
        plateau_threshold_mode: Whether plateau_threshold is relative or
            absolute.
        plateau_cooldown: Epochs to wait after a decay before resuming.
        plateau_eps: Minimum decay; smaller updates are ignored.
    """

    plateau_mode: Literal["min", "max"] = "min"
    plateau_factor: float = pydantic.Field(default=0.1, ge=0.0, le=1.0)
    plateau_patience: int = pydantic.Field(default=10, ge=1)
    plateau_threshold: float = pydantic.Field(default=1e-4, ge=0.0)
    plateau_threshold_mode: Literal["rel", "abs"] = "rel"
    plateau_cooldown: int = pydantic.Field(default=0, ge=0)
    plateau_eps: float = pydantic.Field(default=1e-8, ge=0.0)


class WarmupConfig(pydantic.BaseModel):
    """Configuration for the optional linear warmup prepended to a scheduler.

    Attributes:
        warmup_steps: Steps of linear warmup; 0 disables it.
    """

    warmup_steps: int = pydantic.Field(default=0, ge=0)
