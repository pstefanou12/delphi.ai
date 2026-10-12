# Author: pstefanou12@
"""Optimizer configuration.

One mixin per optimizer, combined with the scheduler mixins into
``OptimizerConfig``, because the optimizer is selected at runtime from
``hyperparameters.optimizer`` and callers pass a single flat dict.

Defaults are transcribed from the ``getattr`` call sites in
``delphi.delphi``, which are the operative values: the ``*_DEFAULTS``
mappings for optimizers in ``delphi.utils.defaults`` are referenced only by
tests and never applied to a live config.
"""

from typing import Literal, Optional

import pydantic

from delphi.configs import schedulers


class SGDConfig(pydantic.BaseModel):
    """Configuration specific to torch.optim.SGD.

    Attributes:
        momentum: Momentum coefficient.
        dampening: Dampening for momentum.
        nesterov: Enable Nesterov momentum.
    """

    momentum: float = pydantic.Field(default=0.0, ge=0.0)
    dampening: float = pydantic.Field(default=0.0, ge=0.0)
    nesterov: bool = False


class AdamConfig(pydantic.BaseModel):
    """Configuration specific to torch.optim.Adam.

    Attributes:
        beta1: Coefficient for the first moment estimate.
        beta2: Coefficient for the second moment estimate.
        adam_eps: Term added to the denominator for numerical stability.
            Named distinctly from the estimators' ``eps``, which is the
            truncation criterion's stability constant and a different
            quantity; sharing the name meant Adam silently received the
            criterion's 1e-5 on any truncated-regression config.
        amsgrad: Use the AMSGrad variant.
        capturable: Allow capture in a CUDA graph.
    """

    beta1: float = pydantic.Field(default=0.9, ge=0.0, le=1.0)
    beta2: float = pydantic.Field(default=0.999, ge=0.0, le=1.0)
    adam_eps: float = pydantic.Field(default=1e-8, ge=0.0)
    amsgrad: bool = False
    capturable: bool = False


class LBFGSConfig(pydantic.BaseModel):
    """Configuration specific to torch.optim.LBFGS.

    Attributes:
        max_iter: Maximum iterations per optimization step.
        max_eval: Maximum function evaluations per step; None derives it.
        tolerance_grad: Termination tolerance on the gradient norm.
        tolerance_change: Termination tolerance on parameter change.
        history_size: Update history size.
        line_search_fn: Line search algorithm; None disables line search.
    """

    max_iter: int = pydantic.Field(default=20, ge=1)
    max_eval: Optional[int] = pydantic.Field(default=None, ge=1)
    tolerance_grad: float = pydantic.Field(default=1e-7, ge=0.0)
    tolerance_change: float = pydantic.Field(default=1e-9, ge=0.0)
    history_size: int = pydantic.Field(default=100, ge=1)
    line_search_fn: Optional[Literal["strong_wolfe"]] = None


class OptimizerConfig(  # pylint: disable=too-many-ancestors
    SGDConfig,
    AdamConfig,
    LBFGSConfig,
    schedulers.StepLRConfig,
    schedulers.MultiStepLRConfig,
    schedulers.ExponentialLRConfig,
    schedulers.CosineLRConfig,
    schedulers.LinearLRConfig,
    schedulers.CyclicLRConfig,
    schedulers.PlateauLRConfig,
    schedulers.WarmupConfig,
):
    """Configuration for optimizers and learning-rate schedulers.

    Attributes:
        optimizer: Optimizer type ('sgd', 'adam', 'adamw', 'lbfgs').
        lr: Learning rate.
        weight_decay: L2 regularization weight.
        maximize: Maximize the objective instead of minimizing.
        foreach: Use the foreach implementation when available.
        differentiable: Enable differentiable optimizer mode.
        fused: Enable fused optimizer implementation when available.
        scheduler: Learning-rate scheduler type; None disables scheduling.
        constant: Hold the learning rate constant, skipping scheduler
            creation entirely. Read by delphi._create_scheduler.
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    optimizer: str = "sgd"
    lr: float = pydantic.Field(default=0.1, gt=0.0)
    weight_decay: float = pydantic.Field(default=0.0, ge=0.0)
    maximize: bool = False
    foreach: Optional[bool] = None
    differentiable: bool = False
    fused: Optional[bool] = None
    scheduler: Optional[str] = None
    constant: bool = False
