# Author: pstefanou12@
"""Optimizer configuration.

One section per optimizer, combined with the scheduler sections into
``OptimizerConfig``, because the optimizer is selected at runtime from
``optimizer`` and callers pass a single flat dict.

Every field a section owns is named ``<section>_<torch name>`` and declares
``serialization_alias`` carrying torch's own parameter name, so
``delphi.delphi`` derives each optimizer's keyword arguments from the section.
The fields declared on ``OptimizerConfig`` itself stay unprefixed: they mean
the same thing to every optimizer and apply to whichever one ``optimizer``
selects, so prefixing them would mean duplicating each per optimizer.

Defaults are transcribed from the ``getattr`` call sites that previously lived
in ``delphi.delphi``, which are the operative values: the optimizer
``*_DEFAULTS`` mappings in ``delphi.utils.defaults`` are referenced only by
tests and never applied to a live config.
"""

from typing import Annotated, Literal, Optional

import pydantic

from delphi.configs import schedulers

# Both coefficients are probabilities.
_Beta = Annotated[float, pydantic.Field(ge=0.0, le=1.0)]


class SGDConfig(pydantic.BaseModel):
    """Configuration specific to torch.optim.SGD.

    Attributes:
        sgd_momentum: Momentum coefficient.
        sgd_dampening: Dampening for momentum.
        sgd_nesterov: Enable Nesterov momentum.
    """

    sgd_momentum: float = pydantic.Field(
        default=0.0, ge=0.0, serialization_alias="momentum"
    )
    sgd_dampening: float = pydantic.Field(
        default=0.0, ge=0.0, serialization_alias="dampening"
    )
    sgd_nesterov: bool = pydantic.Field(default=False, serialization_alias="nesterov")


class AdamConfig(pydantic.BaseModel):
    """Configuration specific to torch.optim.Adam and torch.optim.AdamW.

    Attributes:
        adam_betas: Coefficients for the first and second moment estimates.
        adam_eps: Term added to the denominator for numerical stability. Named
            distinctly from the estimators' ``eps``, which is the truncation
            criterion's constant and a different quantity; sharing the name
            meant Adam silently received the criterion's value.
        adam_amsgrad: Use the AMSGrad variant.
        adam_capturable: Allow capture in a CUDA graph.
    """

    adam_betas: tuple[_Beta, _Beta] = pydantic.Field(
        default=(0.9, 0.999), serialization_alias="betas"
    )
    adam_eps: float = pydantic.Field(default=1e-8, ge=0.0, serialization_alias="eps")
    adam_amsgrad: bool = pydantic.Field(default=False, serialization_alias="amsgrad")
    adam_capturable: bool = pydantic.Field(
        default=False, serialization_alias="capturable"
    )


class LBFGSConfig(pydantic.BaseModel):
    """Configuration specific to torch.optim.LBFGS.

    Attributes:
        lbfgs_max_iter: Maximum iterations per optimization step.
        lbfgs_max_eval: Maximum function evaluations per step; None derives it.
        lbfgs_tolerance_grad: Termination tolerance on the gradient norm.
        lbfgs_tolerance_change: Termination tolerance on parameter change.
        lbfgs_history_size: Update history size.
        lbfgs_line_search_fn: Line search algorithm; None disables it.
    """

    lbfgs_max_iter: int = pydantic.Field(
        default=20, ge=1, serialization_alias="max_iter"
    )
    lbfgs_max_eval: Optional[int] = pydantic.Field(
        default=None, ge=1, serialization_alias="max_eval"
    )
    lbfgs_tolerance_grad: float = pydantic.Field(
        default=1e-7, ge=0.0, serialization_alias="tolerance_grad"
    )
    lbfgs_tolerance_change: float = pydantic.Field(
        default=1e-9, ge=0.0, serialization_alias="tolerance_change"
    )
    lbfgs_history_size: int = pydantic.Field(
        default=100, ge=1, serialization_alias="history_size"
    )
    lbfgs_line_search_fn: Optional[Literal["strong_wolfe"]] = pydantic.Field(
        default=None, serialization_alias="line_search_fn"
    )


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
    schedulers.InverseTimeLRConfig,
    schedulers.FisherLRConfig,
    schedulers.WarmupConfig,
):
    """Configuration for optimizers and learning-rate schedulers.

    These fields are deliberately unprefixed: each means the same thing to
    every optimizer that accepts it, and applies to whichever optimizer
    ``optimizer`` selects.

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
            creation entirely.
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
