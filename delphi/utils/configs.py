# Author: pstefanou12@
"""Pydantic configuration models for delphi.ai algorithms."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


def make_config(args: dict | BaseModel, config_class: type[BaseModel]) -> BaseModel:
    """Construct a Pydantic config from a dict or return an existing config.

    Args:
        args: Hyperparameter dict or an existing Pydantic config.
        config_class: Target config class to instantiate when args is a dict.

    Returns:
        A fully validated instance of config_class.

    Raises:
        TypeError: If args is neither a dict nor a Pydantic BaseModel.
    """
    if isinstance(args, config_class):
        return args
    if isinstance(args, dict):
        return config_class(**args)
    raise TypeError(
        f"args must be a {config_class.__name__} or dict, got {type(args).__name__}."
    )


class TrainerConfig(BaseModel):
    """Configuration for the Trainer.

    Attributes:
        epochs: Maximum training epochs; mutually exclusive with iterations.
        iterations: Maximum training iterations; mutually exclusive with epochs.
        trials: Number of independent training runs.
        ema_decay: Exponential moving-average decay factor.
        tol: Convergence tolerance.
        early_stopping: Enable early stopping based on loss_tol / patience.
        verbose: Enable verbose logging.
        disable_no_grad: Disable torch.no_grad during validation.
        val_interval: Step frequency for mid-epoch validation; None disables it.
        patience: Early-stopping patience in epochs without improvement.
        grad_tol: Gradient norm threshold for stopping.
        grad_tol_window: Window size for smoothed gradient norm stopping check.
        loss_tol: Epoch-level loss improvement tolerance for early stopping.
        log_every: Logging frequency in steps.
        max_grad_norm: Maximum gradient norm for clipping; None disables it.
        tqdm: Enable tqdm progress bars.
        device: Compute device string (e.g. 'cpu', 'cuda').
        use_amp: Enable automatic mixed precision training.
        accumulate_grad_batches: Number of gradient accumulation steps.
        record_params_every: Step frequency for parameter vector snapshots;
            0 disables recording.
        checkpoint_dir: Directory for on-disk best-model checkpoints.
        checkpoint_every: Epoch frequency for periodic checkpoints; 0 disables.
    """

    model_config = ConfigDict(extra="ignore")

    epochs: int | None = Field(default=None, ge=1)
    iterations: int | None = Field(default=None, ge=1)
    trials: int = Field(default=1, ge=1)
    ema_decay: float = Field(default=0.99, ge=0.0, le=1.0)
    tol: float = Field(default=1e-3, ge=0.0)
    early_stopping: bool = False
    verbose: bool = False
    disable_no_grad: bool = False
    val_interval: int | None = Field(default=None, ge=1)
    patience: int | None = Field(default=None, ge=1)
    grad_tol: float = Field(default=0.0, ge=0.0)
    grad_tol_window: int = Field(default=1, ge=1)
    loss_tol: float | None = None
    log_every: int = Field(default=50, ge=1)
    max_grad_norm: float | None = None
    tqdm: bool = False
    device: str = "cpu"
    use_amp: bool = False
    accumulate_grad_batches: int = Field(default=1, ge=1)
    record_params_every: int = Field(default=0, ge=0)
    checkpoint_dir: str | None = None
    checkpoint_every: int = Field(default=0, ge=0)


class OptimizerConfig(BaseModel):
    """Configuration for optimizers.

    Attributes:
        optimizer: Optimizer type ('sgd', 'adam', 'adamw', 'lbfgs').
        lr: Learning rate.
        momentum: SGD momentum coefficient.
        dampening: SGD dampening coefficient.
        weight_decay: L2 regularization weight.
        nesterov: Enable Nesterov momentum for SGD.
        maximize: Maximize the objective instead of minimizing.
        foreach: Use the foreach implementation when available.
        differentiable: Enable differentiable optimizer mode.
        fused: Enable fused optimizer implementation when available.
        scheduler: Learning-rate scheduler type; None disables scheduling.
    """

    model_config = ConfigDict(extra="ignore")

    optimizer: str = "sgd"
    lr: float = Field(default=0.1, gt=0.0)
    momentum: float = Field(default=0.0, ge=0.0)
    dampening: float = Field(default=0.0, ge=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    nesterov: bool = False
    maximize: bool = False
    foreach: bool | None = None
    differentiable: bool = False
    fused: bool | None = None
    scheduler: str | None = None
