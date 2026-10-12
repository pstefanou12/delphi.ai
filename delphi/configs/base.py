# Author: pstefanou12@
"""Base Pydantic configuration models shared across delphi.

TrainerConfig and OptimizerConfig are mixed into every concrete config;
LinearModelConfig is the shared base for the delphi.stats estimators.
"""

from typing import Optional, TypeVar

import pydantic


class TrainerConfig(pydantic.BaseModel):
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

    model_config = pydantic.ConfigDict(extra="ignore")

    epochs: Optional[int] = pydantic.Field(default=None, ge=1)
    iterations: Optional[int] = pydantic.Field(default=None, ge=1)
    trials: int = pydantic.Field(default=1, ge=1)
    ema_decay: float = pydantic.Field(default=0.99, ge=0.0, le=1.0)
    tol: float = pydantic.Field(default=1e-3, ge=0.0)
    early_stopping: bool = False
    verbose: bool = False
    disable_no_grad: bool = False
    val_interval: Optional[int] = pydantic.Field(default=None, ge=1)
    patience: Optional[int] = pydantic.Field(default=None, ge=1)
    grad_tol: float = pydantic.Field(default=0.0, ge=0.0)
    grad_tol_window: int = pydantic.Field(default=1, ge=1)
    loss_tol: Optional[float] = None
    log_every: int = pydantic.Field(default=50, ge=1)
    max_grad_norm: Optional[float] = None
    tqdm: bool = False
    device: str = "cpu"
    use_amp: bool = False
    accumulate_grad_batches: int = pydantic.Field(default=1, ge=1)
    record_params_every: int = pydantic.Field(default=0, ge=0)
    checkpoint_dir: Optional[str] = None
    checkpoint_every: int = pydantic.Field(default=0, ge=0)


class OptimizerConfig(pydantic.BaseModel):
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
        constant: Hold the learning rate constant, skipping scheduler
            creation entirely. Read by delphi._create_scheduler.
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    optimizer: str = "sgd"
    lr: float = pydantic.Field(default=0.1, gt=0.0)
    momentum: float = pydantic.Field(default=0.0, ge=0.0)
    dampening: float = pydantic.Field(default=0.0, ge=0.0)
    weight_decay: float = pydantic.Field(default=0.0, ge=0.0)
    nesterov: bool = False
    maximize: bool = False
    foreach: Optional[bool] = None
    differentiable: bool = False
    fused: Optional[bool] = None
    scheduler: Optional[str] = None
    constant: bool = False


_ConfigT = TypeVar("_ConfigT", bound=TrainerConfig)


class LinearModelConfig(TrainerConfig, OptimizerConfig):
    """Shared configuration for the delphi.stats truncated linear models.

    Transcribed from the pre-pydantic ``*_DEFAULTS`` mappings in
    ``delphi.utils.defaults`` so defaults match the previous
    ``check_and_fill_args`` behaviour exactly.

    Attributes:
        val: Fraction of data held out for validation.
        workers: DataLoader worker processes.
        shuffle: Shuffle the training DataLoader.
        batch_size: Mini-batch size for training.
        alpha: Survival probability lower bound of the truncation set.
        c_s: Scaling constant for the dependent-data survival threshold.
        fit_intercept: Fit an intercept term.
    """

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    val: float = pydantic.Field(default=0.2, ge=0.0, le=1.0)
    workers: int = pydantic.Field(default=0, ge=0)
    shuffle: bool = True
    # -1 is a sentinel for "full batch", read by
    # datasets.make_train_and_val as len(dataset), so this is deliberately
    # unconstrained to match the pre-pydantic defaults.
    batch_size: int = 50
    # The remaining DATASET_DEFAULTS keys. Declared so that the
    # check_and_fill_args call still in make_train_and_val is a no-op against
    # a pydantic config instead of trying to set an undeclared field.
    normalize: bool = False
    pin_memory: bool = True
    drop_last: bool = False
    alpha: float = pydantic.Field(default=1.0, gt=0.0, le=1.0)
    c_s: float = pydantic.Field(default=10.0, gt=0.0)
    fit_intercept: bool = True

    # Must be > 0: fit() reads back best/final params from the trainer, which
    # only records them when this is enabled. TRUNC_REG_DEFAULTS omitted the
    # key entirely, so the pre-pydantic path got None from
    # Parameters.__getattr__ and failed on "iterations % None".
    record_params_every: int = pydantic.Field(default=1, ge=1)
