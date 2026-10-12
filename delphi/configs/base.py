# Author: pstefanou12@
"""Base Pydantic configuration models shared across delphi.

``DelphiConfig`` combines the trainer, optimizer/scheduler and dataset
concerns, and is what every config the delphi model and trainer consume
derives from. ``LinearModelConfig`` is the shared base for the delphi.stats
estimators.
"""

from typing import Optional, TypeVar

import pydantic

from delphi.configs.dataset import DatasetConfig
from delphi.configs.optimizers import OptimizerConfig


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


class DelphiConfig(TrainerConfig, OptimizerConfig, DatasetConfig):
    """Base for every config the delphi model and trainer consume.

    Combining the three concerns here is what lets ``delphi`` read optimizer,
    scheduler and dataset fields as typed attributes. While _ConfigT was bound
    to TrainerConfig alone, those fields were untypeable from the base class
    and every read had to go through ``getattr(..., default)``.
    """


_ConfigT = TypeVar("_ConfigT", bound=DelphiConfig)


class LinearModelConfig(DelphiConfig):
    """Shared configuration for the delphi.stats truncated linear models.

    Transcribed from the pre-pydantic ``*_DEFAULTS`` mappings in
    ``delphi.utils.defaults`` so defaults match the previous
    ``check_and_fill_args`` behaviour exactly.

    Attributes:
        alpha: Survival probability lower bound of the truncation set.
        c_s: Scaling constant for the dependent-data survival threshold.
        fit_intercept: Fit an intercept term.
    """

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Dataset fields (val, workers, shuffle, batch_size, normalize) come from
    # DatasetConfig via DelphiConfig.
    alpha: float = pydantic.Field(default=1.0, gt=0.0, le=1.0)
    c_s: float = pydantic.Field(default=10.0, gt=0.0)
    fit_intercept: bool = True

    # Must be > 0: fit() reads back best/final params from the trainer, which
    # only records them when this is enabled. TRUNC_REG_DEFAULTS omitted the
    # key entirely, so the pre-pydantic path got None from
    # Parameters.__getattr__ and failed on "iterations % None".
    record_params_every: int = pydantic.Field(default=1, ge=1)
