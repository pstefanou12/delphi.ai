# Author: pstefanou12@
"""Base Pydantic configuration models shared across delphi.

``DelphiConfig`` combines the trainer, optimizer/scheduler and dataset
concerns, and is what every config the delphi model and trainer consume
derives from.
"""

from typing import Optional, TypeVar
import warnings

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
        avg_burn_in: Fraction of recorded iterates discarded before
            averaging, so the average covers only the tail of the run.
        tol: Convergence tolerance.
        early_stopping: Enable early stopping based on loss_tol / patience.
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
    ema_decay: float = pydantic.Field(default=0.99, ge=0.0, lt=1.0)
    avg_burn_in: float = pydantic.Field(default=0.5, ge=0.0, lt=1.0)
    tol: float = pydantic.Field(default=1e-3, ge=0.0)
    early_stopping: bool = False
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

    @pydantic.model_validator(mode="before")
    @classmethod
    def _warn_on_unknown_keys(cls, data):
        """Warn about hyperparameter keys this config will discard.

        Args:
            data: The raw input, usually the caller's hyperparameter dict.

        Returns:
            The input unchanged.
        """
        # extra="ignore" means a renamed or misspelled key would otherwise
        # train with a silently different value than the caller asked for.
        if isinstance(data, dict):
            unknown = sorted(set(data) - set(cls.model_fields))
            if unknown:
                warnings.warn(
                    f"{cls.__name__} ignores unknown hyperparameters: "
                    f"{', '.join(unknown)}.",
                    stacklevel=2,
                )
        return data

    def __str__(self):
        """Render the explicitly-set hyperparameters, one per aligned line.

        Only the fields the caller passed, since a config carries 85 and a
        caller typically sets under ten. ``repr`` keeps pydantic's flat form
        and ``model_dump`` the full mapping, so structured sinks are unchanged.
        """
        if not self.model_fields_set:
            return f"{type(self).__name__}: all {len(type(self).model_fields)} defaults"

        names = sorted(self.model_fields_set)
        width = max(len(name) for name in names)
        lines = [f"  {name:<{width}}  {getattr(self, name)}" for name in names]
        hidden = len(type(self).model_fields) - len(names)
        if hidden:
            lines.append(f"  ({hidden} defaults not shown; see model_dump())")
        return "\n".join(lines)


_ConfigT = TypeVar("_ConfigT", bound=DelphiConfig)
