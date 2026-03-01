# Author: pstefanou12@
"""Module used for training models."""

import copy
import os
import random
from collections.abc import Iterable
from time import time

import numpy as np
import torch as ch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pydantic import BaseModel
from tqdm import tqdm
from cox.store import Store

from delphi.delphi import delphi
from delphi.delphi_logger import delphiLogger
from delphi.utils import constants as consts
from delphi.utils.configs import TrainerConfig
from delphi.utils.constants import CheckpointKey, ProcedureStage, StopReason
from delphi.utils.helpers import AverageMeter, setup_store_with_metadata


class Trainer:  # pylint: disable=too-many-instance-attributes
    """Trainer class for fitting delphi models using iterative optimization."""

    def __init__(
        self, model: delphi, args: dict | BaseModel, logger: delphiLogger, store=None
    ):
        """Initialize the Trainer.

        Args:
            model: The delphi model to train.
            args: Hyperparameter dict or Pydantic config. A dict is converted
                to a TrainerConfig with defaults applied.
            logger: Logger instance for training diagnostics.
            store: Optional cox store for experiment logging.
        """
        self.model: delphi = model
        self.args: BaseModel = TrainerConfig(**args) if isinstance(args, dict) else args
        self.logger: delphiLogger = logger
        self.store: Store | None = store

        self.epoch: int = 0
        self.iterations: int = 0
        # Store procedure history in lists to avoid torch.cat overhead.
        self.train_losses: list[ch.Tensor] = []
        self.val_losses: list[ch.Tensor] = []
        # Epoch-level averages for loss_tol and patience stop criteria.
        self.epoch_train_losses: list[float] = []
        self.epoch_val_losses: list[float] = []
        self.param_history: list[ch.Tensor] = []
        self.val_param_history_indices: list[int] = []
        self.grad_norms: list[float] = []
        self._ema_params: ch.Tensor | None = None

        self.t_start: float | None = None
        self.t_end: float | None = None
        self.procedure_duration: float | None = None
        self._best_loss_index: int | None = None
        self._best_param_index: int | None = None
        # Either a state_dict (in-memory) or a filesystem path (disk) to the
        # best checkpoint; None when neither saving mode is active.
        self._best_model_state: dict | str | None = None
        self.stop_reason: StopReason | None = None
        # Tracks whether the cox logs table has been created for this run.
        self._store_logs_ready: bool = False
        # Size of the training dataset; set by train_model for ELBO scaling etc.
        self.dataset_size: int | None = None

        # Gradient accumulation counter.
        self._accum_step: int = 0
        # GradScaler for mixed precision; set by train_model.
        self.scaler: ch.cuda.amp.GradScaler | None = None
        # Most recent global gradient L2 norm; set every step regardless of
        # record_params_every so it is always available for logging.
        self._last_grad_norm: float | None = None

    def make_closure(self, batch):
        """Create an optimizer closure for the given batch.

        Used by the standard (non-AMP, non-accumulation) training path.
        Calls ``model.compute_loss(batch)`` so subclasses can override only
        the loss computation without touching optimizer bookkeeping.
        """

        def closure():
            self.model.optimizer.zero_grad()
            loss = self._forward_loss(batch)
            loss.backward()
            if self.args.max_grad_norm is not None:
                ch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
            return loss

        return closure

    def _forward_loss(self, batch) -> ch.Tensor:
        """Compute forward pass + regularization and return the scalar loss."""
        loss = self.model.compute_loss(batch)
        if loss.ndim > 0:
            loss = loss.sum()
        return loss + self.model.regularize(batch)

    def _compute_grad_norm(self) -> float:
        """Compute the global L2 gradient norm without materialising the parameter vector.

        Iterates over per-parameter gradient norms and combines them into a
        single scalar, avoiding the memory cost of concatenating all gradients.

        Returns:
            Global L2 gradient norm, or 0.0 if no parameters have gradients.
        """
        norms = [
            p.grad.detach().norm()
            for p in self.model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        return ch.stack(norms).norm().item() if norms else 0.0

    def _record_step(self, loss: ch.Tensor) -> None:
        """Record the scalar loss, grad norm, and (when enabled) param vector.

        The gradient norm is computed and stored in ``_last_grad_norm`` every
        step so it is always available for logging. Parameter vector recording
        is skipped when ``record_params_every == 0`` (the default) to avoid
        materialising full-size tensors for large models.
        """
        self.train_losses.append(loss.detach())

        self._last_grad_norm = self._compute_grad_norm()

        # Skip full parameter-vector operations for large models.
        if self.args.record_params_every == 0:
            return

        self.grad_norms.append(self._last_grad_norm)

        if self.iterations % self.args.record_params_every == 0:
            param_vec = ch.nn.utils.parameters_to_vector(
                [p.contiguous() for p in self.model.parameters() if p.requires_grad]
            ).detach()
            self.param_history.append(param_vec)
            if self._ema_params is None:
                self._ema_params = param_vec
            else:
                self._ema_params = (
                    self.args.ema_decay * self._ema_params
                    + (1 - self.args.ema_decay) * param_vec
                )

    def _optimizer_step(self) -> None:
        """Perform the optimizer step, handling AMP and grad clipping."""
        if self.args.use_amp and self.scaler is not None:
            if self.args.max_grad_norm is not None:
                self.scaler.unscale_(self.model.optimizer)
                ch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
            self.scaler.step(self.model.optimizer)
            self.scaler.update()
        else:
            self.model.optimizer.step()

    def _schedule_step(self) -> None:
        """Step the LR scheduler after each optimizer update.

        Skips ``ReduceLROnPlateau`` schedulers, which require a validation
        metric and are stepped separately in ``train_model`` after each
        validation epoch.
        """
        if self.model.schedule is not None and not isinstance(
            self.model.schedule, ReduceLROnPlateau
        ):
            self.model.schedule.step()

    def train_step(self, batch: Iterable) -> ch.Tensor:
        """Run a single training step on the given batch and return the loss.

        Supports three paths:
        1. Custom: ``model.train_batch(batch)`` returns a dict — model owns the
           entire update; trainer skips grad-norm/EMA/param-history recording.
        2. AMP or gradient accumulation: inline forward-backward-step without
           the make_closure pattern (incompatible with LBFGS).
        3. Standard: ``optimizer.step(make_closure)`` for LBFGS compatibility.
        """
        # --- Custom path ---
        custom = self.model.train_batch(batch)
        if custom is not None:
            raw = custom.get("loss", ch.zeros(1))
            loss = raw if isinstance(raw, ch.Tensor) else ch.tensor(float(raw))
            self.train_losses.append(loss.detach())
            self.iterations += 1
            return loss

        accum = self.args.accumulate_grad_batches

        # --- AMP / accumulation path ---
        if self.args.use_amp or accum > 1:
            if self._accum_step == 0:
                self.model.optimizer.zero_grad()

            if self.args.use_amp and self.scaler is not None:
                with ch.autocast(device_type=self.args.device):
                    loss = self._forward_loss(batch)
                self.scaler.scale(loss / accum).backward()
            else:
                loss = self._forward_loss(batch)
                (loss / accum).backward()

            self._accum_step += 1
            if self._accum_step >= accum:
                self._optimizer_step()
                self._schedule_step()
                self._accum_step = 0
                self._record_step(loss)

            self.iterations += 1
            return loss

        # --- Standard path (closure, LBFGS-compatible) ---
        loss = self.model.optimizer.step(self.make_closure(batch))
        self._schedule_step()
        self._record_step(loss)
        self.iterations += 1
        return loss

    def val_step(self, batch: Iterable) -> ch.Tensor:
        """Run a single validation step on the given batch and return the loss.

        Delegates to ``model.compute_val_loss(batch)`` so subclasses can
        override train and val loss separately.
        """
        loss = self.model.compute_val_loss(batch)
        if loss.ndim > 0:
            loss = loss.sum(0)
        self.val_losses.append(loss)
        self.val_param_history_indices.append(len(self.param_history) - 1)
        return loss

    def _loaderless_step(self, loss_: AverageMeter) -> ch.Tensor:
        """Run one loader-free step and return the loss tensor.

        Called by ``run_epoch`` when no DataLoader is provided.  On the
        training side the model is expected to generate its own batch via
        ``train_batch(None)``; on the validation side ``evaluate()`` is used.

        Args:
            loss_: Running average meter updated in-place with the step loss.

        Returns:
            Loss tensor for the completed step.

        Raises:
            ValueError: If the model is training but ``train_batch`` returns
                ``None`` instead of a metrics dict.
        """
        if self.model.training:
            custom = self.model.train_batch(None)
            if custom is None:
                raise ValueError(
                    "train_loader is None but model.train_batch returned None. "
                    "Override train_batch to generate batches internally."
                )
            raw = custom.get("loss", ch.zeros(1))
            loss = raw if isinstance(raw, ch.Tensor) else ch.tensor(float(raw))
            self.train_losses.append(loss.detach())
            self.iterations += 1
            loss_.update(loss.item())
            stop, reason = self.should_stop()
            if stop:
                self.stop_reason = reason
        else:
            eval_metrics = self.model.evaluate()
            raw = eval_metrics.get("loss", ch.zeros(1))
            loss = raw if isinstance(raw, ch.Tensor) else ch.tensor(float(raw))
            self.val_losses.append(loss)
            self.val_param_history_indices.append(len(self.param_history) - 1)
            loss_.update(loss.item())
        return loss

    def _update_metric_meters(self, metric_meters: dict[str, AverageMeter]) -> None:
        """Accumulate per-batch model metrics into running averages.

        Args:
            metric_meters: Dict of metric name → AverageMeter, updated
                in-place from ``model.batch_metrics()``.
        """
        for name, val in self.model.batch_metrics().items():
            if name not in metric_meters:
                metric_meters[name] = AverageMeter()
            metric_meters[name].update(
                val.item() if isinstance(val, ch.Tensor) else float(val)
            )

    def _validate(self, val_loader: DataLoader | None) -> None:
        """Run mid-epoch validation if the configured interval has been reached.

        No-op when the model is not training, ``val_loader`` is ``None``,
        ``val_interval`` is unset, or the current step does not fall on the
        interval boundary.

        Args:
            val_loader: DataLoader for the validation set, or ``None``.
        """
        if (
            not self.model.training
            or val_loader is None
            or self.args.val_interval is None
            or self.iterations % self.args.val_interval != 0
        ):
            return
        self.model.eval()
        for val_batch in val_loader:
            self.update_best(self.val_step(val_batch))
        self.model.train()

    def _batch_epoch(
        self,
        loader: DataLoader,
        val_loader: DataLoader | None,
        mode: ProcedureStage,
        loss_: AverageMeter,
        metric_meters: dict[str, AverageMeter],
    ) -> ch.Tensor:
        """Iterate over a DataLoader for one full epoch.

        Args:
            loader: DataLoader for this epoch.
            val_loader: Optional DataLoader for mid-epoch validation.
            mode: Current procedure stage (used for tqdm descriptions).
            loss_: Running average meter updated in-place with batch losses.
            metric_meters: Per-metric average meters updated in-place.

        Returns:
            Loss tensor from the final batch.
        """
        iterator = (
            tqdm(enumerate(loader), total=len(loader), leave=False)
            if self.args.tqdm
            else enumerate(loader)
        )
        loss = ch.zeros(1)
        for batch_idx, batch in iterator:
            loss = (
                self.train_step(batch) if self.model.training else self.val_step(batch)
            )
            loss_.update(loss.item())
            self._update_metric_meters(metric_meters)

            stop, reason = self.should_stop()
            if stop:
                self.stop_reason = reason
                break

            self._validate(val_loader)

            if self.args.tqdm:
                iterator.set_description(
                    self.model.description(mode, self.epoch, batch_idx, loss_)
                )
        return loss

    def _log_epoch_summary(
        self,
        mode: ProcedureStage,
        loss_: AverageMeter,
        metric_meters: dict[str, AverageMeter],
    ) -> None:
        """Emit the single end-of-epoch log line.

        Args:
            mode: Current procedure stage (``TRAIN`` or ``VAL``).
            loss_: Epoch-averaged loss meter.
            metric_meters: Per-metric average meters for the epoch.
        """
        grad_norm_str = (
            f" grad_norm={self._last_grad_norm:.3e}"
            if self._last_grad_norm is not None
            else ""
        )
        metrics_str = "".join(f" {k}={m.avg:.4f}" for k, m in metric_meters.items())
        self.logger.info(
            f"[{mode}] epoch={self.epoch} step={self.iterations} "
            f"loss={loss_.avg:.4f}{metrics_str}{grad_norm_str}"
        )

    def run_epoch(
        self, loader: DataLoader | None, val_loader: DataLoader | None = None
    ) -> tuple[float, dict[str, float]]:
        """Run one full epoch and return ``(avg_loss, metrics_dict)``.

        Dispatches to ``_loaderless_step`` when no DataLoader is supplied
        (environment-driven algorithms such as RL or online EM), or to
        ``_batch_epoch`` for the standard DataLoader path.  Exactly one
        log line is emitted per call.

        Args:
            loader: DataLoader for the current epoch, or ``None`` for
                loader-free training / evaluation.
            val_loader: Optional DataLoader used for mid-epoch validation.

        Returns:
            Tuple of (avg_loss, metrics_dict) where metrics_dict maps each
            metric name to its epoch-averaged value.
        """
        mode = ProcedureStage.TRAIN if self.model.training else ProcedureStage.VAL
        loss_ = AverageMeter()
        metric_meters: dict[str, AverageMeter] = {}

        if loader is None:
            loss = self._loaderless_step(loss_)
        else:
            loss = self._batch_epoch(loader, val_loader, mode, loss_, metric_meters)

        self._log_epoch_summary(mode, loss_, metric_meters)
        self.model.post_epoch_hook(self.epoch, self.model.training, loss)
        return loss_.avg, {k: m.avg for k, m in metric_meters.items()}

    def should_stop(self):
        """Return (stop, reason) based on current training state.

        Called per step for iteration- and grad-norm-based criteria.
        Epoch-level criteria (loss_tol, patience) are checked in
        ``_check_epoch_stop`` after each full epoch. Also delegates to
        ``model.should_stop()`` so algorithms can inject custom stopping
        criteria (e.g. reward threshold in RL).

        Returns:
            Tuple of (bool, StopReason | None) where the second element
            is the stop reason.
        """
        if self.args.iterations is not None and self.iterations >= self.args.iterations:
            return True, StopReason.MAX_ITERATIONS

        if self.args.early_stopping and self.args.grad_tol > 0:
            window = max(1, self.args.grad_tol_window)
            if len(self.grad_norms) >= window:
                smoothed = sum(self.grad_norms[-window:]) / window
                if smoothed < self.args.grad_tol:
                    return True, StopReason.GRAD_TOL

        model_stop, _ = self.model.should_stop()
        if model_stop:
            return True, StopReason.MODEL_STOP

        return False, None

    def _check_epoch_stop(self) -> bool:
        """Check epoch-level stop criteria and set stop_reason if triggered.

        Compares consecutive epoch-level train losses for loss_tol and
        counts epochs (not batches) for patience-based early stopping.

        Returns:
            True if a stop criterion was met.
        """
        if not self.args.early_stopping:
            return False

        if self.args.loss_tol is not None and len(self.epoch_train_losses) > 1:
            delta = abs(self.epoch_train_losses[-1] - self.epoch_train_losses[-2])
            if delta < self.args.loss_tol:
                self.stop_reason = StopReason.LOSS_TOL
                return True

        patience = self.args.patience
        if patience is not None and len(self.epoch_val_losses) > patience:
            recent = self.epoch_val_losses[-patience - 1 :]
            if recent[-1] > min(recent[:-1]):
                self.stop_reason = StopReason.EARLY_STOP
                return True

        return False

    def _set_rand_seed(self, rand_seed: int) -> None:
        """Seed all RNGs for full reproducibility.

        Seeds the Python, NumPy, and PyTorch (CPU and, when available, all
        CUDA) random number generators.

        Args:
            rand_seed: Integer seed to apply to all RNGs.
        """
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        ch.manual_seed(rand_seed)
        if ch.cuda.is_available():
            ch.cuda.manual_seed_all(rand_seed)

    def _resolve_checkpoint_dir(self) -> str | None:
        """Return the directory for on-disk checkpoints, or None.

        Prefers the explicitly configured ``checkpoint_dir`` arg; falls back
        to ``store.path`` when a cox store is attached to the trainer.
        The caller is responsible for ensuring the directory exists.
        """
        configured = getattr(self.args, "checkpoint_dir", None)
        if configured is not None:
            return configured
        if self.store is not None and hasattr(self.store, "path"):
            return self.store.path
        return None

    def _save_best_model_state(self) -> None:
        """Persist the current model weights as the best checkpoint.

        Writes to disk when a checkpoint directory is available (via
        ``checkpoint_dir`` or an attached cox store).  Falls back to an
        in-memory ``state_dict`` copy when ``record_params_every > 0``
        and no disk target is configured.  When neither condition holds the
        method is a no-op and only the loss/param indices are tracked.
        """
        ckpt_dir = self._resolve_checkpoint_dir()
        if ckpt_dir is not None:
            ckpt_path = os.path.join(ckpt_dir, consts.CKPT_NAME_BEST)
            ch.save(self.model.state_dict(), ckpt_path)
            self._best_model_state = ckpt_path
            return

        if self.args.record_params_every > 0:
            self._best_model_state = copy.deepcopy(self.model.state_dict())
            return

        # Neither disk nor in-memory saving is active; track indices only.
        self._best_model_state = None

    def _save_periodic_checkpoint(self) -> None:
        """Write a full training checkpoint to disk for resumability.

        Saves model weights, optimizer state, scheduler state, epoch
        counter, and RNG states.  Written to ``checkpoint_dir`` (or
        ``store.path``) as ``CKPT_NAME_LATEST``.
        """
        ckpt_dir = self._resolve_checkpoint_dir()
        if ckpt_dir is None:
            return

        checkpoint = {
            CheckpointKey.EPOCH: self.epoch,
            CheckpointKey.MODEL: self.model.state_dict(),
            CheckpointKey.OPTIMIZER: (
                self.model.optimizer.state_dict()
                if self.model.optimizer is not None
                else None
            ),
            CheckpointKey.SCHEDULER: (
                self.model.schedule.state_dict()
                if self.model.schedule is not None
                else None
            ),
            CheckpointKey.RANDOM_STATES: {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "pytorch": ch.get_rng_state(),
            },
        }
        ckpt_path = os.path.join(ckpt_dir, consts.CKPT_NAME_LATEST)
        ch.save(checkpoint, ckpt_path)
        self.logger.info(f"Saved checkpoint at epoch {self.epoch} to {ckpt_path}.")

    def _init_logs_table(
        self,
        store: Store,
        train_metrics: dict,
        val_metrics: dict,
    ) -> None:
        """Create the training logs table with model-specific metric columns.

        Called lazily before the first row is inserted so that the full
        set of metric keys is known.  No-op if the table already exists.

        Args:
            store: Cox store for experiment logging.
            train_metrics: Metrics dict from the current training epoch.
            val_metrics: Metrics dict from the current validation epoch.
        """
        if self._store_logs_ready:
            return
        schema = {
            "epoch": int,
            "train_loss": float,
            "val_loss": float,
            "time": float,
        }
        schema.update({f"train_{k}": float for k in train_metrics})
        schema.update({f"val_{k}": float for k in val_metrics})
        store.add_table(consts.LOGS_TABLE, schema)
        self._store_logs_ready = True

    def _log_epoch(
        self,
        store: Store,
        train_loss: float,
        val_loss: float | None,
        train_metrics: dict,
        val_metrics: dict,
    ) -> None:
        """Build and append one training-log row to the cox store.

        Lazily initialises the logs table on the first call so that
        model-specific metric columns are included in the schema.

        Args:
            store: Cox store for experiment logging.
            train_loss: Average training loss for the epoch.
            val_loss: Average validation loss, or None if unavailable.
            train_metrics: Model-reported train metrics for this epoch.
            val_metrics: Model-reported validation metrics for this epoch.
        """
        self._init_logs_table(store, train_metrics, val_metrics)
        row: dict = {
            "epoch": self.epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time": time() - self.t_start,
        }
        row.update({f"train_{k}": float(v) for k, v in train_metrics.items()})
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        store[consts.LOGS_TABLE].append_row(row)

    def update_best(self, loss: ch.Tensor) -> None:
        """Update the best-loss index and save model weights if loss is new minimum."""
        loss_val = loss.item() if isinstance(loss, ch.Tensor) else float(loss)
        best_val = self.best_loss.item() if self.best_loss is not None else None
        if best_val is None or loss_val < best_val:
            self._best_loss_index = len(self.val_losses) - 1
            self._best_param_index = len(self.param_history) - 1
            self._save_best_model_state()

    def restore_best_weights(self) -> None:
        """Restore model weights from the best validation loss checkpoint.

        Handles both in-memory state dicts and filesystem paths produced
        by disk-based checkpointing.  No-op if no checkpoint is available.
        """
        if self._best_model_state is None:
            return
        if isinstance(self._best_model_state, str):
            state = ch.load(self._best_model_state, weights_only=True)
            self.model.load_state_dict(state)
        else:
            self.model.load_state_dict(self._best_model_state)
        self.logger.info("Restored model weights from best checkpoint.")

    def _val_epoch(self, val_loader: DataLoader | None) -> tuple[float | None, dict]:
        """Run one validation epoch and return ``(val_loss, val_metrics)``.

        Handles two cases:

        * **DataLoader provided**: runs ``run_epoch(val_loader)`` and calls
          ``update_best`` with the returned loss.
        * **No DataLoader**: delegates to ``model.evaluate()``; if the result
          contains a ``"loss"`` key the loss is recorded and ``update_best``
          is called, otherwise ``(None, {})`` is returned.

        In both cases ``ReduceLROnPlateau`` is stepped with the validation
        loss when one is active.

        Args:
            val_loader: DataLoader for the validation set, or ``None``.

        Returns:
            Tuple of ``(val_loss, val_metrics)``.  ``val_loss`` is ``None``
            when no DataLoader is provided and ``evaluate()`` returns no loss.
        """
        with ch.no_grad():
            self.model.eval()
            if val_loader is not None:
                val_loss, val_metrics = self.run_epoch(val_loader)
                self.update_best(val_loss)
            else:
                eval_metrics = self.model.evaluate()
                if "loss" not in eval_metrics:
                    return None, {}
                raw = eval_metrics["loss"]
                val_loss_t = (
                    raw if isinstance(raw, ch.Tensor) else ch.tensor(float(raw))
                )
                self.val_losses.append(val_loss_t)
                self.val_param_history_indices.append(len(self.param_history) - 1)
                self.update_best(val_loss_t)
                val_loss, val_metrics = val_loss_t.item(), {}

        if isinstance(self.model.schedule, ReduceLROnPlateau):
            self.model.schedule.step(val_loss)

        return val_loss, val_metrics

    def train_model(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        rand_seed: int = 0,
        store: Store = None,
        checkpoint=None,
    ):
        """Train the model until a stop condition is met.

        Pass ``train_loader=None`` for environment-driven algorithms (RL,
        online EM) where the model's ``train_batch(None)`` generates data
        internally. Pass ``val_loader=None`` to use ``model.evaluate()``
        for validation.

        Args:
            train_loader: DataLoader for the training set, or ``None``.
            val_loader: DataLoader for the validation set, or ``None``.
            rand_seed: Random seed for reproducibility.
            store: Optional cox store for logging epoch metrics.
            checkpoint: Optional checkpoint dict to resume from.

        Raises:
            ValueError: If train_loader has no samples (mapped datasets only).
        """
        if train_loader is not None:
            try:
                if len(train_loader.dataset) == 0:
                    raise ValueError("No datapoints in train loader.")
            except TypeError:
                pass  # IterableDataset does not support len().

        if store is not None:
            # Metadata table captures hyperparameter config; the logs table
            # is created lazily in _log_epoch after the first epoch so its
            # schema can include model-specific metric columns.
            setup_store_with_metadata(self.args, store)

        self._set_rand_seed(rand_seed)

        # Move model to the configured device.
        self.model.to(self.args.device)

        # Initialise mixed-precision scaler.
        if self.args.use_amp:
            self.scaler = ch.cuda.amp.GradScaler()

        self.t_start = time()
        self.model.pretrain_hook()

        # Record dataset size for algorithms that need it (e.g. mini-batch ELBO).
        if train_loader is not None:
            try:
                self.dataset_size = len(train_loader.dataset)
            except TypeError:
                self.dataset_size = None  # IterableDataset has no len().

        self.model.make_optimizer_and_schedule(checkpoint)
        self.logger.info(
            f"training: {self.model} with the following config:\n {self.args}"
        )

        if (
            self.args.early_stopping
            and self.args.grad_tol > 0
            and self.args.record_params_every == 0
        ):
            self.logger.warning(
                "grad_tol is set but record_params_every=0: grad norms are not "
                "tracked, so the grad_tol stop criterion will never fire. "
                "Set record_params_every > 0 to enable gradient norm tracking."
            )

        if checkpoint:
            self.epoch = checkpoint.get(CheckpointKey.EPOCH, 0)
            if val_loader is not None:
                with ch.no_grad():
                    self.model.eval()
                    self.run_epoch(val_loader)
                self.model.train()

        while True:
            self.epoch += 1
            self.model.train()
            train_loss, train_metrics = self.run_epoch(train_loader, val_loader)
            self.epoch_train_losses.append(train_loss)

            val_loss, val_metrics = self._val_epoch(val_loader)

            if val_loss is not None:
                self.epoch_val_losses.append(
                    val_loss if isinstance(val_loss, float) else float(val_loss)
                )

            # Check epoch-level stop criteria before per-step stop_reason.
            if not self.stop_reason:
                self._check_epoch_stop()

            if store is not None:
                self._log_epoch(store, train_loss, val_loss, train_metrics, val_metrics)

            if (
                self.args.checkpoint_every > 0
                and self.epoch % self.args.checkpoint_every == 0
            ):
                self._save_periodic_checkpoint()

            if self.stop_reason is not None:
                self.t_end = time()
                self.procedure_duration = self.t_end - self.t_start
                self.logger.info(
                    f"stopped due to {self.stop_reason} after {self.iterations} "
                    f"iterations. total time: {self.procedure_duration:.2f} seconds"
                )
                break

            if self.args.epochs is not None and self.epoch == self.args.epochs:
                self.stop_reason = StopReason.MAX_EPOCHS
                self.t_end = time()
                self.procedure_duration = self.t_end - self.t_start
                self.logger.info(
                    f"stopped due to {self.stop_reason} after {self.epoch} epochs. "
                    f"total time: {self.procedure_duration:.2f} seconds"
                )
                break

        self.train_losses = (
            ch.stack(self.train_losses) if self.train_losses else ch.tensor([])
        )
        self.val_losses = (
            ch.stack(self.val_losses) if self.val_losses else ch.tensor([])
        )
        self.param_history = (
            ch.stack(self.param_history) if self.param_history else ch.empty(0)
        )
        self.grad_norms = ch.tensor(self.grad_norms)

        self.model.post_training_hook()

    def eval_model(self, loader: DataLoader, store: Store = None):
        """Evaluate the model on a held-out set and return metrics.

        Args:
            loader: DataLoader for the evaluation set.
            store: Optional cox store for saving results.

        Returns:
            Dict with keys 'test_loss' and 'time', plus any keys from
            ``model.batch_metrics()``.
        """
        start_time = time()

        if store is not None:
            store.add_table(consts.EVAL_LOGS_TABLE, consts.EVAL_LOGS_SCHEMA)

        self.model.eval()
        test_loss, test_metrics = self.run_epoch(loader)

        log_info = {"test_loss": test_loss, "time": time() - start_time, **test_metrics}
        if store:
            store[consts.EVAL_LOGS_TABLE].append_row(log_info)
        return log_info

    @property
    def val_param_history(self):
        """Return parameter history at validation steps, or None if not recorded."""
        if self.param_history.numel() == 0:
            return None
        return self.param_history[self.val_param_history_indices]

    @property
    def best_params(self):
        """Return the parameter vector from the best validation loss step, or None."""
        if self._best_param_index is None or self.param_history.numel() == 0:
            return None
        return self.param_history[self._best_param_index]

    @property
    def best_loss(self):
        """Return the best validation loss seen so far, or None."""
        if self._best_loss_index is None:
            return None
        return self.val_losses[self._best_loss_index]

    @property
    def final_params(self):
        """Return the parameter vector from the last training step, or None."""
        if self.param_history.numel() == 0:
            return None
        return self.param_history[-1]

    @property
    def final_loss(self):
        """Return the loss at the final validation step, or None."""
        if len(self.val_losses) == 0:
            return None
        return self.val_losses[-1]

    @property
    def best_model_state(self):
        """Return the saved state dict from the best validation loss step."""
        return self._best_model_state

    @property
    def ema_params(self):
        """Return the exponential moving average of parameter vectors, or None."""
        return self._ema_params

    @property
    def avg_params(self):
        """Return the mean of all parameter vectors seen during training, or None."""
        if self.param_history.numel() == 0:
            return None
        return self.param_history.mean(0)
