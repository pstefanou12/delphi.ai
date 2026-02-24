# Author: pstefanou12@
"""Module used for training models."""

from time import time
from typing import Iterable
import torch as ch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from cox.store import Store

from delphi.delphi import delphi
from delphi.delphi_logger import delphiLogger
from delphi.utils import constants as consts
from delphi.utils.helpers import AverageMeter, setup_store_with_metadata, Parameters
from delphi.utils.defaults import TRAINER_DEFAULTS, check_and_fill_args


STOP_REASONS = ["grad_tol", "loss_tol", "early_stop", "max_iterations", "max_epochs"]


def ensure_tuple(x):
    """Wrap x in a tuple if it is not already a tuple or list."""
    return x if isinstance(x, (tuple, list)) else (x,)


class Trainer:  # pylint: disable=too-many-instance-attributes
    """Trainer class for fitting delphi models using iterative optimization."""

    def __init__(
        self, model: delphi, args: Parameters, logger: delphiLogger, store=None
    ):
        """Initialize the Trainer.

        Args:
            model: The delphi model to train.
            args: Hyperparameter object; see TRAINER_DEFAULTS for supported keys.
            logger: Logger instance for training diagnostics.
            store: Optional cox store for experiment logging.
        """
        self.model = model
        self.args = check_and_fill_args(args, TRAINER_DEFAULTS)
        self.logger = logger
        self.store = store

        self.epoch, self.iterations = 0, 0
        # Store procedure history in lists to avoid torch.cat overhead.
        self.train_losses = []
        self.val_losses = []
        self.param_history = []
        self.val_param_history_indices = []
        self.grad_norms = []
        self._ema_params = None

        self.t_start, self.t_end = None, None
        self.procedure_duration = None
        self._best_loss_index, self._best_param_index = None, None
        self.stop_reason = None

    def make_closure(self, batch):
        """Create an optimizer closure for the given batch."""
        inp, targ = batch

        def closure():
            self.model.optimizer.zero_grad()
            pred = self.model(inp)
            loss = self.model.criterion(
                *ensure_tuple(pred), targ, *self.model.criterion_params
            )

            if loss.ndim > 0:
                loss = loss.sum()

            reg_term = self.model.regularize(batch)

            if self.args.cuda:
                reg_term = reg_term.cuda()
            loss = loss + reg_term
            loss.backward()
            return loss

        return closure

    def train_step(self, batch: Iterable) -> ch.Tensor:
        """Run a single training step on the given batch and return the loss."""
        loss = self.model.optimizer.step(self.make_closure(batch))
        if self.model.schedule is not None and not isinstance(
            self.model.schedule, ReduceLROnPlateau
        ):
            self.model.schedule.step()

        self.train_losses.append(loss.detach())

        grad_norm = ch.nn.utils.parameters_to_vector(
            [p.grad.contiguous() for p in self.model.parameters() if p.requires_grad]
        ).norm()
        self.grad_norms.append(grad_norm.item())

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

        self.iterations += 1

        return loss

    def val_step(self, batch: Iterable) -> ch.Tensor:
        """Run a single validation step on the given batch and return the loss."""
        inp, targ = batch
        pred = self.model(inp)
        loss = self.model.criterion(
            *ensure_tuple(pred), targ, *self.model.criterion_params
        )
        if loss.ndim > 0:
            loss = loss.sum(0)
        self.val_losses.append(loss)
        self.val_param_history_indices.append(len(self.param_history) - 1)
        return loss

    def run_epoch(self, loader: DataLoader, val_loader: DataLoader | None = None):
        """Run one full epoch over loader and return (avg_loss, avg_prec1, avg_prec5).

        Args:
            loader: DataLoader for the current epoch (train or val).
            val_loader: Optional DataLoader used for mid-epoch validation.

        Returns:
            Tuple of (avg_loss, avg_prec1, avg_prec5) AverageMeter averages.
        """
        mode = "train" if self.model.training else "val"
        loss_, prec1_, prec5_ = AverageMeter(), AverageMeter(), AverageMeter()
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

            stop, reason = self.should_stop()
            if stop:
                self.stop_reason = reason
                break

            # Validate periodically during a training epoch.
            if (
                self.model.training
                and val_loader is not None
                and self.args.val_interval is not None
                and self.iterations % self.args.val_interval == 0
            ):
                self.model.eval()
                for val_batch in val_loader:
                    val_loss = self.val_step(val_batch)
                    self.update_best(val_loss)
                self.model.train()

            if self.args.tqdm:
                desc = self.model.description(
                    self.epoch, batch_idx, loss_, prec1_, prec5_, None
                )
                iterator.set_description(desc)

            if self.model.training and (self.iterations % self.args.log_every == 0):
                self.logger.info(
                    f"[{mode}] epoch={self.epoch} step={self.iterations} "
                    f"loss={loss_.avg:.4f} grad_norm={self.grad_norms[-1]:.3e}"
                )

        grad_norm_str = (
            f" grad_norm={self.grad_norms[-1]:.3e}" if self.grad_norms else ""
        )
        self.logger.info(
            f"[{mode}] epoch={self.epoch} step={self.iterations} "
            f"loss={loss_.avg:.4f}{grad_norm_str}"
        )

        self.model.post_epoch_hook(self.epoch, self.model.training, loss)
        return loss_.avg, prec1_.avg, prec5_.avg

    def should_stop(self):
        """Return (stop, reason) based on current training state.

        Returns:
            Tuple of (bool, str | None) where the string is the stop reason.
        """
        if self.args.iterations is not None and self.iterations >= self.args.iterations:
            return True, "max_iterations"

        if self.args.early_stopping:
            if (
                self.args.grad_tol is not None
                and len(self.grad_norms) > 1
                and self.grad_norms[-1] < self.args.grad_tol
            ):
                return True, "grad_tol"

            if self.args.loss_tol is not None and len(self.train_losses) > 1:
                delta = abs(self.train_losses[-1].item() - self.train_losses[-2].item())
                if delta < self.args.loss_tol:
                    return True, "loss_tol"

            if len(self.val_losses) > self.args.patience:
                recent = self.val_losses[-self.args.patience :]
                if recent[-1].item() > min(r.item() for r in recent[:-1]):
                    return True, "early_stop"

        return False, None

    def update_best(self, loss: ch.Tensor):
        """Update the best-loss index if loss is a new minimum."""
        if self.best_loss is None or loss.item() < self.best_loss.item():
            self._best_loss_index = len(self.val_losses) - 1
            self._best_param_index = len(self.param_history) - 1

    def train_model(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        rand_seed: int = 0,
        store: Store = None,
        checkpoint=None,
    ):
        """Train the model until a stop condition is met.

        Args:
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            rand_seed: Random seed for reproducibility.
            store: Optional cox store for logging epoch metrics.
            checkpoint: Optional checkpoint dict to resume from.

        Raises:
            ValueError: If train_loader contains no samples.
        """
        if len(train_loader.dataset) == 0:
            raise ValueError("No datapoints in train loader.")

        if store is not None:
            store.add_table(
                "logs",
                {
                    "epoch": int,
                    "train_loss": float,
                    "train_prec1": float,
                    "train_prec5": float,
                    "val_loss": float,
                    "val_prec1": float,
                    "val_prec5": float,
                },
            )
            setup_store_with_metadata(self.args, store)

        ch.manual_seed(rand_seed)
        self.t_start = time()
        self.model.pretrain_hook()

        self.model.make_optimizer_and_schedule(self.model.parameters_())
        self.logger.info(
            f"training: {self.model} with the following config:\n {self.args}"
        )

        if checkpoint:
            self.epoch = checkpoint["epoch"]
            _ = (  # pylint: disable=unused-variable
                checkpoint["prec1"]
                if "prec1" in checkpoint
                else self.run_epoch(val_loader)[0]
            )

        while True:
            self.epoch += 1
            self.model.train()
            train_loss, train_prec1, train_prec5 = self.run_epoch(
                train_loader, val_loader
            )

            val_loss, val_prec1, val_prec5 = None, None, None
            if val_loader is not None:
                with ch.no_grad():
                    self.model.eval()
                    val_loss, val_prec1, val_prec5 = self.run_epoch(val_loader)
                    self.update_best(val_loss)

                if isinstance(self.model.schedule, ReduceLROnPlateau):
                    self.model.schedule.step(val_loss)

            if self.stop_reason in STOP_REASONS:
                self.t_end = time()
                self.procedure_duration = self.t_end - self.t_start
                self.logger.info(
                    f"stopped due to {self.stop_reason} after {self.iterations} "
                    f"iterations. total time: {self.procedure_duration:.2f} seconds"
                )
                break

            if store is not None and val_loader is not None:
                store["logs"].append_row(
                    {
                        "epoch": self.epoch,
                        "train_loss": train_loss,
                        "train_prec1": train_prec1,
                        "train_prec5": train_prec5,
                        "val_loss": val_loss,
                        "val_prec1": val_prec1,
                        "val_prec5": val_prec5,
                    }
                )

            if self.args.epochs is not None and self.epoch == self.args.epochs:
                self.stop_reason = "max_epochs"
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
            Dict with keys 'test_prec1', 'test_loss', and 'time'.
        """
        start_time = time()

        if store is not None:
            store.add_table("eval", consts.EVAL_LOGS_SCHEMA)

        self.model.eval()
        test_loss, test_prec1, _ = self.run_epoch(loader)

        log_info = {
            "test_prec1": test_prec1,
            "test_loss": test_loss,
            "time": time() - start_time,
        }
        if store:
            store["eval"].append_row(log_info)
        return log_info

    @property
    def val_param_history(self):
        """Return parameter history at validation steps."""
        return self.param_history[self.val_param_history_indices]

    @property
    def best_params(self):
        """Return the parameter vector from the best validation loss step."""
        if self._best_param_index is None:
            return None
        return self.param_history[self._best_param_index]

    @property
    def best_loss(self):
        """Return the best validation loss seen so far."""
        if self._best_loss_index is None:
            return None
        return self.val_losses[self._best_loss_index]

    @property
    def final_params(self):
        """Return the parameter vector from the last training step."""
        return self.param_history[-1]

    @property
    def final_loss(self):
        """Return the loss at the final validation step."""
        return self.val_param_history[-1]

    @property
    def ema_params(self):
        """Return the exponential moving average of parameter vectors."""
        return self._ema_params

    @property
    def avg_params(self):
        """Return the mean of all parameter vectors seen during training."""
        return self.param_history.mean(0)
