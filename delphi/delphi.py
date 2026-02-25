# Author: pstefanou12@
"""Parent class for models to train in the delphi trainer."""

import random
from typing import Callable, ClassVar, Iterable
import torch as ch
from cox.store import Store
from torch.optim import SGD, LBFGS, Adam, AdamW, lr_scheduler
import numpy as np

from delphi.delphi_logger import delphiLogger
from delphi.utils.constants import (
    CheckpointKey,
    OptimizerType,
    RandomStateKey,
    SchedulerType,
)
from delphi.utils.defaults import (
    check_and_fill_args,
    DELPHI_DEFAULTS,
    SGD_DEFAULTS,
    LBFGS_DEFAULTS,
    ADAM_DEFAULTS,
    ADAMW_DEFAULTS,
)
from delphi.utils.helpers import AverageMeter, Parameters


class delphi(ch.nn.Module):  # pylint: disable=invalid-name,too-many-instance-attributes,abstract-method
    """Parent/abstract class for models to be passed into the trainer.

    Subclasses may register custom optimizers via the class-level registry::

        delphi.register_optimizer("adamw", lambda m, p: AdamW(p, lr=m.args.lr))

    Note: register_optimizer writes to the class on which it is called.
    Subclasses that want an isolated registry should shadow _OPTIMIZER_REGISTRY
    in their own class body before registering.
    """

    _OPTIMIZER_REGISTRY: ClassVar[dict[str, Callable]] = {}

    def __init__(
        self,
        args: Parameters,
        logger: delphiLogger,
        store: Store = None,
        checkpoint=None,
    ):
        """Initialize the delphi model.

        Args:
            args: Hyperparameter object; see DELPHI_DEFAULTS for supported keys.
            logger: Logger instance for training diagnostics.
            store: Optional cox store for experiment logging.
            checkpoint: Optional checkpoint dict to resume from.

        Raises:
            TypeError: If args is not a Parameters instance.
            TypeError: If store is not a cox.store.Store instance or None.
            TypeError: If checkpoint is not a dict or None.
        """
        super().__init__()
        if not isinstance(args, Parameters):
            raise TypeError(
                f"args is type {type(args).__name__}; "
                "expected delphi.utils.helpers.Parameters"
            )
        self.args = check_and_fill_args(args, DELPHI_DEFAULTS)
        self.logger = logger

        self.best_loss, self.best_model = None, None
        self.optimizer, self.schedule = None, None
        self.start_epoch = 0

        if store is not None and not isinstance(store, Store):
            raise TypeError(
                f"store is type {type(store).__name__}; expected cox.store.Store"
            )
        self.store = store

        if checkpoint is not None and not isinstance(checkpoint, dict):
            raise TypeError(
                f"checkpoint is type {type(checkpoint).__name__}; expected dict"
            )
        self.checkpoint = checkpoint

        self.criterion = None
        self.criterion_params = []
        self.model = None
        # Additional optimizer slots for multi-optimizer algorithms (GANs, etc.).
        self.optimizers: dict = {}

    def parameter_groups(self) -> list[dict]:
        """Return optimizer parameter groups.

        Override to assign different hyperparameters (e.g. learning rate or
        weight decay) to different subsets of the model's parameters::

            def parameter_groups(self):
                return [
                    {"params": list(self.backbone.parameters()), "lr": 1e-4},
                    {"params": list(self.head.parameters()), "lr": 1e-3},
                ]

        Returns:
            List of parameter-group dicts accepted by ``torch.optim.Optimizer``.
            The default implementation wraps all trainable parameters in a
            single group with no extra hyperparameters.
        """
        return [{"params": list(self.parameters())}]

    def make_optimizer_and_schedule(self, params: Iterable, checkpoint: dict = None):
        """Create and store the optimizer and learning-rate scheduler.

        The ``params`` argument is accepted for backwards compatibility but
        is ignored when ``parameter_groups()`` is overridden; override
        ``parameter_groups()`` to control per-group hyperparameters.

        Args:
            params: Ignored; kept for API compatibility.
            checkpoint: Optional dict containing optimizer/scheduler state to restore.

        Returns:
            Tuple of (optimizer, scheduler).
        """
        groups = self.parameter_groups()

        self.optimizer = self._create_optimizer(groups)
        self.optimizer.register_step_pre_hook(self.step_pre_hook)
        self.optimizer.register_step_post_hook(self.step_post_hook)

        self.schedule = self._create_scheduler()

        if checkpoint:
            self._load_checkpoint(checkpoint)

        return self.optimizer, self.schedule

    @classmethod
    def register_optimizer(cls, name: str, factory: Callable) -> None:
        """Register a custom optimizer factory under the given name.

        Args:
            name: Lowercase optimizer name (e.g. ``"adamw"``).
            factory: Callable with signature ``(model, params) -> Optimizer``
                where ``model`` is the ``delphi`` instance and ``params``
                is the iterable of parameters to optimise.
        """
        cls._OPTIMIZER_REGISTRY[name.lower()] = factory

    def _create_optimizer(self, params):
        """Create and return the configured optimizer via the registry."""
        optimizer_type = self._get_optimizer_type()

        if optimizer_type not in self._OPTIMIZER_REGISTRY:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        return self._OPTIMIZER_REGISTRY[optimizer_type](self, params)

    def _get_optimizer_type(self):
        """Return the lowercase optimizer type string from args."""
        return self.args.optimizer.lower()

    def _get_arg(self, name, default):
        """Return args.name if set and not None, else default."""
        val = getattr(self.args, name, None)
        return val if val is not None else default

    def _remove_none_config(self, config):
        """Return a copy of config with None values removed."""
        return {k: v for k, v in config.items() if v is not None}

    def _create_sgd(self, params):
        """Create an SGD optimizer from args."""
        check_and_fill_args(self.args, SGD_DEFAULTS)
        config = {
            "lr": self.args.lr,
            "momentum": getattr(self.args, "momentum", 0),
            "dampening": getattr(self.args, "dampening", 0),
            "weight_decay": getattr(self.args, "weight_decay", 0),
            "nesterov": getattr(self.args, "nesterov", False),
            "maximize": getattr(self.args, "maximize", False),
            "foreach": getattr(self.args, "foreach", None),
            "differentiable": getattr(self.args, "differentiable", False),
            "fused": getattr(self.args, "fused", None),
        }
        return SGD(params, **self._remove_none_config(config))

    def _create_lbfgs(self, params):
        """Create an L-BFGS optimizer from args."""
        check_and_fill_args(self.args, LBFGS_DEFAULTS)
        config = {
            "lr": getattr(self.args, "lr", 1.0),
            "max_iter": getattr(self.args, "max_iter", 20),
            "max_eval": getattr(self.args, "max_eval", None),
            "tolerance_grad": getattr(self.args, "tolerance_grad", 1e-7),
            "tolerance_change": getattr(self.args, "tolerance_change", 1e-9),
            "history_size": getattr(self.args, "history_size", 100),
            "line_search_fn": getattr(self.args, "line_search_fn", None),
        }
        return LBFGS(params, **self._remove_none_config(config))

    def _create_adam(self, params):
        """Create an Adam optimizer from args."""
        check_and_fill_args(self.args, ADAM_DEFAULTS)
        config = {
            "lr": getattr(self.args, "lr", 1e-1),
            "betas": (
                getattr(self.args, "beta1", 0.9),
                getattr(self.args, "beta2", 0.999),
            ),
            "eps": getattr(self.args, "eps", 1e-8),
            "weight_decay": getattr(self.args, "weight_decay", 0),
            "amsgrad": getattr(self.args, "amsgrad", False),
            "maximize": getattr(self.args, "maximize", False),
            "foreach": getattr(self.args, "foreach", None),
            "capturable": getattr(self.args, "capturable", False),
            "differentiable": getattr(self.args, "differentiable", False),
            "fused": getattr(self.args, "fused", None),
        }
        return Adam(params, **self._remove_none_config(config))

    def _create_adamw(self, params):
        """Create an AdamW optimizer from args."""
        check_and_fill_args(self.args, ADAMW_DEFAULTS)
        config = {
            "lr": getattr(self.args, "lr", 1e-3),
            "betas": (
                getattr(self.args, "beta1", 0.9),
                getattr(self.args, "beta2", 0.999),
            ),
            "eps": getattr(self.args, "eps", 1e-8),
            "weight_decay": getattr(self.args, "weight_decay", 1e-2),
            "amsgrad": getattr(self.args, "amsgrad", False),
            "maximize": getattr(self.args, "maximize", False),
            "foreach": getattr(self.args, "foreach", None),
            "capturable": getattr(self.args, "capturable", False),
            "differentiable": getattr(self.args, "differentiable", False),
            "fused": getattr(self.args, "fused", None),
        }
        return AdamW(params, **self._remove_none_config(config))

    def _create_scheduler(self):
        """Create and return the configured learning-rate scheduler, or None."""
        if getattr(self.args, "constant", False):
            return None

        scheduler_type = self._get_scheduler_type()
        if not scheduler_type:
            return None

        scheduler_creators = {
            SchedulerType.CYCLIC: self._create_cyclic_scheduler,
            SchedulerType.COSINE: self._create_cosine_scheduler,
            SchedulerType.LINEAR: self._create_linear_scheduler,
            SchedulerType.STEP: self._create_step_scheduler,
            SchedulerType.MULTI_STEP: self._create_multi_step_scheduler,
            SchedulerType.EXPONENTIAL: self._create_exponential_scheduler,
            SchedulerType.REDUCE_ON_PLATEAU: self._create_plateau_scheduler,
        }

        if scheduler_type not in scheduler_creators:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        base_schedule = scheduler_creators[scheduler_type]()

        # Optionally prepend a linear warmup phase.
        warmup_steps = self._get_arg("warmup_steps", 0)
        is_plateau = isinstance(base_schedule, lr_scheduler.ReduceLROnPlateau)
        if warmup_steps > 0 and not is_plateau:
            warmup = lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            return lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, base_schedule],
                milestones=[warmup_steps],
            )

        return base_schedule

    def _get_scheduler_type(self):
        """Return the lowercase scheduler type string from args, or None."""
        # Explicit scheduler arg takes priority over legacy step_lr.
        if hasattr(self.args, "scheduler") and self.args.scheduler:
            return self.args.scheduler.lower()
        return None

    def _create_cyclic_scheduler(self):
        """Create a cyclic LR scheduler using CyclicLR.

        Configurable via args:
            cyclic_base_lr (float): Lower LR bound (default 0.0).
            cyclic_max_lr (float): Upper LR bound (default: optimizer's lr).
            cyclic_step_size_up (int): Steps in ascending half-cycle (default 2000).
            cyclic_step_size_down (int): Steps in descending half-cycle (default: same).
            cyclic_mode (str): 'triangular', 'triangular2', or 'exp_range'.
            cyclic_gamma (float): Multiplicative factor for 'exp_range' mode.
        """
        max_lr = self._get_arg("cyclic_max_lr", self.optimizer.param_groups[0]["lr"])
        config = {
            "base_lr": self._get_arg("cyclic_base_lr", 0.0),
            "max_lr": max_lr,
            "step_size_up": self._get_arg("cyclic_step_size_up", 2000),
            "step_size_down": self._get_arg("cyclic_step_size_down", None),
            "mode": self._get_arg("cyclic_mode", "triangular2"),
            "gamma": self._get_arg("cyclic_gamma", 1.0),
            "cycle_momentum": False,  # Kept False for Adam/AdamW compatibility.
        }
        return lr_scheduler.CyclicLR(self.optimizer, **self._remove_none_config(config))

    def _create_cosine_scheduler(self):
        """Create a cosine annealing scheduler."""
        config = {
            "T_max": self._get_arg("epochs", 100),
            "eta_min": self._get_arg("min_lr", 0),
        }
        return lr_scheduler.CosineAnnealingLR(
            self.optimizer, **self._remove_none_config(config)
        )

    def _create_linear_scheduler(self):
        """Create a linear LR decay scheduler."""
        config = {
            "start_factor": self._get_arg("linear_start_factor", 1.0),
            "end_factor": self._get_arg("linear_end_factor", 0.0),
            "total_iters": self._get_arg("epochs", 100),
        }
        return lr_scheduler.LinearLR(self.optimizer, **self._remove_none_config(config))

    def _create_step_scheduler(self):
        """Create a step LR scheduler."""
        config = {
            "step_size": getattr(self.args, "step_lr", 100),
            "gamma": getattr(self.args, "step_lr_gamma", 0.1),
        }
        return lr_scheduler.StepLR(self.optimizer, **self._remove_none_config(config))

    def _create_multi_step_scheduler(self):
        """Create a multi-step LR scheduler."""
        config = {
            "milestones": getattr(self.args, "milestones", [30, 60, 90]),
            "gamma": getattr(self.args, "gamma", 0.1),
        }
        return lr_scheduler.MultiStepLR(
            self.optimizer, **self._remove_none_config(config)
        )

    def _create_exponential_scheduler(self):
        """Create an exponential LR scheduler."""
        config = {
            "gamma": getattr(self.args, "gamma", 0.95),
        }
        return lr_scheduler.ExponentialLR(self.optimizer, **config)

    def _create_plateau_scheduler(self):
        """Create a reduce-on-plateau scheduler."""
        config = {
            "mode": getattr(self.args, "plateau_mode", "min"),
            "factor": getattr(self.args, "plateau_factor", 0.1),
            "patience": getattr(self.args, "plateau_patience", 10),
            "threshold": getattr(self.args, "plateau_threshold", 1e-4),
            "threshold_mode": getattr(self.args, "plateau_threshold_mode", "rel"),
            "cooldown": getattr(self.args, "plateau_cooldown", 0),
            "min_lr": getattr(self.args, "min_lr", 0),
            "eps": getattr(self.args, "plateau_eps", 1e-8),
        }
        return lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **self._remove_none_config(config)
        )

    def _load_checkpoint(self, checkpoint):
        """Restore model weights, optimizer, scheduler, and random states from checkpoint."""
        if CheckpointKey.MODEL in checkpoint:
            self.load_state_dict(checkpoint[CheckpointKey.MODEL])
            self.logger.info("Loaded model weights from checkpoint.")

        if CheckpointKey.OPTIMIZER in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint[CheckpointKey.OPTIMIZER])
            self.logger.info("Loaded optimizer state from checkpoint.")

        if (
            CheckpointKey.SCHEDULER in checkpoint
            and hasattr(self, "schedule")
            and self.schedule is not None
        ):
            self.schedule.load_state_dict(checkpoint[CheckpointKey.SCHEDULER])
            self.logger.info("Loaded scheduler state from checkpoint.")

            # Advance the scheduler to match the saved training position.
            current_epoch = checkpoint.get(CheckpointKey.EPOCH, 0)
            for _ in range(current_epoch):
                self.schedule.step()
            self.logger.info(f"Advanced scheduler to epoch {current_epoch}.")

        if CheckpointKey.RANDOM_STATES in checkpoint:
            self._load_random_states(checkpoint[CheckpointKey.RANDOM_STATES])

        if CheckpointKey.TRAINING_STATE in checkpoint:
            self._load_training_state(checkpoint[CheckpointKey.TRAINING_STATE])

    def _load_random_states(self, random_states):
        """Restore Python, NumPy, and PyTorch random number generator states."""
        if RandomStateKey.PYTHON in random_states:
            random.setstate(random_states[RandomStateKey.PYTHON])

        if RandomStateKey.NUMPY in random_states:
            np.random.set_state(random_states[RandomStateKey.NUMPY])

        if RandomStateKey.PYTORCH in random_states:
            ch.set_rng_state(random_states[RandomStateKey.PYTORCH])

        self.logger.info("Restored random number generator states.")

    def _load_training_state(self, training_state):
        """Restore epoch counter and best loss from a training state dict."""
        self.start_epoch = training_state.get(CheckpointKey.EPOCH, 0)
        self.best_loss = training_state.get(CheckpointKey.BEST_LOSS, float("inf"))
        self.logger.info(f"Resuming from epoch {self.start_epoch}.")

    def pretrain_hook(self) -> None:
        """Hook called before the training procedure begins."""

    def step_pre_hook(self, optimizer, args, kwargs) -> None:  # pylint: disable=unused-argument
        """Hook called after .backward() but before the optimizer step."""

    def step_post_hook(self, optimizer, args, kwargs) -> None:  # pylint: disable=unused-argument
        """Hook called after each optimizer step."""

    def post_epoch_hook(self, i, is_train, loss) -> None:  # pylint: disable=unused-argument
        """Hook called after each complete pass through the dataset."""

    def post_training_hook(self) -> None:
        """Hook called after the full training procedure completes."""

    def description(
        self, stage: str, epoch: int, step: int, loss_: AverageMeter
    ) -> str:
        """Return a human-readable tqdm progress string for the current step."""
        return f"[{stage}] Epoch {epoch} | Step {step} | Loss {loss_.avg:.4f}"

    def regularize(self, batch) -> ch.Tensor:  # pylint: disable=unused-argument
        """Return the regularization term to add to the loss. Defaults to zero."""
        return ch.zeros(1, 1)

    def parameters_(self):
        """Return the model's trainable parameters."""
        return self.parameters()

    def compute_loss(self, batch) -> ch.Tensor:
        """Compute the loss for a single batch.

        Override to implement custom forward and loss logic. The default
        implementation unpacks ``batch`` as ``(inp, targ)`` and evaluates
        ``self.criterion``.

        Args:
            batch: A single batch from the DataLoader.

        Returns:
            Loss tensor (scalar or vector; the trainer handles reduction).
        """
        inp, targ = batch
        pred = self(inp)
        pred_args = pred if isinstance(pred, (tuple, list)) else (pred,)
        return self.criterion(*pred_args, targ, *self.criterion_params)  # pylint: disable=not-callable

    def compute_val_loss(self, batch) -> ch.Tensor:
        """Compute the validation loss for a single batch.

        Defaults to ``compute_loss``. Override to use a different metric
        during validation (e.g. NLL instead of ELBO).

        Args:
            batch: A single batch from the validation DataLoader.

        Returns:
            Loss tensor (scalar or vector; the trainer handles reduction).
        """
        return self.compute_loss(batch)

    def train_batch(self, batch) -> dict | None:
        """Run a full training update for one batch.

        Return ``None`` (default) to let the trainer use the standard
        ``make_closure`` → ``optimizer.step`` path. Return a metrics dict
        to take full ownership of the update; grad-norm recording, EMA,
        and param-history are then skipped by the trainer. The dict must
        contain at least a ``"loss"`` key.

        This hook is the right place for algorithms that require multiple
        optimizer steps per batch (GANs, actor-critic RL) or that do not
        use gradient-based optimizers (EM, HMC).

        Args:
            batch: A single batch from the DataLoader, or ``None`` when
                training without a loader (e.g. environment-driven RL).

        Returns:
            ``None`` to use the default training path, or a dict with at
            least ``"loss"``.
        """
        return None

    def evaluate(self) -> dict:
        """Compute validation metrics without a DataLoader.

        Called by the trainer when no val_loader is provided. Return a
        dict with at least ``"loss"`` to enable early stopping and
        best-model tracking. The default returns an empty dict.

        Returns:
            Metrics dict, e.g. ``{"loss": tensor, "reward": float}``.
        """
        return {}

    def should_stop(self) -> tuple[bool, str | None]:
        """Return whether training should stop and why.

        Called by the trainer's stop-criterion check in addition to the
        standard criteria (max_iterations, grad_tol, etc.). Return
        ``(True, reason_string)`` to halt training. The trainer records
        the stop reason as ``"model_stop"`` regardless of the returned
        string; log any detail from within this method before returning.

        Returns:
            Tuple of (stop, reason) where reason is a short string or None.
        """
        return False, None

    def batch_metrics(self) -> dict:
        """Return per-batch metrics for the trainer to track.

        Called by the trainer after each train and val step. Return a dict
        of metric names to scalar values; the trainer accumulates running
        averages across the epoch. Override for classification accuracy,
        reward signals, ELBO components, etc.

        Returns:
            Dict mapping metric names to scalar or float values.
        """
        return {}


# Populate the built-in optimizer registry after the class is fully defined.
delphi._OPTIMIZER_REGISTRY = {
    OptimizerType.SGD: delphi._create_sgd,
    OptimizerType.LBFGS: delphi._create_lbfgs,
    OptimizerType.ADAM: delphi._create_adam,
    OptimizerType.ADAMW: delphi._create_adamw,
}
