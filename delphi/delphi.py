# Author: pstefanou12@
"""Parent class for models to train in the delphi trainer."""

import random
from collections.abc import Callable
from typing import ClassVar, Generic, Optional, TypeVar, cast

import numpy as np
import torch as ch
import torch.optim
import torch.optim.lr_scheduler

from delphi.utils import configs, constants, helpers


_T = TypeVar("_T")


class delphi(ch.nn.Module, Generic[configs._ConfigT]):  # pylint: disable=invalid-name,too-many-instance-attributes,abstract-method
    """Parent/abstract class for models to be passed into the trainer.

    Concrete subclasses must define a ``defaults`` class attribute pointing to
    their Pydantic config class::

        class MyModel(delphi[MyConfig]):
            defaults = MyConfig

    Subclasses may also register custom optimizers via the class-level registry::

        delphi.register_optimizer("adamw", lambda m, p: AdamW(p, lr=m.hyperparameters.lr))

    Note: register_optimizer writes to the class on which it is called.
    Subclasses that want an isolated registry should shadow _OPTIMIZER_REGISTRY
    in their own class body before registering.
    """

    _OPTIMIZER_REGISTRY: ClassVar[dict[str, Callable]] = {}
    defaults: type[configs._ConfigT]

    def __init__(self, hyperparameters: dict):
        """Initialize the delphi model.

        Args:
            hyperparameters: Hyperparameter dict, validated and stored as
                ``self.hyperparameters`` via the subclass ``defaults`` config.

        Raises:
            TypeError: If hyperparameters is not a dict.
        """
        super().__init__()
        if not isinstance(hyperparameters, dict):
            raise TypeError(
                f"hyperparameters is type {type(hyperparameters).__name__}; expected a dict."
            )
        self.hyperparameters: configs._ConfigT = self.defaults(**hyperparameters)

        self.optimizer: torch.optim.Optimizer | None = None
        self.schedule: torch.optim.lr_scheduler.LRScheduler | None = None

        self.criterion: Callable | None = None
        self.criterion_params: list = []
        self.model: ch.nn.Module | None = None
        # Additional optimizer slots for multi-optimizer algorithms (GANs, etc.).
        self.optimizers: dict[str, torch.optim.Optimizer] = {}

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

    def make_optimizer_and_schedule(
        self, checkpoint: dict | None = None
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        """Create and store the optimizer and learning-rate scheduler.

        Parameter groups are sourced from ``parameter_groups()``.  Override
        that method to assign different hyperparameters to different parameter
        subsets.

        Args:
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

    def _create_optimizer(self, params: list[dict]) -> torch.optim.Optimizer:
        """Create and return the configured optimizer via the registry."""
        optimizer_type = self._get_optimizer_type()

        if optimizer_type not in self._OPTIMIZER_REGISTRY:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        return self._OPTIMIZER_REGISTRY[optimizer_type](self, params)

    def _get_optimizer_type(self) -> str:
        """Return the lowercase optimizer type string from hyperparameters."""
        return getattr(self.hyperparameters, "optimizer", "sgd").lower()

    @property
    def required_criterion(self) -> Callable:
        """Return the loss criterion, which a subclass must already have set.

        Returns:
            The criterion callable.

        Raises:
            RuntimeError: If no criterion has been set on this model.
        """
        if self.criterion is None:
            raise RuntimeError("criterion has not been set on this model.")
        return self.criterion

    @property
    def required_optimizer(self) -> torch.optim.Optimizer:
        """Return the optimizer, which must already have been created.

        Returns:
            The optimizer built by ``make_optimizer_and_schedule``.

        Raises:
            RuntimeError: If the optimizer has not been created yet.
        """
        if self.optimizer is None:
            raise RuntimeError(
                "optimizer has not been created; "
                "make_optimizer_and_schedule must run before training."
            )
        return self.optimizer

    def _get_arg(self, name: str, default: _T) -> _T:
        """Return the attribute ``name`` from ``self.hyperparameters`` if present and not ``None``, else ``default``."""
        val = getattr(self.hyperparameters, name, None)
        return val if val is not None else default

    def _remove_none_config(self, config: dict) -> dict:
        """Return a copy of config with None values removed."""
        return {k: v for k, v in config.items() if v is not None}

    def _create_sgd(self, params: list[dict]) -> torch.optim.SGD:
        """Create an SGD optimizer from hyperparameters."""
        config = {
            "lr": getattr(self.hyperparameters, "lr", 1e-2),
            "momentum": getattr(self.hyperparameters, "momentum", 0),
            "dampening": getattr(self.hyperparameters, "dampening", 0),
            "weight_decay": getattr(self.hyperparameters, "weight_decay", 0),
            "nesterov": getattr(self.hyperparameters, "nesterov", False),
            "maximize": getattr(self.hyperparameters, "maximize", False),
            "foreach": getattr(self.hyperparameters, "foreach", None),
            "differentiable": getattr(self.hyperparameters, "differentiable", False),
            "fused": getattr(self.hyperparameters, "fused", None),
        }
        return torch.optim.SGD(params, **self._remove_none_config(config))

    def _create_lbfgs(self, params: list[dict]) -> torch.optim.LBFGS:
        """Create an L-BFGS optimizer from hyperparameters."""
        config = {
            "lr": getattr(self.hyperparameters, "lr", 1.0),
            "max_iter": getattr(self.hyperparameters, "max_iter", 20),
            "max_eval": getattr(self.hyperparameters, "max_eval", None),
            "tolerance_grad": getattr(self.hyperparameters, "tolerance_grad", 1e-7),
            "tolerance_change": getattr(self.hyperparameters, "tolerance_change", 1e-9),
            "history_size": getattr(self.hyperparameters, "history_size", 100),
            "line_search_fn": getattr(self.hyperparameters, "line_search_fn", None),
        }
        return torch.optim.LBFGS(params, **self._remove_none_config(config))

    def _create_adam(self, params: list[dict]) -> torch.optim.Adam:
        """Create an Adam optimizer from hyperparameters."""
        config = {
            "lr": getattr(self.hyperparameters, "lr", 1e-1),
            "betas": (
                getattr(self.hyperparameters, "beta1", 0.9),
                getattr(self.hyperparameters, "beta2", 0.999),
            ),
            "eps": getattr(self.hyperparameters, "eps", 1e-8),
            "weight_decay": getattr(self.hyperparameters, "weight_decay", 0),
            "amsgrad": getattr(self.hyperparameters, "amsgrad", False),
            "maximize": getattr(self.hyperparameters, "maximize", False),
            "foreach": getattr(self.hyperparameters, "foreach", None),
            "capturable": getattr(self.hyperparameters, "capturable", False),
            "differentiable": getattr(self.hyperparameters, "differentiable", False),
            "fused": getattr(self.hyperparameters, "fused", None),
        }
        return torch.optim.Adam(params, **self._remove_none_config(config))

    def _create_adamw(self, params: list[dict]) -> torch.optim.AdamW:
        """Create an AdamW optimizer from hyperparameters."""
        config = {
            "lr": getattr(self.hyperparameters, "lr", 1e-3),
            "betas": (
                getattr(self.hyperparameters, "beta1", 0.9),
                getattr(self.hyperparameters, "beta2", 0.999),
            ),
            "eps": getattr(self.hyperparameters, "eps", 1e-8),
            "weight_decay": getattr(self.hyperparameters, "weight_decay", 1e-2),
            "amsgrad": getattr(self.hyperparameters, "amsgrad", False),
            "maximize": getattr(self.hyperparameters, "maximize", False),
            "foreach": getattr(self.hyperparameters, "foreach", None),
            "capturable": getattr(self.hyperparameters, "capturable", False),
            "differentiable": getattr(self.hyperparameters, "differentiable", False),
            "fused": getattr(self.hyperparameters, "fused", None),
        }
        return torch.optim.AdamW(params, **self._remove_none_config(config))

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Create and return the configured learning-rate scheduler, or None."""
        if getattr(self.hyperparameters, "constant", False):
            return None

        scheduler_type = self._get_scheduler_type()
        if not scheduler_type:
            return None

        scheduler_creators = {
            constants.SchedulerType.CYCLIC: self._create_cyclic_scheduler,
            constants.SchedulerType.COSINE: self._create_cosine_scheduler,
            constants.SchedulerType.LINEAR: self._create_linear_scheduler,
            constants.SchedulerType.STEP: self._create_step_scheduler,
            constants.SchedulerType.MULTI_STEP: self._create_multi_step_scheduler,
            constants.SchedulerType.EXPONENTIAL: self._create_exponential_scheduler,
            constants.SchedulerType.REDUCE_ON_PLATEAU: self._create_plateau_scheduler,
        }

        if scheduler_type not in scheduler_creators:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        base_schedule = scheduler_creators[scheduler_type]()

        # Optionally prepend a linear warmup phase.
        warmup_steps = self._get_arg("warmup_steps", 0)
        is_plateau = isinstance(
            base_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau
        )
        if warmup_steps > 0 and not is_plateau:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.required_optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.required_optimizer,
                schedulers=[warmup, base_schedule],
                milestones=[warmup_steps],
            )

        return base_schedule

    def _get_scheduler_type(self) -> Optional[constants.SchedulerType]:
        """Return the configured scheduler type, or None when no scheduler is set.

        Raises:
            ValueError: If the configured name is not a known scheduler.
        """
        scheduler = getattr(self.hyperparameters, "scheduler", None)
        if not scheduler:
            return None
        try:
            return constants.SchedulerType(str(scheduler).lower())
        except ValueError:
            raise ValueError(f"Unsupported scheduler: {scheduler}") from None

    def _create_cyclic_scheduler(self) -> torch.optim.lr_scheduler.CyclicLR:
        """Create a cyclic LR scheduler.

        Configurable via hyperparameters:
            cyclic_base_lr (float): Lower LR bound (default 0.0).
            cyclic_max_lr (float): Upper LR bound (default: optimizer's lr).
            cyclic_step_size_up (int): Steps in ascending half-cycle (default 2000).
            cyclic_step_size_down (int): Steps in descending half-cycle (default: same).
            cyclic_mode (str): 'triangular', 'triangular2', or 'exp_range'.
            cyclic_gamma (float): Multiplicative factor for 'exp_range' mode.
        """
        assert self.optimizer is not None
        max_lr = self._get_arg("cyclic_max_lr", self.optimizer.param_groups[0]["lr"])
        config = {
            "base_lr": self._get_arg("cyclic_base_lr", 0.0),
            "max_lr": max_lr,
            "step_size_up": self._get_arg("cyclic_step_size_up", 2000),
            "step_size_down": self._get_arg("cyclic_step_size_down", None),
            "mode": self._get_arg("cyclic_mode", "triangular2"),
            "gamma": self._get_arg("cyclic_gamma", 1.0),
            "cycle_momentum": False,
        }
        return torch.optim.lr_scheduler.CyclicLR(
            self.optimizer, **self._remove_none_config(config)
        )

    def _create_cosine_scheduler(self) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        """Create a cosine annealing scheduler."""
        assert self.optimizer is not None
        config = {
            "T_max": self._get_arg("epochs", 100),
            "eta_min": self._get_arg("min_lr", 0),
        }
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, **self._remove_none_config(config)
        )

    def _create_linear_scheduler(self) -> torch.optim.lr_scheduler.LinearLR:
        """Create a linear LR decay scheduler."""
        assert self.optimizer is not None
        config = {
            "start_factor": self._get_arg("linear_start_factor", 1.0),
            "end_factor": self._get_arg("linear_end_factor", 0.0),
            "total_iters": self._get_arg("epochs", 100),
        }
        return torch.optim.lr_scheduler.LinearLR(
            self.optimizer, **self._remove_none_config(config)
        )

    def _create_step_scheduler(self) -> torch.optim.lr_scheduler.StepLR:
        """Create a step LR scheduler."""
        assert self.optimizer is not None
        config = {
            "step_size": getattr(self.hyperparameters, "step_lr", 100),
            "gamma": getattr(self.hyperparameters, "step_lr_gamma", 0.1),
        }
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer, **self._remove_none_config(config)
        )

    def _create_multi_step_scheduler(self) -> torch.optim.lr_scheduler.MultiStepLR:
        """Create a multi-step LR scheduler."""
        assert self.optimizer is not None
        config = {
            "milestones": getattr(self.hyperparameters, "milestones", [30, 60, 90]),
            "gamma": getattr(self.hyperparameters, "gamma", 0.1),
        }
        return torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, **self._remove_none_config(config)
        )

    def _create_exponential_scheduler(
        self,
    ) -> torch.optim.lr_scheduler.ExponentialLR:
        """Create an exponential LR scheduler."""
        assert self.optimizer is not None
        gamma = float(self._get_arg("gamma", 0.95))
        return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def _create_plateau_scheduler(
        self,
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Create a reduce-on-plateau scheduler."""
        assert self.optimizer is not None
        config = {
            "mode": getattr(self.hyperparameters, "plateau_mode", "min"),
            "factor": getattr(self.hyperparameters, "plateau_factor", 0.1),
            "patience": getattr(self.hyperparameters, "plateau_patience", 10),
            "threshold": getattr(self.hyperparameters, "plateau_threshold", 1e-4),
            "threshold_mode": getattr(
                self.hyperparameters, "plateau_threshold_mode", "rel"
            ),
            "cooldown": getattr(self.hyperparameters, "plateau_cooldown", 0),
            "min_lr": getattr(self.hyperparameters, "min_lr", 0),
            "eps": getattr(self.hyperparameters, "plateau_eps", 1e-8),
        }
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **self._remove_none_config(config)
        )

    def _load_checkpoint(self, checkpoint: dict) -> None:
        """Restore model weights, optimizer, scheduler, and random states from checkpoint."""
        if constants.CheckpointKey.MODEL in checkpoint:
            self.load_state_dict(checkpoint[constants.CheckpointKey.MODEL])

        if (
            constants.CheckpointKey.OPTIMIZER in checkpoint
            and self.optimizer is not None
        ):
            self.optimizer.load_state_dict(
                checkpoint[constants.CheckpointKey.OPTIMIZER]
            )

        if (
            constants.CheckpointKey.SCHEDULER in checkpoint
            and self.schedule is not None
        ):
            self.schedule.load_state_dict(checkpoint[constants.CheckpointKey.SCHEDULER])

        if constants.CheckpointKey.RANDOM_STATES in checkpoint:
            self._load_random_states(checkpoint[constants.CheckpointKey.RANDOM_STATES])

    def _load_random_states(self, random_states: dict) -> None:
        """Restore Python, NumPy, and PyTorch random number generator states."""
        if constants.PythonFrameworks.PYTHON in random_states:
            random.setstate(random_states[constants.PythonFrameworks.PYTHON])

        if constants.PythonFrameworks.NUMPY in random_states:
            np.random.set_state(random_states[constants.PythonFrameworks.NUMPY])

        if constants.PythonFrameworks.PYTORCH in random_states:
            ch.set_rng_state(random_states[constants.PythonFrameworks.PYTORCH])

    def validate_before_training(self) -> None:
        """Validate that the model is ready to train.

        Called by the trainer after ``make_optimizer_and_schedule`` and before
        the training loop begins.  Override this method for models with more
        complex setups (e.g. multi-optimizer models that populate
        ``self.optimizers`` instead of ``self.optimizer``) to perform
        custom readiness checks.

        Raises:
            RuntimeError: If neither ``self.optimizer`` nor ``self.optimizers``
                is set.
        """
        if self.optimizer is None and not self.optimizers:
            raise RuntimeError(
                f"{self} has no optimizer. Ensure make_optimizer_and_schedule "
                "sets self.optimizer or populates self.optimizers."
            )

    def optimizer_step(self, closure: Callable | None = None) -> ch.Tensor | None:
        """Perform the optimizer step.

        The default implementation calls ``self.optimizer.step()``.  Override
        this method for models with more complex update rules (e.g. GANs or
        actor-critic models that need to step multiple optimizers per update).
        ``validate_before_training`` guarantees ``self.optimizer`` is set
        before this method is called during training.

        Args:
            closure: Optional closure for line-search optimizers (e.g. L-BFGS).

        Returns:
            Loss evaluated by the closure, or None.
        """
        optimizer = cast(torch.optim.Optimizer, self.optimizer)
        if closure is not None:
            # torch stubs type step() as returning float, but line-search
            # optimizers return whatever the closure returns (a Tensor here).
            return optimizer.step(closure)  # type: ignore[return-value]
        return optimizer.step()  # type: ignore[return-value]

    def pretrain_hook(self) -> None:
        """Hook called before the training procedure begins."""

    def step_pre_hook(
        self, optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        """Hook called after .backward() but before the optimizer step."""

    def step_post_hook(
        self, optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict
    ) -> None:  # pylint: disable=unused-argument
        """Hook called after each optimizer step."""

    def post_epoch_hook(self, epoch: int, is_train: bool, loss: ch.Tensor) -> None:  # pylint: disable=unused-argument
        """Hook called after each complete pass through the dataset."""

    def post_training_hook(self) -> None:
        """Hook called after the full training procedure completes."""

    def description(
        self, stage: str, epoch: int, step: int, loss_meter: helpers.AverageMeter
    ) -> str:
        """Return a human-readable tqdm progress string for the current step."""
        return f"[{stage}] Epoch {epoch} | Step {step} | Loss {loss_meter.avg:.4f}"

    def regularize(self, batch) -> ch.Tensor:  # pylint: disable=unused-argument
        """Return the regularization term to add to the loss. Defaults to zero."""
        return ch.tensor(0.0)

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
        return self.required_criterion(*pred_args, targ, *self.criterion_params)  # pylint: disable=not-callable

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


# Register built-in optimizers after the class is fully defined.
delphi.register_optimizer(constants.OptimizerType.SGD, delphi._create_sgd)
delphi.register_optimizer(constants.OptimizerType.LBFGS, delphi._create_lbfgs)
delphi.register_optimizer(constants.OptimizerType.ADAM, delphi._create_adam)
delphi.register_optimizer(constants.OptimizerType.ADAMW, delphi._create_adamw)
