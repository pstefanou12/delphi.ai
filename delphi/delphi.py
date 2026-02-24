# Author: pstefanou12@
"""Parent class for models to train in the delphi trainer."""

import random
from typing import Callable, ClassVar, Iterable
import torch as ch
from cox.store import Store
from torch.optim import SGD, LBFGS, Adam, AdamW, lr_scheduler
import numpy as np

from delphi.delphi_logger import delphiLogger
from delphi.utils.defaults import (
    check_and_fill_args,
    DELPHI_DEFAULTS,
    SGD_DEFAULTS,
    LBFGS_DEFAULTS,
    ADAM_DEFAULTS,
    ADAMW_DEFAULTS,
)
from delphi.utils.helpers import Parameters

# Module-level constants.
BY_ALG = "by algorithm"  # Default parameter depends on algorithm.
ADAM = "adam"
ADAMW = "adamw"

EVAL_LOGS_SCHEMA = {"test_prec1": float, "test_loss": float, "time": float}


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
            "cyclic": self._create_cyclic_scheduler,
            "cosine": self._create_cosine_scheduler,
            "linear": self._create_linear_scheduler,
            "step": self._create_step_scheduler,
            "multi_step": self._create_multi_step_scheduler,
            "exponential": self._create_exponential_scheduler,
            "reduce_on_plateau": self._create_plateau_scheduler,
        }

        if scheduler_type not in scheduler_creators:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        return scheduler_creators[scheduler_type]()

    def _get_scheduler_type(self):
        """Return the lowercase scheduler type string from args, or None."""
        # Explicit scheduler arg takes priority over legacy step_lr.
        if hasattr(self.args, "scheduler") and self.args.scheduler:
            return self.args.scheduler.lower()
        return None

    def _create_cyclic_scheduler(self):
        """Create a cyclic learning-rate scheduler."""
        epochs = getattr(self.args, "epochs", 100)

        def lr_func(t):
            return np.interp([t], [0, epochs * 4 // 15, epochs], [0, 1, 0])[0]

        return lr_scheduler.LambdaLR(self.optimizer, lr_func)

    def _create_cosine_scheduler(self):
        """Create a cosine annealing scheduler."""
        config = {
            "T_max": getattr(self.args, "epochs", 100),
            "eta_min": getattr(self.args, "min_lr", 0),
        }
        return lr_scheduler.CosineAnnealingLR(
            self.optimizer, **self._remove_none_config(config)
        )

    def _create_linear_scheduler(self):
        """Create a linear LR decay scheduler."""
        config = {
            "start_factor": getattr(self.args, "linear_start_factor", 1.0),
            "end_factor": getattr(self.args, "linear_end_factor", 0.0),
            "total_iters": getattr(self.args, "epochs", 100),
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
        """Restore optimizer, scheduler, and random states from a checkpoint dict."""
        if "optimizer" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.logger.info("Loaded optimizer state from checkpoint.")

        if (
            "scheduler" in checkpoint
            and hasattr(self, "schedule")
            and self.schedule is not None
        ):
            self.schedule.load_state_dict(checkpoint["scheduler"])
            self.logger.info("Loaded scheduler state from checkpoint.")

            # Advance the scheduler to match the saved training position.
            current_epoch = checkpoint.get("epoch", 0)
            for _ in range(current_epoch):
                self.schedule.step()
            self.logger.info(f"Advanced scheduler to epoch {current_epoch}.")

        if "random_states" in checkpoint:
            self._load_random_states(checkpoint["random_states"])

        if "training_state" in checkpoint:
            self._load_training_state(checkpoint["training_state"])

    def _load_random_states(self, random_states):
        """Restore Python, NumPy, and PyTorch random number generator states."""
        if "python" in random_states:
            random.setstate(random_states["python"])

        if "numpy" in random_states:
            np.random.set_state(random_states["numpy"])

        if "pytorch" in random_states:
            ch.set_rng_state(random_states["pytorch"])

        self.logger.info("Restored random number generator states.")

    def _load_training_state(self, training_state):
        """Restore epoch counter and best loss from a training state dict."""
        self.start_epoch = training_state.get("epoch", 0)
        self.best_loss = training_state.get("best_loss", float("inf"))
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

    def description(self, epoch, i, loss_, prec1_, prec5_, reg_term):  # pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument
        """Return a human-readable status string for the current iteration."""
        reg_str = f"{float(reg_term):.4f}" if reg_term is not None else "N/A"
        return (
            f"{self.training} Epoch:{epoch} | Loss {loss_.avg:.4f} | "
            f"Prec1: {float(prec1_.avg):.3f} | Prec5: {float(prec5_.avg):.3f} | "
            f"Reg term: {reg_str} ||"
        )

    def regularize(self, batch) -> ch.Tensor:  # pylint: disable=unused-argument
        """Return the regularization term to add to the loss. Defaults to zero."""
        return ch.zeros(1, 1)

    def parameters_(self):
        """Return the model's trainable parameters."""
        return self.parameters()


# Populate the built-in optimizer registry after the class is fully defined.
delphi._OPTIMIZER_REGISTRY = {
    "sgd": delphi._create_sgd,
    "lbfgs": delphi._create_lbfgs,
    ADAM: delphi._create_adam,
    ADAMW: delphi._create_adamw,
}
