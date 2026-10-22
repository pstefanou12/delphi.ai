# Author: pstefanou12@
"""Parent class for models to train in the delphi trainer."""

from collections.abc import Callable
import inspect
import random
from typing import TYPE_CHECKING, ClassVar, Generic, Optional, cast

import numpy as np
import pydantic
import torch as ch
import torch.optim
import torch.optim.lr_scheduler

from delphi import configs
from delphi.optim.lr_scheduler import fisher, inverse_time
from delphi.utils import constants, helpers

if TYPE_CHECKING:
    from delphi import trainer

# Each torch class paired with the config section whose aliases name its
# keyword arguments.
OPTIMIZER_SECTIONS: dict[str, tuple[type, type[pydantic.BaseModel]]] = {
    constants.OptimizerType.SGD: (torch.optim.SGD, configs.SGDConfig),
    constants.OptimizerType.ADAM: (torch.optim.Adam, configs.AdamConfig),
    constants.OptimizerType.ADAMW: (torch.optim.AdamW, configs.AdamConfig),
    constants.OptimizerType.LBFGS: (torch.optim.LBFGS, configs.LBFGSConfig),
}
SCHEDULER_SECTIONS: dict[
    constants.SchedulerType, tuple[type, type[pydantic.BaseModel]]
] = {
    constants.SchedulerType.STEP: (
        torch.optim.lr_scheduler.StepLR,
        configs.StepLRConfig,
    ),
    constants.SchedulerType.MULTI_STEP: (
        torch.optim.lr_scheduler.MultiStepLR,
        configs.MultiStepLRConfig,
    ),
    constants.SchedulerType.EXPONENTIAL: (
        torch.optim.lr_scheduler.ExponentialLR,
        configs.ExponentialLRConfig,
    ),
    constants.SchedulerType.COSINE: (
        torch.optim.lr_scheduler.CosineAnnealingLR,
        configs.CosineLRConfig,
    ),
    constants.SchedulerType.LINEAR: (
        torch.optim.lr_scheduler.LinearLR,
        configs.LinearLRConfig,
    ),
    constants.SchedulerType.CYCLIC: (
        torch.optim.lr_scheduler.CyclicLR,
        configs.CyclicLRConfig,
    ),
    constants.SchedulerType.REDUCE_ON_PLATEAU: (
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        configs.PlateauLRConfig,
    ),
    constants.SchedulerType.INVERSE_TIME: (
        inverse_time.InverseTimeLR,
        configs.InverseTimeLRConfig,
    ),
    constants.SchedulerType.FISHER: (
        fisher.FisherLR,
        configs.FisherLRConfig,
    ),
}


class delphi(ch.nn.Module, Generic[configs._ConfigT]):  # pylint: disable=invalid-name,too-many-instance-attributes,abstract-method
    """Parent/abstract class for models to be passed into the trainer.

    Concrete subclasses must define a ``defaults`` class attribute pointing to
    their Pydantic config class::

        class MyModel(delphi[MyConfig]):
            defaults = MyConfig

    Subclasses may also register custom optimizers via the class-level registry::

        delphi.register_optimizer("adamw", lambda m, p: AdamW(p, lr=m.hyperparameters.lr))

    A factory registered on ``delphi`` is available to every model; one
    registered on a subclass applies only to that subclass and its children.
    Algorithms needing several optimizers at once, such as a GAN, should
    instead override ``make_optimizer_and_schedule`` to populate
    ``self.optimizers`` and drive the updates from ``train_batch``.
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

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.schedule: Optional[torch.optim.lr_scheduler.LRScheduler] = None

        self.criterion: Optional[Callable] = None
        self.criterion_params: list = []
        # The objective of the warm start, which runs only when the config
        # sets a warm-start budget: a criterion with its parameters bound.
        self.warm_start_criterion: Optional[Callable] = None
        # The warm-start criterion's Hessian, which the fisher scheduler needs
        # during the stage; ``hessian`` belongs to the main criterion.
        self.warm_start_hessian: Optional[Callable[..., ch.Tensor]] = None
        self.model: Optional[ch.nn.Module] = None
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
        self,
        checkpoint: Optional[dict] = None,
        hessian: Optional[Callable[..., ch.Tensor]] = None,
    ) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
        """Create and store the optimizer and learning-rate scheduler.

        Parameter groups are sourced from ``parameter_groups()``.  Override
        that method to assign different hyperparameters to different parameter
        subsets.

        Args:
            checkpoint: Optional dict containing optimizer/scheduler state to restore.
            hessian: The trained criterion's Hessian, for a stage whose
                criterion is not the model's own; ``self.hessian`` by default.

        Returns:
            Tuple of (optimizer, scheduler).
        """
        groups = self.parameter_groups()

        self.optimizer = self._create_optimizer(groups)
        self.optimizer.register_step_pre_hook(self.step_pre_hook)
        self.optimizer.register_step_post_hook(self.step_post_hook)

        self.schedule = self._create_scheduler(hessian)

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
                is the iterable of parameters to optimize.
        """
        # Write into this class's own registry, creating it on first use, so
        # the entry does not leak to sibling classes or up to the base.
        if "_OPTIMIZER_REGISTRY" not in cls.__dict__:
            cls._OPTIMIZER_REGISTRY = {}
        cls._OPTIMIZER_REGISTRY[name.lower()] = factory

    def _registered_factory(self, name: str) -> Optional[Callable]:
        """Return the factory registered for name, searching base classes.

        Resolving along the MRO rather than copying at class-definition time
        means a factory registered on ``delphi`` reaches models defined before
        the registration -- the usual case in a notebook -- while a factory
        registered on one model stays with it.

        Args:
            name: Lowercase optimizer name.

        Returns:
            The factory, or None if no class in the MRO registered one.
        """
        for klass in type(self).__mro__:
            registry = klass.__dict__.get("_OPTIMIZER_REGISTRY")
            if registry and name in registry:
                return registry[name]
        return None

    def _create_optimizer(self, params: list[dict]) -> torch.optim.Optimizer:
        """Create and return the configured optimizer.

        A factory registered via ``register_optimizer`` takes precedence, so a
        built-in name can be replaced; otherwise the optimizer is built from
        its entry in ``OPTIMIZER_SECTIONS``.

        Args:
            params: Parameter groups to optimize.

        Returns:
            The constructed optimizer.

        Raises:
            ValueError: If the configured name is neither registered nor
                built in.
        """
        optimizer_type = self._get_optimizer_type()

        factory = self._registered_factory(optimizer_type)
        if factory is not None:
            return factory(self, params)

        if optimizer_type in OPTIMIZER_SECTIONS:
            target, section = OPTIMIZER_SECTIONS[optimizer_type]
            kwargs = self._shared_optimizer_kwargs(target)
            kwargs.update(self._kwargs_for(section))
            return target(params, **kwargs)

        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def _get_optimizer_type(self) -> str:
        """Return the lowercase optimizer type string from hyperparameters."""
        return self.hyperparameters.optimizer.lower()

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

    def _kwargs_for(self, section: type[pydantic.BaseModel]) -> dict:
        """Return the torch keyword arguments a config section declares.

        Each section's fields carry ``serialization_alias`` set to torch's own
        parameter name, so dumping by alias yields the keyword arguments
        directly. Fields left as None are dropped so torch applies its own
        default.

        Args:
            section: The config section whose fields to collect.

        Returns:
            Keyword arguments for the torch class that section describes.
        """
        return self.hyperparameters.model_dump(
            by_alias=True, include=set(section.model_fields), exclude_none=True
        )

    def _shared_optimizer_kwargs(self, target: type) -> dict:
        """Return the unprefixed optimizer arguments ``target`` accepts.

        ``lr`` and friends are declared on OptimizerConfig rather than on a
        per-optimizer section because they apply to whichever optimizer is
        selected, so they are filtered by what the target actually takes.

        Args:
            target: The torch optimizer class.

        Returns:
            The subset of shared arguments in the target's signature.
        """
        shared = {
            "lr": self.hyperparameters.lr,
            "weight_decay": self.hyperparameters.weight_decay,
            "maximize": self.hyperparameters.maximize,
            "foreach": self.hyperparameters.foreach,
            "differentiable": self.hyperparameters.differentiable,
            "fused": self.hyperparameters.fused,
        }
        accepted = inspect.signature(target).parameters
        return {k: v for k, v in shared.items() if k in accepted and v is not None}

    def _create_scheduler(
        self, hessian: Optional[Callable[..., ch.Tensor]] = None
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create and return the configured learning-rate scheduler, or None.

        Args:
            hessian: As ``make_optimizer_and_schedule``.
        """
        if self.hyperparameters.constant:
            return None

        scheduler_type = self._get_scheduler_type()
        if not scheduler_type:
            return None

        if scheduler_type not in SCHEDULER_SECTIONS:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        base_schedule = self._build_scheduler(scheduler_type, hessian)
        return self._with_warmup(base_schedule)

    def _build_scheduler(
        self,
        scheduler_type: constants.SchedulerType,
        hessian: Optional[Callable[..., ch.Tensor]] = None,
    ):
        """Build a scheduler from its config section.

        Args:
            scheduler_type: Key into ``SCHEDULER_SECTIONS``.
            hessian: As ``make_optimizer_and_schedule``.

        Returns:
            The constructed scheduler.
        """
        target, section = SCHEDULER_SECTIONS[scheduler_type]
        kwargs = self._kwargs_for(section)

        # These cannot come from a section: the horizon is the trainer's, and
        # CyclicLR's max_lr is only known once the optimizer exists.
        if scheduler_type is constants.SchedulerType.COSINE:
            kwargs["T_max"] = self._schedule_horizon()
        elif scheduler_type is constants.SchedulerType.LINEAR:
            kwargs["total_iters"] = self._schedule_horizon()
        elif scheduler_type is constants.SchedulerType.CYCLIC:
            kwargs.setdefault("max_lr", self.required_optimizer.param_groups[0]["lr"])
            kwargs["cycle_momentum"] = False
        elif scheduler_type is constants.SchedulerType.FISHER:
            supplied = kwargs.keys() & {"mu", "smoothness"}
            # Skip the sampling round when the caller supplied both.
            if supplied != {"mu", "smoothness"}:
                mu, smoothness = self._hessian_spectrum(hessian=hessian)
                kwargs.setdefault("mu", mu)
                kwargs.setdefault("smoothness", smoothness)
            # A constant the caller fixed is never overwritten by a refresh.
            if not supplied:
                kwargs["curvature"] = lambda: self._hessian_spectrum(
                    current=True, hessian=hessian
                )

        return target(self.required_optimizer, **kwargs)

    def _schedule_horizon(self) -> int:
        """Return the step budget the cosine and linear schedules decay over.

        Returns:
            The configured iteration cap, or the epoch cap when only that is
            set.

        Raises:
            ValueError: If neither cap is set, leaving the decay unbounded.
        """
        # Prefer iterations: _schedule_step runs per optimizer step, not per
        # epoch. Either cap may be None, and torch rejects None here.
        horizon = self.hyperparameters.iterations or self.hyperparameters.epochs
        if horizon is None:
            raise ValueError(
                f"{self.hyperparameters.scheduler} needs iterations or epochs "
                "set; it decays over a fixed number of steps."
            )
        return horizon

    def _hessian_spectrum(
        self,
        current: bool = False,
        hessian: Optional[Callable[..., ch.Tensor]] = None,
    ) -> tuple[float, float]:
        """Return the curvature constants FisherLR decays by.

        Args:
            current: Measure at the current iterate rather than at the
                model's default point, its initialization.
            hessian: As ``make_optimizer_and_schedule``.

        Returns:
            Tuple of (mu, smoothness) from this model's Hessian.

        Raises:
            NotImplementedError: If the model cannot supply a Hessian.
            ValueError: If the estimate has no positive curvature.
        """
        if hessian is None and not hasattr(self, "hessian"):
            raise NotImplementedError(
                f"{type(self).__name__} does not implement hessian(), which "
                f"the {constants.SchedulerType.FISHER.value} scheduler needs; "
                "set fisher_mu and fisher_smoothness explicitly instead."
            )
        if hessian is None:
            # nn.Module.__getattr__ types every attribute as a Tensor.
            hessian = cast(Callable[..., ch.Tensor], getattr(self, "hessian"))
        estimate = hessian(getattr(self, "theta").detach()) if current else hessian()
        return fisher.FisherLR.spectrum(
            estimate, getattr(self, "independent_coordinates", True)
        )

    def _with_warmup(
        self, base_schedule: torch.optim.lr_scheduler.LRScheduler
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Prepend a linear warmup phase to a scheduler when configured.

        Args:
            base_schedule: The scheduler to wrap.

        Returns:
            ``base_schedule`` unchanged, or a SequentialLR that warms up first.
        """
        warmup_steps = self.hyperparameters.warmup_steps
        is_plateau = isinstance(
            base_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau
        )
        if warmup_steps <= 0 or is_plateau:
            return base_schedule

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

    def _get_scheduler_type(self) -> Optional[constants.SchedulerType]:
        """Return the configured scheduler type, or None when none is set.

        Returns:
            The scheduler type, or None.

        Raises:
            ValueError: If the configured name is not a known scheduler.
        """
        scheduler = self.hyperparameters.scheduler
        if not scheduler:
            return None
        try:
            return constants.SchedulerType(str(scheduler).lower())
        except ValueError:
            raise ValueError(f"Unsupported scheduler: {scheduler}") from None

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

    def optimizer_step(self, closure: Optional[Callable] = None) -> Optional[ch.Tensor]:
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

    def pre_phase_hook(self) -> None:
        """Hook called before each phase, once its run is seeded."""

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
        """Hook called once, after the full training procedure completes."""

    def post_warm_start_hook(self, result: "trainer.RunResult") -> None:  # pylint: disable=unused-argument
        """Hook called with the warm start's record, to adopt its result.

        Args:
            result: The warm-start run.
        """

    def post_phase_hook(
        self, result: "trainer.RunResult"
    ) -> tuple[bool, Optional[str]]:  # pylint: disable=unused-argument
        """Decide whether the procedure stops after a phase, and prepare the next.

        A phase changes the problem, such as a projection budget, and
        training resumes from the current parameters. A model that never
        changes its problem keeps this default and trains in one phase.

        Args:
            result: The phase just finished.

        Returns:
            Tuple of (stop, reason), as ``should_stop``.
        """
        return True, "single phase"

    def run_metrics(self) -> dict:
        """Return algorithm-specific facts about the run that just finished.

        Recorded in that run's ``RunResult.metrics``. Defaults to none.
        """
        return {}

    def description(
        self, stage: str, epoch: int, step: int, loss_meter: helpers.AverageMeter
    ) -> str:
        """Return a human-readable tqdm progress string for the current step."""
        return f"[{stage}] Epoch {epoch} | Step {step} | Loss {loss_meter.avg:.4f}"

    def regularize(self, batch) -> ch.Tensor:  # pylint: disable=unused-argument
        """Return the regularization term to add to the loss. Defaults to zero."""
        return ch.tensor(0.0)

    def compute_loss(self, batch, criterion: Optional[Callable] = None) -> ch.Tensor:
        """Compute the loss for a single batch.

        Override to implement custom forward and loss logic. The default
        implementation unpacks ``batch`` as ``(inp, targ)`` and evaluates
        the criterion.

        Args:
            batch: A single batch from the DataLoader.
            criterion: The stage's criterion, its parameters bound; by
                default ``self.criterion`` with ``self.criterion_params``.

        Returns:
            Loss tensor (scalar or vector; the trainer handles reduction).
        """
        inp, targ = batch
        pred = self(inp)
        pred_args = pred if isinstance(pred, (tuple, list)) else (pred,)
        if criterion is not None:
            return criterion(*pred_args, targ)
        return self.required_criterion(*pred_args, targ, *self.criterion_params)  # pylint: disable=not-callable

    def compute_val_loss(
        self, batch, criterion: Optional[Callable] = None
    ) -> ch.Tensor:
        """Compute the validation loss for a single batch.

        Defaults to ``compute_loss``. Override to use a different metric
        during validation (e.g. NLL instead of ELBO).

        Args:
            batch: A single batch from the validation DataLoader.
            criterion: As ``compute_loss``.

        Returns:
            Loss tensor (scalar or vector; the trainer handles reduction).
        """
        return self.compute_loss(batch, criterion)

    def train_batch(self, batch) -> Optional[dict]:
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

    def should_stop(self) -> tuple[bool, Optional[str]]:
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
