# Author: pstefanou12@
"""Pydantic configuration models for delphi.ai algorithms."""

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

    epochs: int | None = pydantic.Field(default=None, ge=1)
    iterations: int | None = pydantic.Field(default=None, ge=1)
    trials: int = pydantic.Field(default=1, ge=1)
    ema_decay: float = pydantic.Field(default=0.99, ge=0.0, le=1.0)
    tol: float = pydantic.Field(default=1e-3, ge=0.0)
    early_stopping: bool = False
    verbose: bool = False
    disable_no_grad: bool = False
    val_interval: int | None = pydantic.Field(default=None, ge=1)
    patience: int | None = pydantic.Field(default=None, ge=1)
    grad_tol: float = pydantic.Field(default=0.0, ge=0.0)
    grad_tol_window: int = pydantic.Field(default=1, ge=1)
    loss_tol: float | None = None
    log_every: int = pydantic.Field(default=50, ge=1)
    max_grad_norm: float | None = None
    tqdm: bool = False
    device: str = "cpu"
    use_amp: bool = False
    accumulate_grad_batches: int = pydantic.Field(default=1, ge=1)
    record_params_every: int = pydantic.Field(default=0, ge=0)
    checkpoint_dir: str | None = None
    checkpoint_every: int = pydantic.Field(default=0, ge=0)


_ConfigT = TypeVar("_ConfigT", bound=TrainerConfig)


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
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    optimizer: str = "sgd"
    lr: float = pydantic.Field(default=0.1, gt=0.0)
    momentum: float = pydantic.Field(default=0.0, ge=0.0)
    dampening: float = pydantic.Field(default=0.0, ge=0.0)
    weight_decay: float = pydantic.Field(default=0.0, ge=0.0)
    nesterov: bool = False
    maximize: bool = False
    foreach: bool | None = None
    differentiable: bool = False
    fused: bool | None = None
    scheduler: str | None = None


class TruncatedExponentialFamilyDistributionConfig(TrainerConfig, OptimizerConfig):
    """Configuration for truncated exponential family distribution algorithms.

    Attributes:
        val: Fraction of data held out for validation.
        eps: Numerical stability constant for the NLL criterion.
        min_radius: Initial NLL budget above the empirical initialization
            for the sublevel-set projection (phase 1).
        max_radius: Maximum NLL budget; the procedure stops when reached.
        rate: Multiplicative budget expansion factor per phase.
        batch_size: Mini-batch size for training.
        num_samples: Monte Carlo samples drawn per NLL evaluation.
        max_phases: Maximum number of radius-expansion phases.
        loss_convergence_tol: Absolute loss improvement threshold for
            stopping between phases.
        relative_loss_tol: Relative loss improvement threshold between phases.
        loss_increase_tol: Loss increase threshold for detecting overshoot.
        project: Enable per-step sublevel-set projection.
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    # Override parent defaults for distribution training.
    tol: float = pydantic.Field(default=1e-1, ge=0.0)
    record_params_every: int = pydantic.Field(default=1, ge=1)
    epochs: int | None = pydantic.Field(default=1, ge=1)

    # Distribution-specific fields.
    val: float = pydantic.Field(default=0.2, ge=0.0, le=1.0)
    eps: float = pydantic.Field(default=1e-5, gt=0.0)
    min_radius: float = pydantic.Field(default=3.0, ge=0.0)
    max_radius: float = pydantic.Field(default=10.0, ge=0.0)
    rate: float = pydantic.Field(default=1.1, gt=1.0)
    batch_size: int = pydantic.Field(default=10, ge=1)
    num_samples: int = pydantic.Field(default=10000, ge=1)
    max_phases: int = pydantic.Field(default=1, ge=1)
    loss_convergence_tol: float = pydantic.Field(default=1e-3, ge=0.0)
    relative_loss_tol: float = pydantic.Field(default=float("inf"), ge=0.0)
    loss_increase_tol: float = pydantic.Field(default=float("inf"), ge=0.0)
    project: bool = True

    @pydantic.model_validator(mode="after")
    def check_radius(self) -> "TruncatedExponentialFamilyDistributionConfig":
        """Validate that min_radius does not exceed max_radius."""
        if self.min_radius > self.max_radius:
            raise ValueError(
                f"min_radius ({self.min_radius}) must be <= "
                f"max_radius ({self.max_radius})."
            )
        return self

    @pydantic.model_validator(mode="after")
    def resolve_epochs_iterations(
        self,
    ) -> "TruncatedExponentialFamilyDistributionConfig":
        """Clear the default epochs when iterations is explicitly provided.

        The trainer uses exactly one stopping criterion. When the user
        supplies iterations, the per-phase epochs default is cleared so
        the trainer stops on iterations instead.
        """
        if (
            "iterations" in self.model_fields_set
            and "epochs" not in self.model_fields_set
        ):
            object.__setattr__(self, "epochs", None)
        return self


class TruncatedMultivariateNormalConfig(TruncatedExponentialFamilyDistributionConfig):
    """Configuration for truncated multivariate normal distributions.

    Attributes:
        eigenvalue_lower_bound: Minimum eigenvalue enforced during the
            negative-definite cone projection of the precision matrix T.
        covariance_matrix_lr: Optional separate learning rate for the
            covariance matrix parameter; falls back to lr when None.
    """

    eigenvalue_lower_bound: float = pydantic.Field(default=1e-2, gt=0.0)
    covariance_matrix_lr: float | None = pydantic.Field(default=None, gt=0.0)


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


class TruncatedRegressionFieldsConfig(LinearModelConfig):
    """Fields shared by the truncated regression and dynamical-system variants.

    ``TruncatedLinearRegression`` reads from both variants depending on its
    ``dependent`` flag, so every field either one needs is declared here and
    the concrete subclasses override only default values. That keeps a single
    type parameter sufficient for the estimator.

    Attributes:
        var_lr: Learning rate for the variance parameter when noise_var is
            unknown.
        eps: Numerical stability constant for the truncated MSE criterion.
        r: Initial projection set radius.
        rate: Multiplicative radius expansion factor.
        num_samples: Monte Carlo samples drawn per criterion evaluation.
        l1: L1 regularization weight.
        c_gamma: Scaling constant for the gamma bound.
        c_eta: Exponent used to derive the learning rate from alpha.
        constant: Disable learning-rate scheduling.
    """

    # None disables the separate variance learning rate; parameters_() in
    # TruncatedLinearRegression branches on that explicitly.
    var_lr: Optional[float] = pydantic.Field(default=1e-2, gt=0.0)
    eps: float = pydantic.Field(default=1e-5, gt=0.0)
    r: float = pydantic.Field(default=1.0, gt=0.0)
    rate: float = pydantic.Field(default=1.5, gt=0.0)
    num_samples: int = pydantic.Field(default=1000, ge=1)
    l1: float = pydantic.Field(default=0.0, ge=0.0)
    c_gamma: float = pydantic.Field(default=2.0, gt=0.0)
    c_eta: float = pydantic.Field(default=0.5, gt=0.0)
    constant: bool = False


class TruncatedRegressionConfig(TruncatedRegressionFieldsConfig):
    """Configuration for truncated linear regression (TRUNC_REG_DEFAULTS)."""

    # Override parent defaults for truncated regression.
    iterations: int | None = pydantic.Field(default=1500, ge=1)
    val_interval: int | None = pydantic.Field(default=50, ge=1)


class TruncatedLassoRegressionConfig(TruncatedRegressionConfig):
    """Configuration for truncated LASSO regression (TRUNC_LASSO_DEFAULTS)."""

    # Override parent defaults for the LASSO variant.
    num_samples: int = pydantic.Field(default=10000, ge=1)


class TruncatedElasticNetRegressionConfig(TruncatedLassoRegressionConfig):
    """Configuration for truncated elastic net regression.

    The pre-pydantic code derived its defaults from TRUNC_LASSO_DEFAULTS, so
    this adds no fields of its own; weight_decay comes from OptimizerConfig.
    """


class TruncatedRidgeRegressionConfig(TruncatedRegressionConfig):
    """Configuration for truncated ridge regression.

    The pre-pydantic code derived its defaults from TRUNC_REG_DEFAULTS, so
    this adds no fields of its own; weight_decay comes from OptimizerConfig.
    """


class TruncatedLinearDynamicalSystemConfig(TruncatedRegressionFieldsConfig):
    """Configuration for truncated linear dynamical systems (TRUNC_LDS_DEFAULTS)."""

    # Override parent defaults for dependent data, which is consumed in order.
    num_samples: int = pydantic.Field(default=50, ge=1)
    shuffle: bool = False
    constant: bool = True


class TruncatedLogisticRegressionConfig(LinearModelConfig):
    """Configuration for truncated logistic regression (TRUNC_LOG_REG_DEFAULTS).

    Attributes:
        l1: L1 regularization weight.
        eps: Numerical stability constant for the criterion.
        r: Initial projection set radius.
        rate: Multiplicative radius expansion factor.
        num_samples: Monte Carlo samples drawn per criterion evaluation.
    """

    l1: float = pydantic.Field(default=0.0, ge=0.0)
    eps: float = pydantic.Field(default=1e-5, gt=0.0)
    r: float = pydantic.Field(default=1.0, gt=0.0)
    rate: float = pydantic.Field(default=1.5, gt=0.0)
    num_samples: int = pydantic.Field(default=1000, ge=1)

    # Override parent defaults for logistic regression.
    epochs: int | None = pydantic.Field(default=1, ge=1)
    batch_size: int = pydantic.Field(default=10, ge=1)


class TruncatedProbitRegressionConfig(TruncatedLogisticRegressionConfig):
    """Configuration for truncated probit regression (TRUNC_PROB_REG_DEFAULTS).

    Attributes:
        tol: Convergence tolerance.
    """

    tol: float = pydantic.Field(default=1e-3, ge=0.0)

    # The probit defaults did not override epochs.
    epochs: int | None = pydantic.Field(default=None, ge=1)


class TruncatedLQRConfig(LinearModelConfig):
    """Configuration for truncated linear quadratic regulator estimation.

    Transcribed from TRUNC_LQR_DEFAULTS. R, U_A, U_B, delta and gamma were
    marked REQ there and so are required here.

    Attributes:
        target_thickness: Target thickness of the collected trajectories.
        num_traj_phase_one: Trajectory budget for phase one.
        num_traj_phase_two: Trajectory budget for phase two.
        num_traj_gen_samples_A: Trajectory budget for generating A samples.
        num_traj_gen_samples_B: Trajectory budget for generating B samples.
        T_phase_one: Step budget for phase one.
        T_phase_two: Step budget for phase two.
        T_gen_samples_A: Step budget for generating A samples.
        T_gen_samples_B: Step budget for generating B samples.
        R: Radius of the truncation ball.
        U_A: Spectral norm bound on A.
        U_B: Spectral norm bound on B.
        delta: Failure probability.
        eps1: Phase one accuracy target.
        eps2: Phase two accuracy target.
        repeat: Number of repetitions; None runs once.
        gamma: Thickness parameter.
    """

    target_thickness: float = float("inf")
    num_traj_phase_one: int = pydantic.Field(default=int(1e100), ge=1)
    num_traj_phase_two: int = pydantic.Field(default=int(1e100), ge=1)
    num_traj_gen_samples_A: int = pydantic.Field(default=int(1e100), ge=1)
    num_traj_gen_samples_B: int = pydantic.Field(default=int(1e100), ge=1)
    T_phase_one: int = pydantic.Field(default=int(1e100), ge=1)
    T_phase_two: int = pydantic.Field(default=int(1e100), ge=1)
    T_gen_samples_A: int = pydantic.Field(default=int(1e100), ge=1)
    T_gen_samples_B: int = pydantic.Field(default=int(1e100), ge=1)
    R: float = pydantic.Field(gt=0.0)
    U_A: float = pydantic.Field(gt=0.0)
    U_B: float = pydantic.Field(gt=0.0)
    delta: float = pydantic.Field(gt=0.0, lt=1.0)
    eps1: float = pydantic.Field(default=0.9, gt=0.0)
    eps2: float = pydantic.Field(default=0.9, gt=0.0)
    repeat: int | None = pydantic.Field(default=None, ge=1)
    gamma: float = pydantic.Field(gt=0.0)


class SoftmaxRegressionConfig(LinearModelConfig):
    """Configuration for truncated softmax (multinomial) regression.

    Attributes:
        num_samples: Monte Carlo samples drawn per criterion evaluation.
        multi_class: Multiclass strategy used for the empirical initialization.
    """

    num_samples: int = pydantic.Field(default=1000, ge=1)
    multi_class: str = "ovr"
