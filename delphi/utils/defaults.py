# Author: pstefanou12@
"""Default parameters for running algorithms in delphi.ai."""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Union, get_args, get_origin

import torch as ch

from delphi.utils.constants import OptimizerType, SchedulerType
from delphi.utils.helpers import has_attr

if TYPE_CHECKING:
    from delphi.utils.helpers import Parameters


# Module-level constants.
REQ = "required"

OPTIMIZER_DEFAULTS = {
    "optimizer": (str, OptimizerType.SGD),
    "scheduler": (
        [
            SchedulerType.CYCLIC,
            SchedulerType.COSINE,
            SchedulerType.LINEAR,
            SchedulerType.STEP,
            SchedulerType.MULTI_STEP,
            SchedulerType.EXPONENTIAL,
            SchedulerType.REDUCE_ON_PLATEAU,
            None,
        ],
        None,
    ),
    "weight_decay": (float, 0.0, {"min": 0}),
}

SGD_DEFAULTS = {
    "lr": (float, 1e-1, {"min": 0}),
    "momentum": (float, 0.0, {"min": 0}),
    "dampening": (float, 0.0, {"min": 0}),
    "nesterov": (bool, False),
    "maximize": (bool, False),
    "foreach": (bool | None, None),
    "differentiable": (bool, False),
    "fused": (bool | None, None),
}

LBFGS_DEFAULTS = {
    "lr": (float, 1.0, {"min": 0}),
    "max_iter": (int, 20, {"min": 1}),
    "max_eval": (int | None, None),
    "tolerance_grad": (float, 1e-7),
    "tolerance_change": (float, 1e-9),
    "history_size": (int, 100),
    "line_search_fn": (["strong_wolfe", None], "strong_wolfe"),
}

ADAM_DEFAULTS = {
    "beta1": (float, 0.9, {"min": 0, "max": 1}),
    "beta2": (float, 0.999, {"min": 0, "max": 1}),
    "eps": (float, 1e-8, {"min": 0}),
    "amsgrad": (bool, False),
    "capturable": (bool, False),
}

ADAMW_DEFAULTS = {
    "beta1": (float, 0.9, {"min": 0, "max": 1}),
    "beta2": (float, 0.999, {"min": 0, "max": 1}),
    "eps": (float, 1e-8, {"min": 0}),
    "weight_decay": (float, 1e-2, {"min": 0}),
    "amsgrad": (bool, False),
    "capturable": (bool, False),
}

STEP_LR_DEFAULTS = {
    "step_lr": (int, 100, {"min": 1}),
    "step_lr_gamma": (float, 0.9, {"min": 0, "max": 1}),
    "min_lr": (float, 0.0, {"min": 0}),
    "milestones": (list, [30, 60, 90]),
    "gamma": (float, 0.1, {"min": 0, "max": 1}),
    "warmup_steps": (int, 0, {"min": 0}),
}

CYCLIC_LR_DEFAULTS = {
    "cyclic_base_lr": (float, 0.0, {"min": 0}),
    "cyclic_step_size_up": (int, 2000, {"min": 1}),
    "cyclic_mode": (["triangular", "triangular2", "exp_range"], "triangular2"),
    "cyclic_gamma": (float, 1.0, {"min": 0}),
}

PLATEAU_SCHEDULER_DEFAULTS = {
    "plateau_mode": (["min", "max"], "min"),
    "plateau_factor": (float, 0.1, {"min": 0, "max": 1}),
    "plateau_patience": (int, 10, {"min": 1}),
    "plateau_threshold": (float, 1e-4, {"min": 0}),
    "plateau_threshold_mode": (["rel", "abs"], "rel"),
    "plateau_cooldown": (int, 0, {"min": 0}),
    "plateau_eps": (float, 1e-8, {"min": 0}),
}

TRAINER_DEFAULTS = {
    "trials": (int, 1),
    "ema_decay": (float, 0.99),
    "tol": (float, 1e-3),
    "early_stopping": (bool, False),
    "verbose": (bool, False),
    "disable_no_grad": (bool, False),
    "val_interval": (int | None, None, {"min": 1}),
    "patience": (int, float("inf"), {"min": 1}),
    "grad_tol": (float, 0.0, {"min": 0}),
    "grad_tol_window": (int, 1, {"min": 1}),
    "loss_tol": (float | None, None),
    "log_every": (int, 50, {"min": 1}),
    "max_grad_norm": (float | None, None),
    "tqdm": (bool, False),
    "device": (str, "cpu"),
    "use_amp": (bool, False),
    "accumulate_grad_batches": (int, 1, {"min": 1}),
    # When 0 (default) no in-memory parameter vectors are recorded; suitable
    # for large DNNs.  Set to a positive integer to record parameter vectors
    # at that step frequency (e.g. for SGD iterate-averaging or EMA).
    "record_params_every": (int, 0, {"min": 0}),
    # Directory for on-disk checkpoints.  When set (or when a cox store is
    # provided at training time), the trainer writes the best state dict to
    # disk.  Falls back to in-memory saving only when record_params_every > 0
    # and no disk target is configured.
    "checkpoint_dir": (str | None, None),
    # Save a full checkpoint (model, optimizer, scheduler, epoch) to disk
    # every this many epochs so that training can be resumed after failure.
    # 0 disables periodic checkpointing.
    "checkpoint_every": (int, 0, {"min": 0}),
}

DATASET_DEFAULTS = {
    "workers": (int, 1),
    "batch_size": (int, 100),
    "val": (float, 0.2),
    "normalize": (bool, False),
    "shuffle": (bool, True),
    "pin_memory": (bool, True),
    "drop_last": (bool, False),
}

DELPHI_DEFAULTS = {**OPTIMIZER_DEFAULTS, "device": (str, "cpu")}

TRUNC_REG_DEFAULTS = {
    "val": (float, 0.2),
    "var_lr": (float, 1e-2),
    "weight_decay": (float, 0.0),
    "eps": (float, 1e-5),
    "r": (float, 1.0),
    "rate": (float, 1.5),
    "batch_size": (int, 50),
    "workers": (int, 0),
    "num_samples": (int, 1000),
    "shuffle": (bool, True),
    "iterations": (int, 1500),
    "val_interval": (int, 50),
}

TRUNC_LASSO_DEFAULTS = {
    "val": (float, 0.2),
    "l1": (float, 0.0),
    "weight_decay": (float, 0.0),
    "eps": (float, 1e-5),
    "r": (float, 1.0),
    "rate": (float, 1.5),
    "batch_size": (int, 50),
    "workers": (int, 0),
    "num_samples": (int, 10000),
    "shuffle": (bool, True),
    "iterations": (int, 1500),
    "val_interval": (int, 50),
}

TRUNC_LDS_DEFAULTS = {
    "val": (float, 0.2),
    "l1": (float, 0.0),
    "weight_decay": (float, 0.0),
    "eps": (float, 1e-5),
    "r": (float, 1.0),
    "rate": (float, 1.5),
    "batch_size": (int, 50),
    "workers": (int, 0),
    "num_samples": (int, 50),
    "c_gamma": (float, 2.0),
    "shuffle": (bool, False),
    "constant": (bool, True),
    "c_eta": (float, 0.5),
    "c_s": (float, 10.0),
}

TRUNC_LOG_REG_DEFAULTS = {
    "epochs": (int, 1),
    "val": (float, 0.2),
    "l1": (float, 0.0),
    "eps": (float, 1e-5),
    "r": (float, 1.0),
    "rate": (float, 1.5),
    "batch_size": (int, 10),
    "workers": (int, 0),
    "num_samples": (int, 1000),
}

TRUNC_PROB_REG_DEFAULTS = {
    "val": (float, 0.2),
    "l1": (float, 0.0),
    "eps": (float, 1e-5),
    "r": (float, 1.0),
    "rate": (float, 1.5),
    "batch_size": (int, 10),
    "tol": (float, 1e-3),
    "workers": (int, 0),
    "num_samples": (int, 1000),
}

TRUNC_EXP_FAMILY_DISTR_DEFAULTS = {
    "val": (float, 0.2),
    "eps": (float, 1e-5),
    # NLL budget above the empirical initialization for the Karatapanis
    # sublevel-set projection.  min_radius is the starting budget (phase 1)
    # and max_radius is the maximum budget (typically log(1/alpha) + 2).
    "min_radius": (float, 3.0),
    "max_radius": (float, 10.0),
    "rate": (float, 1.1),  # Multiplicative budget expansion per phase.
    "batch_size": (int, 10),
    "tol": (float, 1e-1),
    "num_samples": (int, 10000),
    "optimizer": (str, OptimizerType.SGD),
    "max_phases": (int, 1),
    "loss_convergence_tol": (float, 1e-3),
    "relative_loss_tol": (float, float("inf")),
    # Disabled by default; set to a finite value to stop when the truncated
    # NLL increases by more than this amount between consecutive phases.
    "loss_increase_tol": (float, float("inf")),
    # Set to False to skip the per-step sublevel-set projection entirely,
    # running unconstrained gradient descent instead.
    "project": (bool, True),
    # Number of steps between parameter vector snapshots; must be > 0 so
    # that fit() can read back best/final/ema/avg params after each phase.
    "record_params_every": (int, 1, {"min": 1}),
    # Maximum number of training epochs per phase.
    "epochs": (int, 1, {"min": 1}),
}

TRUNC_MULTI_NORM_DEFAULTS = {
    **TRUNC_EXP_FAMILY_DISTR_DEFAULTS,
    "covariance_matrix": (ch.Tensor, None),
    "covariance_matrix_lr": (float, 1e-2),
    "eigenvalue_lower_bound": (float, 1e-2),
}


TRUNC_BOOL_PROD_DEFAULTS = {**TRUNC_EXP_FAMILY_DISTR_DEFAULTS}


TRUNC_POISS_DEFAULTS = {**TRUNC_EXP_FAMILY_DISTR_DEFAULTS}


TRUNC_EXP_DEFAULTS = {**TRUNC_EXP_FAMILY_DISTR_DEFAULTS}


TRUNC_WEIBULL_DEFAULTS = {**TRUNC_EXP_FAMILY_DISTR_DEFAULTS}


UNKNOWN_TRUNC_MULTI_NORM_DEFAULTS = {
    "val": (float, 0.2),
    "min_radius": (float, 3.0),
    "max_radius": (float, 10.0),
    "eps": (float, 1e-5),
    "r": (float, 1.0),
    "rate": (float, 1.5),
    "batch_size": (int, 10),
    "tol": (float, 1e-1),
    "workers": (int, 0),
    "num_samples": (int, 10),
    "covariance_matrix_lr": (float | None, None),
}


TRUNC_LQR_DEFAULTS = {
    "target_thickness": (float, float("inf")),
    "num_traj_phase_one": (int, int(1e100)),
    "num_traj_phase_two": (int, int(1e100)),
    "num_traj_gen_samples_B": (int, int(1e100)),
    "num_traj_gen_samples_A": (int, int(1e100)),
    "T_phase_one": (int, int(1e100)),
    "T_phase_two": (int, int(1e100)),
    "T_gen_samples_B": (int, int(1e100)),
    "T_gen_samples_A": (int, int(1e100)),
    "R": (float, REQ),
    "U_A": (float, REQ),
    "U_B": (float, REQ),
    "delta": (float, REQ),
    "eps1": (float, 0.9),
    "eps2": (float, 0.9),
    "repeat": (int | None, None),
    "gamma": (float, REQ),
    "alpha": (float, 1.0),
}


def check_and_fill_args(  # pylint: disable=too-many-branches
    args: Parameters, defaults: Mapping[str, tuple]
) -> Parameters:
    """Validate and fill missing arguments from a defaults mapping.

    For each key in ``defaults``, if ``args`` does not already have that
    attribute, the default value is set on ``args``.  If ``args`` already
    has the attribute, its value is type-checked and constraint-validated
    against the spec.

    Args:
        args: Parameter object to validate and fill.
        defaults: Mapping of argument names to ``(type_spec, default)`` or
            ``(type_spec, default, constraints)`` tuples.  ``default`` may
            be ``REQ`` to indicate a required argument with no default.

    Returns:
        The updated ``args`` object with missing values filled from defaults.

    Raises:
        ValueError: If a required argument is missing, an argument has an
            invalid type, or a value fails a numeric or choice constraint.
    """

    def is_valid_value(value: object, type_spec: object) -> bool:
        """Return True if value matches the type specification."""
        if isinstance(type_spec, (list, tuple)) and not isinstance(type_spec, type):
            return value in type_spec

        origin = get_origin(type_spec)
        if origin is Union or origin is types.UnionType:
            type_args = get_args(type_spec)
            return any(is_valid_value(value, arg) for arg in type_args)

        if isinstance(type_spec, type):
            # Allow int/float interoperability for numeric fields.
            if type_spec in (int, float) and isinstance(value, (int, float)):
                return True
            return isinstance(value, type_spec)

        return False

    def validate_constraints(value: object, constraints: dict) -> bool:
        """Return True if value satisfies all numeric and choice constraints."""
        if "min" in constraints and value < constraints["min"]:
            return False
        if "max" in constraints and value > constraints["max"]:
            return False
        if "choices" in constraints and value not in constraints["choices"]:
            return False
        return True

    for arg_name, spec in defaults.items():
        type_spec, default = spec[0], spec[1]
        constraints = spec[2] if len(spec) == 3 else {}

        if not has_attr(args, arg_name):
            # Don't auto-fill epochs if iterations is already set, and vice
            # versa — only one training-duration parameter should be active.
            if arg_name == "epochs" and has_attr(args, "iterations"):
                continue
            if arg_name == "iterations" and has_attr(args, "epochs"):
                continue
            if default == REQ:
                raise ValueError(f"Required argument '{arg_name}' is missing.")
            setattr(args, arg_name, default)
            continue

        value = getattr(args, arg_name)

        # None is always a valid "not provided" sentinel; skip type-checking.
        if value is None:
            continue

        if not is_valid_value(value, type_spec):
            if isinstance(type_spec, (list, tuple)):
                expected = f"one of {type_spec}"
            else:
                expected = f"type {type_spec}"
            raise ValueError(
                f"Argument '{arg_name}' has invalid type. "
                f"Got {type(value).__name__} with value {value}, "
                f"expected {expected}"
            )

        if value is not None and not validate_constraints(value, constraints):
            constraints_desc = []
            if "min" in constraints:
                constraints_desc.append(f">= {constraints['min']}")
            if "max" in constraints:
                constraints_desc.append(f"<= {constraints['max']}")
            if "choices" in constraints:
                constraints_desc.append(f"in {constraints['choices']}")
            raise ValueError(
                f"Argument '{arg_name}' failed constraints. "
                f"Value {value} must be {', '.join(constraints_desc)}"
            )

    # Enforce that exactly one of epochs/iterations is set, but only when
    # the defaults dict defines training-duration parameters.
    if "epochs" in defaults or "iterations" in defaults:
        epochs_provided = getattr(args, "epochs", None) is not None
        iterations_provided = getattr(args, "iterations", None) is not None

        if epochs_provided and iterations_provided:
            raise ValueError(
                "You must provide exactly ONE of 'epochs' or 'iterations' (not both)."
            )

        if not epochs_provided and not iterations_provided:
            raise ValueError(
                "You must provide exactly ONE of 'epochs' or 'iterations',"
                " but neither was supplied."
            )

    return args
