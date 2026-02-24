# Author: pstefanou12@
"""
Default parameters for running algorithms in delphi.ai.
"""

from typing import Optional, Union, get_origin, get_args

import torch as ch

from delphi.utils.helpers import has_attr

# Module-level constants.
REQ = "required"

OPTIMIZER_DEFAULTS = {
    "optimizer": (str, "sgd"),
    "scheduler": (
        Optional[  # noqa: F821
            [
                "cyclic",  # noqa: F821
                "cosine",  # noqa: F821
                "linear",  # noqa: F821
                "step",  # noqa: F821
                "multi_step",  # noqa: F821
                "exponential",  # noqa: F821
                "reduce_on_plateau",  # noqa: F821
            ]
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
    "foreach": (Optional[bool], None),
    "differentiable": (bool, False),
    "fused": (Optional[bool], None),
}

LBFGS_DEFAULTS = {
    "lr": (float, 1.0, {"min": 0}),
    "max_iter": (int, 20, {"min": 1}),
    "max_eval": (Optional[int], None),
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
    "val_interval": (int, None, {"min": 1}),
    "patience": (int, float("inf"), {"min": 1}),
    "grad_tol": (float, 0, {"min": 0}),
    "log_every": (int, 50, {"min": 1}),
    "max_grad_norm": (float, None),
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

DELPHI_DEFAULTS = {**OPTIMIZER_DEFAULTS, **{"device": (str, "cpu")}}

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
    "suffle": (bool, True),
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
    "min_radius": (float, 3.0),
    "max_radius": (float, 10.0),
    "rate": (float, 1.1),  # increase radius size by 10% each trial
    "batch_size": (int, 10),
    "tol": (float, 1e-1),
    "num_samples": (int, 10000),
    "optimizer": (str, "sgd"),
    "max_phases": (int, 1),
    "loss_convergence_tol": (float, 1e-3),
    "relative_loss_tol": (float, float("inf")),
}

TRUNC_MULTI_NORM_DEFAULTS = {
    **TRUNC_EXP_FAMILY_DISTR_DEFAULTS,
    **{
        "covariance_matrix": (ch.Tensor, None),
        "covariance_matrix_lr": (float, 1e-2),
        "eigenvalue_lower_bound": (float, 1e-2),
    },
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
    "covariance_matrix_lr": (float, None),
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
    "repeat": (int, None),
    "gamma": (float, REQ),
    "alpha": (float, 1.0),
}


def check_and_fill_args(args, defaults):  # pylint: disable=too-many-branches
    """
    Validate and fill in missing arguments from defaults.

    Args:
        args (Parameters): parameter object to validate and fill
        defaults (dict): mapping of argument names to (type_spec, default) or
            (type_spec, default, constraints) tuples

    Returns:
        The updated args object with missing values filled from defaults.

    Raises:
        ValueError: if an argument has an invalid type or fails a constraint.
    """

    def is_valid_value(value, type_spec):
        """Check if value matches type specification."""
        if isinstance(type_spec, (list, tuple)) and not isinstance(type_spec, type):
            return value in type_spec

        origin = get_origin(type_spec)
        if origin is Union or (
            hasattr(type_spec, "__origin__") and get_origin(type_spec) is Union
        ):
            # Check all type arguments in the Union (Optional[T] is Union[T, None])
            args = get_args(type_spec)
            return any(is_valid_value(value, arg) for arg in args)

        # Handle type classes
        if isinstance(type_spec, type):
            # Special handling for numeric types
            if type_spec in (int, float) and isinstance(value, (int, float)):
                return True
            return isinstance(value, type_spec)

        return False

    def validate_constraints(value, constraints):
        """Validate value against constraints."""
        if not constraints:
            return True

        if "min" in constraints and value < constraints["min"]:
            return False
        if "max" in constraints and value > constraints["max"]:
            return False
        if "choices" in constraints and value not in constraints["choices"]:
            return False

        return True

    for arg_name, spec in defaults.items():
        # Parse specification (support both 2-tuple and 3-tuple formats)
        if len(spec) == 2:
            type_spec, default = spec
            constraints = {}
        else:
            type_spec, default, constraints = spec

        # Check if argument exists using standard has_attr
        if not has_attr(args, arg_name):
            if default is not None:
                setattr(args, arg_name, default)
            continue

        # Get current value using standard getattr
        value = getattr(args, arg_name)

        # Skip validation for None values (unless required)
        if value is None and default is not None:
            continue

        # Type validation
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

        # Constraint validation (only for non-None values)
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

    # verify that exactly one of iterations/epochs is provided
    epochs_provided_by_user = getattr(args, "epochs", None) is not None
    steps_provided_by_user = getattr(args, "iterations", None) is not None

    if epochs_provided_by_user and steps_provided_by_user:
        raise ValueError(
            "You must provide exactly ONE of 'epochs' or 'iterations' (not both)."
        )

    if not epochs_provided_by_user and not steps_provided_by_user:
        raise ValueError(
            "You must provide exactly ONE of 'epochs' or 'iterations', "
            "but neither was supplied."
        )

    return args
