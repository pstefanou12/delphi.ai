# Author: pstefanou12@
"""Tests for delphi.utils.defaults.check_and_fill_args."""

import pytest

from delphi.utils.defaults import (
    REQ,
    DELPHI_DEFAULTS,
    OPTIMIZER_DEFAULTS,
    SGD_DEFAULTS,
    TRAINER_DEFAULTS,
    TRUNC_LQR_DEFAULTS,
    TRUNC_LOG_REG_DEFAULTS,
    TRUNC_REG_DEFAULTS,
    check_and_fill_args,
)
from delphi.utils.helpers import Parameters


def make_args(**kwargs) -> Parameters:
    """Return a Parameters object populated from keyword arguments."""
    return Parameters(kwargs)


# Minimal defaults dicts reused across tests.
_SIMPLE = {"lr": (float, 1e-2)}
_WITH_CONSTRAINTS = {"lr": (float, 1e-2, {"min": 0, "max": 1})}
_LIST_ENUM = {"mode": (["fast", "slow", None], "fast")}
_NULLABLE = {"path": (str | None, None)}
_REQUIRED = {"alpha": (float, REQ)}
_EPOCHS_KEY = {"epochs": (int, 5)}
_ITER_KEY = {"iterations": (int, 100)}
_ITER_NONE_KEY = {"iterations": (int | None, None)}


def test_default_filling() -> None:
    """Missing attributes are filled from defaults; existing values are kept."""
    # Missing attribute receives the default value.
    args = make_args()
    check_and_fill_args(args, _SIMPLE)
    assert args.lr == 1e-2

    # Missing attribute with a None default is explicitly set to None.
    check_and_fill_args(args, _NULLABLE)
    assert args.path is None

    # Pre-existing attribute is not overwritten.
    args = make_args(lr=0.5)
    check_and_fill_args(args, _SIMPLE)
    assert args.lr == 0.5

    # All keys in the defaults dict are filled when absent.
    args = make_args()
    check_and_fill_args(args, SGD_DEFAULTS)
    assert args.lr == 1e-1
    assert args.momentum == 0.0
    assert args.nesterov is False

    # The returned object is the same instance that was passed in.
    args = make_args()
    result = check_and_fill_args(args, _SIMPLE)
    assert result is args


def test_required_args() -> None:
    """REQ sentinel raises a clear ValueError when the argument is absent."""
    # Absent required argument raises with the argument name.
    args = make_args()
    with pytest.raises(ValueError, match="Required argument 'alpha' is missing"):
        check_and_fill_args(args, _REQUIRED)

    # Present required argument passes without error.
    args = make_args(alpha=0.5)
    check_and_fill_args(args, _REQUIRED)
    assert args.alpha == 0.5

    # TRUNC_LQR_DEFAULTS raises for the first missing required field.
    args = make_args()
    with pytest.raises(ValueError, match="Required argument '.*' is missing"):
        check_and_fill_args(args, TRUNC_LQR_DEFAULTS)


def test_type_validation() -> None:
    """Valid types are accepted; mismatched types raise ValueError."""
    # Valid str passes.
    args = make_args(optimizer="adam")
    check_and_fill_args(args, OPTIMIZER_DEFAULTS)

    # Non-str for a str field raises.
    args = make_args(optimizer=123)
    with pytest.raises(ValueError, match="'optimizer'"):
        check_and_fill_args(args, OPTIMIZER_DEFAULTS)

    # Valid bool passes.
    args = make_args(nesterov=True)
    check_and_fill_args(args, {"nesterov": (bool, False)})

    # Non-bool for a bool field raises.
    args = make_args(nesterov="yes")
    with pytest.raises(ValueError, match="'nesterov'"):
        check_and_fill_args(args, {"nesterov": (bool, False)})

    # int accepted for a float field (numeric interoperability).
    args = make_args(lr=1, iterations=100)
    check_and_fill_args(args, TRUNC_REG_DEFAULTS)

    # float accepted for an int field (numeric interoperability).
    args = make_args(batch_size=32.0, iterations=100)
    check_and_fill_args(args, TRUNC_REG_DEFAULTS)


def test_union_type_validation() -> None:
    """X | None type specs accept valid values and None; reject wrong types."""
    # Typed (non-None) value is accepted.
    args = make_args(path="/tmp")
    check_and_fill_args(args, _NULLABLE)
    assert args.path == "/tmp"

    # None is accepted for a nullable field.
    args = make_args(path=None)
    check_and_fill_args(args, {"path": (str | None, "/default")})

    # Wrong type for a nullable field raises.
    args = make_args(path=42)
    with pytest.raises(ValueError, match="'path'"):
        check_and_fill_args(args, _NULLABLE)

    # Explicit None when the default is non-None skips type-checking entirely.
    args = make_args(path=None)
    check_and_fill_args(args, {"path": (str, "/default")})


def test_list_enum_validation() -> None:
    """List-valued type specs act as an allowlist; unlisted values raise."""
    # Value in the enum list is accepted.
    args = make_args(mode="slow")
    check_and_fill_args(args, _LIST_ENUM)

    # None is accepted when explicitly listed.
    args = make_args(mode=None)
    check_and_fill_args(args, _LIST_ENUM)

    # Value absent from the enum list raises.
    args = make_args(mode="turbo")
    with pytest.raises(ValueError, match="'mode'"):
        check_and_fill_args(args, _LIST_ENUM)

    # Valid scheduler name passes OPTIMIZER_DEFAULTS.
    args = make_args(scheduler="cosine")
    check_and_fill_args(args, OPTIMIZER_DEFAULTS)

    # None scheduler is valid.
    args = make_args(scheduler=None)
    check_and_fill_args(args, OPTIMIZER_DEFAULTS)

    # Unknown scheduler name raises.
    args = make_args(scheduler="unknown")
    with pytest.raises(ValueError, match="'scheduler'"):
        check_and_fill_args(args, OPTIMIZER_DEFAULTS)


def test_constraint_validation() -> None:
    """Numeric min/max constraints are enforced; None values bypass them."""
    # Value at the min boundary passes.
    args = make_args(lr=0.0)
    check_and_fill_args(args, _WITH_CONSTRAINTS)

    # Value below min raises.
    args = make_args(lr=-0.01)
    with pytest.raises(ValueError, match="'lr'"):
        check_and_fill_args(args, _WITH_CONSTRAINTS)

    # Value at the max boundary passes.
    args = make_args(lr=1.0)
    check_and_fill_args(args, _WITH_CONSTRAINTS)

    # Value above max raises.
    args = make_args(lr=1.01)
    with pytest.raises(ValueError, match="'lr'"):
        check_and_fill_args(args, _WITH_CONSTRAINTS)

    # None value is not checked against numeric constraints.
    args = make_args(val_interval=None)
    check_and_fill_args(args, {"val_interval": (int | None, 50, {"min": 1})})


def test_epochs_iterations_guard() -> None:
    """Exactly one of epochs/iterations must be set when the defaults dict
    includes either key; the guard is skipped for unrelated defaults dicts."""
    # Only epochs provided — passes.
    args = make_args(epochs=10)
    check_and_fill_args(args, _EPOCHS_KEY)

    # Only iterations provided — passes.
    args = make_args(iterations=500)
    check_and_fill_args(args, _ITER_KEY)

    # Both provided — raises.
    args = make_args(epochs=5, iterations=100)
    with pytest.raises(ValueError, match="exactly ONE"):
        check_and_fill_args(args, _ITER_KEY)

    # Neither provided (None default for iterations) — raises.
    args = make_args()
    with pytest.raises(ValueError, match="neither was supplied"):
        check_and_fill_args(args, _ITER_NONE_KEY)

    # Defaults without epochs/iterations key do not trigger the guard,
    # even when args also lacks both.
    args = make_args()
    check_and_fill_args(args, _SIMPLE)
    check_and_fill_args(args, SGD_DEFAULTS)
    check_and_fill_args(args, TRAINER_DEFAULTS)


def test_integration() -> None:
    """Real defaults dicts fill expected keys and respect user overrides."""
    # TRUNC_REG_DEFAULTS fills missing iterations with 1500.
    args = make_args()
    check_and_fill_args(args, TRUNC_REG_DEFAULTS)
    assert args.iterations == 1500

    # User-supplied iterations is preserved.
    args = make_args(iterations=200)
    check_and_fill_args(args, TRUNC_REG_DEFAULTS)
    assert args.iterations == 200

    # TRUNC_LOG_REG_DEFAULTS fills missing epochs with 1.
    args = make_args()
    check_and_fill_args(args, TRUNC_LOG_REG_DEFAULTS)
    assert args.epochs == 1

    # Providing iterations alongside TRUNC_LOG_REG_DEFAULTS (epochs-based) raises.
    args = make_args(iterations=100)
    with pytest.raises(ValueError, match="exactly ONE"):
        check_and_fill_args(args, TRUNC_LOG_REG_DEFAULTS)

    # DELPHI_DEFAULTS fills optimizer and device.
    args = make_args()
    check_and_fill_args(args, DELPHI_DEFAULTS)
    assert args.optimizer == "sgd"
    assert args.device == "cpu"

    # Calling check_and_fill_args twice is idempotent.
    args = make_args(iterations=100)
    check_and_fill_args(args, TRUNC_REG_DEFAULTS)
    lr_first = args.lr
    check_and_fill_args(args, TRUNC_REG_DEFAULTS)
    assert args.lr == lr_first
    assert args.iterations == 100
