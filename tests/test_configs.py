# Author: pstefanou12@
"""Tests for the delphi.configs Pydantic models.

These lock in the invariants established when the optimizer, scheduler and
dataset groups were extracted: that every hyperparameter the library reads is
declared, that the declared defaults match the values that were actually in
effect beforehand, and that the four previously-colliding field names stay
distinct.
"""

import pathlib
import re

import pydantic
import pytest

from delphi import configs

# The defaults that were operative before the extraction, taken from the
# getattr/_get_arg fallbacks in delphi.delphi rather than from the *_DEFAULTS
# mappings, which were referenced only by tests and never applied to a live
# config.
OPERATIVE_DEFAULTS = {
    "adam_eps": 1e-8,
    "amsgrad": False,
    "beta1": 0.9,
    "beta2": 0.999,
    "capturable": False,
    "constant": False,
    "cyclic_base_lr": 0.0,
    "cyclic_gamma": 1.0,
    "cyclic_max_lr": None,
    "cyclic_mode": "triangular2",
    "cyclic_step_size_down": None,
    "cyclic_step_size_up": 2000,
    "dampening": 0.0,
    "differentiable": False,
    "exponential_gamma": 0.95,
    "foreach": None,
    "fused": None,
    "history_size": 100,
    "line_search_fn": None,
    "linear_end_factor": 0.0,
    "linear_start_factor": 1.0,
    "lr": 0.1,
    "max_eval": None,
    "max_iter": 20,
    "maximize": False,
    "milestones": [30, 60, 90],
    "min_lr": 0.0,
    "momentum": 0.0,
    "multi_step_gamma": 0.1,
    "nesterov": False,
    "optimizer": "sgd",
    "plateau_cooldown": 0,
    "plateau_eps": 1e-8,
    "plateau_factor": 0.1,
    "plateau_mode": "min",
    "plateau_patience": 10,
    "plateau_threshold": 1e-4,
    "plateau_threshold_mode": "rel",
    "scheduler": None,
    "step_lr": 100,
    "step_lr_gamma": 0.1,
    "tolerance_change": 1e-9,
    "tolerance_grad": 1e-7,
    "warmup_steps": 0,
    "weight_decay": 0.0,
}


def test_every_hyperparameter_delphi_reads_is_declared() -> None:
    """No hyperparameter access in delphi.delphi falls through undeclared."""
    source = pathlib.Path("delphi/delphi.py").read_text()
    read = set(re.findall(r"self\.hyperparameters\.([A-Za-z_0-9]+)", source))
    undeclared = read - set(configs.DelphiConfig.model_fields)
    assert not undeclared, f"undeclared fields read by delphi.py: {sorted(undeclared)}"


def test_no_defensive_hyperparameter_reads_remain() -> None:
    """delphi.delphi reads declared attributes, not getattr with a fallback."""
    source = pathlib.Path("delphi/delphi.py").read_text()
    assert "getattr(self.hyperparameters" not in source
    assert "_get_arg(" not in source


def test_operative_defaults_are_preserved() -> None:
    """Declaring the fields did not change any value that was in effect."""
    config = configs.DelphiConfig()
    mismatched = {
        name: (expected, getattr(config, name))
        for name, expected in OPERATIVE_DEFAULTS.items()
        if getattr(config, name) != expected
    }
    assert not mismatched, f"defaults drifted: {mismatched}"


def test_dataset_fields_come_from_the_dataset_config() -> None:
    """Dataset concerns are declared once and inherited, not per model."""
    dataset_fields = set(configs.DatasetConfig.model_fields)
    assert dataset_fields == {"val", "workers", "shuffle", "batch_size", "normalize"}
    for model in (
        configs.LinearModelConfig,
        configs.TruncatedExponentialFamilyDistributionConfig,
    ):
        assert dataset_fields <= set(model.model_fields)


def test_unread_dataset_fields_are_not_declared() -> None:
    """pin_memory and drop_last were never read, so they are gone."""
    declared = set(configs.DelphiConfig.model_fields)
    for name in ("pin_memory", "drop_last"):
        assert name not in declared


def test_batch_size_accepts_the_full_batch_sentinel() -> None:
    """-1 means "use the whole training set", per make_train_and_val."""
    assert configs.TruncatedRegressionConfig(batch_size=-1).batch_size == -1


def test_adam_epsilon_is_distinct_from_the_criterion_epsilon() -> None:
    """Adam's eps no longer collides with the truncation criterion's eps."""
    config = configs.TruncatedRegressionConfig()
    assert config.adam_eps == 1e-8
    assert config.eps == 1e-5
    # Setting one must not disturb the other.
    config = configs.TruncatedRegressionConfig(eps=1e-3)
    assert config.adam_eps == 1e-8


def test_scheduler_gamma_is_distinct_from_the_lqr_gamma() -> None:
    """The LQR thickness parameter no longer shadows a scheduler decay rate."""
    config = configs.TruncatedLQRConfig(
        phi=lambda x: x, R=3.0, U_A=1.0, U_B=1.0, delta=0.9, gamma=2.0
    )
    assert config.gamma == 2.0
    assert config.multi_step_gamma == 0.1
    assert config.exponential_gamma == 0.95


def test_constraints_are_enforced() -> None:
    """Out-of-range values raise instead of being silently accepted."""
    with pytest.raises(pydantic.ValidationError):
        configs.DelphiConfig(beta1=2.0)
    with pytest.raises(pydantic.ValidationError):
        configs.DelphiConfig(plateau_patience=0)
    with pytest.raises(pydantic.ValidationError):
        # mypy rejects this too, which is the point: it is a Literal.
        configs.DelphiConfig(cyclic_mode="not_a_mode")  # type: ignore[arg-type]


def test_required_lqr_fields_must_be_supplied() -> None:
    """Fields the old specs marked REQ have no default."""
    with pytest.raises(pydantic.ValidationError):
        # mypy also rejects this, which is the point: the fields are required.
        configs.TruncatedLQRConfig()  # type: ignore[call-arg]


def test_unknown_keys_are_ignored() -> None:
    """Callers pass one flat dict, so extra keys must not raise."""
    # Splatting a caller-supplied dict is exactly what delphi.__init__ does.
    hyperparameters = {"not_a_field": 1, "lr": 0.5}
    config = configs.TruncatedRegressionConfig(**hyperparameters)  # type: ignore[arg-type]
    assert config.lr == 0.5


def test_dependent_variant_overrides_only_defaults() -> None:
    """The LDS config differs from the regression config by defaults alone."""
    regression = configs.TruncatedRegressionConfig()
    dynamical = configs.TruncatedLinearDynamicalSystemConfig()
    assert (regression.shuffle, regression.constant) == (True, False)
    assert (dynamical.shuffle, dynamical.constant) == (False, True)
    assert set(configs.TruncatedRegressionFieldsConfig.model_fields) <= set(
        type(dynamical).model_fields
    )
