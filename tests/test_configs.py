# Author: pstefanou12@
"""Tests for the delphi.configs Pydantic models.

These lock in the invariants established when the optimizer, scheduler and
dataset groups were extracted: that every hyperparameter the library reads is
declared, that the declared defaults match the values that were actually in
effect beforehand, and that the four previously-colliding field names stay
distinct.
"""

import ast
import inspect
import pathlib
import re

import pydantic
import pytest
import torch

from delphi import configs

# Taken from the getattr fallbacks delphi.delphi used, not from the
# *_DEFAULTS mappings, which were never applied to a live config.
OPERATIVE_DEFAULTS = {
    "adam_eps": 1e-8,
    "adam_amsgrad": False,
    "adam_betas": (0.9, 0.999),
    "adam_capturable": False,
    "constant": False,
    "cyclic_base_lr": 0.0,
    "cyclic_gamma": 1.0,
    "cyclic_max_lr": None,
    "cyclic_mode": "triangular2",
    "cyclic_step_size_down": None,
    "cyclic_step_size_up": 2000,
    "sgd_dampening": 0.0,
    "differentiable": False,
    "exponential_gamma": 0.95,
    "foreach": None,
    "fused": None,
    "lbfgs_history_size": 100,
    "lbfgs_line_search_fn": None,
    "linear_end_factor": 0.0,
    "linear_start_factor": 1.0,
    "lr": 0.1,
    "lbfgs_max_eval": None,
    "lbfgs_max_iter": 20,
    "maximize": False,
    "multi_step_milestones": [30, 60, 90],
    "plateau_min_lr": 0.0,
    "cosine_eta_min": 0.0,
    "sgd_momentum": 0.0,
    "multi_step_gamma": 0.1,
    "sgd_nesterov": False,
    "optimizer": "sgd",
    "plateau_cooldown": 0,
    "plateau_eps": 1e-8,
    "plateau_factor": 0.1,
    "plateau_mode": "min",
    "plateau_patience": 10,
    "plateau_threshold": 1e-4,
    "plateau_threshold_mode": "rel",
    "scheduler": None,
    "step_lr_step_size": 100,
    "step_lr_gamma": 0.1,
    "lbfgs_tolerance_change": 1e-9,
    "lbfgs_tolerance_grad": 1e-7,
    "warmup_steps": 0,
    "weight_decay": 0.0,
}


_REPO_ROOT = pathlib.Path(__file__).parent.parent


def test_every_hyperparameter_delphi_reads_is_declared() -> None:
    """No hyperparameter access in delphi.delphi falls through undeclared."""
    source = (_REPO_ROOT / "delphi/delphi.py").read_text()
    read = set(re.findall(r"self\.hyperparameters\.([A-Za-z_0-9]+)", source))
    # model_dump and friends are pydantic's API, not hyperparameters.
    read = {name for name in read if not name.startswith("model_")}
    undeclared = read - set(configs.DelphiConfig.model_fields)
    assert not undeclared, f"undeclared fields read by delphi.py: {sorted(undeclared)}"


def test_no_defensive_hyperparameter_reads_remain() -> None:
    """delphi.delphi reads declared attributes, not getattr with a fallback."""
    source = (_REPO_ROOT / "delphi/delphi.py").read_text()
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
        configs.DelphiConfig(adam_betas=(2.0, 0.999))
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


# Sections that name a torch class's keyword arguments, with the prefix each
# one's fields must carry.
_SECTIONS: dict[str, tuple[type, type[pydantic.BaseModel]]] = {
    "sgd_": (torch.optim.SGD, configs.SGDConfig),
    "adam_": (torch.optim.Adam, configs.AdamConfig),
    "lbfgs_": (torch.optim.LBFGS, configs.LBFGSConfig),
    "step_lr_": (torch.optim.lr_scheduler.StepLR, configs.StepLRConfig),
    "multi_step_": (torch.optim.lr_scheduler.MultiStepLR, configs.MultiStepLRConfig),
    "exponential_": (
        torch.optim.lr_scheduler.ExponentialLR,
        configs.ExponentialLRConfig,
    ),
    "cosine_": (torch.optim.lr_scheduler.CosineAnnealingLR, configs.CosineLRConfig),
    "linear_": (torch.optim.lr_scheduler.LinearLR, configs.LinearLRConfig),
    "cyclic_": (torch.optim.lr_scheduler.CyclicLR, configs.CyclicLRConfig),
    "plateau_": (torch.optim.lr_scheduler.ReduceLROnPlateau, configs.PlateauLRConfig),
}

# Declared on OptimizerConfig itself: these apply to whichever optimizer
# ``optimizer`` selects, so they are deliberately unprefixed.
_SHARED_FIELDS = {
    "lr",
    "weight_decay",
    "maximize",
    "foreach",
    "differentiable",
    "fused",
    "optimizer",
    "scheduler",
    "constant",
}


def test_every_emitted_name_is_a_real_torch_parameter() -> None:
    """Each section's aliases match its torch class's signature.

    This is what makes a redundant alias safe: a typo'd or stale one fails
    here rather than passing a wrong keyword or being silently dropped.
    """
    for prefix, (target, section) in _SECTIONS.items():
        accepted = set(inspect.signature(target).parameters)
        emitted = {
            field.serialization_alias or name
            for name, field in section.model_fields.items()
        }
        unknown = emitted - accepted
        assert not unknown, f"{prefix}: {sorted(unknown)} not parameters of {target}"


def test_every_section_field_is_prefixed_and_aliased() -> None:
    """One naming rule, no exceptions within a section."""
    for prefix, (_, section) in _SECTIONS.items():
        for name, field in section.model_fields.items():
            assert name.startswith(prefix), f"{name} lacks the {prefix} prefix"
            assert field.serialization_alias, f"{name} declares no serialization_alias"


def test_only_optimizer_config_fields_are_unprefixed() -> None:
    """The bare fields are exactly the ones OptimizerConfig declares itself."""
    section_owned: set[str] = set()
    for _, (_, section) in _SECTIONS.items():
        section_owned |= set(section.model_fields)
    section_owned |= set(configs.WarmupConfig.model_fields)
    bare = set(configs.OptimizerConfig.model_fields) - section_owned
    assert bare == _SHARED_FIELDS


def test_plateau_and_cosine_floors_are_independent() -> None:
    """Separate fields: one value no longer feeds both schedulers."""
    config = configs.DelphiConfig(plateau_min_lr=0.5)
    assert config.plateau_min_lr == 0.5
    assert config.cosine_eta_min == 0.0


def _dotted(node: ast.AST) -> str:
    """Return the dotted name of an attribute chain, or ""."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def test_configs_construct_nothing() -> None:
    """The configs declare names; delphi.delphi builds the torch objects.

    Checked on the syntax tree so docstrings naming the torch class a section
    describes do not count; a Tensor type annotation is also fine. What must
    not appear is a reference to torch.optim, which would mean a config had
    started constructing optimizers or schedulers.
    """
    for path in (_REPO_ROOT / "delphi/configs").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute):
                name = _dotted(node)
                assert "optim" not in name.split("."), f"{path} references {name}"
