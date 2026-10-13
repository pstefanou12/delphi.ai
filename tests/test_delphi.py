# Author: pstefanou12@
"""Tests for optimizer and scheduler construction in delphi.delphi."""

import inspect

import pydantic
import pytest
import torch

from delphi import configs
from delphi.delphi import OPTIMIZER_SECTIONS, SCHEDULER_SECTIONS, delphi


class Model(delphi[configs.TruncatedRegressionConfig]):
    """Minimal concrete model with a parameter to optimise."""

    defaults = configs.TruncatedRegressionConfig

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        self.linear = torch.nn.Linear(2, 1)


class OtherModel(Model):
    """A sibling, to check registrations do not leak between subclasses."""


def test_builtin_optimizers_build_from_their_sections() -> None:
    """Each built-in is constructed without needing a registered factory."""
    for name, (target, _) in OPTIMIZER_SECTIONS.items():
        optimizer, _ = Model(
            {"optimizer": name, "lr": 0.05, "constant": True}
        ).make_optimizer_and_schedule()
        assert isinstance(optimizer, target)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_every_scheduler_builds() -> None:
    """Each scheduler is constructed from its section."""
    for name, (target, _) in SCHEDULER_SECTIONS.items():
        _, schedule = Model(
            {"scheduler": name.value, "epochs": 5, "constant": False}
        ).make_optimizer_and_schedule()
        assert isinstance(schedule, target)


def test_section_aliases_match_their_torch_signature() -> None:
    """Every name a section emits is a real parameter of its torch class."""
    for target, section in list(OPTIMIZER_SECTIONS.values()) + list(
        SCHEDULER_SECTIONS.values()
    ):
        accepted = set(inspect.signature(target).parameters)
        emitted = {
            field.serialization_alias or name
            for name, field in section.model_fields.items()
        }
        assert emitted <= accepted, f"{sorted(emitted - accepted)} not on {target}"


def test_registry_is_isolated_per_subclass() -> None:
    """Registering on one model does not affect siblings or the base."""

    def factory(model, params):  # pylint: disable=unused-argument
        return torch.optim.RMSprop(params, lr=0.1)

    Model.register_optimizer("isolated_probe", factory)
    assert "isolated_probe" in Model._OPTIMIZER_REGISTRY  # pylint: disable=protected-access
    assert "isolated_probe" not in OtherModel._OPTIMIZER_REGISTRY  # pylint: disable=protected-access
    assert "isolated_probe" not in delphi._OPTIMIZER_REGISTRY  # pylint: disable=protected-access


def test_registered_factory_overrides_a_builtin_for_that_model_only() -> None:
    """A custom factory wins, and the override does not escape the subclass."""

    class Overriding(Model):
        """Replaces SGD for itself alone."""

    Overriding.register_optimizer(
        "sgd", lambda model, params: torch.optim.RMSprop(params, lr=0.9)
    )
    overridden, _ = Overriding(
        {"optimizer": "sgd", "constant": True}
    ).make_optimizer_and_schedule()
    untouched, _ = Model(
        {"optimizer": "sgd", "constant": True}
    ).make_optimizer_and_schedule()
    assert isinstance(overridden, torch.optim.RMSprop)
    assert isinstance(untouched, torch.optim.SGD)


def test_unknown_optimizer_raises() -> None:
    """A name that is neither registered nor built in is rejected."""
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        Model(
            {"optimizer": "not_an_optimizer", "constant": True}
        ).make_optimizer_and_schedule()


def test_unknown_scheduler_raises() -> None:
    """A scheduler name outside SchedulerType is rejected."""
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        Model(
            {"scheduler": "not_a_scheduler", "constant": False}
        ).make_optimizer_and_schedule()


def test_sections_are_module_level_not_model_state() -> None:
    """The lookup tables are constants, not attributes of the model."""
    assert not hasattr(delphi, "_OPTIMIZER_SECTIONS")
    assert not hasattr(delphi, "_SCHEDULER_SECTIONS")
    assert isinstance(OPTIMIZER_SECTIONS, dict)
    assert all(
        issubclass(section, pydantic.BaseModel)
        for _, section in OPTIMIZER_SECTIONS.values()
    )
