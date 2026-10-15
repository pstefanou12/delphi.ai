# Author: pstefanou12@
"""Tests for optimizer and scheduler construction in delphi.delphi."""

import inspect

import pydantic
import pytest
import torch

from delphi import configs
from delphi.delphi import OPTIMIZER_SECTIONS, SCHEDULER_SECTIONS, delphi
from delphi.optim.lr_scheduler import fisher
from delphi.utils import constants


class Model(delphi[configs.TruncatedRegressionConfig]):
    """Minimal concrete model with a parameter to optimise."""

    defaults = configs.TruncatedRegressionConfig

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        self.linear = torch.nn.Linear(2, 1)


class SiblingModel(delphi[configs.TruncatedRegressionConfig]):
    """A sibling of Model, deriving from delphi independently.

    Separate from Model rather than a child of it, because inheriting a
    parent's registrations is correct; what must not happen is one model
    seeing another's.
    """

    defaults = configs.TruncatedRegressionConfig

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        self.linear = torch.nn.Linear(2, 1)


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
        hyperparameters: dict = {
            "scheduler": name.value,
            "epochs": 5,
            "constant": False,
        }
        if name is constants.SchedulerType.FISHER:
            # Model declares no hessian(), which is the path a caller takes
            # when supplying the curvature constants directly.
            hyperparameters |= {"fisher_mu": 0.5, "fisher_smoothness": 8.0}
        _, schedule = Model(hyperparameters).make_optimizer_and_schedule()
        assert isinstance(schedule, target)


def test_fisher_scheduler_refuses_a_model_without_a_hessian() -> None:
    """Deriving the spectrum needs a Hessian, so say so rather than guess."""
    with pytest.raises(NotImplementedError, match="hessian"):
        Model(
            {"scheduler": "fisher", "epochs": 5, "constant": False}
        ).make_optimizer_and_schedule()


def test_fisher_scheduler_derives_its_constants_from_a_hessian() -> None:
    """A model that reports curvature needs no learning rate of its own."""

    class CurvedModel(Model):  # pylint: disable=too-few-public-methods
        """Model whose Hessian has a known spectrum."""

        independent_coordinates = True

        def hessian(self) -> torch.Tensor:
            """Return a diagonal Hessian with eigenvalues 2 and 50."""
            return torch.diag(torch.Tensor([2.0, 50.0]))

    _, schedule = CurvedModel(
        {"scheduler": "fisher", "epochs": 5, "constant": False}
    ).make_optimizer_and_schedule()
    assert isinstance(schedule, fisher.FisherLR)
    assert (schedule.mu, schedule.smoothness) == (2.0, 50.0)
    # Starts at 1 / smoothness, the largest step that stays stable.
    assert schedule.get_last_lr()[0] == pytest.approx(1.0 / 50.0)


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


def test_global_registration_reaches_models_defined_earlier() -> None:
    """A factory registered on delphi works for pre-existing subclasses.

    This is the notebook case: the estimators are defined at import time and
    the user registers afterwards. Resolving along the MRO rather than copying
    at class-definition time is what makes it work.
    """
    delphi.register_optimizer(
        "late_probe", lambda model, params: torch.optim.RMSprop(params, lr=0.3)
    )
    optimizer, _ = Model(
        {"optimizer": "late_probe", "constant": True}
    ).make_optimizer_and_schedule()
    assert isinstance(optimizer, torch.optim.RMSprop)


def test_multi_optimizer_models_bypass_the_registry() -> None:
    """A GAN-style model builds its own optimizers and still validates."""

    class Gan(Model):
        """Two optimizers stepped from train_batch, as a GAN needs."""

        def make_optimizer_and_schedule(self, checkpoint=None):
            self.optimizers = {
                "generator": torch.optim.Adam(self.linear.parameters(), lr=1e-3),
                "discriminator": torch.optim.Adam(self.linear.parameters(), lr=4e-4),
            }
            return None, None

    model = Gan({"constant": True})
    model.make_optimizer_and_schedule()
    model.validate_before_training()
    assert sorted(model.optimizers) == ["discriminator", "generator"]


def test_registration_does_not_leak_to_siblings_or_the_base() -> None:
    """One model's factory stays its own; a child still inherits it."""

    def factory(model, params):  # pylint: disable=unused-argument
        return torch.optim.RMSprop(params, lr=0.1)

    class Child(Model):
        """Inherits Model's registrations, which is intended."""

    Model.register_optimizer("isolated_probe", factory)
    probe = {"optimizer": "sgd", "constant": True}
    assert Model(probe)._registered_factory("isolated_probe") is factory  # pylint: disable=protected-access
    assert Child(probe)._registered_factory("isolated_probe") is factory  # pylint: disable=protected-access
    assert SiblingModel(probe)._registered_factory("isolated_probe") is None  # pylint: disable=protected-access
    assert "isolated_probe" not in delphi.__dict__.get("_OPTIMIZER_REGISTRY", {})


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
