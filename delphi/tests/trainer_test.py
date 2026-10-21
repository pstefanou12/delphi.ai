# Author: pstefanou12@
"""Tests for the trainer's procedure: the warm start and the phases."""

from typing import Optional

import pytest
import torch as ch
from torch.utils import data

from delphi import configs
import delphi.delphi
import delphi.trainer
from delphi.utils import constants

_TARGET = 3.0


def _squared_error(pred, targ, offset=0.0):
    return ((pred - (targ + offset)) ** 2).mean()


class Mean(delphi.delphi.delphi[configs.DelphiConfig]):
    """Estimates a mean by squared error, recording what its hooks receive."""

    defaults = configs.DelphiConfig

    def __init__(self, hyperparameters: dict, phases: int = 1):
        super().__init__(hyperparameters)
        self.mu = ch.nn.Parameter(ch.zeros(1))
        self.criterion = _squared_error
        self.phases = phases
        self.warm_start_results: list = []
        self.phase_results: list = []

    def forward(self, x):
        return self.mu.expand(x.size(0))

    def warm_start_end_hook(self, result):
        self.warm_start_results.append(result)

    def next_phase_hook(self, result):
        self.phase_results.append(result)
        return len(self.phase_results) < self.phases


def _fit(model: Mean) -> delphi.trainer.Trainer:
    ch.manual_seed(0)
    samples = _TARGET + 0.1 * ch.randn(200)
    loader = data.DataLoader(
        data.TensorDataset(ch.zeros(200, 1), samples), batch_size=20
    )
    procedure = delphi.trainer.Trainer(model, model.hyperparameters)
    procedure.fit(loader, loader)
    return procedure


def _model(phases: int = 1, **hyperparameters) -> Mean:
    settings: dict = {"iterations": 100, "lr": 0.1, "record_params_every": 1}
    return Mean(settings | hyperparameters, phases)


def _estimate(params: Optional[ch.Tensor]) -> float:
    assert params is not None
    return float(params[0])


def test_a_model_without_phases_trains_once_without_a_warm_start():
    procedure = _fit(_model())
    assert len(procedure.phases) == 1
    assert procedure.warm_start_result is None
    assert procedure.procedure_stop_reason == "no_next_phase"
    assert _estimate(procedure.avg_params) == pytest.approx(_TARGET, abs=0.05)


def test_a_warm_start_budget_without_a_criterion_raises():
    with pytest.raises(ValueError, match="warm_start_criterion"):
        _fit(_model(warm_start_iterations=10))


def test_the_warm_start_trains_its_own_criterion_and_hands_over_its_result():
    model = _model(warm_start_iterations=100)
    model.warm_start_criterion = _squared_error
    model.warm_start_criterion_params = [7.0]
    procedure = _fit(model)
    (warm,) = model.warm_start_results
    assert warm is procedure.warm_start_result
    assert warm.iterations == 100
    assert _estimate(warm.final_params) == pytest.approx(_TARGET + 7.0, abs=0.05)
    # The main run went back to the model's own criterion and settings.
    assert _estimate(procedure.avg_params) == pytest.approx(_TARGET, abs=0.05)
    assert procedure.phases[0].iterations == 100
    assert model.hyperparameters.iterations == 100
    assert model.hyperparameters.warm_start_iterations == 100


def test_the_warm_start_stops_on_its_own_criteria():
    model = _model(
        warm_start_iterations=10_000,
        warm_start_early_stopping=True,
        warm_start_grad_tol=1e-2,
    )
    model.warm_start_criterion = _squared_error
    procedure = _fit(model)
    warm = procedure.warm_start_result
    assert warm is not None
    assert warm.stop_reason == constants.StopReason.GRAD_TOL
    assert warm.iterations < 10_000
    # The main run keeps its own budget rather than the stage's criteria.
    assert procedure.phases[0].iterations == 100


def test_the_model_ends_the_phases_and_sees_each_result():
    model = _model(phases=3, phase_loss_tol=None)
    procedure = _fit(model)
    assert len(procedure.phases) == 3
    assert model.phase_results == procedure.phases
    assert procedure.procedure_stop_reason == "no_next_phase"


def test_a_converged_loss_ends_the_phases_early():
    # Every phase fits the same problem, so the second barely improves.
    procedure = _fit(_model(phases=5, phase_loss_tol=1e-3))
    assert len(procedure.phases) == 2
    assert procedure.procedure_stop_reason.startswith("phase_loss_converged")


def test_max_phases_caps_the_phases():
    procedure = _fit(_model(phases=5, phase_loss_tol=None, max_phases=2))
    assert len(procedure.phases) == 2
    assert procedure.procedure_stop_reason == "max_phases"


class CurvedMean(Mean):
    """Mean whose criteria report curvature, recording who asked for it."""

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        self.curvature_calls: list[str] = []

    def hessian(self, theta=None):  # pylint: disable=unused-argument
        self.curvature_calls.append("main")
        return 2.0 * ch.eye(1)

    def stage_hessian(self, theta=None):  # pylint: disable=unused-argument
        self.curvature_calls.append("warm start")
        return 2.0 * ch.eye(1)


def _curved(**hyperparameters) -> CurvedMean:
    settings: dict = {
        "iterations": 50,
        "record_params_every": 1,
        "scheduler": "fisher",
        "warm_start_iterations": 50,
    }
    model = CurvedMean(settings | hyperparameters)
    model.warm_start_criterion = _squared_error
    return model


def test_a_fisher_warm_start_without_its_own_hessian_raises():
    with pytest.raises(ValueError, match="warm_start_hessian"):
        _fit(_curved())


def test_the_warm_start_takes_its_step_sizes_from_its_own_hessian():
    model = _curved()
    model.warm_start_hessian = model.stage_hessian
    _fit(model)
    assert model.curvature_calls[0] == "warm start"
    assert "main" in model.curvature_calls
    first_main = model.curvature_calls.index("main")
    assert set(model.curvature_calls[first_main:]) == {"main"}
