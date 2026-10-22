# Author: pstefanou12@
"""Tests for the trainer's procedure: the warm start and the phases."""

import functools

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
    """Estimates a mean by squared error, keeping the base class's hooks."""

    defaults = configs.DelphiConfig

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        self.mu = ch.nn.Parameter(ch.zeros(1))
        self.criterion = _squared_error

    def forward(self, x):
        return self.mu.expand(x.size(0))


class PhasedMean(Mean):
    """Mean that runs a set number of phases, recording what its hooks see."""

    def __init__(self, hyperparameters: dict, phases: int = 1):
        super().__init__(hyperparameters)
        self.phases = phases
        self.warm_start_results: list = []
        self.phase_results: list = []
        self.trainings_finished = 0

    def post_warm_start_hook(self, result):
        self.warm_start_results.append(result)

    def post_phase_hook(self, result):
        self.phase_results.append(result)
        if len(self.phase_results) < self.phases:
            return False, None
        return True, f"ran {self.phases} phase(s)"

    def run_metrics(self):
        return {"phase": len(self.phase_results) + 1}

    def post_training_hook(self):
        self.trainings_finished += 1


def _loader() -> data.DataLoader:
    ch.manual_seed(0)
    samples = _TARGET + 0.1 * ch.randn(200)
    return data.DataLoader(data.TensorDataset(ch.zeros(200, 1), samples), batch_size=20)


def _fit(model: Mean) -> delphi.trainer.ProcedureResult:
    loader = _loader()
    return delphi.trainer.Trainer(model, model.hyperparameters).train_model(
        loader, loader
    )


_SETTINGS: dict = {"iterations": 100, "lr": 0.1, "record_params_every": 1}


def _phased(phases: int = 1, **hyperparameters) -> PhasedMean:
    return PhasedMean(_SETTINGS | hyperparameters, phases)


def _estimate(params) -> float:
    assert params is not None
    return float(params[0])


def test_a_model_keeping_the_base_hooks_trains_once():
    result = _fit(Mean(_SETTINGS))
    assert len(result.phases) == 1
    assert result.warm_start is None
    assert result.stop_reason == "single phase"
    assert result.final.metrics == {}
    assert _estimate(result.final.avg_params) == pytest.approx(_TARGET, abs=0.05)


def test_a_warm_start_budget_without_a_criterion_raises():
    with pytest.raises(ValueError, match="warm_start_criterion"):
        _fit(_phased(warm_start_iterations=10))


def test_the_warm_start_trains_its_own_criterion_and_hands_over_its_record():
    model = _phased(warm_start_iterations=100)
    model.warm_start_criterion = functools.partial(_squared_error, offset=7.0)
    result = _fit(model)
    (warm,) = model.warm_start_results
    assert warm is result.warm_start
    assert warm.stage == "warm start"
    assert warm.iterations == 100
    assert _estimate(warm.final_params) == pytest.approx(_TARGET + 7.0, abs=0.05)
    # The phases went back to the model's own criterion and settings.
    assert _estimate(result.final.avg_params) == pytest.approx(_TARGET, abs=0.05)
    assert result.final.iterations == 100
    assert model.hyperparameters.iterations == 100


def test_the_warm_start_stops_on_its_own_criteria():
    model = _phased(
        warm_start_iterations=10_000,
        warm_start_early_stopping=True,
        warm_start_grad_tol=1e-2,
    )
    model.warm_start_criterion = _squared_error
    result = _fit(model)
    assert result.warm_start is not None
    assert result.warm_start.stop_reason == constants.StopReason.GRAD_TOL
    assert result.warm_start.iterations < 10_000
    assert result.warm_start.config.early_stopping
    # The phases keep their own budget rather than the stage's criteria.
    assert result.final.iterations == 100
    assert not result.final.config.early_stopping


def test_the_model_ends_the_phases_and_sees_each_record():
    model = _phased(phases=3)
    result = _fit(model)
    assert len(result.phases) == 3
    assert model.phase_results == list(result.phases)
    assert result.stop_reason == "ran 3 phase(s)"
    assert result.series("phase") == [1, 2, 3]
    assert result.iterations == 300


def test_max_phases_caps_the_phases():
    result = _fit(_phased(phases=5, max_phases=2))
    assert len(result.phases) == 2
    assert result.stop_reason == "max_phases"


def test_training_finishes_once_however_many_runs():
    model = _phased(phases=3, warm_start_iterations=50)
    model.warm_start_criterion = _squared_error
    _fit(model)
    assert model.trainings_finished == 1


def test_the_trainer_reads_its_estimates_from_the_last_run():
    model = _phased(phases=2)
    loader = _loader()
    procedure = delphi.trainer.Trainer(model, model.hyperparameters)
    result = procedure.train_model(loader, loader)
    assert ch.equal(procedure.avg_params, result.final.avg_params)
    assert ch.equal(procedure.best_params, result.final.best_params)


class CurvedMean(PhasedMean):
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


def _curved() -> CurvedMean:
    model = CurvedMean(
        {
            "iterations": 50,
            "record_params_every": 1,
            "scheduler": "fisher",
            "warm_start_iterations": 50,
        }
    )
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
    first_main = model.curvature_calls.index("main")
    assert set(model.curvature_calls[first_main:]) == {"main"}
