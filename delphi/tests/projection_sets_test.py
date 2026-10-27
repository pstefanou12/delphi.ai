# Author: pstefanou12@
"""Tests for the projection sets."""

import math

import torch as ch
import torch.nn.functional

import delphi.projection_sets


def _natural_parameter_set() -> delphi.projection_sets.NaturalParameterSet:
    residual_var = ch.tensor([1.0, 4.0])
    anchor = ch.cat([ch.zeros(2, 2), (1.0 / residual_var)[None]])
    return delphi.projection_sets.NaturalParameterSet(
        residual_var, 2.0, anchor, bound=1.5
    )


def test_natural_parameter_set_bounds_the_precision_by_the_survival_mass():
    projection = _natural_parameter_set()
    lower, upper = projection.precision_bounds()
    # alpha = 1 / radius = 1 / 2 in Definition 3.1.
    residual_var = ch.tensor([1.0, 4.0])
    assert ch.allclose(lower, 0.25 / (96.0 * residual_var))
    assert ch.allclose(upper, 8.0 * (5.0 + 2.0 * math.log(2.0)) / residual_var)


def test_natural_parameter_set_projects_to_the_nearest_point():
    projection = _natural_parameter_set()
    lower, upper = projection.precision_bounds()
    ch.manual_seed(0)
    # Feasible points, to check no candidate beats the projection.
    precision = lower + (upper - lower) * ch.rand(20000, 2)
    direction = torch.nn.functional.normalize(ch.randn(20000, 2, 2), dim=1)
    radius = 1.5 * precision * ch.rand(20000, 2)
    feasible = ch.cat([direction * radius[:, None], precision[:, None]], dim=1)
    for scale in [0.1, 3.0, 100.0]:
        point = scale * ch.randn(3, 2)
        projected = projection.project(point)
        assert projection.contains(projected + 1e-6 * (projection.anchor - projected))
        nearest = (feasible - point).norm(dim=1).min(0).values
        assert ((projected - point).norm(dim=0) <= nearest + 1e-5).all()
        assert ch.allclose(projection.project(projected), projected)
