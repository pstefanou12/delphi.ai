# Author: pstefanou12@
"""Tests for the truncation set library."""

import functools
import math

import pytest
import torch as ch
from torch import distributions

from delphi.truncated.distributions.tests import truncation_sets

ch.set_printoptions(precision=4, sci_mode=False)

_REFERENCE = distributions.MultivariateNormal(ch.zeros(2), 2 * ch.eye(2))


def test_annulus_excludes_the_mode_and_the_tail():
    phi = truncation_sets.annulus(1.0, 2.0)
    points = ch.Tensor([[0.0, 0.0], [1.5, 0.0], [0.0, -1.5], [3.0, 0.0]])
    assert phi(points).tolist() == [False, True, True, False]


def test_crescent_removes_its_wedge():
    phi = truncation_sets.crescent(1.0, 2.0, 0.0, math.pi / 2)
    # The first point sits in the wedge; its mirror image does not.
    points = ch.Tensor([[1.0, 1.0], [-1.0, -1.0]])
    assert phi(points).tolist() == [False, True]


def test_triangle_is_orientation_independent():
    vertices = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    points = ch.Tensor([[0.5, 0.5], [1.5, 1.5]])
    clockwise = truncation_sets.triangle(vertices)
    anticlockwise = truncation_sets.triangle(vertices[::-1])
    assert clockwise(points).tolist() == [True, False]
    assert anticlockwise(points).tolist() == [True, False]


def test_triangle_rejects_degenerate_vertices():
    with pytest.raises(ValueError, match="collinear"):
        truncation_sets.triangle([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="three vertices"):
        truncation_sets.triangle([[0.0, 0.0], [1.0, 1.0]])


def test_checkerboard_alternates_between_neighbouring_cells():
    phi = truncation_sets.checkerboard(1.0)
    points = ch.Tensor([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5]])
    assert phi(points).tolist() == [True, False, True]
    with pytest.raises(ValueError, match="cell_size"):
        truncation_sets.checkerboard(0.0)


def test_spiral_band_follows_its_arms():
    phi = truncation_sets.spiral_band(0.2, 0.5, turns=2)
    # On the positive x axis the arms sit at the offset and one turn out.
    on_arm = ch.Tensor([[0.5, 0.0], [0.5 + 2.0 * math.pi * 0.5, 0.0]])
    between = ch.Tensor([[0.5 + math.pi * 0.5, 0.0]])
    assert phi(on_arm).tolist() == [True, True]
    assert phi(between).tolist() == [False]
    with pytest.raises(ValueError, match="positive"):
        truncation_sets.spiral_band(0.0, 0.5)
    with pytest.raises(ValueError, match="turns"):
        truncation_sets.spiral_band(0.2, 0.5, turns=0)


def test_random_sets_depend_only_on_the_seed():
    points = _REFERENCE.sample([256])
    factories = [
        functools.partial(truncation_sets.random_half_space, dims=2, offset=0.5),
        functools.partial(
            truncation_sets.random_polytope, dims=2, num_faces=3, offset=1.0
        ),
        functools.partial(truncation_sets.random_ellipsoid, dims=2, level=2.0),
        functools.partial(
            truncation_sets.random_disc_union, dims=2, num_discs=4, radius=1.0
        ),
    ]
    for factory in factories:
        repeated = factory(seed=7)(points)
        assert ch.equal(factory(seed=7)(points), repeated)
        assert not ch.equal(factory(seed=8)(points), repeated)


def test_random_sets_reject_empty_shapes():
    with pytest.raises(ValueError, match="num_faces"):
        truncation_sets.random_polytope(0, 2, 0, 1.0)
    with pytest.raises(ValueError, match="num_discs"):
        truncation_sets.random_disc_union(0, 2, 0, 1.0)


def test_random_draws_leave_the_global_generator_alone():
    ch.manual_seed(0)
    before = ch.randn(4)
    ch.manual_seed(0)
    truncation_sets.calibrate(
        functools.partial(truncation_sets.random_half_space, 0, 2),
        _REFERENCE,
        bounds=(-10.0, 10.0),
        num_samples=2000,
    )
    assert ch.equal(ch.randn(4), before)


def test_calibrate_lands_in_the_requested_band():
    band = (0.25, 0.35)
    for seed in range(4):
        result = truncation_sets.calibrate(
            functools.partial(truncation_sets.random_disc_union, seed, 2, 6),
            _REFERENCE,
            target_alpha=band,
            bounds=(1e-4, 1e3),
            num_samples=4000,
        )
        assert band[0] <= result.alpha <= band[1]
        fresh = _REFERENCE.sample([20000])
        measured = float(result.phi(fresh).sum()) / fresh.size(0)
        assert abs(measured - result.alpha) < 0.05


def test_calibrate_reports_an_unreachable_band():
    with pytest.raises(ValueError, match="no scale"):
        truncation_sets.calibrate(
            functools.partial(truncation_sets.random_half_space, 0, 2),
            _REFERENCE,
            target_alpha=(0.3, 0.31),
            bounds=(5.0, 10.0),
            num_samples=2000,
        )
    with pytest.raises(ValueError, match="band inside"):
        truncation_sets.calibrate(
            functools.partial(truncation_sets.random_half_space, 0, 2),
            _REFERENCE,
            target_alpha=(0.4, 0.2),
        )
