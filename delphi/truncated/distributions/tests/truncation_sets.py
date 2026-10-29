# Author: pstefanou12@
"""Truncation sets with geometry, for testing truncated estimators.

Two kinds of set live here. The named ones -- annulus, crescent, triangle,
checkerboard, spiral_band -- each break one property a threshold on a single
coordinate quietly supplies: convexity, axis alignment, connectedness, or a
support wide enough to be cheap to sample. The random ones are drawn from a
seed, so they cannot have been chosen to suit the estimator, and a seed that
breaks a fit is a reproducible counterexample rather than an anecdote.

Every set reads the first two coordinates of a batch unless it says
otherwise, which leaves the remaining coordinates free and so keeps a
two-dimensional shape usable as a higher-dimensional test. Returned masks are
shaped ``(n,)``; ``generate_truncated_distribution_dataset`` reads them
through ``.nonzero()``.

Survival mass under a random set swings with the seed -- six unit discs over
seeds 0 to 5 measured 0.4864 to 0.6006 on a centered ``N(0, 2I)`` -- and
rejection cost is one draw per alpha accepted samples, so a calibration search
bisects one scale parameter until the mass lands in a requested band.
"""

from collections.abc import Callable, Sequence
import math
from typing import NamedTuple

import torch as ch
from torch.distributions import Distribution

from delphi.truncated.distributions.tests import helpers


def _radius_and_angle(z: ch.Tensor) -> tuple[ch.Tensor, ch.Tensor]:
    """Return the polar coordinates of the first two columns of z.

    Args:
        z: Batch of shape (n, d) with d >= 2.

    Returns:
        Radius and angle, the angle wrapped into [0, 2*pi).
    """
    radius = z[:, :2].norm(dim=-1)
    angle = ch.atan2(z[:, 1], z[:, 0]) % (2.0 * math.pi)
    return radius, angle


def annulus(r_inner: float, r_outer: float) -> helpers.Oracle:
    """Return membership in the ring between two radii.

    Excludes the mode of any centered distribution, so the truncated mean is
    unbiased while the truncated covariance is badly inflated, and the set is
    not convex.

    Args:
        r_inner: Inner radius, excluded.
        r_outer: Outer radius, included.

    Returns:
        Oracle over the first two coordinates of a batch.
    """

    def phi(z: ch.Tensor) -> ch.Tensor:
        radius = z[:, :2].norm(dim=-1)
        return (radius > r_inner) & (radius <= r_outer)

    return phi


def crescent(
    r_inner: float, r_outer: float, wedge_start: float, wedge_end: float
) -> helpers.Oracle:
    """Return membership in an annulus with an angular wedge removed.

    Cutting one wedge out of a ring leaves a set that is neither convex nor
    symmetric, so it biases the mean in a known direction.

    Args:
        r_inner: Inner radius of the ring, excluded.
        r_outer: Outer radius of the ring, included.
        wedge_start: Angle in radians where the removed wedge begins.
        wedge_end: Angle in radians where the removed wedge ends; must exceed
            wedge_start.

    Returns:
        Oracle over the first two coordinates of a batch.
    """

    def phi(z: ch.Tensor) -> ch.Tensor:
        radius, angle = _radius_and_angle(z)
        ring = (radius > r_inner) & (radius <= r_outer)
        wedge = (angle >= wedge_start) & (angle < wedge_end)
        return ring & ~wedge

    return phi


def triangle(vertices: Sequence[Sequence[float]]) -> helpers.Oracle:
    """Return membership in the triangle spanned by three vertices.

    Convex, and aligned with no axis unless the caller asks for it: the three
    half-planes mix both coordinates, so a per-coordinate account of the set
    cannot describe it.

    Args:
        vertices: Three (x, y) pairs, in either orientation.

    Returns:
        Oracle over the first two coordinates of a batch.

    Raises:
        ValueError: If vertices does not hold exactly three points, or if the
            three points are collinear.
    """
    if len(vertices) != 3:
        raise ValueError(f"a triangle needs three vertices; got {len(vertices)}.")
    corners = ch.tensor(vertices, dtype=ch.float32)
    edges = corners.roll(-1, dims=0) - corners
    # Twice the signed area; zero exactly when the corners are collinear.
    area = float(edges[0, 0] * edges[1, 1] - edges[0, 1] * edges[1, 0])
    if area == 0.0:
        raise ValueError(f"the vertices {vertices} are collinear.")
    orientation = math.copysign(1.0, area)

    def phi(z: ch.Tensor) -> ch.Tensor:
        inside = ch.ones(z.size(0), dtype=ch.bool)
        for corner, edge in zip(corners, edges):
            offset = z[:, :2] - corner
            cross = edge[0] * offset[:, 1] - edge[1] * offset[:, 0]
            inside &= orientation * cross >= 0.0
        return inside

    return phi


def sierpinski(level: int) -> helpers.Oracle:
    """Return membership in a Sierpinski triangle on the unit base.

    Non-convex, with holes at every scale, inside the triangle with vertices
    (0, 0), (1, 0) and (1/2, sqrt(3)/2). A point belongs when, at each of
    ``level`` halvings, it falls in one of the three corner triangles rather
    than the removed middle one, which in barycentric coordinates means one
    coordinate is at least 1/2.

    Args:
        level: Number of halvings; level 0 is the whole triangle.

    Returns:
        Oracle over the first two coordinates of a batch.
    """
    height = math.sqrt(3.0) / 2.0

    def phi(z: ch.Tensor) -> ch.Tensor:
        # Barycentric weights of the corners (1/2, h), (1, 0) and (0, 0).
        top = z[:, 1] / height
        right = z[:, 0] - top / 2.0
        weights = ch.stack([top, right, 1.0 - top - right], dim=1)
        inside = (weights >= 0.0).all(dim=1)
        for _ in range(level):
            corner = weights.argmax(dim=1)
            inside &= weights.gather(1, corner[:, None]).squeeze(1) >= 0.5
            # Zoom into that corner: its weight maps [1/2, 1] onto [0, 1].
            weights = 2.0 * weights
            weights.scatter_(
                1, corner[:, None], weights.gather(1, corner[:, None]) - 1.0
            )
        return inside

    return phi


def checkerboard(cell_size: float) -> helpers.Oracle:
    """Return membership in the even cells of a square grid.

    Splits the support into many components with no interior path between
    them, which is where a connectedness assumption would show up. No
    distribution test uses it: with cells narrower than a standard deviation
    the parity of one coordinate is near-independent of the other, which
    leaves the moments almost untouched -- on a centered ``N(0, 2I)`` the
    naive mean and covariance measure 0.0002 and 0.0026 against a 0.2
    tolerance, so nothing can tell a fit from doing nothing.

    Args:
        cell_size: Side length of one cell.

    Returns:
        Oracle over the first two coordinates of a batch.

    Raises:
        ValueError: If cell_size is not positive.
    """
    if cell_size <= 0.0:
        raise ValueError(f"cell_size must be positive; got {cell_size}.")

    def phi(z: ch.Tensor) -> ch.Tensor:
        cells = ch.floor(z[:, :2] / cell_size)
        return cells.sum(dim=-1) % 2.0 == 0.0

    return phi


def spiral_band(
    width: float, growth: float, turns: int = 3, offset: float = 0.5
) -> helpers.Oracle:
    """Return membership in a band following an Archimedean spiral.

    The thinnest and most curved set here: ``r = offset + growth * t``
    thickened by ``width``, which leaves a support of a few percent survival
    mass and no straight piece anywhere. No distribution test uses it either:
    on a centered ``N(0, 2I)`` the arms sample the angle evenly, so at
    ``(0.15, 0.5)`` over four turns the mass is 0.0456 -- 22 draws per
    accepted sample -- while the naive mean and covariance only reach 0.0299
    and 0.0349.

    Args:
        width: Full thickness of the band in radius.
        growth: Radius gained per radian along the spiral.
        turns: Number of turns to include.
        offset: Radius at the start of the spiral.

    Returns:
        Oracle over the first two coordinates of a batch.

    Raises:
        ValueError: If width or growth is not positive, or turns is below one.
    """
    if width <= 0.0 or growth <= 0.0:
        raise ValueError(f"width and growth must be positive; got {width}, {growth}.")
    if turns < 1:
        raise ValueError(f"turns must be at least one; got {turns}.")

    def phi(z: ch.Tensor) -> ch.Tensor:
        radius, angle = _radius_and_angle(z)
        inside = ch.zeros(z.size(0), dtype=ch.bool)
        for turn in range(turns):
            arm = offset + growth * (angle + 2.0 * math.pi * turn)
            inside |= (radius - arm).abs() <= width / 2.0
        return inside

    return phi


def sum_threshold(threshold: float, below: bool = False) -> helpers.Oracle:
    """Return membership in a lower bound on the coordinate sum.

    One scalar condition, so rejection sampling stays cheap, but it biases
    every coordinate rather than one.

    Args:
        threshold: Lower bound on the sum. Survival mass decreases as this
            grows, so it is the knob to search over.
        below: Make threshold an upper bound instead, keeping the lower tail,
            for families whose sufficient statistic flattens in the upper one.

    Returns:
        Oracle over a batch.
    """

    def phi(z: ch.Tensor) -> ch.Tensor:
        total = z.sum(dim=-1)
        return total < threshold if below else total > threshold

    return phi


def norm_threshold(threshold: float) -> helpers.Oracle:
    """Return membership outside a ball centered at the origin.

    Couples every coordinate, and excludes the mode as a one-sided bound
    on a sum does not.

    Args:
        threshold: Radius of the excluded ball. Survival mass decreases as
            this grows, so it is the knob to search over.

    Returns:
        Oracle over a batch.
    """

    def phi(z: ch.Tensor) -> ch.Tensor:
        return z.norm(dim=-1) > threshold

    return phi


def coordinate_threshold(
    index: int, threshold: float, below: bool = False
) -> helpers.Oracle:
    """Return membership in a one-sided bound on a single coordinate.

    The simplest truncation there is, kept here so a test can state the
    survival mass it wants and have the threshold searched for, rather than
    hard-coding a constant whose mass nobody measured.

    Args:
        index: Coordinate the bound applies to.
        threshold: Lower bound. Survival mass decreases as this grows, so
            it is the knob to search over.
        below: As ``sum_threshold``.

    Returns:
        Oracle over a batch.
    """

    def phi(z: ch.Tensor) -> ch.Tensor:
        return z[:, index] < threshold if below else z[:, index] > threshold

    return phi


def _unit_directions(seed: int, dims: int, count: int) -> ch.Tensor:
    """Return count directions drawn uniformly on the unit sphere.

    Args:
        seed: Seed for the generator backing the draw.
        dims: Dimension of each direction.
        count: Number of directions.

    Returns:
        Tensor of shape (count, dims) with unit rows.
    """
    generator = ch.Generator().manual_seed(seed)
    directions = ch.randn(count, dims, generator=generator)
    return directions / directions.norm(dim=-1, keepdim=True)


def random_half_space(seed: int, dims: int, offset: float) -> helpers.Oracle:
    """Return membership in a half-space with a randomly rotated normal.

    Args:
        seed: Seed fixing the normal direction.
        dims: Number of dimensions.
        offset: Signed distance of the boundary from the origin. Survival mass
            decreases as this grows, so it is the knob to search over.

    Returns:
        Oracle over a batch.
    """
    normal = _unit_directions(seed, dims, 1)[0]

    def phi(z: ch.Tensor) -> ch.Tensor:
        return z @ normal >= offset

    return phi


def random_polytope(
    seed: int, dims: int, num_faces: int, offset: float
) -> helpers.Oracle:
    """Return membership in an intersection of randomly oriented half-spaces.

    Convex like a half-space, but with num_faces unrelated normals no
    coordinate is left untouched and the boundary has corners.

    Args:
        seed: Seed fixing the face normals.
        dims: Number of dimensions.
        num_faces: Number of half-spaces to intersect.
        offset: Distance of every face from the origin. Survival mass grows
            with it, so it is the knob to search over.

    Returns:
        Oracle over a batch.

    Raises:
        ValueError: If num_faces is below one.
    """
    if num_faces < 1:
        raise ValueError(f"num_faces must be at least one; got {num_faces}.")
    normals = _unit_directions(seed, dims, num_faces)

    def phi(z: ch.Tensor) -> ch.Tensor:
        return (z @ normals.T <= offset).all(dim=-1)

    return phi


def random_ellipsoid(seed: int, dims: int, level: float) -> helpers.Oracle:
    """Return membership in an ellipsoid with a random shape and center.

    The quadratic form is ``A A^T + I`` for a random ``A``, which is positive
    definite by construction, so the set is a bounded ellipsoid whose axes
    agree with no coordinate direction.

    Args:
        seed: Seed fixing the shape matrix and the center.
        dims: Number of dimensions.
        level: Right-hand side of the quadratic form. Survival mass grows with
            it, so it is the knob to search over.

    Returns:
        Oracle over a batch.
    """
    generator = ch.Generator().manual_seed(seed)
    factor = ch.randn(dims, dims, generator=generator) / math.sqrt(dims)
    shape = factor @ factor.T + ch.eye(dims)
    center = ch.randn(dims, generator=generator)

    def phi(z: ch.Tensor) -> ch.Tensor:
        offset = z - center
        return ((offset @ shape) * offset).sum(dim=-1) <= level

    return phi


def random_disc_union(
    seed: int, dims: int, num_discs: int, radius: float
) -> helpers.Oracle:
    """Return membership in a union of balls with random centers.

    Disconnected whenever the balls are small enough to stay apart, and the
    number of components is itself a function of the seed.

    Args:
        seed: Seed fixing the centers.
        dims: Number of dimensions.
        num_discs: Number of balls in the union.
        radius: Shared radius. Survival mass grows with it, so it is the
            knob to search over.

    Returns:
        Oracle over a batch.

    Raises:
        ValueError: If num_discs is below one.
    """
    if num_discs < 1:
        raise ValueError(f"num_discs must be at least one; got {num_discs}.")
    generator = ch.Generator().manual_seed(seed)
    centers = ch.randn(num_discs, dims, generator=generator)

    def phi(z: ch.Tensor) -> ch.Tensor:
        distances = (z[:, None, :] - centers[None, :, :]).norm(dim=-1)
        return (distances <= radius).any(dim=-1)

    return phi


class CalibratedSet(NamedTuple):
    """A truncation set whose survival mass has been measured.

    Attributes:
        phi: The calibrated oracle.
        alpha: Survival mass under the reference distribution, measured on
            the draw the bisection used.
        scale: Value the scale parameter was calibrated to.
    """

    phi: helpers.Oracle
    alpha: float
    scale: float


def _reference_draw(reference: Distribution, num_samples: int, seed: int) -> ch.Tensor:
    """Return a draw from reference that depends only on seed.

    ``Distribution.sample`` takes no generator, so the global RNG is seeded
    inside ``fork_rng``, which restores whatever state the caller had.

    Args:
        reference: Distribution to draw from.
        num_samples: Samples to draw.
        seed: Seed fixing the draw.

    Returns:
        Tensor of shape (num_samples, dims).
    """
    with ch.random.fork_rng():
        ch.manual_seed(seed)
        return reference.sample([num_samples])


def calibrate(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    predicate_factory: Callable[[float], helpers.Oracle],
    reference: Distribution,
    target_alpha: tuple[float, float] = (0.2, 0.4),
    bounds: tuple[float, float] = (1e-3, 1e2),
    num_samples: int = 20000,
    seed: int = 0,
    steps: int = 60,
) -> CalibratedSet:
    """Bisect a set's scale parameter until its survival mass hits a band.

    Every candidate is scored on one fixed draw, which makes the measured
    mass monotone in the scale for a nested family of sets and so makes the
    bisection exact rather than approximate.

    Args:
        predicate_factory: Maps a scale to an oracle. Survival mass must be
            monotone in the scale; the direction is read off the bounds.
        reference: Distribution the mass is measured against, normally the
            untruncated target of the test.
        target_alpha: Inclusive band the measured mass must land in.
        bounds: Scales to search between, in either order.
        num_samples: Draw size backing each measurement.
        seed: Seed fixing the draw.
        steps: Bisection steps before giving up.

    Returns:
        The calibrated oracle, its measured mass, and the scale it came from.

    Raises:
        ValueError: If target_alpha is not a band inside (0, 1), or if no
            scale within bounds puts the mass in the band.
    """
    low_alpha, high_alpha = target_alpha
    if not 0.0 < low_alpha <= high_alpha < 1.0:
        raise ValueError(
            f"target_alpha must be a band inside (0, 1); got {target_alpha}."
        )
    samples = _reference_draw(reference, num_samples, seed)

    def measure(scale: float) -> float:
        return float(predicate_factory(scale)(samples).sum()) / num_samples

    # Orient the search so the mass increases from low to high.
    low, high = bounds
    if measure(low) > measure(high):
        low, high = high, low

    midpoint = (low_alpha + high_alpha) / 2.0
    for _ in range(steps):
        scale = (low + high) / 2.0
        alpha = measure(scale)
        if low_alpha <= alpha <= high_alpha:
            return CalibratedSet(predicate_factory(scale), alpha, scale)
        if alpha < midpoint:
            low = scale
        else:
            high = scale
    raise ValueError(
        f"no scale in {bounds} gives survival mass in {target_alpha}; the "
        f"bounds measure {measure(bounds[0]):.4f} and {measure(bounds[1]):.4f}."
    )
