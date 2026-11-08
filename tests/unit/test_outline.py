import math

import numpy as np
import pytest
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from argo_geoai.analysis.outline import straighten

CELL = 10.0


def staircase_strip(angle_deg: float, length_m: float = 400, width_m: float = 100) -> Polygon:
    slope = math.tan(math.radians(angle_deg))
    columns = int(length_m / CELL)
    lower, upper = [], []
    for i in range(columns):
        x0, x1 = i * CELL, (i + 1) * CELL
        bottom = math.floor((x0 + CELL / 2) * slope / CELL) * CELL
        lower += [(x0, bottom), (x1, bottom)]
        upper += [(x0, bottom + width_m), (x1, bottom + width_m)]
    traced = Polygon(lower + upper[::-1]).simplify(0)
    assert isinstance(traced, Polygon)
    return traced


def ideal_strip(angle_deg: float, length_m: float = 400, width_m: float = 100) -> Polygon:
    slope = math.tan(math.radians(angle_deg))
    return Polygon(
        [(0, 0), (length_m, length_m * slope), (length_m, length_m * slope + width_m), (0, width_m)]
    )


@pytest.mark.parametrize("angle", [20, 35, 45])
def test_diagonal_staircase_becomes_a_straight_edge(angle):
    steps = staircase_strip(angle)

    straight = straighten(steps, 7)

    assert isinstance(straight, Polygon)
    assert len(steps.exterior.coords) > 60
    assert len(straight.exterior.coords) <= 16
    assert straight.hausdorff_distance(steps) < 1.5 * CELL
    assert straight.area == pytest.approx(steps.area, rel=0.02)


def test_axis_aligned_rectangle_keeps_its_shape():
    rectangle = box(0, 0, 300, 200)

    straight = straighten(rectangle, 7)

    assert straight.equals(rectangle)


def test_zero_tolerance_returns_the_source():
    steps = staircase_strip(30)
    assert straighten(steps, 0) is steps


def test_holes_and_parts_are_kept():
    field = Polygon(box(0, 0, 300, 300).exterior, [box(100, 100, 200, 200).exterior.coords])
    parts = MultiPolygon([field, box(400, 0, 500, 100)])

    straight = straighten(parts, 7)

    assert isinstance(straight, MultiPolygon)
    assert len(straight.geoms) == 2
    assert len(straight.geoms[0].interiors) == 1
    assert straight.is_valid


def traced(shape: Polygon) -> Polygon:
    minx, miny, maxx, maxy = shape.bounds
    cells = [
        box(x, y, x + CELL, y + CELL)
        for x in np.arange(math.floor(minx / CELL) * CELL, maxx, CELL)
        for y in np.arange(math.floor(miny / CELL) * CELL, maxy, CELL)
        if shape.contains(Point(x + CELL / 2, y + CELL / 2))
    ]
    merged = unary_union(cells).simplify(0)
    assert isinstance(merged, Polygon)
    return merged


def corner_angles(polygon: Polygon) -> list[float]:
    points = np.asarray(polygon.exterior.coords)[:-1]
    before, after = np.roll(points, 1, axis=0) - points, np.roll(points, -1, axis=0) - points
    cosine = (before * after).sum(axis=1) / np.hypot(*before.T) / np.hypot(*after.T)
    return list(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


@pytest.mark.parametrize("angle", [15, 30, 45])
def test_turned_rectangle_keeps_its_corners(angle):
    true = rotate(box(0, 0, 400, 250), angle, origin=(0, 0))

    straight = straighten(traced(true), 7)

    assert isinstance(straight, Polygon)
    assert straight.hausdorff_distance(true) < 1.1 * CELL
    for corner in np.asarray(true.exterior.coords)[:-1]:
        assert Point(corner).distance(straight) < 1.1 * CELL


@pytest.mark.parametrize("angle", [30, 45])
def test_corners_are_sharp_not_cut(angle):
    straight = straighten(traced(rotate(box(0, 0, 400, 250), angle, origin=(0, 0))), 7)

    assert isinstance(straight, Polygon)
    corners = [turn for turn in corner_angles(straight) if turn < 150]
    assert len(corners) == 4
    assert corners == pytest.approx([90] * 4, abs=20)


def test_round_edge_stays_round():
    circle = Point(0, 0).buffer(150, quad_segs=32)

    straight = straighten(traced(circle), 7)

    assert isinstance(straight, Polygon)
    assert straight.hausdorff_distance(circle) < 1.5 * CELL
    assert len(straight.exterior.coords) > 12
    assert min(corner_angles(straight)) > 120
