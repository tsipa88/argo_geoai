import numpy as np
import pytest

from argo_geoai.sources.dem import slope_aspect

RESOLUTION = 30.0


def plane(dz_east: float, dz_north: float, size: int = 9) -> np.ndarray:
    rows, cols = np.mgrid[0:size, 0:size]
    north = (size - 1 - rows) * RESOLUTION
    east = cols * RESOLUTION
    heights: np.ndarray = dz_east * east + dz_north * north + 150.0
    return heights


def test_flat_ground_has_no_slope():
    slope, _ = slope_aspect(np.full((5, 5), 150.0), RESOLUTION)

    assert slope == pytest.approx(0)


def test_slope_rising_to_the_east_faces_west():
    gradient = np.tan(np.radians(5))
    slope, aspect = slope_aspect(plane(gradient, 0), RESOLUTION)

    assert slope == pytest.approx(5)
    assert aspect == pytest.approx(270)


@pytest.mark.parametrize(
    ("dz_east", "dz_north", "expected_aspect"),
    [(0, 0.05, 180), (0, -0.05, 0), (-0.05, 0, 90), (-0.05, -0.05, 45)],
)
def test_aspect_is_the_direction_the_slope_faces(dz_east, dz_north, expected_aspect):
    _, aspect = slope_aspect(plane(dz_east, dz_north), RESOLUTION)

    assert aspect == pytest.approx(expected_aspect)
