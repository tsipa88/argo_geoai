import numpy as np
import pytest

from argo_geoai.analysis.hydrology import fill_depressions, flow_accumulation, wetness_index

CELL = 30.0


def valley(size: int = 21, side_slope: float = 0.05, down_slope: float = 0.01) -> np.ndarray:
    rows, cols = np.mgrid[0:size, 0:size]
    centre = size // 2
    heights: np.ndarray = (
        100.0 + side_slope * CELL * np.abs(cols - centre) + down_slope * CELL * (size - 1 - rows)
    )
    return heights


def test_pit_is_filled_to_its_spill_level():
    heights = np.full((7, 7), 10.0)
    heights[3, 3] = 5.0

    filled = fill_depressions(heights)

    assert filled[3, 3] >= 10.0
    assert filled[3, 3] == pytest.approx(10.0, abs=0.01)


def test_filled_surface_drains_everywhere():
    heights = valley()
    heights[10, 10] -= 3

    filled = fill_depressions(heights)

    assert (filled >= heights).all()
    assert filled[10, 10] >= filled[11, 10]


def test_accumulation_grows_downstream_and_sums_the_catchment():
    heights = valley()
    area, edge = flow_accumulation(fill_depressions(heights), CELL)

    centre = heights.shape[1] // 2
    bottom = area[:, centre]
    assert (np.diff(bottom) > 0).all()
    assert area.max() == pytest.approx(heights.size * CELL**2, rel=0.05)
    assert area[0, 0] == pytest.approx(CELL**2)
    assert edge.shape == heights.shape


def test_wetness_is_highest_in_the_valley_bottom():
    heights = valley()
    area, _ = flow_accumulation(fill_depressions(heights), CELL)
    slope = np.full(heights.shape, 3.0)

    twi = wetness_index(area, slope, CELL)

    centre = heights.shape[1] // 2
    assert twi[15, centre] > twi[15, 0] + 2
    assert np.isfinite(twi).all()


def test_flat_ground_does_not_divide_by_zero():
    twi = wetness_index(np.full((3, 3), CELL**2), np.zeros((3, 3)), CELL)

    assert np.isfinite(twi).all()


def test_flow_from_outside_the_window_is_flagged():
    heights = valley()
    _, edge = flow_accumulation(fill_depressions(heights), CELL)

    centre = heights.shape[1] // 2
    assert edge[20, centre] == pytest.approx(80 / 441, abs=0.05)
    assert edge[10, 0] == pytest.approx(1.0)


def test_no_edge_water_on_a_hill_inside_the_window():
    rows, cols = np.mgrid[0:21, 0:21]
    hill = 150.0 - 2.0 * np.hypot(rows - 10, cols - 10)

    _, edge = flow_accumulation(fill_depressions(hill), CELL)

    assert edge[10, 10] == 0
    assert edge[8:13, 8:13].max() == 0
