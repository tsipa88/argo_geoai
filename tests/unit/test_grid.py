import pytest
from shapely.geometry import box

from argo_geoai.analysis.grid import cells_of, parcel_grid
from argo_geoai.parcels import Parcel

AREA = box(28.40, 47.265, 28.42, 47.275)
LAND = box(28.405, 47.265, 28.41, 47.275)


def test_cells_of_a_land_inside_the_area():
    grid = parcel_grid(Parcel(geometry=AREA, name="area"), 20)

    whole = cells_of(grid, LAND)
    inner = cells_of(grid, LAND, inner_edge_m=10)

    assert whole.sum() / grid.cells == pytest.approx(0.25, abs=0.03)
    assert 0 < inner.sum() < whole.sum()
    assert not (inner & ~whole).any()


def test_a_land_too_small_for_the_inner_edge_is_taken_whole():
    grid = parcel_grid(Parcel(geometry=AREA, name="area"), 20)
    tiny = box(28.4100, 47.2700, 28.4103, 47.2702)

    assert cells_of(grid, tiny, inner_edge_m=15).sum() == cells_of(grid, tiny).sum()
