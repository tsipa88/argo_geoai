from datetime import UTC, datetime

import numpy as np
import pystac
import pytest

from argo_geoai.sources import sentinel2


def item(item_id: str, day: int, cloud: float, baseline: str | None = "05.12") -> pystac.Item:
    properties: dict[str, object] = {"eo:cloud_cover": cloud}
    if baseline is not None:
        properties["s2:processing_baseline"] = baseline
    return pystac.Item(
        id=item_id,
        geometry=None,
        bbox=None,
        datetime=datetime(2026, 6, day, 9, tzinfo=UTC),
        properties=properties,
    )


def test_reflectance_removes_the_offset_from_baseline_4():
    digital_numbers = np.array([1000, 3000])

    assert sentinel2.reflectance(digital_numbers, 5.12) == pytest.approx([0.0, 0.2])
    assert sentinel2.reflectance(digital_numbers, 3.0) == pytest.approx([0.1, 0.3])


def test_ndvi_of_bare_soil_and_dense_crop():
    red = np.array([0.1226, 0.03])
    nir = np.array([0.2068, 0.45])

    assert sentinel2.ndvi(red, nir) == pytest.approx([0.2556, 0.875], abs=1e-3)


def test_ndvi_is_nan_where_undefined():
    result = sentinel2.ndvi(np.array([0.0, np.nan]), np.array([0.0, 0.3]))

    assert np.isnan(result).all()


def test_ndvi_with_and_without_offset_differ_on_bare_soil():
    red, nir = np.array([2226]), np.array([3068])

    with_offset = sentinel2.ndvi(sentinel2.reflectance(red, 5.12), sentinel2.reflectance(nir, 5.12))
    without = sentinel2.ndvi(sentinel2.reflectance(red, 3.0), sentinel2.reflectance(nir, 3.0))

    assert with_offset[0] == pytest.approx(0.256, abs=1e-3)
    assert without[0] == pytest.approx(0.159, abs=1e-3)


@pytest.mark.parametrize(
    ("baseline", "expected"), [("05.12", 5.12), ("04.00", 4.0), (None, 0.0), ("n/a", 0.0)]
)
def test_processing_baseline(baseline, expected):
    assert sentinel2.processing_baseline(item("x", 1, 5, baseline)) == expected


def test_least_cloudy_scene_is_kept_per_day():
    items = [item("a", 2, 15), item("b", 2, 3), item("c", 1, 9), item("d", 5, 0)]

    assert [i.id for i in sentinel2.least_cloudy_per_day(items)] == ["c", "b", "d"]
