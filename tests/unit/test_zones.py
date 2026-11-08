from datetime import date

import numpy as np
import pytest
import xarray as xr

from argo_geoai.analysis import zones
from argo_geoai.tools.vegetation import (
    ZonePoint,
    describe_zones,
    development,
    plural_uk,
    season_timing,
    zone_findings,
)

DATES = [date(2026, 5, 6), date(2026, 5, 26), date(2026, 6, 30), date(2026, 8, 16), date(2026, 9, 28)]
WINTER = [0.80, 0.82, 0.60, 0.20, 0.18]
SPRING = [0.22, 0.25, 0.85, 0.45, 0.22]


def field(
    curves: list[list[float]], strip_shares: list[float], size: int = 40, noise: float = 0.03, seed: int = 1
):
    rng = np.random.default_rng(seed)
    bounds = np.cumsum([0, *strip_shares]) * size
    strip = np.searchsorted(bounds, np.arange(size), side="right") - 1
    rasters = {}
    for column, day in enumerate(DATES):
        values = np.array([curves[s][column] for s in strip])[None, :].repeat(size, axis=0)
        values = values + rng.normal(0, noise, values.shape)
        data = xr.DataArray(
            values.astype("float32"),
            coords={"y": np.arange(size)[::-1] * 10.0, "x": np.arange(size) * 10.0},
            dims=("y", "x"),
        ).rio.write_crs("EPSG:32636")
        rasters[day] = data
    return rasters


def test_two_crops_in_strips_give_two_zones():
    found, silhouette, raster = describe_zones(
        field([WINTER, SPRING, WINTER, SPRING], [0.25, 0.25, 0.25, 0.25])
    )

    assert len(found) == 2
    assert [z.share for z in found] == pytest.approx([0.5, 0.5], abs=0.03)
    assert silhouette is not None
    assert silhouette > 0.5
    assert raster is not None
    assert set(np.unique(raster[np.isfinite(raster)])) == {1.0, 2.0}


def test_zones_describe_their_own_season():
    found, _, _ = describe_zones(field([WINTER, SPRING], [0.7, 0.3]))
    main, second = found

    assert main.share == pytest.approx(0.7, abs=0.03)
    assert main.timing == "early"
    assert main.timing_label.uk == "рання"
    assert main.peak.date == date(2026, 5, 26)
    assert second.timing == "late"
    assert second.peak.date == date(2026, 6, 30)
    assert second.peak.ndvi == pytest.approx(0.85, abs=0.02)


def test_uniform_field_is_one_zone():
    found, silhouette, _ = describe_zones(field([SPRING], [1.0], noise=0.05))

    assert len(found) == 1
    assert silhouette is None


def test_slightly_different_strips_are_not_split():
    similar = [v + 0.1 for v in SPRING]

    found, _, _ = describe_zones(field([SPRING, similar], [0.5, 0.5], noise=0.01))

    assert len(found) == 1


def test_tiny_patch_is_not_a_zone():
    found, _, _ = describe_zones(field([SPRING, WINTER], [0.97, 0.03], noise=0.01))

    assert len(found) == 1


def test_cloudy_pixels_still_get_a_zone():
    rasters = field([WINTER, SPRING], [0.5, 0.5])
    rasters[DATES[1]].values[:10, :] = np.nan

    ndvi_stack = zones.stack(rasters)
    split = zones.find_zones(ndvi_stack.values)

    assert split.count == 2
    assert len(split.labels) == 40 * 40


def test_zone_findings():
    found, _, _ = describe_zones(field([WINTER, SPRING], [0.6, 0.4]))

    findings = zone_findings(found)

    assert [f.code for f in findings] == ["zones", "zone_1", "zone_2"]
    assert "2 зони" in findings[0].text
    assert "озимих" in findings[1].text
    assert "соняшник" in findings[2].text


def test_no_zone_findings_for_uniform_field():
    found, _, _ = describe_zones(field([SPRING], [1.0]))

    assert zone_findings(found) == []


def point(day: date, ndvi: float) -> ZonePoint:

    return ZonePoint(date=day, ndvi=ndvi)


@pytest.mark.parametrize(
    ("curve", "expected"),
    [
        ([(5, 6, 0.58), (5, 26, 0.74), (6, 30, 0.5), (9, 28, 0.24)], "early"),
        ([(5, 6, 0.24), (5, 26, 0.3), (6, 30, 0.78), (9, 28, 0.23)], "late"),
        ([(5, 6, 0.5), (5, 26, 0.55), (6, 30, 0.7)], "other"),
        ([(7, 1, 0.3), (8, 1, 0.8)], "other"),
    ],
)
def test_season_timing(curve, expected):

    assert season_timing([point(date(2026, m, d), v) for m, d, v in curve]) == expected


def patchy_field(size: int = 40, seed: int = 2):
    rng = np.random.default_rng(seed)
    weak = np.zeros((size, size), dtype=bool)
    for row, col in [(5, 5), (25, 8), (12, 28), (30, 30)]:
        weak[row : row + 8, col : col + 8] = True
    rasters = {}
    for column, day in enumerate(DATES):
        strong_value, weak_value = SPRING[column], 0.2 + (SPRING[column] - 0.2) * 0.4
        values = np.where(weak, weak_value, strong_value) + rng.normal(0, 0.02, (size, size))
        rasters[day] = xr.DataArray(
            values.astype("float32"),
            coords={"y": np.arange(size)[::-1] * 10.0, "x": np.arange(size) * 10.0},
            dims=("y", "x"),
        ).rio.write_crs("EPSG:32636")
    return rasters


def test_weak_spots_of_one_crop_are_a_weaker_area_not_another_crop():
    found, _, _ = describe_zones(patchy_field())
    main, weak = found

    assert main.role == "main"
    assert main.development == "strong"
    assert weak.role == "weaker_area"
    assert weak.role_label.uk == "слабша ділянка"
    assert weak.development == "weak"
    assert weak.correlation_with_main is not None
    assert weak.correlation_with_main > 0.85
    assert weak.patches == 4
    assert weak.largest_patch_share < 0.7


def test_weaker_area_texts():
    findings = zone_findings(describe_zones(patchy_field())[0])

    assert "однією культурою" in findings[0].text
    assert findings[2].level == "notice"
    assert "слабша" in findings[2].text
    assert "4 окремі ділянки" in findings[2].text
    assert "проблемних місць" in findings[2].text
    assert "соняшник" not in findings[2].text


def test_strips_of_two_crops_are_separate_crops():
    found, _, _ = describe_zones(field([WINTER, SPRING, WINTER, SPRING], [0.25, 0.25, 0.25, 0.25]))

    assert [z.role for z in found] == ["main", "separate_crop"]
    assert found[1].correlation_with_main is not None
    assert found[1].correlation_with_main < 0


@pytest.mark.parametrize(("peak", "expected"), [(0.78, "strong"), (0.6, "moderate"), (0.48, "weak")])
def test_development(peak, expected):
    assert development(peak) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, "1 зона"),
        (2, "2 зони"),
        (4, "4 зони"),
        (5, "5 зон"),
        (11, "11 зон"),
        (21, "21 зона"),
        (22, "22 зони"),
    ],
)
def test_plural_uk(count, expected):
    assert plural_uk(count, "зона", "зони", "зон") == expected
