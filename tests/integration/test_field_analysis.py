"""Season 2026 of the four test parcels (Fields of The World outlines in Moldova, see tests/data/README.md).

The numbers are what the tools gave in November 2026; Planetary Computer reprocessing can move them a little.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from argo_geoai.runner import run_tool
from argo_geoai.schema import RunStatus
from argo_geoai.tools.activity import ActivityParams, ActivityTool
from argo_geoai.tools.relief import ReliefParams, ReliefTool
from argo_geoai.tools.vegetation import VegetationParams, VegetationTool
from argo_geoai.tools.water import WaterParams, WaterTool
from tests.conftest import load_parcel

pytestmark = pytest.mark.network

DATA = Path(__file__).parents[1] / "data"
END_OF_SEASON = date(2026, 10, 2)


@pytest.fixture(scope="module")
def output(tmp_path_factory):
    return tmp_path_factory.mktemp("output")


def vegetation_run(parcel, output):
    result, _ = run_tool(VegetationTool(), parcel, VegetationParams(end_date=END_OF_SEASON), output)
    assert result.run.status is RunStatus.OK
    return result


def as_context(result):
    """The vegetation result the way the HTTP caller would send it back in `context.vegetation`."""
    return json.loads(result.model_dump_json())


def codes(result):
    return [f.code for f in result.summary]


def texts(result):
    return {f.code: f.text for f in result.summary}


# --- field_slope: one late crop on an east-facing slope, no inflow ---------------------------------


@pytest.fixture(scope="module")
def field_slope():
    return load_parcel(DATA / "field_slope.geojson")


@pytest.fixture(scope="module")
def slope_vegetation(field_slope, output):
    return vegetation_run(field_slope, output)


def test_field_slope_vegetation_one_late_crop(slope_vegetation):
    metrics = slope_vegetation.metrics
    assert metrics is not None

    assert slope_vegetation.warnings == []
    assert metrics.dates_counted == 24
    assert metrics.peak.date == date(2026, 7, 30)
    assert metrics.peak.ndvi.mean == pytest.approx(0.67, abs=0.01)
    assert metrics.max_spread.date == date(2026, 8, 9)
    assert metrics.latest.date == date(2026, 10, 1)
    assert metrics.latest.ndvi.mean == pytest.approx(0.43, abs=0.01)
    assert metrics.longest_gap.days == 28
    assert metrics.land_cover[0].code == "cropland"
    assert metrics.land_cover[0].share == pytest.approx(0.99, abs=0.01)
    assert [(z.role, z.timing) for z in metrics.zones] == [("main", "late")]
    assert all(h.parcel_area_ha == pytest.approx(84.6, abs=0.3) for h in metrics.history)
    assert codes(slope_vegetation) == [
        "land_cover_main",
        "clear_dates",
        "peak",
        "uneven_at_peak",
        "latest",
        "data_gap",
    ]


def test_field_slope_relief_uses_the_vegetation_dates(field_slope, slope_vegetation, output):
    result, _ = run_tool(
        ReliefTool(), field_slope, ReliefParams(), output, vegetation_result=as_context(slope_vegetation)
    )
    metrics = result.metrics
    assert metrics is not None

    assert result.run.status is RunStatus.OK
    assert result.warnings == []
    assert metrics.elevation_m.min == pytest.approx(363.7, abs=0.5)
    assert metrics.elevation_m.max == pytest.approx(392.7, abs=0.5)
    assert metrics.slope_deg.mean == pytest.approx(3.27, abs=0.05)
    shares = {c.code: c.share for c in metrics.slope_classes}
    assert shares["strong"] + shares["steep"] == pytest.approx(0.21, abs=0.01)
    assert metrics.aspect_of_slopes[0].code == "e"
    assert not metrics.ndvi_mixed_crops
    assert [(r.date, r.reason) for r in metrics.ndvi_relation] == [
        (date(2026, 7, 30), "vegetation_peak"),
        (date(2026, 8, 9), "vegetation_max_spread"),
    ]
    assert metrics.ndvi_relation[1].vs_elevation.rho == pytest.approx(-0.30, abs=0.05)
    found = texts(result)
    assert found["elevation_range"] == "Перепад висот на полі 29 м (364–393 м над рівнем моря)."
    assert found["erosion_risk"].startswith("21% поля на ухилах 5° і більше")
    assert found["aspect"] == "Схили переважно звернені на схід і північний схід (50%)."
    assert codes(result).count("ndvi_vs_relief") == 2


def test_field_slope_relief_without_a_vegetation_result_has_no_ndvi(field_slope, tmp_path):
    result, _ = run_tool(ReliefTool(), field_slope, ReliefParams(), tmp_path)

    assert result.run.status is RunStatus.OK
    assert [w.code for w in result.warnings] == ["no_vegetation_result"]
    assert result.metrics is not None
    assert result.metrics.ndvi_relation == []
    assert "ndvi_vs_relief" not in codes(result)


def test_field_slope_water_drains_itself(field_slope, slope_vegetation, output):
    result, _ = run_tool(
        WaterTool(), field_slope, WaterParams(), output, vegetation_result=as_context(slope_vegetation)
    )
    metrics = result.metrics
    assert metrics is not None

    assert result.warnings == []
    assert not metrics.flat
    assert metrics.wet_share == pytest.approx(0.015, abs=0.005)
    assert metrics.max_upslope_ha < metrics.area_ha
    assert metrics.flow_lines_share < 0.01
    assert metrics.catchment_truncated_share == 0
    assert metrics.open_water_on_parcel.share_ever == 0
    found = texts(result)
    assert found["wet_share"] == "Місць, де збирається вода, мало: 2% поля."
    assert "inflow" not in found
    assert "flow_lines" not in found
    assert codes(result).count("ndvi_vs_wetness") == 2


# --- field_ravines: steep dissected slopes, runoff from beyond the field, two late crops -----------


@pytest.fixture(scope="module")
def field_ravines():
    return load_parcel(DATA / "field_ravines.geojson")


@pytest.fixture(scope="module")
def ravines_vegetation(field_ravines, output):
    return vegetation_run(field_ravines, output)


def test_field_ravines_two_late_crops(ravines_vegetation):
    metrics = ravines_vegetation.metrics
    assert metrics is not None

    assert ravines_vegetation.warnings == []
    assert metrics.peak.date == date(2026, 6, 28)
    assert metrics.peak.ndvi.mean == pytest.approx(0.74, abs=0.02)
    assert [(z.role, z.timing, round(z.share, 1)) for z in metrics.zones] == [
        ("main", "late", 0.6),
        ("separate_crop", "late", 0.4),
    ]
    assert metrics.zoning_silhouette is not None
    assert metrics.zoning_silhouette > 0.5
    assert codes(ravines_vegetation)[-3:] == ["zones", "zone_1", "zone_2"]
    assert "див. зони" in texts(ravines_vegetation)["peak"]


def test_field_ravines_relief_is_steep_and_skips_ndvi_across_crops(field_ravines, ravines_vegetation, output):
    result, _ = run_tool(
        ReliefTool(), field_ravines, ReliefParams(), output, vegetation_result=as_context(ravines_vegetation)
    )
    metrics = result.metrics
    assert metrics is not None

    assert metrics.elevation_m.range == pytest.approx(100, abs=1)
    assert metrics.slope_deg.mean == pytest.approx(6.8, abs=0.1)
    assert metrics.aspect_of_slopes[0].code == "ne"
    assert metrics.ndvi_mixed_crops
    found = texts(result)
    assert found["erosion_risk"].startswith("86% поля на ухилах 5° і більше")
    assert "довше тримають вологу" in found["aspect"]
    assert "ndvi_vs_relief_mixed_crops" in found
    assert "ndvi_vs_relief" not in found


def test_field_ravines_water_runoff_through_ravines(field_ravines, ravines_vegetation, output):
    result, _ = run_tool(
        WaterTool(), field_ravines, WaterParams(), output, vegetation_result=as_context(ravines_vegetation)
    )
    metrics = result.metrics
    assert metrics is not None

    assert metrics.max_upslope_ha == pytest.approx(99, abs=5)
    assert metrics.max_upslope_ha > metrics.area_ha
    assert metrics.flow_lines_share == pytest.approx(0.086, abs=0.01)
    assert metrics.catchment_truncated_share == 0
    assert metrics.ndvi_mixed_crops
    found = texts(result)
    assert "99 га" in found["inflow"]
    assert "промоїни" in found["flow_lines"]
    assert "ndvi_vs_wetness_mixed_crops" in found
    assert "ndvi_vs_wetness" not in found


# --- field_two_crops: two neighbouring flat fields, an early and a late crop, under one polygon ---


@pytest.fixture(scope="module")
def field_two_crops():
    return load_parcel(DATA / "field_two_crops.geojson")


@pytest.fixture(scope="module")
def two_crops_vegetation(field_two_crops, output):
    return vegetation_run(field_two_crops, output)


def test_field_two_crops_has_an_early_and_a_late_half(two_crops_vegetation):
    metrics = two_crops_vegetation.metrics
    assert metrics is not None

    assert two_crops_vegetation.warnings == []
    assert [(z.timing, round(z.share, 1)) for z in metrics.zones] == [("early", 0.5), ("late", 0.5)]
    assert metrics.zoning_silhouette is not None
    assert metrics.zoning_silhouette > 0.6
    assert metrics.zones[0].peak.date == date(2026, 5, 24)
    assert metrics.zones[1].peak.date == date(2026, 6, 17)
    assert metrics.zones[1].peak.ndvi == pytest.approx(0.84, abs=0.03)
    assert metrics.latest.ndvi.mean < 0.3
    assert codes(two_crops_vegetation)[-3:] == ["zones", "zone_1", "zone_2"]
    assert "див. зони" in texts(two_crops_vegetation)["peak"]


def test_field_two_crops_relief_is_flat_and_skips_ndvi_across_crops(
    field_two_crops, two_crops_vegetation, output
):
    result, _ = run_tool(
        ReliefTool(),
        field_two_crops,
        ReliefParams(),
        output,
        vegetation_result=as_context(two_crops_vegetation),
    )
    metrics = result.metrics
    assert metrics is not None

    assert metrics.slope_deg.p90 < 1
    assert metrics.elevation_m.range == pytest.approx(3.7, abs=1)
    assert metrics.ndvi_mixed_crops
    found = codes(result)
    assert "flat_field" in found
    assert "aspect" not in found
    assert "erosion_risk" not in found
    assert "ndvi_vs_relief_mixed_crops" in found
    assert "ndvi_vs_relief" not in found


def test_field_two_crops_water_lowland_with_inflow(field_two_crops, two_crops_vegetation, output):
    result, _ = run_tool(
        WaterTool(),
        field_two_crops,
        WaterParams(),
        output,
        vegetation_result=as_context(two_crops_vegetation),
    )
    metrics = result.metrics
    assert metrics is not None

    assert metrics.flat
    assert metrics.wet_share > 0.3
    assert metrics.max_upslope_ha > 10 * metrics.area_ha
    assert metrics.open_water_on_parcel.share_ever == 0
    assert metrics.open_water_nearby.share_ever > 0
    found = texts(result)
    assert found["wet_share"].startswith("45% поля")
    assert "flat_field" in found
    assert "inflow" in found
    assert "flow_lines" not in found
    assert "ndvi_vs_wetness" not in found
    assert "ndvi_vs_wetness_mixed_crops" not in found or metrics.ndvi_mixed_crops


# --- field_two_parts: two separate flat fields with early crops as one MultiPolygon ----------------


@pytest.fixture(scope="module")
def field_two_parts():
    return load_parcel(DATA / "field_two_parts.geojson")


@pytest.fixture(scope="module")
def two_parts_vegetation(field_two_parts, output):
    return vegetation_run(field_two_parts, output)


def test_field_two_parts_early_crops_with_different_starts(field_two_parts, two_parts_vegetation):
    assert field_two_parts.parts == 2
    assert field_two_parts.area_ha == pytest.approx(242.1, abs=0.5)

    metrics = two_parts_vegetation.metrics
    assert metrics is not None
    assert two_parts_vegetation.warnings == []
    assert metrics.peak.date == date(2026, 5, 26)
    assert metrics.peak.ndvi.mean == pytest.approx(0.82, abs=0.02)
    assert metrics.latest.ndvi.mean < 0.25
    assert metrics.land_cover[0].code == "cropland"
    assert metrics.land_cover[0].share > 0.99
    assert [(z.role, z.timing, z.development) for z in metrics.zones] == [
        ("main", "early", "strong"),
        ("separate_crop", "early", "strong"),
    ]
    assert metrics.zones[1].share == pytest.approx(0.44, abs=0.03)
    assert all(h.parcel_area_ha == pytest.approx(242.0, abs=0.5) for h in metrics.history)


def test_field_two_parts_relief_is_flat(field_two_parts, two_parts_vegetation, output):
    result, _ = run_tool(
        ReliefTool(),
        field_two_parts,
        ReliefParams(),
        output,
        vegetation_result=as_context(two_parts_vegetation),
    )
    metrics = result.metrics
    assert metrics is not None

    assert metrics.slope_deg.p90 < 2
    assert metrics.elevation_m.range == pytest.approx(55.8, abs=1)
    found = codes(result)
    assert "flat_field" in found
    assert "aspect" not in found
    assert "ndvi_vs_relief" not in found


def test_field_two_parts_water_flat_field(field_two_parts, two_parts_vegetation, output):
    result, _ = run_tool(
        WaterTool(),
        field_two_parts,
        WaterParams(),
        output,
        vegetation_result=as_context(two_parts_vegetation),
    )
    metrics = result.metrics
    assert metrics is not None

    assert metrics.flat
    assert metrics.wet_share == pytest.approx(0.20, abs=0.02)
    assert metrics.catchment_truncated_share == 0
    found = texts(result)
    assert "flat_field" in found
    assert found["wet_part"] == "Вологіші ділянки зосереджені переважно на заході поля."
    assert "flow_lines" not in found
    assert "ndvi_vs_wetness" not in found


def test_field_two_parts_activity_finds_two_harvests(field_two_parts, output):
    params = ActivityParams(months=20, end_date=date(2026, 10, 11))
    result, _ = run_tool(ActivityTool(), field_two_parts, params, output)
    metrics = result.metrics
    assert metrics is not None
    assert result.run.status is RunStatus.OK

    harvests = [e for e in metrics.events if e.code == "harvest"]
    assert [(e.date_from.year, e.date_to.month) for e in harvests] == [(2025, 7), (2026, 7)]
    assert harvests[-1].date_from == date(2026, 6, 27)
    assert all(e.share and e.share > 0.95 for e in harvests)
    assert [e.code for e in metrics.events].count("sowing") == 2
    assert texts(result)["latest_harvest"] == "Останнє збирання: 27.06.2026–15.07.2026."
