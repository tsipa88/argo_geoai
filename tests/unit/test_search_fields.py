import io
import json
import urllib.request
from datetime import UTC, datetime

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import shapely
from fastapi.testclient import TestClient
from shapely.geometry import Polygon, box, mapping

from argo_geoai import server
from argo_geoai.parcels import Parcel
from argo_geoai.runner import run_tool
from argo_geoai.sources import field_boundaries
from argo_geoai.sources.field_boundaries import FieldRecord, read_fields
from argo_geoai.tools import TOOLS
from argo_geoai.tools.search_fields import (
    SearchFieldsTool,
    relation_of,
    search_findings,
    search_metrics,
)

X0, Y0, DX, DY = 28.60, 45.95, 0.01292, 0.009
PARCEL = box(X0, Y0, X0 + DX, Y0 + DY)


def square(x: float, y: float, size: float = 0.5) -> Polygon:
    return box(X0 + x * DX, Y0 + y * DY, X0 + (x + size) * DX, Y0 + (y + size) * DY)


FIELDS = {
    "inside": square(0.25, 0.25),
    "crosses": square(0.8, 0.1),
    "touches_edge": square(1.0, 0.2),
    "touches_corner": square(-0.5, -0.5),
    "far": square(3, 3),
    "contains": square(-1, -1, size=3),
}


def record(name: str, year: int = 2025, confidence: float | None = 80) -> FieldRecord:
    return FieldRecord(id=name, geometry=FIELDS[name], year=year, confidence=confidence)


@pytest.fixture
def parcel() -> Parcel:
    return Parcel(geometry=PARCEL, name="square")


def test_relation_counts_a_single_shared_point():
    assert relation_of(PARCEL, FIELDS["inside"]) == "inside"
    assert relation_of(PARCEL, FIELDS["crosses"]) == "crosses"
    assert relation_of(PARCEL, FIELDS["touches_edge"]) == "touches"
    assert relation_of(PARCEL, FIELDS["touches_corner"]) == "touches"
    assert relation_of(PARCEL, FIELDS["contains"]) == "contains"
    assert relation_of(PARCEL, FIELDS["far"]) is None


def test_metrics_keep_every_field_that_meets_the_polygon(parcel):
    names = ["inside", "crosses", "touches_edge", "touches_corner", "far"]
    metrics = search_metrics(parcel, [record(n) for n in names], year=2025, years=[2025])

    assert {f.id for f in metrics.fields.features} == {"inside", "crosses", "touches_edge", "touches_corner"}
    assert metrics.counts == {"inside": 1, "crosses": 1, "contains": 0, "touches": 2}
    inside = next(f for f in metrics.fields.features if f.id == "inside")
    assert inside.properties.area_ha == pytest.approx(25, rel=0.02)
    assert inside.properties.overlap_share == 1
    crosses = next(f for f in metrics.fields.features if f.id == "crosses")
    assert crosses.properties.overlap_share == pytest.approx(0.4, abs=0.01)
    touches = next(f for f in metrics.fields.features if f.id == "touches_edge")
    assert touches.properties.overlap_ha == 0
    assert metrics.fields.features[0].id == "inside"
    assert metrics.covered_share == pytest.approx(0.25 + 0.2 * 0.5, abs=0.01)


def test_feature_collection_is_geojson_with_lng_lat(parcel):
    metrics = search_metrics(parcel, [record("inside")], year=2025, years=[2025], straighten_m=0)
    collection = json.loads(metrics.fields.model_dump_json())

    feature = collection["features"][0]
    assert collection["type"] == "FeatureCollection"
    assert feature["geometry"]["type"] == "Polygon"
    assert shapely.geometry.shape(feature["geometry"]).equals_exact(FIELDS["inside"], 1e-6)
    assert feature["properties"]["area_ha"] > 0


def test_returned_outline_is_straightened_but_numbers_are_of_the_source(parcel):
    source = search_metrics(parcel, [record("crosses")], year=2025, years=[2025], straighten_m=0)
    straight = search_metrics(parcel, [record("crosses")], year=2025, years=[2025])

    outline = shapely.geometry.shape(straight.fields.features[0].geometry.model_dump())
    assert outline.hausdorff_distance(FIELDS["crosses"]) < 6 / 70_000
    assert straight.fields.features[0].properties == source.fields.features[0].properties


def test_findings_texts(parcel):
    metrics = search_metrics(
        parcel, [record("inside", confidence=40), record("crosses", confidence=None)], year=2025, years=[2025]
    )
    texts = [f.text for f in search_findings(metrics, covered=True)]

    assert texts[0].startswith("Знайдено 2 поля загальною площею")
    assert "всередині ділянки — 1, перетинає межу ділянки — 1" in texts[0]
    assert "покривають лише 35% ділянки" in texts[1]
    assert texts[2].startswith("У 1 поля низька довіра моделі")


def test_small_polygons_are_skipped(parcel):
    sliver = FieldRecord(id="sliver", geometry=square(0.9, 0.9, size=0.07), year=2025, confidence=90)
    records = [record("inside"), sliver]

    metrics = search_metrics(parcel, records, year=2025, years=[2025])
    everything = search_metrics(parcel, records, year=2025, years=[2025], min_area_ha=0)

    assert [f.id for f in metrics.fields.features] == ["inside"]
    assert metrics.small_skipped == 1
    assert {f.id for f in everything.fields.features} == {"inside", "sliver"}
    assert everything.small_skipped == 0
    texts = [f.text for f in search_findings(metrics, covered=True)]
    assert (
        "Не показано 1 дрібний полігон (менше 1 га): ймовірно, клаптики між полями, а не окремі поля."
        in texts
    )


def test_findings_without_fields_or_data(parcel):
    empty = search_metrics(parcel, [], year=2025, years=[2025])
    assert search_findings(empty, covered=True)[0].code == "no_fields"
    assert search_findings(empty, covered=False)[0].code == "outside_field_map"


def write_ftw(path, records: list[FieldRecord], *, row_group_size: int) -> None:
    bounds = [r.geometry.bounds for r in records]
    table = pa.table(
        {
            "id": [r.id for r in records],
            "geometry": [shapely.to_wkb(r.geometry) for r in records],
            "bbox": [{"xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]} for b in bounds],
            "determination:datetime": pa.array(
                [datetime(r.year, 1, 1, tzinfo=UTC) for r in records], pa.timestamp("ms", tz="UTC")
            ),
            "confidence": pa.array([r.confidence for r in records], pa.float32()),
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def test_read_fields_prunes_by_bbox_and_keeps_years(tmp_path):
    path = tmp_path / "UA.parquet"
    records = [record("inside"), record("far"), record("crosses", year=2024, confidence=None)]
    write_ftw(path, records, row_group_size=1)

    found = read_fields(path, PARCEL.bounds)

    assert sorted((r.id, r.year, r.confidence) for r in found) == [
        ("crosses", 2024, None),
        ("inside", 2025, 80),
    ]
    assert found[0].geometry.equals(FIELDS[found[0].id])


@pytest.fixture
def ftw_cache(tmp_path, monkeypatch):
    path = tmp_path / "UA.parquet"
    write_ftw(
        path,
        [record(n) for n in FIELDS] + [record("inside", year=2024)],
        row_group_size=2,
    )
    monkeypatch.setattr(field_boundaries, "country_file", lambda partition: path)


def test_tool_over_http(ftw_cache, monkeypatch):
    monkeypatch.setitem(server.CURRENT_TOOLS, "search_fields", "search_fields")
    monkeypatch.setitem(TOOLS, "search_fields", SearchFieldsTool())
    feature = {"type": "Feature", "geometry": mapping(PARCEL), "properties": {"name": "square"}}

    response = TestClient(server.app).post("/v1/runs", json={"tool": "search_fields", "geometry": feature})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    metrics = body["result"]["metrics"]
    assert metrics["year"] == 2025
    assert metrics["years_available"] == [2024, 2025]
    assert metrics["fields_found"] == 5
    assert {f["id"] for f in metrics["fields"]["features"]} == set(FIELDS) - {"far"}
    assert [f["name"] for f in body["figures"]] == ["maps", "table"]


def test_tool_year_and_confidence_params(ftw_cache, tmp_path):
    parcel = Parcel(geometry=PARCEL, name="square")

    tool = SearchFieldsTool()
    result, _ = run_tool(tool, parcel, tool.params_model(year=2024), tmp_path / "a")
    assert result.metrics is not None
    assert [f.id for f in result.metrics.fields.features] == ["inside"]

    result, _ = run_tool(tool, parcel, tool.params_model(year=2026), tmp_path / "b")
    assert result.metrics is not None
    assert result.metrics.fields_found == 0
    assert [w.code for w in result.warnings] == ["year_not_available"]

    result, _ = run_tool(tool, parcel, tool.params_model(min_confidence=90), tmp_path / "c")
    assert result.metrics is not None
    assert result.metrics.fields_found == 0


def test_parcel_outside_the_countries(monkeypatch, tmp_path):
    def no_download(partition):
        raise AssertionError("no download for a parcel outside the partitions")

    monkeypatch.setattr(field_boundaries, "country_file", no_download)

    parcel = Parcel(geometry=box(10, 50, 10.01, 50.01), name="germany")
    result, _ = run_tool(SearchFieldsTool(), parcel, SearchFieldsTool.params_model(), tmp_path)

    assert result.run.status == "ok"
    assert [f.code for f in result.summary] == ["outside_field_map"]


def test_country_download_sends_a_user_agent(tmp_path, monkeypatch):
    sent = []

    def urlopen(request):
        sent.append(request)
        response = io.BytesIO(b"data")
        response.headers = {"Content-Length": "4"}  # type: ignore[attr-defined]
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    target = tmp_path / "UA.parquet"

    field_boundaries._download("https://example.org/UA.parquet", target)

    assert sent[0].get_header("User-agent") == field_boundaries.USER_AGENT
    assert target.read_bytes() == b"data"
