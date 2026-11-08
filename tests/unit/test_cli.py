import json

import pytest
from typer.testing import CliRunner

from argo_geoai import cli
from argo_geoai.tools import TOOLS

from .test_runner import DummyTool

runner = CliRunner()


@pytest.fixture
def dummy_vegetation(monkeypatch):
    class DummyVegetation(DummyTool):
        name = "vegetation"

    monkeypatch.setitem(TOOLS, "vegetation", DummyVegetation())


def test_tools_list():
    result = runner.invoke(cli.app, ["tools"])

    assert result.exit_code == 0
    assert "vegetation" in result.output
    assert "Рельєф" in result.output


def test_geometry_is_required():
    result = runner.invoke(cli.app, ["run", "relief"])

    assert result.exit_code == 2
    assert "--geometry" in result.output


def test_run_prints_one_json_line(dummy_vegetation, field_slope_geojson, tmp_path):
    result = runner.invoke(
        cli.app, ["run", "vegetation", "--geometry", str(field_slope_geojson), "--output-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    line = json.loads(result.stdout.strip().splitlines()[-1])
    assert line["parcel"] == "field_slope"
    assert [run["tool"] for run in line["runs"]] == ["vegetation"]
    assert line["runs"][0]["status"] == "ok"
    assert line["runs"][0]["result"].endswith("result.json")
    assert line["runs"][0]["artifacts"][0].endswith("map.png")


def test_run_all_reads_the_polygon_from_stdin(dummy_vegetation, monkeypatch, field_slope_geojson, tmp_path):
    class DummyVegetation(DummyTool):
        name = "vegetation_v2"

    class DummyRelief(DummyTool):
        name = "relief_v2"

    class DummyWater(DummyTool):
        name = "water_v2"

    class DummyMoisture(DummyTool):
        name = "moisture"

    monkeypatch.setitem(TOOLS, "vegetation_v2", DummyVegetation())
    monkeypatch.setitem(TOOLS, "relief_v2", DummyRelief())
    monkeypatch.setitem(TOOLS, "water_v2", DummyWater())
    monkeypatch.setitem(TOOLS, "moisture", DummyMoisture())
    feature = json.loads(field_slope_geojson.read_text())
    feature["properties"]["name"] = "field_two_parts"

    result = runner.invoke(
        cli.app, ["run-all", "--geometry", "-", "--output-dir", str(tmp_path)], input=json.dumps(feature)
    )

    assert result.exit_code == 0, result.output
    line = json.loads(result.stdout.strip().splitlines()[-1])
    assert line["parcel"] == "field_two_parts"
    assert [run["tool"] for run in line["runs"]] == ["vegetation_v2", "relief_v2", "water_v2", "moisture"]
    assert (tmp_path / "field_two_parts" / "relief_v2").is_dir()


def test_broken_geometry_input(tmp_path):
    path = tmp_path / "broken.geojson"
    path.write_text('{"type": "Feature"')

    result = runner.invoke(cli.app, ["run", "relief", "--geometry", str(path)])

    assert result.exit_code == 2


def test_schema_command(tmp_path):
    result = runner.invoke(cli.app, ["schema", str(tmp_path)])

    assert result.exit_code == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "activity.schema.json",
        "field_details.schema.json",
        "moisture.schema.json",
        "relief.schema.json",
        "relief_v2.schema.json",
        "request.schema.json",
        "search_fields.schema.json",
        "vegetation.schema.json",
        "vegetation_v2.schema.json",
        "water.schema.json",
        "water_v2.schema.json",
    ]
