import json
from pathlib import Path

import pytest

from argo_geoai.parcels import Parcel
from argo_geoai.results import RESULT_FILE, latest_result
from argo_geoai.runner import run_tool
from argo_geoai.schema import Artifact, Finding, Label, Message, Model, RunStatus, ToolResult
from argo_geoai.tools import TOOLS
from argo_geoai.tools.base import RunContext, Tool, ToolOutput
from argo_geoai.tools.common import vegetation_metrics_for
from argo_geoai.tools.vegetation_v2 import VegetationV2Tool


class DummyParams(Model):
    fail: bool = False
    partial: bool = False


class DummyMetrics(Model):
    area_ha: float


class DummyTool(Tool[DummyParams, DummyMetrics]):
    name = "dummy"
    version = "0.0.1"
    title = Label(en="Dummy", uk="Тестовий")
    params_model = DummyParams
    metrics_model = DummyMetrics

    def run(self, params: DummyParams, context: RunContext) -> ToolOutput[DummyMetrics]:
        if getattr(params, "fail", False):
            raise RuntimeError("no data")
        (context.workdir / "map.png").write_bytes(b"png")
        return ToolOutput(
            metrics=DummyMetrics(area_ha=round(context.parcel.area_ha, 1)),
            summary=[Finding(code="area", level="info", text="Площа 69 га.")],
            artifacts=[Artifact(path="map.png", media_type="image/png", description="Карта.")],
            warnings=[Message(code="cloudy", text="one date skipped")]
            if getattr(params, "partial", False)
            else [],
            partial=getattr(params, "partial", False),
        )


def run(parcel: Parcel, tmp_path: Path, **params: bool):
    return run_tool(DummyTool(), parcel, DummyParams(**params), tmp_path)


def test_ok_run_writes_the_template(field_slope, tmp_path):
    result, folder = run(field_slope, tmp_path)

    assert folder.parent == tmp_path / "field_slope" / "dummy"
    assert (folder / "map.png").is_file()
    data = json.loads((folder / RESULT_FILE).read_text(encoding="utf-8"))
    assert list(data) == [
        "schema_version",
        "tool",
        "run",
        "parcel",
        "parameters",
        "data_sources",
        "summary",
        "metrics",
        "artifacts",
        "warnings",
        "errors",
    ]
    assert data["run"]["status"] == "ok"
    assert data["run"]["status_label"] == {"en": "ok", "uk": "успішно"}
    assert data["metrics"] == {"area_ha": round(field_slope.area_ha, 1)}
    assert data["summary"][0]["text"] == "Площа 69 га."
    assert "Площа" in (folder / RESULT_FILE).read_text(encoding="utf-8")
    assert result.parcel.name == "field_slope"
    assert result.parcel.metadata["number"] == "1"


def test_result_validates_against_the_model(field_slope, tmp_path):
    _, folder = run(field_slope, tmp_path)

    parsed = ToolResult[DummyMetrics, DummyParams].model_validate_json((folder / RESULT_FILE).read_text())

    assert parsed.metrics == DummyMetrics(area_ha=round(field_slope.area_ha, 1))


def test_partial_run(field_slope, tmp_path):
    result, _ = run(field_slope, tmp_path, partial=True)

    assert result.run.status is RunStatus.PARTIAL
    assert result.warnings[0].code == "cloudy"


def test_failing_tool_still_leaves_a_result(field_slope, tmp_path):
    result, folder = run(field_slope, tmp_path, fail=True)

    assert result.run.status is RunStatus.ERROR
    assert result.metrics is None
    assert result.errors == [Message(code="RuntimeError", text="no data")]
    assert json.loads((folder / RESULT_FILE).read_text())["run"]["status_label"]["uk"] == "помилка"


def test_latest_result_skips_failed_runs(field_slope, tmp_path, monkeypatch):
    stamps = iter(["2026-10-01_100000", "2026-10-02_100000"])
    monkeypatch.setattr(
        "argo_geoai.runner.run_folder", lambda root, parcel, tool, _: root / parcel.name / tool / next(stamps)
    )
    run(field_slope, tmp_path)
    run(field_slope, tmp_path, fail=True)

    latest = latest_result(tmp_path, field_slope, "dummy")

    assert latest is not None
    assert latest["run"]["status"] == "ok"


def test_latest_result_without_runs(field_slope, tmp_path):
    assert latest_result(tmp_path, field_slope, "dummy") is None


@pytest.mark.parametrize("tool_name", ["vegetation", "relief"])
def test_registered_tools_have_a_json_schema(tool_name):
    tool = TOOLS[tool_name]
    schema = ToolResult[tool.metrics_model, tool.params_model].model_json_schema()  # type: ignore[name-defined]

    assert set(schema["required"]) >= {"tool", "run", "parcel", "parameters"}
    assert "metrics" in schema["properties"]


def vegetation_result(root: Path, version: str) -> None:
    folder = root / "field_slope" / "vegetation_v2" / "2026-10-01_100000"
    folder.mkdir(parents=True)
    result = {
        "run": {"status": "ok"},
        "tool": {"name": "vegetation_v2", "version": version},
        "metrics": {"peak": {"date": "2026-06-30"}, "zones": [{"role": "main"}, {"role": "separate_crop"}]},
    }
    (folder / "result.json").write_text(json.dumps(result))


def context_for(parcel: Parcel, root: Path) -> RunContext:
    return RunContext(parcel=parcel, workdir=root, output_root=root)


def test_relief_uses_the_current_vegetation_result(field_slope, tmp_path):
    vegetation_result(tmp_path, VegetationV2Tool.version)
    warnings: list[Message] = []

    metrics = vegetation_metrics_for(context_for(field_slope, tmp_path), warnings)

    assert metrics["peak"]["date"] == "2026-06-30"
    assert warnings == []


def test_relief_ignores_a_vegetation_result_of_another_version(field_slope, tmp_path):
    vegetation_result(tmp_path, "0.9.0")
    warnings: list[Message] = []

    metrics = vegetation_metrics_for(context_for(field_slope, tmp_path), warnings)

    assert metrics == {}
    assert [w.code for w in warnings] == ["stale_vegetation_result"]


def test_relief_without_a_vegetation_result(field_slope, tmp_path):
    assert vegetation_metrics_for(context_for(field_slope, tmp_path), []) == {}


def test_relief_uses_the_vegetation_result_sent_by_the_caller(field_slope, tmp_path):
    sent = {"tool": {"name": "vegetation_v2", "version": VegetationV2Tool.version}, "metrics": {"zones": []}}
    context = RunContext(parcel=field_slope, workdir=tmp_path, output_root=tmp_path, vegetation_result=sent)

    assert vegetation_metrics_for(context, []) == {"zones": []}
