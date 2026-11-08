import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from argo_geoai import server
from argo_geoai.config import get_settings
from argo_geoai.schema import Artifact
from argo_geoai.tools import TOOLS
from argo_geoai.tools.base import RunContext, ToolOutput

from .test_runner import DummyMetrics, DummyParams, DummyTool

client = TestClient(server.app)


class DummyRelief(DummyTool):
    name = "relief_v2"
    seen_vegetation: dict[str, Any] | None = None

    def run(self, params: DummyParams, context: RunContext) -> ToolOutput[DummyMetrics]:
        DummyRelief.seen_vegetation = context.vegetation_result
        output = super().run(params, context)
        figures = {
            "maps": {"data": [], "layout": {}},
            "classes": {"data": [], "layout": {"title": {"text": "Класи"}}},
        }
        (context.workdir / "relief_figures.json").write_text(json.dumps(figures))
        output.artifacts.append(
            Artifact(path="relief_figures.json", media_type="application/json", description="Plotly.")
        )
        return output


@pytest.fixture
def relief(monkeypatch):
    monkeypatch.setitem(TOOLS, "relief_v2", DummyRelief())


@pytest.fixture
def feature(field_slope_geojson):
    return json.loads(field_slope_geojson.read_text())


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_tools_are_the_current_implementations():
    tools = {t["tool"]: t["implementation"] for t in client.get("/v1/tools").json()}

    assert tools == {
        "vegetation": "vegetation_v2",
        "relief": "relief_v2",
        "water": "water_v2",
        "moisture": "moisture",
        "activity": "activity",
        "search_fields": "search_fields",
        "field_details": "field_details",
    }


def test_run_returns_result_and_figures(relief, feature, tmp_path):
    vegetation = {"tool": {"name": "vegetation_v2", "version": "1.0.0"}, "metrics": {"zones": []}}

    response = client.post(
        "/v1/runs", json={"tool": "relief", "geometry": feature, "context": {"vegetation": vegetation}}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tool"] == "relief"
    assert body["tool_version"] == "relief_v2/0.0.1"
    assert body["status"] == "ok"
    assert body["result"]["parcel"]["name"].startswith("parcel_")
    assert body["result"]["metrics"]["area_ha"] > 0
    assert body["figures"] == [
        {"name": "maps", "title": "Карти ділянки", "figure": {"data": [], "layout": {}}},
        {"name": "classes", "title": "Класи", "figure": {"data": [], "layout": {}}},
    ]
    assert DummyRelief.seen_vegetation == vegetation


def test_unknown_tool(feature):
    response = client.post("/v1/runs", json={"tool": "relief_v2", "geometry": feature})

    assert response.status_code == 422
    assert "unknown tool" in response.json()["detail"]


def test_broken_polygon(relief):
    response = client.post(
        "/v1/runs", json={"tool": "relief", "geometry": {"type": "Feature", "geometry": {"type": "Polygon"}}}
    )

    assert response.status_code == 422


def test_invalid_params(relief, feature):
    response = client.post("/v1/runs", json={"tool": "relief", "geometry": feature, "params": {"fail": "x"}})

    assert response.status_code == 422


@pytest.fixture
def protected(monkeypatch):
    monkeypatch.setenv("ARGO_GEOAI_API_TOKEN", "s3cret")
    get_settings.cache_clear()
    yield TestClient(server.app), TestClient(server.create_app())
    get_settings.cache_clear()


def test_token_is_required_when_configured(protected, relief, feature):
    client_with_routes, _ = protected
    body = {"tool": "relief", "geometry": feature}

    assert client_with_routes.post("/v1/runs", json=body).status_code == 401
    assert client_with_routes.get("/v1/tools", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert (
        client_with_routes.post("/v1/runs", json=body, headers={"Authorization": "Bearer s3cret"}).status_code
        == 200
    )
    assert client_with_routes.get("/health").status_code == 200


def test_api_docs_are_off_when_protected(protected):
    _, fresh = protected

    assert fresh.get("/docs").status_code == 404
    assert fresh.get("/openapi.json").status_code == 404


def test_responses_are_compressed(relief, feature):
    response = client.post(
        "/v1/runs", json={"tool": "relief", "geometry": feature}, headers={"Accept-Encoding": "gzip"}
    )

    assert response.headers["content-encoding"] == "gzip"
