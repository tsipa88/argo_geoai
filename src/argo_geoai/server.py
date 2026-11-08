import json
import logging
import secrets
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, ValidationError

from argo_geoai import __version__
from argo_geoai.config import get_settings
from argo_geoai.parcels import ParcelError, from_request
from argo_geoai.request import ParcelRequest
from argo_geoai.runner import run_tool
from argo_geoai.schema import RunStatus
from argo_geoai.tools import TOOLS

log = logging.getLogger(__name__)

CURRENT_TOOLS = {
    "vegetation": "vegetation_v2",
    "relief": "relief_v2",
    "water": "water_v2",
    "moisture": "moisture",
    "activity": "activity",
    "search_fields": "search_fields",
    "field_details": "field_details",
}
FIGURES_MEDIA_TYPE = "application/json"


class RunContextIn(BaseModel):
    vegetation: dict[str, Any] | None = Field(
        None,
        description="relief, water, moisture: the latest vegetation result of this parcel (its result JSON)",
    )


class RunRequest(BaseModel):
    tool: str = Field(description=f"one of: {', '.join(CURRENT_TOOLS)}")
    geometry: dict[str, Any] = Field(description="the parcel as a GeoJSON Feature (see request.schema.json)")
    params: dict[str, Any] = Field(default_factory=dict, description="tool parameters; defaults when empty")
    context: RunContextIn = Field(default_factory=RunContextIn)


class FigureOut(BaseModel):
    name: str
    title: str = Field(description="for the card header; removed from the figure layout")
    figure: dict[str, Any] = Field(description="Plotly spec: Plotly.newPlot(el, figure.data, figure.layout)")


class RunResponse(BaseModel):
    tool: str
    tool_version: str
    status: RunStatus
    result: dict[str, Any] = Field(description="the result JSON, same template as result.json of the CLI")
    figures: list[FigureOut] = Field(description="in display order (a list: jsonb does not keep key order)")


def create_app() -> FastAPI:
    protected = bool(get_settings().api_token)
    application = FastAPI(
        title="argo_geoai",
        version=__version__,
        docs_url=None if protected else "/docs",
        redoc_url=None,
        openapi_url=None if protected else "/openapi.json",
    )
    application.add_middleware(GZipMiddleware, minimum_size=1000)
    return application


app = create_app()
_run_slots = threading.BoundedSemaphore(max(1, get_settings().max_parallel_runs))


def require_token(request: Request) -> None:
    token = get_settings().api_token
    if not token:
        return
    given = request.headers.get("authorization", "")
    if not secrets.compare_digest(given.encode(), f"Bearer {token}".encode()):
        raise HTTPException(401, "invalid or missing API token", headers={"WWW-Authenticate": "Bearer"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/v1/tools", dependencies=[Depends(require_token)])
def tools() -> list[dict[str, str]]:
    return [
        {
            "tool": public,
            "implementation": name,
            "version": TOOLS[name].version,
            "title": TOOLS[name].title.uk,
        }
        for public, name in CURRENT_TOOLS.items()
    ]


@app.post("/v1/runs", dependencies=[Depends(require_token)])
def run(request: RunRequest) -> RunResponse:
    """Runs one tool synchronously (vegetation takes minutes); a plain def runs in the thread pool."""
    name = CURRENT_TOOLS.get(request.tool)
    if name is None:
        raise HTTPException(422, f"unknown tool {request.tool!r}; available: {', '.join(CURRENT_TOOLS)}")
    tool = TOOLS[name]
    try:
        parcel = from_request(ParcelRequest.model_validate(request.geometry))
        params = tool.params_model.model_validate(request.params)
    except (ValidationError, ParcelError, ValueError) as error:
        raise HTTPException(422, str(error)) from error

    with _run_slots, tempfile.TemporaryDirectory(prefix="argo_geoai_") as root:
        result, folder = run_tool(
            tool, parcel, params, Path(root), vegetation_result=request.context.vegetation
        )
        figures = _figures(folder, result.artifacts, maps_title=tool.maps_title)
    log.info(
        "%s %s for %s: %s in %.1f s",
        name,
        tool.version,
        parcel.name,
        result.run.status,
        result.run.duration_s,
    )
    return RunResponse(
        tool=request.tool,
        tool_version=f"{name}/{tool.version}",
        status=result.run.status,
        result=result.model_dump(mode="json"),
        figures=figures,
    )


def _figures(folder: Path, artifacts: list[Any], *, maps_title: str) -> list[FigureOut]:
    for artifact in artifacts:
        if artifact.media_type == FIGURES_MEDIA_TYPE:
            specs: dict[str, Any] = json.loads((folder / artifact.path).read_text(encoding="utf-8"))
            return [
                FigureOut(
                    name=name, title=_pop_title(spec) or (maps_title if name == "maps" else name), figure=spec
                )
                for name, spec in specs.items()
            ]
    return []


def _pop_title(spec: dict[str, Any]) -> str | None:
    layout = spec.get("layout", {})
    title = layout.pop("title", None)
    text = title.get("text") if isinstance(title, dict) else title
    return str(text) if text else None
