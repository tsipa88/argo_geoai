import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError

from argo_geoai import __version__
from argo_geoai.api import analyze
from argo_geoai.parcels import ParcelError
from argo_geoai.request import ParcelRequest
from argo_geoai.results import RESULT_FILE
from argo_geoai.schema import RunStatus, ToolResult
from argo_geoai.tools import TOOLS
from argo_geoai.tools.activity import ActivityParams
from argo_geoai.tools.field_details import FieldDetailsParams
from argo_geoai.tools.moisture import MoistureParams
from argo_geoai.tools.relief import ReliefParams
from argo_geoai.tools.vegetation import VegetationParams
from argo_geoai.tools.water import WaterParams

app = typer.Typer(no_args_is_help=True, help="Satellite analysis tools for a parcel polygon.")
run_app = typer.Typer(no_args_is_help=True, help="Run one tool for a parcel.")
app.add_typer(run_app, name="run")

Geometry = Annotated[
    str, typer.Option("--geometry", help="parcel as a GeoJSON Feature file, or - to read it from stdin")
]
OutputDir = Annotated[
    Path | None, typer.Option("--output-dir", help="default: ARGO_GEOAI_OUTPUT_DIR or ./output")
]
Months = Annotated[int, typer.Option(min=1, max=36, help="vegetation: period length")]
EndDate = Annotated[
    datetime | None, typer.Option(formats=["%Y-%m-%d"], help="vegetation: last day, today by default")
]
MaxSceneCloud = Annotated[float, typer.Option(min=0, max=100, help="vegetation: skip cloudier scenes, %")]
MinClearShare = Annotated[
    float, typer.Option(min=0, max=1, help="vegetation: clear share for a date to count")
]
NdviDates = Annotated[
    list[datetime] | None,
    typer.Option(
        "--ndvi-date",
        formats=["%Y-%m-%d"],
        help="relief, repeatable; default: peak and max-spread of the latest vegetation run",
    ),
]
BufferM = Annotated[float, typer.Option(min=0, max=1000, help="relief: margin read around the parcel, m")]
CatchmentBufferM = Annotated[
    float,
    typer.Option(min=300, max=5000, help="water: relief read around the parcel to route upslope water, m"),
]


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="debug logging")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("rasterio", "rio_tiler", "urllib3", "matplotlib", "PIL", "geoai", "botocore", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # known, harmless warnings of the geospatial stack (see README)
    warnings.filterwarnings("ignore", message="File object not available, clipping in-memory")
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, module="pystac_client")
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="rioxarray")


def read_request(geometry: str) -> ParcelRequest:
    try:
        if geometry == "-":
            return ParcelRequest.from_json(sys.stdin.read())
        path = Path(geometry)
        return ParcelRequest.from_json(path.read_text(encoding="utf-8-sig"), default_name=path.stem)
    except (OSError, ValueError, ValidationError) as error:
        raise typer.BadParameter(f"--geometry: {error}") from error


def execute(geometry: str, params: dict[str, BaseModel], output_dir: Path | None) -> None:
    request = read_request(geometry)
    try:
        outcomes = analyze(request, tools=list(params), params=params, output_root=output_dir)
    except ParcelError as error:
        raise typer.BadParameter(str(error)) from error

    runs = []
    for outcome in outcomes:
        result: ToolResult[BaseModel, BaseModel] = outcome.result
        runs.append(
            {
                "tool": outcome.tool,
                "status": result.run.status.value,
                "result": str((outcome.folder / RESULT_FILE).resolve()),
                "artifacts": [str((outcome.folder / a.path).resolve()) for a in result.artifacts],
            }
        )
    typer.echo(json.dumps({"parcel": request.name, "runs": runs}, ensure_ascii=False))
    if any(o.result.run.status is RunStatus.ERROR for o in outcomes):
        raise typer.Exit(code=1)


def vegetation_params(
    months: int, end_date: datetime | None, max_scene_cloud: float, min_clear_share: float
) -> VegetationParams:
    return VegetationParams(
        months=months,
        end_date=end_date.date() if end_date else None,
        max_scene_cloud=max_scene_cloud,
        min_clear_share=min_clear_share,
    )


def moisture_params(
    months: int, end_date: datetime | None, max_scene_cloud: float, min_clear_share: float
) -> MoistureParams:
    return MoistureParams(
        months=months,
        end_date=end_date.date() if end_date else None,
        max_scene_cloud=max_scene_cloud,
        min_clear_share=min_clear_share,
    )


def relief_params(ndvi_date: list[datetime] | None, buffer_m: float) -> ReliefParams:
    return ReliefParams(ndvi_dates=[d.date() for d in ndvi_date or []], buffer_m=buffer_m)


def water_params(ndvi_date: list[datetime] | None, catchment_buffer_m: float) -> WaterParams:
    return WaterParams(ndvi_dates=[d.date() for d in ndvi_date or []], catchment_buffer_m=catchment_buffer_m)


@app.command("run-all")
def run_all(
    geometry: Geometry,
    months: Months = 6,
    end_date: EndDate = None,
    max_scene_cloud: MaxSceneCloud = 20,
    min_clear_share: MinClearShare = 0.8,
    ndvi_date: NdviDates = None,
    buffer_m: BufferM = 120,
    catchment_buffer_m: CatchmentBufferM = 1500,
    output_dir: OutputDir = None,
) -> None:
    """All current tools for one parcel: vegetation first, then relief and water (they use its dates)."""
    execute(
        geometry,
        {
            "vegetation_v2": vegetation_params(months, end_date, max_scene_cloud, min_clear_share),
            "relief_v2": relief_params(ndvi_date, buffer_m),
            "water_v2": water_params(ndvi_date, catchment_buffer_m),
            "moisture": moisture_params(months, end_date, max_scene_cloud, min_clear_share),
        },
        output_dir,
    )


@run_app.command("vegetation")
def run_vegetation(
    geometry: Geometry,
    months: Months = 6,
    end_date: EndDate = None,
    max_scene_cloud: MaxSceneCloud = 20,
    min_clear_share: MinClearShare = 0.8,
    output_dir: OutputDir = None,
) -> None:
    """NDVI history over a period (Sentinel-2), zones of the field and land cover (ESA WorldCover)."""
    execute(
        geometry,
        {"vegetation": vegetation_params(months, end_date, max_scene_cloud, min_clear_share)},
        output_dir,
    )


@run_app.command("vegetation_v2")
def run_vegetation_v2(
    geometry: Geometry,
    months: Months = 6,
    end_date: EndDate = None,
    max_scene_cloud: MaxSceneCloud = 20,
    min_clear_share: MinClearShare = 0.8,
    output_dir: OutputDir = None,
) -> None:
    """vegetation with interactive output: a static HTML page with Plotly maps and charts."""
    execute(
        geometry,
        {"vegetation_v2": vegetation_params(months, end_date, max_scene_cloud, min_clear_share)},
        output_dir,
    )


@run_app.command("relief")
def run_relief(
    geometry: Geometry, ndvi_date: NdviDates = None, buffer_m: BufferM = 120, output_dir: OutputDir = None
) -> None:
    """Elevation, slope and aspect (Copernicus DEM 30 m) and how NDVI relates to them."""
    execute(geometry, {"relief": relief_params(ndvi_date, buffer_m)}, output_dir)


@run_app.command("relief_v2")
def run_relief_v2(
    geometry: Geometry, ndvi_date: NdviDates = None, buffer_m: BufferM = 120, output_dir: OutputDir = None
) -> None:
    """relief with interactive output: a static HTML page with Plotly maps and charts."""
    execute(geometry, {"relief_v2": relief_params(ndvi_date, buffer_m)}, output_dir)


@run_app.command("water")
def run_water(
    geometry: Geometry,
    ndvi_date: NdviDates = None,
    catchment_buffer_m: CatchmentBufferM = 1500,
    output_dir: OutputDir = None,
) -> None:
    """Where water collects (TWI), concentrated runoff, inflow, open water (JRC) and NDVI vs wetness."""
    execute(geometry, {"water": water_params(ndvi_date, catchment_buffer_m)}, output_dir)


@run_app.command("water_v2")
def run_water_v2(
    geometry: Geometry,
    ndvi_date: NdviDates = None,
    catchment_buffer_m: CatchmentBufferM = 1500,
    output_dir: OutputDir = None,
) -> None:
    """water with interactive output: a static HTML page with Plotly maps and charts."""
    execute(geometry, {"water_v2": water_params(ndvi_date, catchment_buffer_m)}, output_dir)


@run_app.command("moisture")
def run_moisture(
    geometry: Geometry,
    months: Months = 6,
    end_date: EndDate = None,
    max_scene_cloud: MaxSceneCloud = 20,
    min_clear_share: MinClearShare = 0.8,
    output_dir: OutputDir = None,
) -> None:
    """Water stress of plants: NDMI (water in leaves) over the season, Sentinel-2 20 m."""
    execute(
        geometry,
        {"moisture": moisture_params(months, end_date, max_scene_cloud, min_clear_share)},
        output_dir,
    )


@run_app.command("activity")
def run_activity(
    geometry: Geometry,
    start_date: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="first day to look at; --months back if empty"),
    ] = None,
    months: Annotated[int, typer.Option(min=1, max=120, help="period length without --start-date")] = 24,
    end_date: EndDate = None,
    radar: Annotated[bool, typer.Option(help="read Sentinel-1 radar (harvest dates)")] = True,
    output_dir: OutputDir = None,
) -> None:
    """What was done on the field and when: harvest, tillage, emergence, green cut (Sentinel-2 + radar)."""
    params = ActivityParams(
        start_date=start_date.date() if start_date else None,
        months=months,
        end_date=end_date.date() if end_date else None,
        radar=radar,
    )
    execute(geometry, {"activity": params}, output_dir)


@run_app.command("field_details")
def run_field_details(
    geometry: Geometry,
    start_year: Annotated[int, typer.Option(min=2015, help="first year of the history")] = 2017,
    end_date: EndDate = None,
    max_scenes: Annotated[int, typer.Option(min=20, max=3000, help="scenes read at most")] = 400,
    snapshots_per_year: Annotated[int, typer.Option(min=1, max=12, help="true-colour images per year")] = 3,
    output_dir: OutputDir = None,
) -> None:
    """Details of one parcel for a report: land cover, relief, years of use and snapshots (Sentinel-2)."""
    params = FieldDetailsParams(
        start_year=start_year,
        end_date=end_date.date() if end_date else None,
        max_scenes=max_scenes,
        snapshots_per_year=snapshots_per_year,
    )
    execute(geometry, {"field_details": params}, output_dir)


@app.command("tools")
def list_tools() -> None:
    """Available tools."""
    for tool in TOOLS.values():
        typer.echo(f"{tool.name:14} {tool.version:8} {tool.title.uk}")


@app.command("schema")
def write_schemas(
    directory: Annotated[Path, typer.Argument(help="where to write <tool>.schema.json")] = Path("schemas"),
) -> None:
    """JSON Schema of the request and of result.json for every tool."""
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "request.schema.json"
    request_path.write_text(
        json.dumps(ParcelRequest.model_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(str(request_path))
    for tool in TOOLS.values():
        model = ToolResult[tool.metrics_model, tool.params_model]  # type: ignore[name-defined]
        path = directory / f"{tool.name}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        typer.echo(str(path))


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="interface to listen on, localhost by default")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="TCP port")] = 8000,
) -> None:
    """HTTP server: POST /v1/runs with a tool type and a parcel polygon (see argo_geoai.server)."""
    import uvicorn  # noqa: PLC0415 - only this command needs the server stack

    uvicorn.run("argo_geoai.server:app", host=host, port=port, log_level="info")


@app.command("version")
def version() -> None:
    typer.echo(__version__)
