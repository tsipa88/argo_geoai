from datetime import date
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from rasterio.enums import Resampling

from argo_geoai.parcels import Parcel
from argo_geoai.results import latest_result
from argo_geoai.schema import Message, Model
from argo_geoai.sources import planetary, sentinel2
from argo_geoai.tools.base import RunContext
from argo_geoai.tools.vegetation import VegetationTool
from argo_geoai.tools.vegetation_v2 import VegetationV2Tool

FloatArray = npt.NDArray[np.float64]
NdviDateReason = Literal["requested", "vegetation_peak", "vegetation_max_spread"]

MIN_NDVI_DIFFERENCE = 0.03
FLAT_FIELD_P90_DEG = 2


class Correlation(Model):
    rho: float
    p_value: float


def describe_strength(rho: float, p_value: float) -> str | None:
    if p_value >= 0.05 or abs(rho) < 0.1:
        return None
    if abs(rho) < 0.3:
        return "слабкий"
    if abs(rho) < 0.5:
        return "помірний"
    return "сильний"


VEGETATION_VERSIONS = {
    VegetationV2Tool.name: VegetationV2Tool.version,
    VegetationTool.name: VegetationTool.version,
}


def vegetation_metrics_for(context: RunContext, warnings: list[Message]) -> dict[str, Any]:
    vegetation = context.vegetation_result
    if vegetation is None:
        vegetation = latest_result(context.output_root, context.parcel, VegetationV2Tool.name)
    if not vegetation:
        return {}
    tool = vegetation.get("tool") or {}
    expected = VEGETATION_VERSIONS.get(tool.get("name", ""))
    if tool.get("version") != expected:
        warnings.append(
            Message(
                code="stale_vegetation_result",
                text=(
                    f"the vegetation result is {tool.get('name')} {tool.get('version')}, current is "
                    f"{VegetationV2Tool.name} {VegetationV2Tool.version}: ignored, rerun vegetation"
                ),
            )
        )
        return {}
    metrics: dict[str, Any] = vegetation.get("metrics") or {}
    return metrics


def has_mixed_crops(vegetation_metrics: dict[str, Any]) -> bool:
    return any(zone.get("role") == "separate_crop" for zone in vegetation_metrics.get("zones", []))


def ndvi_dates(
    requested: list[date], vegetation_metrics: dict[str, Any], warnings: list[Message]
) -> list[tuple[date, NdviDateReason]]:
    if requested:
        return [(day, "requested") for day in requested]
    if not vegetation_metrics:
        warnings.append(
            Message(
                code="no_vegetation_result",
                text="NDVI comparison skipped: run the vegetation tool first or pass ndvi dates",
            )
        )
        return []

    dates: list[tuple[date, NdviDateReason]] = []
    reasons: tuple[tuple[str, NdviDateReason], ...] = (
        ("peak", "vegetation_peak"),
        ("max_spread", "vegetation_max_spread"),
    )
    for key, reason in reasons:
        point = vegetation_metrics.get(key)
        if point and date.fromisoformat(point["date"]) not in {d for d, _ in dates}:
            dates.append((date.fromisoformat(point["date"]), reason))
    return dates


def ndvi_on_grid(
    day: date, parcel: Parcel, grid: xr.DataArray, *, buffer_m: float, warnings: list[Message]
) -> tuple[FloatArray, str] | None:
    items = planetary.search(sentinel2.COLLECTION, parcel, time_range=f"{day}/{day}")
    scenes = sentinel2.least_cloudy_per_day(items)
    scene = sentinel2.scene_ndvi(scenes[0], parcel, buffer_m=buffer_m) if scenes else None
    if scene is None:
        warnings.append(Message(code="ndvi_scene_missing", text=f"no Sentinel-2 scene for {day}"))
        return None
    values: FloatArray = scene.ndvi.rio.reproject_match(grid, resampling=Resampling.average).values.astype(
        "float64"
    )
    return values, scene.item_id
