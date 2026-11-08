from dataclasses import asdict

import numpy as np
import xarray as xr
from rasterio.features import geometry_mask

from argo_geoai import interactive
from argo_geoai.analysis import hydrology
from argo_geoai.schema import Artifact, DataSource, Finding, Label, Message
from argo_geoai.sources import dem, sentinel2, surface_water
from argo_geoai.tools.base import RunContext, ToolOutput
from argo_geoai.tools.common import FLAT_FIELD_P90_DEG, has_mixed_crops, vegetation_metrics_for
from argo_geoai.tools.water import (
    FLOW_LINE_HA,
    TRUNCATED_CELL_SHARE,
    TRUNCATED_EDGE_SHARE,
    WET_TWI,
    WETNESS_CLASSES,
    BoolArray,
    FloatArray,
    OpenWater,
    WaterMetrics,
    WaterParams,
    WaterTool,
    Wetness,
    WetnessClassShare,
    wetness_findings,
)

WETNESS_CLASS_COLORS = ["#fed98e", "#c7e9b4", "#41b6c4", "#225ea8"]
TWI_RANGE = (5.0, 15.0)
UPSLOPE_TICKS_HA = [0.1, 1, 10, 100, 1000, 10000]
HTML_FILE = "water.html"
FIGURES_FILE = "water_figures.json"


# run is a copy of WaterTool.run with another output step: metrics and summary must stay identical
class WaterV2Tool(WaterTool):
    name = "water_v2"
    version = "1.0.0"
    maps_title = "Карти вологості, водозбору і NDVI"
    title = Label(
        en="Water, interactive: wetness, runoff, open water",
        uk="Вода, інтерактивно: вологість, стік, відкрита вода",
    )

    def run(self, params: WaterParams, context: RunContext) -> ToolOutput[WaterMetrics]:
        parcel = context.parcel
        grid, dem_items = dem.elevation(parcel, buffer_m=params.catchment_buffer_m)
        heights = grid.values.astype("float64")
        slope, _ = dem.slope_aspect(heights, dem.RESOLUTION_M)
        area, edge_share = hydrology.flow_accumulation(hydrology.fill_depressions(heights), dem.RESOLUTION_M)
        twi = hydrology.wetness_index(area, slope, dem.RESOLUTION_M)
        inside: BoolArray = ~geometry_mask(
            parcel.in_crs(grid.rio.crs).geometry, out_shape=grid.shape, transform=grid.rio.transform()
        ) & np.isfinite(twi)

        warnings: list[Message] = []
        values, upslope_ha = twi[inside], area[inside] / 10_000
        truncated = float((edge_share[inside] >= TRUNCATED_EDGE_SHARE).mean())
        if truncated > TRUNCATED_CELL_SHARE:
            warnings.append(
                Message(
                    code="catchment_truncated",
                    text=(
                        f"{truncated:.0%} of the parcel gets water from beyond the "
                        f"{params.catchment_buffer_m:.0f} m "
                        "window: wetness is underestimated there, increase catchment_buffer_m"
                    ),
                )
            )

        on_parcel, nearby, water_items = surface_water.surface_water(parcel, nearby_m=params.nearby_water_m)
        vegetation_metrics = vegetation_metrics_for(context, warnings)
        relations, ndvi_maps = self._ndvi_relations(
            params=params,
            context=context,
            grid=grid,
            twi=twi,
            inside=inside,
            vegetation_metrics=vegetation_metrics,
            warnings=warnings,
        )

        metrics = WaterMetrics(
            pixels=int(inside.sum()),
            area_ha=round(inside.sum() * dem.RESOLUTION_M**2 / 10_000, 2),
            flat=bool(np.percentile(slope[inside], 90) < FLAT_FIELD_P90_DEG),
            twi=Wetness(
                mean=round(float(values.mean()), 2),
                median=round(float(np.median(values)), 2),
                p10=round(float(np.percentile(values, 10)), 2),
                p90=round(float(np.percentile(values, 90)), 2),
                max=round(float(values.max()), 2),
            ),
            wetness_classes=[
                WetnessClassShare(
                    code=code,
                    label=label,
                    min_twi=None if np.isinf(lo) else lo,
                    max_twi=None if np.isinf(hi) else hi,
                    share=round(float(((values >= lo) & (values < hi)).mean()), 3),
                )
                for code, lo, hi, label in WETNESS_CLASSES
            ],
            wet_share=round(float((values >= WET_TWI).mean()), 3),
            wet_part_offset=self._wet_offset(grid, twi, inside),
            max_upslope_ha=round(float(upslope_ha.max()), 1),
            flow_lines_share=round(float((upslope_ha >= FLOW_LINE_HA).mean()), 3),
            catchment_truncated_share=round(truncated, 3),
            open_water_on_parcel=OpenWater(**asdict(on_parcel)),
            open_water_nearby=OpenWater(**asdict(nearby)),
            ndvi_relation=relations,
            ndvi_mixed_crops=has_mixed_crops(vegetation_metrics),
        )

        summary = wetness_findings(metrics)
        sources = [
            dem.DATA_SOURCE.model_copy(update={"items": dem_items}),
            surface_water.DATA_SOURCE.model_copy(update={"items": water_items}),
        ]
        if relations:
            sources.append(
                sentinel2.DATA_SOURCE.model_copy(update={"items": [r.scene_id for r in relations]})
            )
        artifacts = self._interactive(
            context=context,
            grid=grid,
            twi=twi,
            area=area,
            inside=inside,
            ndvi_maps=ndvi_maps,
            metrics=metrics,
            summary=summary,
            sources=sources,
        )
        return ToolOutput(
            metrics=metrics,
            summary=summary,
            artifacts=artifacts,
            data_sources=sources,
            warnings=warnings,
            partial=any(w.code == "ndvi_scene_missing" for w in warnings),
        )

    def _interactive(
        self,
        *,
        context: RunContext,
        grid: xr.DataArray,
        twi: FloatArray,
        area: FloatArray,
        inside: BoolArray,
        ndvi_maps: dict[str, FloatArray],
        metrics: WaterMetrics,
        summary: list[Finding],
        sources: list[DataSource],
    ) -> list[Artifact]:
        upslope_ha = np.where(inside, area / 10_000, np.nan)
        layers = [
            interactive.MapLayer(
                title=f"Індекс вологості (TWI), червоним — стік з понад {FLOW_LINE_HA} га",
                hover_label="TWI",
                unit="",
                values=np.where(inside, twi, np.nan),
                colorscale="YlGnBu",
                zmin=TWI_RANGE[0],
                zmax=TWI_RANGE[1],
                decimals=1,
                overlay=inside & (area / 10_000 >= FLOW_LINE_HA),
            ),
            interactive.MapLayer(
                title="Площа водозбору вище точки",
                hover_label="Водозбір",
                unit="га",
                values=np.log10(upslope_ha),
                hover_values=upslope_ha,
                colorscale="Blues",
                zmin=float(np.log10(UPSLOPE_TICKS_HA[0])),
                zmax=max(1.0, float(np.ceil(np.nanmax(np.log10(upslope_ha))))),
                decimals=1,
                ticks=([float(np.log10(t)) for t in UPSLOPE_TICKS_HA], [f"{t:g}" for t in UPSLOPE_TICKS_HA]),
            ),
            *[
                interactive.MapLayer(
                    title=f"NDVI {label}",
                    hover_label=f"NDVI {label}",
                    unit="",
                    values=values,
                    colorscale="RdYlGn",
                    zmin=0,
                    zmax=0.9,
                )
                for label, values in ndvi_maps.items()
            ],
        ]
        geometry = interactive.grid_geometry(grid.x.values, grid.y.values, grid.rio.crs, inside)
        maps = interactive.map_panels(context.parcel, geometry, layers)
        classes = interactive.bar_chart(
            "Класи вологості (TWI)",
            [f"{label.uk} ({_twi_range(lo, hi)})" for _, lo, hi, label in WETNESS_CLASSES],
            [c.share for c in metrics.wetness_classes],
            WETNESS_CLASS_COLORS,
        )

        interactive.write_page(
            context.workdir / HTML_FILE,
            title=f"{context.parcel.name}: вода",
            subtitle=(
                f"{context.parcel.area_ha:.1f} га · середній TWI {metrics.twi.mean:.1f} · "
                f"вологих ділянок {metrics.wet_share:.0%} · Copernicus DEM 30 м, JRC Global Surface Water"
            ),
            summary=summary,
            figures=[maps, classes],
            sources=sources,
        )
        interactive.write_figures_json(
            context.workdir / FIGURES_FILE, {"maps": maps, "wetness_classes": classes}
        )
        return [
            Artifact(
                path=HTML_FILE,
                media_type="text/html",
                description=(
                    "Інтерактивна сторінка: TWI з лініями стоку, площа водозбору, NDVI, класи вологості."
                ),
            ),
            Artifact(
                path=FIGURES_FILE,
                media_type="application/json",
                description="Специфікації графіків Plotly (дані і вигляд) для plotly.js.",
            ),
        ]


def _twi_range(lo: float, hi: float) -> str:
    if np.isinf(lo):
        return f"<{hi:g}"
    if np.isinf(hi):
        return f"≥{lo:g}"
    return f"{lo:g}–{hi:g}"
