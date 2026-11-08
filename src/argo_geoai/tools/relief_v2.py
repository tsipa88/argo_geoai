import numpy as np
import xarray as xr
from rasterio.features import geometry_mask

from argo_geoai import interactive
from argo_geoai.schema import Artifact, DataSource, Finding, Label, Message
from argo_geoai.sources import dem, sentinel2
from argo_geoai.tools.base import RunContext, ToolOutput
from argo_geoai.tools.common import has_mixed_crops, vegetation_metrics_for
from argo_geoai.tools.relief import (
    ASPECTS,
    MIN_SLOPE_FOR_ASPECT_DEG,
    SLOPE_CLASSES,
    BoolArray,
    ClassShare,
    Elevation,
    FloatArray,
    Offset,
    ReliefMetrics,
    ReliefParams,
    ReliefTool,
    Slope,
    SlopeClassShare,
    relief_findings,
    share,
)

SLOPE_CLASS_COLORS = ["#ffffcc", "#c2e699", "#fed976", "#fd8d3c", "#e31a1c"]
HTML_FILE = "relief.html"
FIGURES_FILE = "relief_figures.json"


# run is a copy of ReliefTool.run with another output step: metrics and summary must stay identical
class ReliefV2Tool(ReliefTool):
    name = "relief_v2"
    version = "1.0.0"
    maps_title = "Карти висоти, ухилу і NDVI"
    title = Label(
        en="Relief, interactive (elevation, slope, aspect)",
        uk="Рельєф, інтерактивно (висота, ухил, експозиція)",
    )

    def run(self, params: ReliefParams, context: RunContext) -> ToolOutput[ReliefMetrics]:
        parcel = context.parcel
        grid, dem_items = dem.elevation(parcel, buffer_m=params.buffer_m)
        heights = grid.values.astype("float64")
        slope, aspect = dem.slope_aspect(heights, dem.RESOLUTION_M)
        inside: BoolArray = ~geometry_mask(
            parcel.in_crs(grid.rio.crs).geometry, out_shape=grid.shape, transform=grid.rio.transform()
        ) & np.isfinite(heights)

        z, s = heights[inside], slope[inside]
        p5, p95 = np.percentile(z, [5, 95])
        sloped = inside & (slope >= MIN_SLOPE_FOR_ASPECT_DEG)
        sectors = (np.nan_to_num((aspect + 22.5) // 45) % 8).astype(int)
        aspects = (
            [
                ClassShare(code=code, label=label, share=share(sectors == k, sloped))
                for k, (code, label, _) in enumerate(ASPECTS)
            ]
            if sloped.any()
            else []
        )

        rows, cols = np.nonzero(inside)
        ys, xs = grid.y.values[rows], grid.x.values[cols]
        low, high = z <= np.percentile(z, 33), z >= np.percentile(z, 67)

        warnings: list[Message] = []
        vegetation_metrics = vegetation_metrics_for(context, warnings)
        relations, ndvi_maps = self._ndvi_relations(
            vegetation_metrics=vegetation_metrics,
            params=params,
            context=context,
            grid=grid,
            heights=heights,
            slope=slope,
            inside=inside,
            warnings=warnings,
        )

        metrics = ReliefMetrics(
            pixels=int(inside.sum()),
            area_ha=round(inside.sum() * dem.RESOLUTION_M**2 / 10_000, 2),
            elevation_m=Elevation(
                min=round(float(z.min()), 1),
                max=round(float(z.max()), 1),
                mean=round(float(z.mean()), 1),
                p5=round(float(p5), 1),
                p95=round(float(p95), 1),
                range=round(float(z.max() - z.min()), 1),
                range_p5_p95=round(float(p95 - p5), 1),
            ),
            slope_deg=Slope(
                mean=round(float(s.mean()), 2),
                median=round(float(np.median(s)), 2),
                p90=round(float(np.percentile(s, 90)), 2),
                max=round(float(s.max()), 2),
            ),
            slope_classes=[
                SlopeClassShare(
                    code=code,
                    label=label,
                    min_deg=lo,
                    max_deg=hi,
                    share=share((slope >= lo) & (slope < hi), inside),
                )
                for code, lo, hi, label in SLOPE_CLASSES
            ],
            aspect_of_slopes=sorted(aspects, key=lambda a: -a.share),
            low_third_offset_from_high_third=Offset(
                north_m=round(float(ys[low].mean() - ys[high].mean())),
                east_m=round(float(xs[low].mean() - xs[high].mean())),
            ),
            ndvi_relation=relations,
            ndvi_mixed_crops=has_mixed_crops(vegetation_metrics),
        )

        summary = relief_findings(metrics)
        sources = [dem.DATA_SOURCE.model_copy(update={"items": dem_items})]
        if relations:
            sources.append(
                sentinel2.DATA_SOURCE.model_copy(update={"items": [r.scene_id for r in relations]})
            )
        artifacts = self._interactive(
            context=context,
            grid=grid,
            heights=heights,
            slope=slope,
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
        heights: FloatArray,
        slope: FloatArray,
        inside: BoolArray,
        ndvi_maps: dict[str, FloatArray],
        metrics: ReliefMetrics,
        summary: list[Finding],
        sources: list[DataSource],
    ) -> list[Artifact]:
        slope_inside = np.where(inside, slope, np.nan)
        layers = [
            interactive.MapLayer(
                title="Висота (горизонталі через 2 м)",
                hover_label="Висота",
                unit="м",
                values=np.where(inside, heights, np.nan),
                colorscale="terrain",
                contour_step=2,
                decimals=1,
            ),
            interactive.MapLayer(
                title="Ухил",
                hover_label="Ухил",
                unit="°",
                values=slope_inside,
                colorscale="magma_r",
                zmin=0,
                zmax=max(3.0, float(np.nanmax(slope_inside))),
                decimals=1,
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
        slope_classes = interactive.bar_chart(
            "Класи ухилу",
            [
                f"{label.uk} ({lo:g}–{hi:g}°)" if hi < 90 else f"{label.uk} (>{lo:g}°)"
                for _, lo, hi, label in SLOPE_CLASSES
            ],
            [c.share for c in metrics.slope_classes],
            SLOPE_CLASS_COLORS,
        )
        by_code = {a.code: a.share for a in metrics.aspect_of_slopes}
        rose = interactive.rose_chart(
            "Експозиція схилів (ухил від 1°)",
            [label.uk for _, label, _ in ASPECTS],
            [by_code.get(code, 0.0) for code, _, _ in ASPECTS],
        )

        e = metrics.elevation_m
        interactive.write_page(
            context.workdir / HTML_FILE,
            title=f"{context.parcel.name}: рельєф",
            subtitle=(
                f"{context.parcel.area_ha:.1f} га · висоти {e.min:.0f}–{e.max:.0f} м · "
                f"середній ухил {metrics.slope_deg.mean:.1f}° · Copernicus DEM 30 м"
            ),
            summary=summary,
            figures=[maps, slope_classes, rose],
            sources=sources,
        )
        interactive.write_figures_json(
            context.workdir / FIGURES_FILE, {"maps": maps, "slope_classes": slope_classes, "aspect": rose}
        )
        return [
            Artifact(
                path=HTML_FILE,
                media_type="text/html",
                description=(
                    "Інтерактивна сторінка: висота з горизонталями, ухил, NDVI, класи ухилу, експозиція."
                ),
            ),
            Artifact(
                path=FIGURES_FILE,
                media_type="application/json",
                description="Специфікації графіків Plotly (дані і вигляд) для plotly.js.",
            ),
        ]
