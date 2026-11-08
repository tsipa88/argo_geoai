import logging
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import geometry_mask

from argo_geoai import interactive, render
from argo_geoai.analysis import zones as zoning
from argo_geoai.schema import Artifact, DataSource, Finding, Label, Message
from argo_geoai.sources import planetary, sentinel2, worldcover
from argo_geoai.tools.base import RunContext, ToolOutput
from argo_geoai.tools.vegetation import (
    DAYS_PER_MONTH,
    Gap,
    LandCoverShare,
    NdviDate,
    SeasonPoint,
    VegetationMetrics,
    VegetationParams,
    VegetationTool,
    describe_zones,
    season_findings,
    stats,
    uk_date,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

log = logging.getLogger(__name__)

HTML_FILE = "vegetation.html"
FIGURES_FILE = "vegetation_figures.json"
NDVI_RANGE = (0.0, 0.9)


# run is a copy of VegetationTool.run with another output step: metrics and summary must stay identical
class VegetationV2Tool(VegetationTool):
    name = "vegetation_v2"
    version = "1.0.0"
    maps_title = "Карти NDVI і зон поля"
    title = Label(
        en="Vegetation (NDVI) and land cover, interactive",
        uk="Вегетація (NDVI) і покрив ділянки, інтерактивно",
    )

    def run(self, params: VegetationParams, context: RunContext) -> ToolOutput[VegetationMetrics]:
        parcel = context.parcel
        end = params.end_date or datetime.now(UTC).astimezone().date()
        start = end - timedelta(days=params.months * DAYS_PER_MONTH)

        items = planetary.search(
            sentinel2.COLLECTION,
            parcel,
            time_range=f"{start.isoformat()}/{end.isoformat()}",
            query={"eo:cloud_cover": {"lt": params.max_scene_cloud}},
        )

        warnings: list[Message] = []
        history: list[NdviDate] = []
        rasters: dict[date, xr.DataArray] = {}
        for item in sentinel2.least_cloudy_per_day(items):
            try:
                scene = sentinel2.scene_ndvi(item, parcel)
            except Exception as error:  # one broken scene must not stop the whole history
                log.warning("scene %s failed: %s", item.id, error)
                warnings.append(Message(code="scene_failed", text=f"{item.id}: {error}"))
                continue
            if scene is None:
                continue
            values = scene.values
            counted = scene.clear_share >= params.min_clear_share and values.size > 0
            history.append(
                NdviDate(
                    date=scene.day,
                    scene_id=scene.item_id,
                    scene_cloud_cover=round(scene.scene_cloud_cover, 2),
                    clear_share=round(scene.clear_share, 3),
                    parcel_area_ha=round(scene.parcel_pixels * scene.pixel_area_m2 / 10_000, 2),
                    counted=counted,
                    ndvi=stats(values) if values.size else None,
                )
            )
            if counted:
                rasters[scene.day] = scene.ndvi
            log.info("%s clear %.0f%%", scene.day, scene.clear_share * 100)

        season = [SeasonPoint(date=h.date, ndvi=h.ndvi) for h in history if h.counted and h.ndvi]
        peak = max(season, key=lambda p: p.ndvi.mean, default=None)
        widest = max(season, key=lambda p: p.ndvi.spread, default=None)
        latest = season[-1] if season else None
        gaps = [Gap(start=a.date, end=b.date, days=(b.date - a.date).days) for a, b in pairwise(season)]

        zones, silhouette, zone_map = describe_zones(rasters)

        cover, cover_items = worldcover.land_cover(parcel)
        if not cover:
            warnings.append(Message(code="no_land_cover", text="ESA WorldCover does not cover the parcel"))

        metrics = VegetationMetrics(
            period_start=start,
            period_end=end,
            scenes_found=len(items),
            dates_total=len(history),
            dates_counted=len(season),
            peak=peak,
            max_spread=widest,
            latest=latest,
            change_peak_to_latest=round(latest.ndvi.mean - peak.ndvi.mean, 3) if peak and latest else None,
            longest_gap=max(gaps, key=lambda g: g.days, default=None),
            history=history,
            land_cover=[
                LandCoverShare(code=c.code, label=c.label, share=round(c.share, 3))
                for c in cover
                if round(c.share, 3) > 0
            ],
            zones=zones,
            zoning_silhouette=silhouette,
        )

        summary = season_findings(metrics)
        s2_source = sentinel2.DATA_SOURCE.model_copy(update={"items": [h.scene_id for h in history]})
        cover_source = worldcover.DATA_SOURCE.model_copy(update={"items": cover_items})
        sources = [s2_source, cover_source]
        artifacts = self._interactive(
            context=context,
            metrics=metrics,
            season=season,
            rasters=rasters,
            zone_map=zone_map,
            summary=summary,
            sources=sources,
        )
        return ToolOutput(
            metrics=metrics,
            summary=summary,
            artifacts=artifacts,
            data_sources=sources,
            warnings=warnings,
            partial=bool(warnings) or peak is None,
        )

    def _interactive(
        self,
        *,
        context: RunContext,
        metrics: VegetationMetrics,
        season: list[SeasonPoint],
        rasters: dict[date, xr.DataArray],
        zone_map: interactive.FloatArray | None,
        summary: list[Finding],
        sources: list[DataSource],
    ) -> list[Artifact]:
        figures: dict[str, go.Figure] = {}
        parcel = context.parcel

        if season:
            figures["timeseries"] = interactive.line_chart(
                "NDVI за період",
                [
                    interactive.Series(
                        name="середнє NDVI",
                        x=[p.date.isoformat() for p in season],
                        y=[p.ndvi.mean for p in season],
                        color="#1b7837",
                        band=([p.ndvi.p10 for p in season], [p.ndvi.p90 for p in season]),
                        band_name="10–90% поля",
                    )
                ],
                y_title="NDVI",
                y_range=NDVI_RANGE,
            )

        if rasters:
            template = zoning.stack(rasters).template
            inside = ~geometry_mask(
                parcel.in_crs(template.rio.crs).geometry,
                out_shape=template.shape,
                transform=template.rio.transform(),
            )
            grid = interactive.grid_geometry(template.x.values, template.y.values, template.rio.crs, inside)

            def on_grid(day: date) -> interactive.FloatArray:
                raster = rasters[day]
                if raster.shape != template.shape or not np.array_equal(raster.x, template.x):
                    raster = raster.rio.reproject_match(template, resampling=Resampling.nearest)
                values: interactive.FloatArray = np.where(inside, raster.values.astype("float64"), np.nan)
                return values

            layers: list[interactive.MapLayer] = []
            shown: set[date] = set()
            for title, point in (
                ("пік", metrics.peak),
                ("найбільший розкид", metrics.max_spread),
                ("остання дата", metrics.latest),
            ):
                if point is None or point.date in shown:
                    continue
                shown.add(point.date)
                layers.append(
                    interactive.MapLayer(
                        title=f"NDVI {uk_date(point.date)} ({title})",
                        hover_label=f"NDVI {uk_date(point.date)}",
                        unit="",
                        values=on_grid(point.date),
                        colorscale="RdYlGn",
                        zmin=NDVI_RANGE[0],
                        zmax=NDVI_RANGE[1],
                    )
                )
            if len(metrics.zones) > 1 and zone_map is not None:
                count = len(metrics.zones)
                layers.append(
                    interactive.MapLayer(
                        title="Зони поля за динамікою NDVI",
                        hover_label="Зона",
                        unit="",
                        values=np.where(inside, zone_map, np.nan),
                        colorscale="tab10",
                        zmin=0.5,
                        zmax=count + 0.5,
                        decimals=0,
                        plotly_colorscale=interactive.categorical_colorscale(render.ZONE_COLORS[:count]),
                        ticks=(
                            [float(z.id) for z in metrics.zones],
                            [f"{z.id}: {z.role_label.uk} ({z.share:.0%})" for z in metrics.zones],
                        ),
                    )
                )
            figures["maps"] = interactive.map_panels(parcel, grid, layers)
            figures["by_date"] = interactive.date_slider_map(
                parcel,
                grid,
                title="NDVI по датах (повзунок під картою)",
                frames={uk_date(day): on_grid(day) for day in sorted(rasters)},
                colorscale="RdYlGn",
                zmin=NDVI_RANGE[0],
                zmax=NDVI_RANGE[1],
            )

        if len(metrics.zones) > 1:
            figures["zones"] = interactive.line_chart(
                "NDVI по зонах",
                [
                    interactive.Series(
                        name=f"Зона {z.id} ({z.role_label.uk}, {z.share:.0%})",
                        x=[p.date.isoformat() for p in z.curve],
                        y=[p.ndvi for p in z.curve],
                        color=render.ZONE_COLORS[(z.id - 1) % len(render.ZONE_COLORS)],
                    )
                    for z in metrics.zones
                ],
                y_title="NDVI",
                y_range=NDVI_RANGE,
            )
        if metrics.land_cover:
            figures["land_cover"] = interactive.bar_chart(
                "Покрив ділянки (ESA WorldCover)",
                [c.label.uk for c in metrics.land_cover],
                [c.share for c in metrics.land_cover],
                ["#4daf4a"] * len(metrics.land_cover),
            )

        interactive.write_page(
            context.workdir / HTML_FILE,
            title=f"{parcel.name}: вегетація (NDVI)",
            subtitle=(
                f"{parcel.area_ha:.1f} га · {uk_date(metrics.period_start)}–{uk_date(metrics.period_end)} · "
                f"чистих дат {metrics.dates_counted} з {metrics.dates_total} · Sentinel-2 10 м"
            ),
            summary=summary,
            figures=list(figures.values()),
            sources=sources,
        )
        interactive.write_figures_json(context.workdir / FIGURES_FILE, figures)
        return [
            Artifact(
                path=HTML_FILE,
                media_type="text/html",
                description=(
                    "Інтерактивна сторінка: NDVI за сезоном, карти NDVI і зон, повзунок по датах, покрив."
                ),
            ),
            Artifact(
                path=FIGURES_FILE,
                media_type="application/json",
                description="Специфікації графіків Plotly (дані і вигляд) для plotly.js.",
            ),
        ]
