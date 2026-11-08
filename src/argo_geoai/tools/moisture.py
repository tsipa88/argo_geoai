import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import xarray as xr
from pydantic import Field
from rasterio.enums import Resampling
from rasterio.features import geometry_mask

from argo_geoai import interactive
from argo_geoai.analysis import zones as zoning
from argo_geoai.parcels import Parcel
from argo_geoai.schema import Artifact, DataSource, Finding, Label, Message, Model
from argo_geoai.sources import planetary, sentinel2
from argo_geoai.tools.base import RunContext, Tool, ToolOutput
from argo_geoai.tools.common import has_mixed_crops, vegetation_metrics_for
from argo_geoai.tools.vegetation import DAYS_PER_MONTH, plural_uk, uk_date
from argo_geoai.tools.water import WET_PART_MIN_OFFSET_M, locative_direction

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

VEGETATED_NDVI = 0.4
MIN_VEGETATED_SHARE = 0.3
DENSE_NDVI = 0.7
MIN_DENSE_SHARE = 0.3
RIPENING_NDVI_SHARE = 0.8
# NDMI is comparable on a closed canopy: a pixel is assessed when its NDVI is within this of its own season
# peak. Below it and before the peak the canopy is still closing (NDMI lags behind a fast green-up), after
# the peak it is ripening. Per pixel, because a field may hold a winter crop ripening in June next to
# a spring crop still growing.
PIXEL_CANOPY_TOLERANCE = 0.05
STRESS_CLASSES: list[tuple[str, float, float, Label, str]] = [
    ("strong", -np.inf, 0.1, Label(en="strong stress", uk="сильний стрес"), "#d7301f"),
    ("moderate", 0.1, 0.2, Label(en="moderate stress", uk="помірний стрес"), "#fdae61"),
    ("none", 0.2, np.inf, Label(en="enough water", uk="вологи достатньо"), "#4575b4"),
]
STRONG_SHARE_WARNING = 0.2
STRESSED_SHARE_NOTICE = 0.2
GREEN_NDVI = 0.6
RELATIVE_DRY_NDMI = 0.05
PERSISTENT_SHARE = 0.5
MIN_DATES_FOR_DRY_ZONE = 3
DRY_ZONE_NOTICE = 0.05
NEGLIGIBLE_SHARE = 0.01
NDMI_RANGE = (-0.2, 0.5)
NDVI_RANGE = (0.0, 0.9)
HTML_FILE = "moisture.html"
FIGURES_FILE = "moisture_figures.json"

Phase = Literal["bare", "sparse", "growing", "ripening"]


class MoistureParams(Model):
    months: int = Field(6, ge=1, le=36, description="period length, counted back from end_date")
    end_date: date | None = Field(None, description="last day of the period; today by default")
    max_scene_cloud: float = Field(20, ge=0, le=100, description="skip scenes cloudier than this, %")
    min_clear_share: float = Field(
        0.8, ge=0, le=1, description="a date counts when at least this share of the parcel is cloud-free"
    )


class NdmiStats(Model):
    mean: float
    median: float
    p10: float
    p90: float


class StressShare(Model):
    code: str
    label: Label
    min_ndmi: float | None
    max_ndmi: float | None
    share: float = Field(description=f"share of the dense canopy (NDVI >= {DENSE_NDVI}) of the date")


class MoistureDate(Model):
    date: date
    scene_id: str
    scene_cloud_cover: float
    clear_share: float
    counted: bool = Field(description="clear enough to count")
    vegetated_share: float | None = Field(
        None, description=f"share of the clear area with NDVI >= {VEGETATED_NDVI}"
    )
    dense_share: float | None = Field(None, description=f"share of the clear area with NDVI >= {DENSE_NDVI}")
    ndvi_mean: float | None = None
    ndmi: NdmiStats | None = Field(None, description="NDMI of the green pixels")
    phase: Phase | None = Field(
        None,
        description="bare: too little green; sparse: canopy not dense yet; ripening: after the NDVI peak",
    )
    stress: list[StressShare] = Field(default_factory=list, description="only for growing dates")
    stressed_share: float | None = Field(
        None, description="strong + moderate stress, share of the dense canopy"
    )
    developing_share: float | None = Field(
        None, description="share of the dense canopy left out: before its own NDVI peak, canopy still closing"
    )
    ripening_share: float | None = Field(
        None, description="share of the dense canopy left out: past its own NDVI peak (ripening crop)"
    )


class Offset(Model):
    north_m: float
    east_m: float


class DryZone(Model):
    share: float = Field(description="share of the green field drier than its NDVI explains on most dates")
    offset: Offset | None = Field(description="from the field centre to the dry part's centre")
    dates_used: int


class MoistureMetrics(Model):
    period_start: date
    period_end: date
    scenes_found: int
    dates_total: int
    dates_counted: int
    dates_assessed: int = Field(description="growing dates: dense canopy and not ripening")
    ndvi_peak: date | None
    worst: MoistureDate | None = Field(description="assessed date with the most water stress")
    latest: MoistureDate | None = Field(description="the latest assessed date")
    dry_zone: DryZone | None
    mixed_crops: bool = Field(
        description="vegetation found several crops: a drier part may be another crop, not a lack of water"
    )
    history: list[MoistureDate]


def percent(share: float) -> str:
    return "менш ніж 1%" if 0 < share < NEGLIGIBLE_SHARE else f"{share:.0%}"


def stress_of(record: MoistureDate, code: str) -> float:
    return next((s.share for s in record.stress if s.code == code), 0.0)


def worst_finding(worst: MoistureDate) -> Finding:
    strong, moderate = stress_of(worst, "strong"), stress_of(worst, "moderate")
    day = uk_date(worst.date)
    if strong >= STRONG_SHARE_WARNING:
        return Finding(
            code="water_stress_strong",
            level="warning",
            text=(
                f"Найсильніший водний стрес {day}: на {percent(strong)} густого посіву листя сухе "
                f"(NDMI нижче 0.1), ще на {percent(moderate)} — помірний стрес."
            ),
        )
    if strong + moderate >= STRESSED_SHARE_NOTICE:
        return Finding(
            code="water_stress_moderate",
            level="notice",
            text=(
                f"Найбільше ознак водного стресу {day}: помірний стрес на {percent(strong + moderate)} "
                "густого "
                f"посіву (NDMI нижче 0.2), сильного — {percent(strong)}."
            ),
        )
    return Finding(
        code="no_water_stress",
        level="info",
        text=(
            f"Ознак водного стресу за період не видно: навіть {day} вологи в листі достатньо "
            f"на {percent(stress_of(worst, 'none'))} густого посіву."
        ),
    )


def dry_zone_finding(zone: DryZone) -> Finding:
    if zone.share < DRY_ZONE_NOTICE:
        return Finding(
            code="no_dry_zone",
            level="info",
            text=(
                "Стійко сухіших частин поля немає: місця, де листя сухіше, ніж за його зеленістю, "
                "змінюються від дати до дати."
            ),
        )
    offset = zone.offset
    where = (
        f"переважно на {locative_direction(offset.north_m, offset.east_m)} поля"
        if offset and np.hypot(offset.north_m, offset.east_m) >= WET_PART_MIN_OFFSET_M
        else "місця розкидані по полю"
    )
    return Finding(
        code="dry_zone",
        level="notice",
        text=(
            f"На частині поля ({percent(zone.share)}) листя на більшості дат сухіше, ніж очікується за його "
            f"зеленістю, {where}. "
            "Такі місця висихають першими: схили, вершини, легші ґрунти — порівняйте з рельєфом і водою."
        ),
    )


def moisture_findings(metrics: MoistureMetrics) -> list[Finding]:
    if metrics.worst is None:
        return [
            Finding(
                code="no_green_dates",
                level="info",
                text=(
                    f"Немає дат, коли посів був густий (NDVI від {DENSE_NDVI} хоча б на "
                    f"{MIN_DENSE_SHARE:.0%} площі) і ще не дозрівав: водний стрес рослин оцінити неможливо. "
                    "На рідкому посіві NDMI низький через ґрунт між рослинами, а не через нестачу води."
                ),
            )
        ]
    findings = [
        Finding(
            code="assessed_dates",
            level="info",
            text=(
                "Вологість листя (NDMI) оцінено на "
                f"{plural_uk(metrics.dates_assessed, 'даті', 'датах', 'датах')} "
                f"з {metrics.dates_counted} чистих: коли посів густий (NDVI від {DENSE_NDVI}) "
                "і ще не дозріває."
            ),
        ),
        worst_finding(metrics.worst),
    ]
    worst = metrics.worst
    if (worst.ndvi_mean or 0) >= GREEN_NDVI and (worst.stressed_share or 0) >= STRESSED_SHARE_NOTICE:
        findings.append(
            Finding(
                code="green_but_dry",
                level="notice",
                text=(
                    f"{uk_date(worst.date)} поле ще зелене (NDVI {worst.ndvi_mean:.2f}), "
                    "але листю вже бракує "
                    "води: це ранній сигнал, NDVI зазвичай падає пізніше."
                ),
            )
        )
    latest = metrics.latest
    if latest is not None and latest.date != worst.date:
        findings.append(
            Finding(
                code="latest_moisture",
                level="info",
                text=(
                    f"Остання оцінена дата {uk_date(latest.date)}: водний стрес на "
                    f"{percent(latest.stressed_share or 0)} густого посіву, сильний — "
                    f"{percent(stress_of(latest, 'strong'))}."
                ),
            )
        )
    ripening = sum(1 for h in metrics.history if h.phase == "ripening")
    if ripening and metrics.ndvi_peak:
        findings.append(
            Finding(
                code="ripening_dates",
                level="info",
                text=(
                    f"Після піку вегетації {uk_date(metrics.ndvi_peak)} поле дозріває: природне висихання "
                    f"листя не є стресом, тому дати після піку ({ripening}) не оцінюються."
                ),
            )
        )
    if metrics.dry_zone is not None and metrics.mixed_crops:
        findings.append(
            Finding(
                code="dry_zone_mixed_crops",
                level="info",
                text=(
                    "На ділянці кілька культур (див. вегетацію): у різних культур листя різне, тому сухіші "
                    "частини поля тут не визначаються — це може бути інша культура, а не нестача води."
                ),
            )
        )
    elif metrics.dry_zone is not None:
        findings.append(dry_zone_finding(metrics.dry_zone))
    return findings


def ndmi_stats(values: FloatArray) -> NdmiStats:
    p10, p90 = np.percentile(values, [10, 90])
    return NdmiStats(
        mean=round(float(values.mean()), 3),
        median=round(float(np.median(values)), 3),
        p10=round(float(p10), 3),
        p90=round(float(p90), 3),
    )


def stress_shares(values: FloatArray) -> list[StressShare]:
    return [
        StressShare(
            code=code,
            label=label,
            min_ndmi=None if np.isinf(lo) else lo,
            max_ndmi=None if np.isinf(hi) else hi,
            share=round(float(((values >= lo) & (values < hi)).mean()), 3),
        )
        for code, lo, hi, label, _ in STRESS_CLASSES
    ]


def phases(records: list[MoistureDate]) -> tuple[date | None, dict[date, Phase]]:
    green = [r for r in records if r.counted and r.ndvi_mean is not None]
    if not green:
        return None, {}
    peak = max(green, key=lambda r: r.ndvi_mean or 0)
    peak_ndvi = peak.ndvi_mean or 0
    result: dict[date, Phase] = {}
    for record in green:
        if (record.vegetated_share or 0) < MIN_VEGETATED_SHARE:
            result[record.date] = "bare"
        elif (record.dense_share or 0) < MIN_DENSE_SHARE:
            result[record.date] = "sparse"
        elif record.date > peak.date and (record.ndvi_mean or 0) < RIPENING_NDVI_SHARE * peak_ndvi:
            result[record.date] = "ripening"
        else:
            result[record.date] = "growing"
    return peak.date, result


@dataclass(frozen=True)
class PixelPeaks:
    template: xr.DataArray
    value: FloatArray
    day: npt.NDArray[np.datetime64]


def on_grid(template: xr.DataArray, raster: xr.DataArray) -> FloatArray:
    if raster.shape != template.shape or not np.array_equal(raster.x, template.x):
        raster = raster.rio.reproject_match(template, resampling=Resampling.nearest)
    values: FloatArray = raster.values.astype("float64")
    return values


def pixel_peaks(scenes: dict[date, sentinel2.SceneMoisture]) -> PixelPeaks:
    days = sorted(scenes)
    template = zoning.stack({day: scenes[day].ndvi for day in days}).template
    cube = np.stack([on_grid(template, scenes[day].ndvi) for day in days], axis=-1)
    filled = np.where(np.isfinite(cube), cube, -np.inf)
    value = filled.max(axis=-1)
    value[~np.isfinite(value)] = np.nan
    return PixelPeaks(
        template=template, value=value, day=np.array(days, dtype="datetime64[D]")[filled.argmax(axis=-1)]
    )


@dataclass(frozen=True)
class ClosedCanopy:
    ndmi: FloatArray
    developing: float
    ripening: float


def closed_canopy_ndmi(scene: sentinel2.SceneMoisture, peaks: PixelPeaks) -> ClosedCanopy:
    ndvi, ndmi = on_grid(peaks.template, scene.ndvi), on_grid(peaks.template, scene.ndmi)
    dense = np.isfinite(ndmi) & (ndvi >= DENSE_NDVI)
    below_peak = dense & (ndvi < peaks.value - PIXEL_CANOPY_TOLERANCE)
    after_peak = np.datetime64(scene.day, "D") > peaks.day
    count = max(int(dense.sum()), 1)
    return ClosedCanopy(
        ndmi=ndmi[dense & ~below_peak],
        developing=float((below_peak & ~after_peak).sum() / count),
        ripening=float((below_peak & after_peak).sum() / count),
    )


def ndmi_residual(scene: sentinel2.SceneMoisture) -> xr.DataArray:
    ndvi, ndmi = scene.ndvi.values.astype("float64"), scene.ndmi.values.astype("float64")
    green = np.isfinite(ndvi) & np.isfinite(ndmi) & (ndvi >= VEGETATED_NDVI)
    residual = np.full(ndmi.shape, np.nan)
    if green.sum() >= 3 and np.ptp(ndvi[green]) > 0:
        slope, intercept = np.polyfit(ndvi[green], ndmi[green], 1)
        residual[green] = ndmi[green] - (slope * ndvi[green] + intercept)
    result: xr.DataArray = scene.ndmi.copy(data=residual.astype("float32"))
    return result


class MoistureTool(Tool[MoistureParams, MoistureMetrics]):
    name = "moisture"
    version = "1.0.0"
    title = Label(en="Water stress of plants (NDMI)", uk="Водний стрес рослин (NDMI)")
    maps_title = "Карти вологості листя (NDMI)"
    params_model = MoistureParams
    metrics_model = MoistureMetrics

    def run(self, params: MoistureParams, context: RunContext) -> ToolOutput[MoistureMetrics]:
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
        history: list[MoistureDate] = []
        scenes: dict[date, sentinel2.SceneMoisture] = {}
        for item in sentinel2.least_cloudy_per_day(items):
            try:
                scene = sentinel2.scene_moisture(item, parcel)
            except Exception as error:  # one broken scene must not stop the whole season
                log.warning("scene %s failed: %s", item.id, error)
                warnings.append(Message(code="scene_failed", text=f"{item.id}: {error}"))
                continue
            if scene is None:
                continue
            history.append(self._describe(scene, params))
            if history[-1].counted:
                scenes[scene.day] = scene
            log.info("%s clear %.0f%%", scene.day, scene.clear_share * 100)

        peak, phase_of = phases(history)
        peaks = pixel_peaks(scenes) if scenes else None
        history = [self._assess(record, phase_of.get(record.date), scenes, peaks) for record in history]
        assessed = [r for r in history if r.phase == "growing"]
        worst = max(
            assessed,
            key=lambda r: (stress_of(r, "strong") >= STRONG_SHARE_WARNING, r.stressed_share or 0),
            default=None,
        )
        dry_zone, dry_frequency = self._dry_zone({r.date: scenes[r.date] for r in assessed})
        mixed_crops = has_mixed_crops(vegetation_metrics_for(context, warnings))

        metrics = MoistureMetrics(
            period_start=start,
            period_end=end,
            scenes_found=len(items),
            dates_total=len(history),
            dates_counted=len(scenes),
            dates_assessed=len(assessed),
            ndvi_peak=peak,
            worst=worst,
            latest=assessed[-1] if assessed else None,
            dry_zone=dry_zone,
            mixed_crops=mixed_crops,
            history=history,
        )
        summary = moisture_findings(metrics)
        sources = [
            sentinel2.DATA_SOURCE.model_copy(
                update={"items": [h.scene_id for h in history], "resolution_m": 20}
            )
        ]
        artifacts = self._interactive(
            context=context,
            metrics=metrics,
            scenes=scenes,
            dry_frequency=None if mixed_crops else dry_frequency,
            summary=summary,
            sources=sources,
        )
        return ToolOutput(
            metrics=metrics,
            summary=summary,
            artifacts=artifacts,
            data_sources=sources,
            warnings=warnings,
            partial=any(w.code == "scene_failed" for w in warnings),
        )

    def _describe(self, scene: sentinel2.SceneMoisture, params: MoistureParams) -> MoistureDate:
        ndvi = scene.ndvi.values.astype("float64")
        clear = np.isfinite(ndvi) & np.isfinite(scene.ndmi.values)
        counted = scene.clear_share >= params.min_clear_share and bool(clear.any())
        green = clear & (ndvi >= VEGETATED_NDVI)
        dense = clear & (ndvi >= DENSE_NDVI)
        ndmi = scene.ndmi.values.astype("float64")
        return MoistureDate(
            date=scene.day,
            scene_id=scene.item_id,
            scene_cloud_cover=round(scene.scene_cloud_cover, 2),
            clear_share=round(scene.clear_share, 3),
            counted=counted,
            vegetated_share=round(float(green.sum() / clear.sum()), 3) if clear.any() else None,
            dense_share=round(float(dense.sum() / clear.sum()), 3) if clear.any() else None,
            ndvi_mean=round(float(ndvi[clear].mean()), 3) if clear.any() else None,
            ndmi=ndmi_stats(ndmi[green]) if green.any() else None,
        )

    def _assess(
        self,
        record: MoistureDate,
        phase: Phase | None,
        scenes: dict[date, sentinel2.SceneMoisture],
        peaks: PixelPeaks | None,
    ) -> MoistureDate:
        if phase != "growing" or peaks is None:
            return record.model_copy(update={"phase": phase})
        canopy = closed_canopy_ndmi(scenes[record.date], peaks)
        left_out = {
            "developing_share": round(canopy.developing, 3),
            "ripening_share": round(canopy.ripening, 3),
        }
        if (record.dense_share or 0) * (1 - canopy.developing - canopy.ripening) < MIN_DENSE_SHARE:
            late = canopy.ripening > canopy.developing
            return record.model_copy(update={"phase": "ripening" if late else "sparse", **left_out})
        stress = stress_shares(canopy.ndmi)
        none = next(s.share for s in stress if s.code == "none")
        return record.model_copy(
            update={
                "phase": phase,
                "stress": stress,
                "stressed_share": round(1 - none, 3),
                **left_out,
            }
        )

    def _dry_zone(
        self, scenes: dict[date, sentinel2.SceneMoisture]
    ) -> tuple[DryZone | None, xr.DataArray | None]:
        if len(scenes) < MIN_DATES_FOR_DRY_ZONE:
            return None, None
        cube = zoning.stack({day: ndmi_residual(s) for day, s in scenes.items()})
        values = cube.values
        seen = np.isfinite(values)
        dry = seen & (values < -RELATIVE_DRY_NDMI)
        frequency = dry.sum(axis=1) / np.maximum(seen.sum(axis=1), 1)
        persistent = frequency >= PERSISTENT_SHARE
        if not len(frequency):
            return None, None

        raster = np.full(cube.template.shape, np.nan)
        raster[cube.rows, cube.cols] = frequency
        frequency_map = xr.DataArray(raster, coords=cube.template.coords, dims=cube.template.dims)
        frequency_map = frequency_map.rio.write_crs(cube.template.rio.crs)

        offset = None
        if persistent.any():
            y, x = cube.template.y.values, cube.template.x.values
            offset = Offset(
                north_m=round(float(y[cube.rows[persistent]].mean() - y[cube.rows].mean())),
                east_m=round(float(x[cube.cols[persistent]].mean() - x[cube.cols].mean())),
            )
        zone = DryZone(share=round(float(persistent.mean()), 3), offset=offset, dates_used=len(scenes))
        return zone, frequency_map

    def _interactive(
        self,
        *,
        context: RunContext,
        metrics: MoistureMetrics,
        scenes: dict[date, sentinel2.SceneMoisture],
        dry_frequency: xr.DataArray | None,
        summary: list[Finding],
        sources: list[DataSource],
    ) -> list[Artifact]:
        parcel = context.parcel
        figures: dict[str, go.Figure] = {}
        with_ndmi = [h for h in metrics.history if h.counted and h.ndmi is not None]
        if with_ndmi:
            figures["timeseries"] = interactive.line_chart(
                "Вода в листі (NDMI) і зеленість (NDVI) за період",
                [
                    interactive.Series(
                        name="NDMI зелених рослин",
                        x=[h.date.isoformat() for h in with_ndmi],
                        y=[h.ndmi.mean for h in with_ndmi if h.ndmi],
                        color="#2171b5",
                        band=(
                            [h.ndmi.p10 for h in with_ndmi if h.ndmi],
                            [h.ndmi.p90 for h in with_ndmi if h.ndmi],
                        ),
                        band_name="NDMI 10–90% поля",
                    ),
                    interactive.Series(
                        name="NDVI поля",
                        x=[h.date.isoformat() for h in with_ndmi],
                        y=[h.ndvi_mean or 0 for h in with_ndmi],
                        color="#1b7837",
                    ),
                ],
                y_title="індекс",
                y_range=(-0.3, 0.9),
            )
        assessed = [h for h in metrics.history if h.phase == "growing"]
        if assessed:
            figures["stress"] = interactive.stacked_bar_chart(
                "Водний стрес густого посіву по датах",
                [uk_date(h.date) for h in assessed],
                [
                    (label.uk, [stress_of(h, code) for h in assessed], color)
                    for code, _, _, label, color in STRESS_CLASSES
                ],
                y_title="% густого посіву",
            )
        if scenes:
            figures.update(self._maps(parcel, metrics, scenes, dry_frequency))

        interactive.write_page(
            context.workdir / HTML_FILE,
            title=f"{parcel.name}: водний стрес рослин",
            subtitle=(
                f"{parcel.area_ha:.1f} га · {uk_date(metrics.period_start)}–{uk_date(metrics.period_end)} · "
                f"оцінено дат {metrics.dates_assessed} з {metrics.dates_counted} чистих · Sentinel-2 20 м"
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
                    "Інтерактивна сторінка: NDMI і NDVI за сезоном, стрес по датах, карти, сухіші частини."
                ),
            ),
            Artifact(
                path=FIGURES_FILE,
                media_type="application/json",
                description="Специфікації графіків Plotly (дані і вигляд) для plotly.js.",
            ),
        ]

    def _maps(
        self,
        parcel: Parcel,
        metrics: MoistureMetrics,
        scenes: dict[date, sentinel2.SceneMoisture],
        dry_frequency: xr.DataArray | None,
    ) -> dict[str, go.Figure]:
        template = zoning.stack({day: s.ndmi for day, s in scenes.items()}).template
        inside = ~geometry_mask(
            parcel.in_crs(template.rio.crs).geometry,
            out_shape=template.shape,
            transform=template.rio.transform(),
        )
        grid = interactive.grid_geometry(template.x.values, template.y.values, template.rio.crs, inside)

        def masked(raster: xr.DataArray) -> FloatArray:
            values: FloatArray = np.where(inside, on_grid(template, raster), np.nan)
            return values

        def ndmi_layer(day: date, note: str) -> interactive.MapLayer:
            return interactive.MapLayer(
                title=f"NDMI {uk_date(day)} ({note})",
                hover_label=f"NDMI {uk_date(day)}",
                unit="",
                values=masked(scenes[day].ndmi),
                colorscale="RdYlBu",
                zmin=NDMI_RANGE[0],
                zmax=NDMI_RANGE[1],
            )

        layers: list[interactive.MapLayer] = []
        if metrics.worst:
            day = metrics.worst.date
            layers.append(ndmi_layer(day, "найсухіша оцінена дата"))
            layers.append(
                interactive.MapLayer(
                    title=f"NDVI {uk_date(day)}",
                    hover_label=f"NDVI {uk_date(day)}",
                    unit="",
                    values=masked(scenes[day].ndvi),
                    colorscale="RdYlGn",
                    zmin=NDVI_RANGE[0],
                    zmax=NDVI_RANGE[1],
                )
            )
        if metrics.latest and (metrics.worst is None or metrics.latest.date != metrics.worst.date):
            layers.append(ndmi_layer(metrics.latest.date, "остання оцінена дата"))
        if dry_frequency is not None:
            frequency = masked(dry_frequency)
            layers.append(
                interactive.MapLayer(
                    title="Частка дат, коли листя сухіше за очікуване",
                    hover_label="Частка дат, коли сухіше",
                    unit="",
                    values=frequency,
                    colorscale="YlOrRd",
                    zmin=0,
                    zmax=1,
                    overlay=np.nan_to_num(frequency) >= PERSISTENT_SHARE,
                )
            )
        figures: dict[str, go.Figure] = {}
        if layers:
            figures["maps"] = interactive.map_panels(parcel, grid, layers)
        figures["by_date"] = interactive.date_slider_map(
            parcel,
            grid,
            title="NDMI по датах (повзунок під картою)",
            frames={uk_date(day): masked(scenes[day].ndmi) for day in sorted(scenes)},
            colorscale="RdYlBu",
            zmin=NDMI_RANGE[0],
            zmax=NDMI_RANGE[1],
        )
        return figures
