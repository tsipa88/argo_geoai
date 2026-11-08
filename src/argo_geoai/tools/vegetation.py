import logging
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import Field

from argo_geoai import render
from argo_geoai.analysis import zones as zoning
from argo_geoai.schema import Artifact, Finding, Label, Message, Model
from argo_geoai.sources import planetary, sentinel2, worldcover
from argo_geoai.tools.base import RunContext, Tool, ToolOutput

log = logging.getLogger(__name__)

DAYS_PER_MONTH = 31


class VegetationParams(Model):
    months: int = Field(6, ge=1, le=36, description="period length, counted back from end_date")
    end_date: date | None = Field(None, description="last day of the period; today by default")
    max_scene_cloud: float = Field(20, ge=0, le=100, description="skip scenes cloudier than this, %")
    min_clear_share: float = Field(
        0.8,
        ge=0,
        le=1,
        description="a date counts in the season only if this share of the parcel is cloud-free",
    )


class NdviStats(Model):
    mean: float
    median: float
    p10: float
    p90: float
    spread: float = Field(description="p90 - p10: how uneven the parcel is")


class NdviDate(Model):
    date: date
    scene_id: str
    scene_cloud_cover: float
    clear_share: float
    parcel_area_ha: float
    counted: bool = Field(description="clear enough to count in the season")
    ndvi: NdviStats | None


class SeasonPoint(Model):
    date: date
    ndvi: NdviStats


class Gap(Model):
    start: date
    end: date
    days: int


class LandCoverShare(Model):
    code: str
    label: Label
    share: float


EARLY_PEAK_LAST_MONTH = 5
LATE_PEAK_FROM = (6, 15)
LATE_START_MAX_NDVI = 0.35
STRONG_PEAK_NDVI = 0.7
MODERATE_PEAK_NDVI = 0.5
DEVELOPMENT_LABELS: dict[str, Label] = {
    "strong": Label(en="strong", uk="сильний"),
    "moderate": Label(en="moderate", uk="помірний"),
    "weak": Label(en="weak", uk="слабкий"),
}
SAME_CROP_MIN_CORRELATION = 0.85
COMPACT_ZONE_SHARE = 0.7
ROLE_LABELS: dict[str, Label] = {
    "main": Label(en="main", uk="основна"),
    "separate_crop": Label(en="separate crop", uk="інша культура або строк"),
    "weaker_area": Label(en="weaker area", uk="слабша ділянка"),
    "stronger_area": Label(en="stronger area", uk="сильніша ділянка"),
}
TIMING_LABELS: dict[str, Label] = {
    "early": Label(en="early", uk="рання"),
    "late": Label(en="late", uk="пізня"),
    "other": Label(en="other", uk="інша"),
}


class ZonePoint(Model):
    date: date
    ndvi: float


class Zone(Model):
    id: int = Field(description="1 is the largest zone")
    share: float
    area_ha: float
    timing: Literal["early", "late", "other"] = Field(
        description="early: peak by the end of May; late: bare start, peak from mid-June; other: the rest"
    )
    timing_label: Label
    development: Literal["strong", "moderate", "weak"] = Field(description="strength of the zone peak")
    development_label: Label
    role: Literal["main", "separate_crop", "weaker_area", "stronger_area"] = Field(
        description="relation to zone 1: another crop (different curve) or the same crop, weaker or stronger"
    )
    role_label: Label
    correlation_with_main: float | None = Field(
        description="Pearson correlation of the NDVI curve with zone 1"
    )
    patches: int = Field(description=f"separate patches on the map (from {zoning.MIN_PATCH_HA} ha)")
    largest_patch_share: float
    first: ZonePoint
    peak: ZonePoint
    latest: ZonePoint
    curve: list[ZonePoint]


class VegetationMetrics(Model):
    period_start: date
    period_end: date
    scenes_found: int
    dates_total: int
    dates_counted: int
    peak: SeasonPoint | None
    max_spread: SeasonPoint | None
    latest: SeasonPoint | None
    change_peak_to_latest: float | None
    longest_gap: Gap | None
    history: list[NdviDate]
    land_cover: list[LandCoverShare]
    zones: list[Zone] = Field(
        default_factory=list,
        description="parts of the parcel with different NDVI curves; one zone means a uniform field",
    )
    zoning_silhouette: float | None = Field(None, description="how clear-cut the split into zones is, 0..1")


def stats(values: npt.NDArray[np.float64]) -> NdviStats:
    p10, p90 = np.percentile(values, [10, 90])
    return NdviStats(
        mean=round(float(values.mean()), 3),
        median=round(float(np.median(values)), 3),
        p10=round(float(p10), 3),
        p90=round(float(p90), 3),
        spread=round(float(p90 - p10), 3),
    )


def plural_uk(count: int, one: str, few: str, many: str) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return f"{count} {one}"
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return f"{count} {few}"
    return f"{count} {many}"


def uk_date(day: date) -> str:
    return day.strftime("%d.%m.%Y")


def season_timing(curve: list[ZonePoint]) -> Literal["early", "late", "other"]:
    first, peak = curve[0], max(curve, key=lambda point: point.ndvi)
    if first.date.month > EARLY_PEAK_LAST_MONTH:
        return "other"
    if peak.date.month <= EARLY_PEAK_LAST_MONTH:
        return "early"
    if first.ndvi <= LATE_START_MAX_NDVI and (peak.date.month, peak.date.day) >= LATE_PEAK_FROM:
        return "late"
    return "other"


def development(peak_ndvi: float) -> Literal["strong", "moderate", "weak"]:
    if peak_ndvi >= STRONG_PEAK_NDVI:
        return "strong"
    if peak_ndvi >= MODERATE_PEAK_NDVI:
        return "moderate"
    return "weak"


def zone_role(
    curve: list[ZonePoint], main: list[ZonePoint] | None
) -> tuple[Literal["main", "separate_crop", "weaker_area", "stronger_area"], float | None]:
    if main is None:
        return "main", None
    main_by_day = {point.date: point.ndvi for point in main}
    pairs = [(point.ndvi, main_by_day[point.date]) for point in curve if point.date in main_by_day]
    if len(pairs) < 3:
        return "separate_crop", None
    values, reference = np.array(pairs).T
    correlation = float(np.corrcoef(values, reference)[0, 1]) if values.std() and reference.std() else 0.0
    if correlation < SAME_CROP_MIN_CORRELATION:
        return "separate_crop", round(correlation, 3)
    return ("weaker_area" if values.mean() < reference.mean() else "stronger_area"), round(correlation, 3)


def describe_zones(
    rasters: dict[date, xr.DataArray],
) -> tuple[list[Zone], float | None, npt.NDArray[np.float64] | None]:
    if len(rasters) < 2:
        return [], None, None
    ndvi_stack = zoning.stack(rasters)
    split = zoning.find_zones(ndvi_stack.values)
    raster = zoning.zone_raster(ndvi_stack, split)
    zones: list[Zone] = []
    for index in range(split.count):
        pixels = ndvi_stack.values[split.labels == index]
        curve = [
            ZonePoint(date=day, ndvi=round(float(np.nanmean(pixels[:, column])), 3))
            for column, day in enumerate(ndvi_stack.dates)
            if np.isfinite(pixels[:, column]).any()
        ]
        timing = season_timing(curve)
        peak = max(curve, key=lambda point: point.ndvi)
        role, correlation = zone_role(curve, zones[0].curve if zones else None)
        spots = zoning.patches(raster, index + 1, ndvi_stack.pixel_area_m2)
        zones.append(
            Zone(
                id=index + 1,
                share=round(len(pixels) / len(ndvi_stack.values), 3),
                area_ha=round(len(pixels) * ndvi_stack.pixel_area_m2 / 10_000, 1),
                timing=timing,
                timing_label=TIMING_LABELS[timing],
                first=curve[0],
                peak=peak,
                development=development(peak.ndvi),
                development_label=DEVELOPMENT_LABELS[development(peak.ndvi)],
                role=role,
                role_label=ROLE_LABELS[role],
                correlation_with_main=correlation,
                patches=spots.count,
                largest_patch_share=spots.largest_share,
                latest=curve[-1],
                curve=curve,
            )
        )
    return zones, split.silhouette, raster


def zone_text(zone: Zone, main: Zone) -> str:
    size = f"{zone.share:.0%} поля, {zone.area_ha:.0f} га"
    if zone.role == "main":
        text = (
            f"Зона {zone.id} — основна ({size}): {uk_date(zone.first.date)} NDVI {zone.first.ndvi:.2f}, "
            f"пік {zone.peak.ndvi:.2f} {uk_date(zone.peak.date)} ({zone.development_label.uk} розвиток), "
            f"на {uk_date(zone.latest.date)} — {zone.latest.ndvi:.2f}."
        )
    elif zone.role == "separate_crop":
        text = (
            f"Зона {zone.id} ({size}): інша динаміка, ніж у основної зони — ймовірно, інша культура "
            f"або строк сівби. {uk_date(zone.first.date)} NDVI {zone.first.ndvi:.2f}, "
            f"пік {zone.peak.ndvi:.2f} {uk_date(zone.peak.date)} ({zone.development_label.uk} розвиток), "
            f"на {uk_date(zone.latest.date)} — {zone.latest.ndvi:.2f}."
        )
    else:
        weaker = zone.role == "weaker_area"
        text = (
            f"Зона {zone.id} ({size}): та сама динаміка, що й у основної зони, але "
            f"{'слабша' if weaker else 'сильніша'} — пік {zone.peak.ndvi:.2f} {uk_date(zone.peak.date)} "
            f"проти {main.peak.ndvi:.2f} ({zone.development_label.uk} розвиток)."
        )
        if zone.largest_patch_share >= COMPACT_ZONE_SHARE:
            text += " Зосереджена переважно в одному місці."
        else:
            spots = plural_uk(zone.patches, "окрема ділянка", "окремі ділянки", "окремих ділянок")
            text += f" Розкидана по полю: {spots}."
        if weaker:
            text += (
                " Типово для проблемних місць (ґрунт, рельєф, волога, пошкодження посівів), "
                "а не іншої культури."
            )
        return text

    if zone.timing == "early":
        text += " Рання вегетація (пік до кінця травня) типова для озимих або багаторічних культур."
    elif zone.timing == "late":
        text += (
            " Пізній старт і пік від середини червня типові для ярих просапних культур (соняшник, кукурудза)."
        )
    return text


def zone_findings(zones: list[Zone]) -> list[Finding]:
    if len(zones) < 2:
        return []
    one_crop = all(zone.role in {"main", "weaker_area", "stronger_area"} for zone in zones)
    headline = (
        "Поле, ймовірно, під однією культурою, але розвивалось нерівномірно: "
        f"{plural_uk(len(zones), 'зона', 'зони', 'зон')} "
        "з різною силою вегетації. Кожну зону описано окремо."
        if one_crop
        else (
            f"Поле неоднорідне за сезоном: {plural_uk(len(zones), 'зона', 'зони', 'зон')} "
            "з різною динамікою NDVI. "
            "Середні значення по полю їх змішують, тому кожну зону описано окремо."
        )
    )
    findings = [Finding(code="zones", level="notice", text=headline)]
    for zone in zones:
        level: Literal["info", "notice", "warning"] = "notice" if zone.role == "weaker_area" else "info"
        findings.append(Finding(code=f"zone_{zone.id}", level=level, text=zone_text(zone, zones[0])))
    return findings


def peak_finding(peak: SeasonPoint, *, zoned: bool) -> Finding:
    if zoned:
        level, verdict = "notice", "це середнє по полю з різними частинами, див. зони"
    elif peak.ndvi.mean >= STRONG_PEAK_NDVI:
        level, verdict = "info", "густа, добре розвинена рослинність"
    elif peak.ndvi.mean >= MODERATE_PEAK_NDVI:
        level, verdict = "notice", "помірно розвинена рослинність"
    else:
        level, verdict = "warning", "слабкий розвиток рослинності за весь період"
    return Finding(
        code="peak",
        level=level,
        text=f"Пік вегетації {uk_date(peak.date)}: NDVI {peak.ndvi.mean:.2f} — {verdict}.",
    )


def season_findings(metrics: VegetationMetrics) -> list[Finding]:
    findings: list[Finding] = []

    if metrics.land_cover:
        main = metrics.land_cover[0]
        findings.append(
            Finding(
                code="land_cover_main",
                level="info",
                text=f"Основний тип покриву ділянки — {main.label.uk.lower()} ({main.share:.0%}).",
            )
        )

    if metrics.peak is None:
        findings.append(
            Finding(
                code="no_clear_dates",
                level="warning",
                text="За період немає жодної дати, коли ділянка була б достатньо вільна від хмар.",
            )
        )
        return findings

    findings.append(
        Finding(
            code="clear_dates",
            level="info",
            text=(
                f"Знімків за період: {metrics.scenes_found}, з них дат над ділянкою: {metrics.dates_total}, "
                f"чистих від хмар: {metrics.dates_counted}."
            ),
        )
    )

    peak = metrics.peak
    findings.append(peak_finding(peak, zoned=len(metrics.zones) > 1))

    if peak.ndvi.spread >= 0.15:
        findings.append(
            Finding(
                code="uneven_at_peak",
                level="notice",
                text=(
                    f"На піку поле неоднорідне: у 10% поля NDVI нижче {peak.ndvi.p10:.2f}, "
                    f"а в найкращих 10% вище {peak.ndvi.p90:.2f}."
                ),
            )
        )

    widest = metrics.max_spread
    if widest is not None and widest.ndvi.spread >= 0.25:
        findings.append(
            Finding(
                code="max_spread",
                level="notice",
                text=(
                    f"Найбільша неоднорідність {uk_date(widest.date)} "
                    f"(розкид NDVI {widest.ndvi.spread:.2f}): "
                    "частини поля розвиваються або дозрівають неодночасно."
                ),
            )
        )

    latest = metrics.latest
    if latest is not None and latest.date != peak.date:
        if latest.ndvi.mean < 0.3 and peak.ndvi.mean >= 0.5:
            state = "рослинності майже немає, ймовірно після збирання або висихання"
        elif latest.ndvi.mean >= 0.5:
            state = "зелена рослинність"
        else:
            state = "рослинність слабка або згасає"
        findings.append(
            Finding(
                code="latest",
                level="info",
                text=f"Остання чиста дата {uk_date(latest.date)}: NDVI {latest.ndvi.mean:.2f}, {state}.",
            )
        )

    gap = metrics.longest_gap
    if gap is not None and gap.days > 21:
        findings.append(
            Finding(
                code="data_gap",
                level="notice",
                text=(
                    f"Найдовша перерва без чистих знімків: {uk_date(gap.start)}–{uk_date(gap.end)} "
                    f"({gap.days} днів), зміни в цей час не видно."
                ),
            )
        )
    return findings + zone_findings(metrics.zones)


class VegetationTool(Tool[VegetationParams, VegetationMetrics]):
    name = "vegetation"
    version = "1.2.0"
    title = Label(en="Vegetation (NDVI) and land cover", uk="Вегетація (NDVI) і покрив ділянки")
    params_model = VegetationParams
    metrics_model = VegetationMetrics

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

        artifacts = self._render(
            context=context, season=season, rasters=rasters, peak=peak, widest=widest, latest=latest
        )
        if len(zones) > 1 and zone_map is not None:
            artifacts += self._render_zones(context=context, zones=zones, zone_map=zone_map)
        s2_source = sentinel2.DATA_SOURCE.model_copy(update={"items": [h.scene_id for h in history]})
        cover_source = worldcover.DATA_SOURCE.model_copy(update={"items": cover_items})
        return ToolOutput(
            metrics=metrics,
            summary=season_findings(metrics),
            artifacts=artifacts,
            data_sources=[s2_source, cover_source],
            warnings=warnings,
            partial=bool(warnings) or peak is None,
        )

    def _render_zones(
        self, *, context: RunContext, zones: list[Zone], zone_map: npt.NDArray[np.float64]
    ) -> list[Artifact]:
        name = context.parcel.name
        labels = [f"Зона {z.id} ({z.share:.0%})" for z in zones]
        map_path = context.workdir / "zones_map.png"
        render.zone_map(map_path, f"{name}: зони поля за динамікою NDVI", zone_map, labels)
        curves_path = context.workdir / "zones_timeseries.png"
        render.zone_timeseries(
            curves_path,
            f"{name}: NDVI по зонах",
            curves=[([p.date for p in z.curve], [p.ndvi for p in z.curve]) for z in zones],
            labels=labels,
        )
        return [
            Artifact(
                path=map_path.name,
                media_type="image/png",
                description="Карта зон поля з різною динамікою NDVI.",
            ),
            Artifact(
                path=curves_path.name,
                media_type="image/png",
                description="Графік NDVI окремо для кожної зони.",
            ),
        ]

    def _render(
        self,
        *,
        context: RunContext,
        season: list[SeasonPoint],
        rasters: dict[date, xr.DataArray],
        peak: SeasonPoint | None,
        widest: SeasonPoint | None,
        latest: SeasonPoint | None,
    ) -> list[Artifact]:
        artifacts: list[Artifact] = []
        name = context.parcel.name
        if season:
            path = context.workdir / "ndvi_timeseries.png"
            render.ndvi_timeseries(
                path,
                f"{name}: NDVI за період",
                days=[p.date for p in season],
                mean=[p.ndvi.mean for p in season],
                p10=[p.ndvi.p10 for p in season],
                p90=[p.ndvi.p90 for p in season],
            )
            artifacts.append(
                Artifact(
                    path=path.name,
                    media_type="image/png",
                    description="Графік NDVI за період: середнє по полю і діапазон 10–90% пікселів.",
                )
            )

        maps = [
            ("peak", peak, "Карта NDVI на піку вегетації."),
            ("max_spread", widest, "Карта NDVI в день найбільшої неоднорідності поля."),
            ("latest", latest, "Карта NDVI на останню чисту дату."),
        ]
        drawn: set[date] = set()
        for code, point, description in maps:
            if point is None or point.date in drawn:
                continue
            drawn.add(point.date)
            path = context.workdir / f"ndvi_{code}_{point.date.isoformat()}.png"
            render.ndvi_map(path, f"{name}: NDVI {uk_date(point.date)}", rasters[point.date])
            artifacts.append(Artifact(path=path.name, media_type="image/png", description=description))
        return artifacts
