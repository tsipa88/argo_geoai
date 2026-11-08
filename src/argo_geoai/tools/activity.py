import logging
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Literal

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import pystac
from pydantic import Field, model_validator

from argo_geoai import interactive
from argo_geoai.analysis import activity
from argo_geoai.analysis.grid import ParcelGrid, cells_of, parcel_grid
from argo_geoai.parcels import Parcel
from argo_geoai.schema import Artifact, DataSource, Finding, Label, Message, Model
from argo_geoai.sources import planetary, sentinel1, sentinel2
from argo_geoai.tools.base import RunContext, Tool, ToolOutput
from argo_geoai.tools.vegetation import DAYS_PER_MONTH, plural_uk, uk_date

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

GRID_RESOLUTION_M = 20
MAX_SEARCH_ITEMS = 5000
RADAR_MONTHS = range(5, 13)
PEAK_NDVI = 0.5
SEASON_SPLIT_MONTH = 1
FIELD_LINE_COLOR = "#7f8c8d"
LAND_LINE_COLOR = "#1b7837"
TOUCHED_BAND_COLOR = "#c0392b"
UNTOUCHED_BAND_COLOR = "#95a5a6"
RADAR_PLOT_SMOOTHING = 5
RADAR_PLOT_GAP_DAYS = 20
RADAR_PLOT_RANGE = (-6.0, 4.0)
# the longest period read in one request: a run behind the HTTP server must stay below 10 minutes
MAX_PERIOD_DAYS = 3 * 365
HTML_FILE = "activity.html"
FIGURES_FILE = "activity_figures.json"

EVENT_LABELS: dict[str, Label] = {
    "harvest": Label(en="harvest", uk="Збирання врожаю"),
    "tillage": Label(en="tillage", uk="Обробіток ґрунту"),
    "sowing": Label(en="seedbed and sowing", uk="Передпосівний обробіток / сівба"),
    "emergence": Label(en="emergence", uk="Сходи"),
    "green_cut": Label(en="green cover cut", uk="Скошування / знищення зеленого покриву"),
    "peak": Label(en="vegetation peak", uk="Пік вегетації"),
}
EVENT_TEXTS = {
    "harvest": "Радар: рослини, що стояли на полі після вегетації, зникли — поле зібрано.",
    "tillage": "Рослинні рештки зароблено в ґрунт: оранка, дискування або культивація.",
    "emergence": "На голому полі з'явилась зелень: сходи посіву (або бур'яни).",
    "green_cut": (
        "Зелений покрив зник за кілька днів: покіс, збирання на силос або дискування падалиці чи бур'янів."
    ),
}
EVENT_COLORS = {
    "harvest": "#d95f02",
    "tillage": "#7570b3",
    "sowing": "#386cb0",
    "emergence": "#1b9e77",
    "green_cut": "#e7298a",
    "peak": "#66a61e",
}


HIGHLIGHT_EDGE_M = 10
HIGHLIGHT_MOSTLY_SHARE = 0.5
HIGHLIGHT_SEEN_SHARE = 0.5
SMALL_HIGHLIGHT_CELLS = 25
SOWING_TO_EMERGENCE_DAYS = 45


def sowing_before(tillage: activity.Event, emergence: activity.Event) -> bool:
    if tillage.code != "tillage":
        return False
    lag = (emergence.date_from - tillage.date_to).days
    return emergence.date_from >= tillage.date_from and 0 <= lag <= SOWING_TO_EMERGENCE_DAYS


def uk_period(event: activity.Event) -> str:
    if event.date_from == event.date_to:
        return uk_date(event.date_to)
    return f"{uk_date(event.date_from)}–{uk_date(event.date_to)}"


def moving_average(values: FloatArray, window: int) -> FloatArray:
    finite = np.isfinite(values)
    kernel = np.ones(window)
    total = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    count = np.convolve(finite.astype("float64"), kernel, mode="same")
    with np.errstate(all="ignore"):
        result: FloatArray = np.where(count > 0, total / count, np.nan)
    return result


class ActivityParams(Model):
    start_date: date | None = Field(
        None,
        description="first day of the period (the caller decides how far back to look); months when empty",
    )
    months: int = Field(
        24, ge=1, le=120, description="period length counted back from end_date, without start_date"
    )
    end_date: date | None = Field(None, description="last day of the period; today by default")
    max_scene_cloud: float = Field(
        90,
        ge=0,
        le=100,
        description="skip Sentinel-2 scenes cloudier than this, %; the parcel's own cloud mask decides then",
    )
    min_clear_share: float = Field(
        0.6,
        ge=0,
        le=1,
        description="a Sentinel-2 date counts when at least this share of the parcel is clear",
    )
    radar: bool = Field(
        True, description="read Sentinel-1 radar (harvest dates); False: optical only, faster"
    )
    highlight_tolerance: float = Field(
        0.05,
        ge=0,
        le=1,
        description="the highlight (a land inside the analysed area) counts as touched by an event when at "
        "least this share of it changed; less is the error of the method (the caller decides)",
    )

    @model_validator(mode="after")
    def _period(self) -> "ActivityParams":
        if self.start_date is None:
            return self
        if self.end_date is not None and self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.start_date > datetime.now(UTC).date():
            raise ValueError("start_date is in the future")
        return self

    def period(self, today: date) -> tuple[date, date]:
        end = self.end_date or today
        if self.start_date is not None:
            return max(self.start_date, end - timedelta(days=MAX_PERIOD_DAYS)), end
        return end - timedelta(days=self.months * DAYS_PER_MONTH), end


class ActivityEvent(Model):
    code: str = Field(
        description="harvest, tillage, sowing (tillage followed by emergence), emergence, green_cut, peak"
    )
    label: Label
    date_from: date = Field(description="the last observation before the change")
    date_to: date = Field(description="the first observation that shows the change")
    share: float | None = Field(description="share of the field where it happened; None for a season peak")
    area_ha: float | None
    source: str = Field(description="sentinel-1 (radar) or sentinel-2 (optical)")
    text: str = Field(description="Ukrainian explanation for people")
    highlight_share: float | None = Field(
        None, description="share of the highlight (the land) where it happened; None without a highlight"
    )
    highlight_area_ha: float | None = None


class OpticalDate(Model):
    date: date
    scene_id: str
    clear_share: float
    snow_share: float
    counted: bool
    ndvi_median: float | None = None
    ndti_median: float | None = None
    highlight_clear_share: float | None = Field(None, description="share of the highlight seen clearly")
    highlight_ndvi_median: float | None = None
    highlight_ndti_median: float | None = None


class TimelineEntry(Model):
    date: date
    kind: Literal["event", "observation"] = Field(
        description="event: something was done or happened; observation: what a clear image shows"
    )
    code: str = Field(description="event code, or the ground state of an observation")
    text: str = Field(description="Ukrainian sentence for people: the activity column of the table")
    highlight_text: str | None = Field(
        None,
        description="what happened on the highlight (the land): the land column; None without a highlight",
    )
    highlight_touched: bool = Field(
        False, description="the event changed at least highlight_tolerance of the highlight"
    )


class ActivityMetrics(Model):
    period_start: date
    period_end: date
    optical_scenes: int = Field(description="Sentinel-2 days found")
    optical_dates: int = Field(description="clear enough to count")
    radar_acquisitions: int
    events: list[ActivityEvent] = Field(description="in time order")
    counts: dict[str, int] = Field(description="events per code")
    timeline: list[TimelineEntry] = Field(
        description="date | activity table, oldest first: events (on the first date they are seen) "
        "and the state of the field on every clear image"
    )
    history: list[OpticalDate]


def event_period(event: ActivityEvent) -> str:
    if event.date_from == event.date_to:
        return uk_date(event.date_to)
    return f"{uk_date(event.date_from)}–{uk_date(event.date_to)}"


def event_line(event: ActivityEvent) -> str:
    where = f" на {event.share:.0%} поля" if event.share is not None else ""
    return f"{event.label.uk.lower()} {event_period(event)}{where}"


def activity_findings(metrics: ActivityMetrics, *, tolerance: float = 0.05) -> list[Finding]:
    period = f"{uk_date(metrics.period_start)}–{uk_date(metrics.period_end)}"
    operations = [e for e in metrics.events if e.code != "peak"]
    if not operations:
        return [
            Finding(
                code="no_events",
                level="info",
                text=(
                    f"За період {period} змін на полі не знайдено "
                    f"(чистих знімків {metrics.optical_dates}, радарних — {metrics.radar_acquisitions})."
                ),
            )
        ]
    counts = ", ".join(
        f"{EVENT_LABELS[code].uk.lower()} — {count}"
        for code, count in metrics.counts.items()
        if code != "peak" and count
    )
    findings = [
        Finding(
            code="events_found",
            level="info",
            text=(
                f"За період {period} знайдено {plural_uk(len(operations), 'подію', 'події', 'подій')} "
                f"на полі: {counts}."
            ),
        ),
        Finding(code="latest_event", level="info", text=f"Остання: {event_line(operations[-1])}."),
    ]
    land = highlight_finding(metrics.events, tolerance=tolerance)
    if land is not None:
        findings.append(land)
    last_harvest = next((e for e in reversed(operations) if e.code == "harvest"), None)
    if last_harvest and last_harvest is not operations[-1]:
        findings.append(
            Finding(
                code="latest_harvest", level="info", text=f"Останнє збирання: {event_period(last_harvest)}."
            )
        )
    findings.append(
        Finding(
            code="dates_are_windows",
            level="info",
            text=(
                "Дата події — проміжок між останнім знімком до зміни і першим, де її видно. "
                "Під хмарами і снігом оптичні знімки нічого не бачать, тому обробіток ґрунту взимку "
                "чи в тривалу негоду може бути пропущений."
            ),
        )
    )
    return findings


def without_doubled_harvests(events: list[activity.Event]) -> list[activity.Event]:
    cuts = [e for e in events if e.code == "green_cut"]
    return [
        e
        for e in events
        if e.code != "harvest" or not any(c.date_from <= e.date_to and e.date_from <= c.date_to for c in cuts)
    ]


def skip_failed[R](
    read: Callable[[pystac.Item], R | None], warnings: list[Message]
) -> Callable[[pystac.Item], R | None]:
    def safe(item: pystac.Item) -> R | None:
        try:
            return read(item)
        except Exception as error:
            log.warning("scene %s failed: %s", item.id, error)
            warnings.append(Message(code="scene_failed", text=f"{item.id}: {error}"))
            return None

    return safe


DENSE_CROP_NDVI = 0.6
GROWING_CROP_NDVI = 0.4
SOME_GREEN_NDVI = 0.25
RESIDUE_NDTI = 0.08
SNOW_SHARE = 0.3


def highlight_area_ha(parcel: Parcel) -> float | None:
    if parcel.highlight is None:
        return None
    frame = gpd.GeoSeries([parcel.highlight], crs="EPSG:4326").to_crs(parcel.utm_crs)
    return round(float(frame.area.iloc[0]) / 10_000, 2)


def highlight_part(
    event: activity.Event, highlight: BoolArray | None, land_ha: float | None
) -> dict[str, float | None]:
    if highlight is None or event.cells is None or not highlight.any():
        return {}
    share = float((event.cells & highlight).sum() / highlight.sum())
    return {
        "highlight_share": round(share, 3),
        "highlight_area_ha": round(share * land_ha, 1) if land_ha is not None else None,
    }


def ground_state(record: OpticalDate) -> tuple[str, str]:
    ndvi, ndti = record.ndvi_median or 0.0, record.ndti_median or 0.0
    rules: list[tuple[bool, str, str]] = [
        (record.snow_share >= SNOW_SHARE, "snow", f"Сніг на {record.snow_share:.0%} поля."),
        (ndvi >= DENSE_CROP_NDVI, "dense_crop", f"Густий зелений посів (NDVI {ndvi:.2f})."),
        (
            ndvi >= GROWING_CROP_NDVI,
            "growing_crop",
            f"Посів є, але не зімкнувся або вже жовтіє (NDVI {ndvi:.2f}).",
        ),
        (
            ndvi >= SOME_GREEN_NDVI,
            "sparse_green",
            f"Мало зелені: сходи, бур'яни або дозрілий посів (NDVI {ndvi:.2f}).",
        ),
        (
            ndti >= RESIDUE_NDTI,
            "residue",
            f"Рослин немає, на полі рослинні рештки — стерня або солома (NDTI {ndti:.2f}).",
        ),
        (
            ndti < activity.TILLED_NDTI,
            "bare_soil",
            f"Голий оброблений ґрунт, решток майже немає (NDTI {ndti:.2f}).",
        ),
    ]
    return next(
        ((code, text) for matches, code, text in rules if matches),
        ("no_plants", f"Рослин немає, рештки частково зароблені (NDTI {ndti:.2f})."),
    )


def counted_days(history: list[OpticalDate]) -> set[date]:
    return {h.date for h in history if h.counted}


def event_sentence(event: ActivityEvent) -> str:
    if event.share is None:
        return f"{event.label.uk}. {event.text}"
    where = f"на {event.share:.0%} поля ({event.area_ha:.1f} га)" if event.area_ha is not None else ""
    when = (
        f", зміна між {uk_date(event.date_from)} і {uk_date(event.date_to)}"
        if event.date_from != event.date_to
        else ""
    )
    return f"{event.label.uk} {where}{when}. {event.text}"


def uk_form(count: int, one: str, few: str, many: str) -> str:
    return plural_uk(count, one, few, many).split(" ", 1)[1]


def highlight_event_text(event: ActivityEvent, *, tolerance: float) -> tuple[str, bool]:
    share = event.highlight_share or 0.0
    area = f" ({event.highlight_area_ha:.1f} га)" if event.highlight_area_ha is not None else ""
    if share < tolerance:
        return "Не зачеплена.", False
    if share < HIGHLIGHT_MOSTLY_SHARE:
        return f"Частково: ~{share:.0%} ділянки{area}.", True
    return f"Ділянку теж: ~{share:.0%}{area}.", True


def highlight_state_text(record: OpticalDate) -> str:
    if (record.highlight_clear_share or 0.0) < HIGHLIGHT_SEEN_SHARE:
        return "Ділянку не видно (хмари)."
    land = record.model_copy(
        update={"ndvi_median": record.highlight_ndvi_median, "ndti_median": record.highlight_ndti_median}
    )
    field_code, _ = ground_state(record)
    land_code, land_text = ground_state(land)
    if land_code == field_code:
        return "Так само, як поле."
    return f"Інакше, ніж поле: {land_text[0].lower()}{land_text[1:]}"


def timeline(
    events: list[ActivityEvent],
    history: list[OpticalDate],
    *,
    tolerance: float = 0.05,
    small_highlight: bool = False,
) -> list[TimelineEntry]:
    rough = " Ділянка мала, оцінка приблизна." if small_highlight else ""
    rows = []
    for e in events:
        land_text, touched = None, False
        if e.highlight_share is not None:
            land_text, touched = highlight_event_text(e, tolerance=tolerance)
            land_text += rough
        elif e.code == "peak" and e.highlight_area_ha is not None:
            land_text = "—"
        rows.append(
            TimelineEntry(
                date=e.date_to,
                kind="event",
                code=e.code,
                text=event_sentence(e),
                highlight_text=land_text,
                highlight_touched=touched,
            )
        )
    for record in history:
        if record.counted:
            code, text = ground_state(record)
            land_text = None if record.highlight_clear_share is None else highlight_state_text(record)
            rows.append(
                TimelineEntry(
                    date=record.date,
                    kind="observation",
                    code=code,
                    text=f"Знімок: {text}",
                    highlight_text=land_text,
                )
            )
    return sorted(rows, key=lambda r: (r.date, r.kind != "event"))


def highlight_finding(events: list[ActivityEvent], *, tolerance: float) -> Finding | None:
    operations = [e for e in events if e.code != "peak" and e.highlight_share is not None]
    if not any(e.highlight_share is not None for e in events if e.code != "peak"):
        return None
    touched = [e for e in operations if (e.highlight_share or 0.0) >= tolerance]
    if not touched:
        count = len(operations)
        return Finding(
            code="highlight_untouched",
            level="info",
            text=(
                f"На ділянці за період змін не видно (похибка до {tolerance:.0%}), хоча на полі навколо "
                f"{uk_form(count, 'була', 'були', 'було')} {plural_uk(count, 'подія', 'події', 'подій')}."
            ),
        )
    listed = "; ".join(
        f"{e.label.uk.lower()} {event_period(e)} (~{e.highlight_share or 0.0:.0%})" for e in touched
    )
    return Finding(
        code="highlight_touched",
        level="warning",
        text=f"Ділянку зачепили ({plural_uk(len(touched), 'раз', 'рази', 'разів')}): {listed}.",
    )


def season_peaks(observations: list[activity.OpticalObservation]) -> list[tuple[date, float]]:
    best: dict[int, tuple[date, float]] = {}
    for observation in observations:
        median = float(np.nanmedian(observation.ndvi))
        year = observation.day.year
        if median >= PEAK_NDVI and (year not in best or median > best[year][1]):
            best[year] = (observation.day, median)
    return [best[year] for year in sorted(best)]


class ActivityTool(Tool[ActivityParams, ActivityMetrics]):
    name = "activity"
    version = "1.4.0"
    title = Label(
        en="Field activity: harvest, tillage, emergence", uk="Роботи на полі: збирання, обробіток, сходи"
    )
    params_model = ActivityParams
    metrics_model = ActivityMetrics

    def run(self, params: ActivityParams, context: RunContext) -> ToolOutput[ActivityMetrics]:
        parcel = context.parcel
        start, end = params.period(datetime.now(UTC).astimezone().date())
        time_range = f"{start.isoformat()}/{end.isoformat()}"
        grid = parcel_grid(parcel, GRID_RESOLUTION_M)
        warnings: list[Message] = []
        highlight = None
        if parcel.highlight is not None:
            highlight = cells_of(grid, parcel.highlight, inner_edge_m=HIGHLIGHT_EDGE_M)
            if not highlight.any():
                warnings.append(
                    Message(code="highlight_outside", text="the highlight is outside the analysed area")
                )
                highlight = None

        items = sentinel2.least_cloudy_per_day(
            planetary.search(
                sentinel2.COLLECTION,
                parcel,
                time_range=time_range,
                query={"eo:cloud_cover": {"lt": params.max_scene_cloud}},
                max_items=MAX_SEARCH_ITEMS,
            )
        )
        read_optical = skip_failed(lambda item: sentinel2.scene_surface(item, parcel, grid), warnings)
        scenes = [s for s in planetary.read_parallel(read_optical, items) if s is not None]
        history = [self._describe(scene, params, grid, highlight) for scene in scenes]
        optical = [
            activity.OpticalObservation(day=s.day, ndvi=s.ndvi[grid.inside], ndti=s.ndti[grid.inside])
            for s, h in zip(scenes, history, strict=True)
            if h.counted and s.ndvi is not None and s.ndti is not None
        ]
        log.info("optical: %d days, %d counted", len(scenes), len(optical))

        radar: list[activity.RadarObservation] = []
        radar_ids: list[str] = []
        if params.radar:
            radar_items = [
                item
                for item in planetary.search(
                    sentinel1.COLLECTION, parcel, time_range=time_range, max_items=MAX_SEARCH_ITEMS
                )
                if item.datetime is not None and item.datetime.month in RADAR_MONTHS
            ]
            read_radar = skip_failed(lambda item: sentinel1.scene(item, parcel, grid), warnings)
            for scene in planetary.read_parallel(read_radar, radar_items):
                if scene is None:
                    continue
                with np.errstate(all="ignore"):
                    ratio = 10 * np.log10(scene.vh / scene.vv)
                radar.append(
                    activity.RadarObservation(
                        acquired=scene.acquired, orbit=scene.orbit, ratio_db=ratio[grid.inside]
                    )
                )
                radar_ids.append(scene.item_id)
            log.info("radar: %d acquisitions", len(radar))

        detections = activity.optical_detections(optical) + activity.harvest_detections(radar, optical)
        merged = without_doubled_harvests(activity.merge(detections, grid.cells))
        events = self._events(merged, season_peaks(optical), parcel, highlight)
        metrics = ActivityMetrics(
            period_start=start,
            period_end=end,
            optical_scenes=len(scenes),
            optical_dates=len(optical),
            radar_acquisitions=len(radar),
            events=events,
            counts=dict(Counter(e.code for e in events)),
            timeline=timeline(
                events,
                history,
                tolerance=params.highlight_tolerance,
                small_highlight=highlight is not None and int(highlight.sum()) < SMALL_HIGHLIGHT_CELLS,
            ),
            history=history,
        )
        summary = activity_findings(metrics, tolerance=params.highlight_tolerance)
        sources = [
            sentinel2.DATA_SOURCE.model_copy(
                update={
                    "items": [h.scene_id for h in history if h.counted],
                    "resolution_m": GRID_RESOLUTION_M,
                }
            )
        ]
        if params.radar:
            sources.append(sentinel1.DATA_SOURCE.model_copy(update={"items": radar_ids}))
        ndvi_maps = {s.day: s.ndvi for s in scenes if s.ndvi is not None and s.day in counted_days(history)}
        artifacts = self._interactive(
            context=context,
            metrics=metrics,
            grid=grid,
            ndvi_maps=ndvi_maps,
            radar=radar,
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

    @staticmethod
    def _describe(
        scene: sentinel2.SceneSurface, params: ActivityParams, grid: ParcelGrid, highlight: BoolArray | None
    ) -> OpticalDate:
        counted = scene.ndvi is not None and scene.clear_share >= params.min_clear_share
        ndvi = scene.ndvi[grid.inside] if scene.ndvi is not None else None
        ndti = scene.ndti[grid.inside] if scene.ndti is not None else None
        seen = ndvi is not None and bool(np.isfinite(ndvi).any())
        land: dict[str, float | None] = {}
        if highlight is not None:
            land_ndvi = ndvi[highlight] if ndvi is not None else np.array([])
            land_seen = np.isfinite(land_ndvi)
            land["highlight_clear_share"] = round(float(land_seen.mean()), 3) if land_ndvi.size else 0.0
            if land_seen.any() and ndti is not None:
                land["highlight_ndvi_median"] = round(float(np.nanmedian(land_ndvi)), 3)
                land["highlight_ndti_median"] = round(float(np.nanmedian(ndti[highlight])), 3)
        return OpticalDate(
            date=scene.day,
            scene_id=scene.item_id,
            clear_share=round(scene.clear_share, 3),
            snow_share=round(scene.snow_share, 3),
            counted=counted and seen,
            ndvi_median=round(float(np.nanmedian(ndvi)), 3) if seen and ndvi is not None else None,
            ndti_median=round(float(np.nanmedian(ndti)), 3) if seen and ndti is not None else None,
            **land,
        )

    @staticmethod
    def _events(
        merged: list[activity.Event],
        peaks: list[tuple[date, float]],
        parcel: Parcel,
        highlight: BoolArray | None,
    ) -> list[ActivityEvent]:
        land_ha = highlight_area_ha(parcel)
        emergences = [e for e in merged if e.code == "emergence"]
        events = []
        for e in merged:
            code: str = e.code
            text = EVENT_TEXTS[e.code]
            sprouts = next((m for m in emergences if sowing_before(e, m)), None)
            if sprouts is not None:
                code = "sowing"
                text = f"Після обробітку з'явились сходи ({uk_period(sprouts)}): ймовірно сівба."
            events.append(
                ActivityEvent(
                    code=code,
                    label=EVENT_LABELS[code],
                    date_from=e.date_from,
                    date_to=e.date_to,
                    share=e.share,
                    area_ha=round(e.share * parcel.area_ha, 1),
                    source=e.source,
                    text=text,
                    **highlight_part(e, highlight, land_ha),
                )
            )
        events += [
            ActivityEvent(
                code="peak",
                label=EVENT_LABELS["peak"],
                date_from=day,
                date_to=day,
                share=None,
                area_ha=None,
                source="sentinel-2",
                text=f"Найзеленіше поле за рік: NDVI {ndvi:.2f} (медіана поля).",
                highlight_area_ha=land_ha,
            )
            for day, ndvi in peaks
        ]
        return sorted(events, key=lambda e: (e.date_to, e.date_from))

    def _interactive(
        self,
        *,
        context: RunContext,
        metrics: ActivityMetrics,
        grid: ParcelGrid,
        ndvi_maps: dict[date, FloatArray],
        radar: list[activity.RadarObservation],
        summary: list[Finding],
        sources: list[DataSource],
    ) -> list[Artifact]:
        parcel = context.parcel
        figures: dict[str, go.Figure] = {}
        if ndvi_maps:
            figures["by_date"] = self._slider(parcel, metrics, grid, ndvi_maps)
        touched = {(e.date, e.code) for e in metrics.timeline if e.kind == "event" and e.highlight_touched}
        with_land = any(h.highlight_ndvi_median is not None for h in metrics.history)
        figures["timeline"] = (
            self._field_and_land(metrics, touched) if with_land else self._timeline(metrics, radar)
        )
        interactive.write_page(
            context.workdir / HTML_FILE,
            title=f"{parcel.name}: роботи на полі",
            subtitle=(
                f"{parcel.area_ha:.1f} га · {uk_date(metrics.period_start)}–{uk_date(metrics.period_end)} · "
                f"чистих знімків {metrics.optical_dates}, радарних {metrics.radar_acquisitions}"
            ),
            summary=summary,
            figures=[self._table(metrics), *figures.values()],
            sources=sources,
            note=(
                "Карта: повзунок під нею перемикає дати чистих знімків, над картою — що видно на полі і "
                "які події помічено до цієї дати. Графік: наближення рамкою, подвійний клік — повернутись."
            ),
        )
        interactive.write_figures_json(context.workdir / FIGURES_FILE, figures)
        return [
            Artifact(
                path=HTML_FILE,
                media_type="text/html",
                description=(
                    "Інтерактивна сторінка: таблиця дата / активність, карта NDVI з повзунком по датах, "
                    "графік NDVI, рослинних решток і радара з подіями."
                ),
            ),
            Artifact(
                path=FIGURES_FILE,
                media_type="application/json",
                description="Специфікації графіків Plotly (дані і вигляд) для plotly.js.",
            ),
        ]

    @staticmethod
    def _slider(
        parcel: Parcel, metrics: ActivityMetrics, grid: ParcelGrid, ndvi_maps: dict[date, FloatArray]
    ) -> go.Figure:
        geometry = interactive.grid_geometry(grid.x, grid.y, grid.crs, grid.inside)
        states = {e.date: e.text for e in metrics.timeline if e.kind == "observation"}
        days = sorted(ndvi_maps)
        captions: dict[str, str] = {}
        for index, day in enumerate(days):
            previous = days[index - 1] if index else date.min
            happened = [
                f"{e.label.uk} ({event_period(e)})"
                for e in metrics.events
                if e.code != "peak" and previous < e.date_to <= day
            ]
            caption = states.get(day, "").removeprefix("Знімок: ")
            if happened:
                caption += " Помічено: " + "; ".join(happened) + "."
            captions[uk_date(day)] = caption
        return interactive.date_slider_map(
            parcel,
            geometry,
            title="NDVI поля по датах чистих знімків",
            frames={uk_date(day): np.where(grid.inside, ndvi_maps[day], np.nan) for day in days},
            captions=captions,
            colorscale="RdYlGn",
            zmin=0.0,
            zmax=0.9,
        )

    @staticmethod
    def _table(metrics: ActivityMetrics) -> go.Figure:
        ordered = sorted(metrics.timeline, key=lambda e: (e.date, e.kind == "event"), reverse=True)
        with_land = any(entry.highlight_text is not None for entry in ordered)
        rows = []
        for entry in ordered:
            row = [uk_date(entry.date), ("▶ " if entry.kind == "event" else "") + entry.text]
            if with_land:
                row.append(("⚠ " if entry.highlight_touched else "") + (entry.highlight_text or ""))
            rows.append(row)
        if with_land:
            return interactive.table_figure(
                "Хронологія: дата, поле і ділянка",
                ["Дата", "Поле (уявний полігон)", "Ділянка"],
                rows,
                column_widths=(1, 5, 3),
            )
        return interactive.table_figure(
            "Хронологія: дата і активність на полі", ["Дата", "Активність"], rows, column_widths=(1, 7)
        )

    @staticmethod
    def _field_and_land(metrics: ActivityMetrics, touched: set[tuple[date, str]]) -> go.Figure:
        counted = [h for h in metrics.history if h.counted and h.ndvi_median is not None]
        land = [h for h in counted if h.highlight_ndvi_median is not None]
        figure = interactive.line_chart(
            "Поле і ділянка: зеленість (NDVI) і події",
            [
                interactive.Series(
                    name="Поле (масиви і їхні ділянки)",
                    x=[h.date.isoformat() for h in counted],
                    y=[h.ndvi_median or 0 for h in counted],
                    color=FIELD_LINE_COLOR,
                ),
                interactive.Series(
                    name="Ця ділянка",
                    x=[h.date.isoformat() for h in land],
                    y=[h.highlight_ndvi_median or 0 for h in land],
                    color=LAND_LINE_COLOR,
                ),
            ],
            y_title="NDVI",
            y_range=(-0.1, 1.0),
        )
        operations = [e for e in metrics.events if e.code != "peak"]
        for index, event in enumerate(operations):
            hit = (event.date_to, event.code) in touched
            figure.add_vrect(
                x0=event.date_from.isoformat(),
                x1=(event.date_to + timedelta(days=1)).isoformat(),
                fillcolor=TOUCHED_BAND_COLOR if hit else UNTOUCHED_BAND_COLOR,
                opacity=0.3 if hit else 0.2,
                line_width=0,
                annotation_text=event.label.uk.split(" / ")[0] + (" ⚠" if hit else ""),
                annotation_position="top left" if index % 2 == 0 else "bottom left",
                annotation_font_size=10,
            )
        figure.update_layout(height=420, margin={"l": 50, "r": 20, "t": 50, "b": 40})
        return figure

    @staticmethod
    def _timeline(metrics: ActivityMetrics, radar: list[activity.RadarObservation]) -> go.Figure:
        counted = [h for h in metrics.history if h.counted and h.ndvi_median is not None]
        series = [
            interactive.Series(
                name="NDVI поля (медіана)",
                x=[h.date.isoformat() for h in counted],
                y=[h.ndvi_median or 0 for h in counted],
                color="#1b7837",
            ),
            interactive.Series(
                name="NDTI — рослинні рештки",
                x=[h.date.isoformat() for h in counted],
                y=[h.ndti_median or 0 for h in counted],
                color="#8c6d31",
            ),
        ]
        figure = interactive.line_chart(
            "Стан поля і події за період", series, y_title="індекс", y_range=(-0.1, 1.0)
        )
        if radar:
            ordered, ratio = activity.normalized_ratio(radar)
            with np.errstate(all="ignore"):
                field = np.nanmedian(ratio, axis=1)
            smooth = moving_average(field, RADAR_PLOT_SMOOTHING)
            x: list[str | None] = []
            y: list[float | None] = []
            for i, observation in enumerate(ordered):
                if i and (observation.acquired - ordered[i - 1].acquired).days > RADAR_PLOT_GAP_DAYS:
                    x.append(None)
                    y.append(None)
                x.append(observation.acquired.date().isoformat())
                y.append(round(float(smooth[i]), 2) if np.isfinite(smooth[i]) else None)
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name="Радар VH/VV, дБ (відхилення від звичного)",
                    line={"color": "#6a51a3", "width": 1.5},
                    yaxis="y2",
                    connectgaps=False,
                    hovertemplate="радар: %{y:.1f} дБ<br>%{x}<extra></extra>",
                )
            )
            figure.update_layout(
                yaxis2={
                    "title": "радар, дБ",
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": False,
                    "range": list(RADAR_PLOT_RANGE),
                    "dtick": 2,
                }
            )
        operations = [e for e in metrics.events if e.code != "peak"]
        for index, event in enumerate(operations):
            figure.add_vrect(
                x0=event.date_from.isoformat(),
                x1=(event.date_to + timedelta(days=1)).isoformat(),
                fillcolor=EVENT_COLORS[event.code],
                opacity=0.25,
                line_width=0,
                annotation_text=event.label.uk.split(" / ")[0],
                annotation_position="top left" if index % 2 == 0 else "bottom left",
                annotation_font_size=10,
            )
        figure.update_layout(height=460, margin={"l": 50, "r": 60, "t": 50, "b": 40})
        return figure
