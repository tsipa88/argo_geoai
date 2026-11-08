import base64
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from itertools import pairwise
from math import atan2, cos, degrees, log2, radians
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import pystac
import shapely
from pydantic import Field
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, shapes
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from argo_geoai import interactive
from argo_geoai.analysis.grid import ParcelGrid, parcel_grid
from argo_geoai.parcels import Parcel
from argo_geoai.schema import Artifact, DataSource, Finding, Label, Message, Model
from argo_geoai.sources import dem, planetary, sentinel1, sentinel2, worldcover
from argo_geoai.tools import field_details_images as pictures
from argo_geoai.tools.base import RunContext, Tool, ToolOutput
from argo_geoai.tools.vegetation import plural_uk, uk_date

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
Rgb = npt.NDArray[np.uint8]

GRID_RESOLUTION_M = 10
MAX_SEARCH_ITEMS = 5000
BANDS = ("B02", "B03", "B04", "B08", "B11")
RGB_BANDS = (2, 1, 0)
BRIGHTNESS_BANDS = (0, 1, 2)
COMPOSITE_MONTHS = range(4, 11)
MIN_COMPOSITE_DATES = 3
RADAR_MONTHS = range(4, 11)
RADAR_PER_YEAR = 12
RADAR_STRUCTURE_DB = 2.0
BRIGHT_STEP = 0.03
CONFIRM_DATES = 2
STRADDLE_MIN_SHARE = 0.05
BOUNDARY_SIMPLIFY_M = 4.0
BOUNDARY_PAD_M = 25.0
MIN_SIDE_M = 20.0
CONFIDENT_CELLS = 5
CONFIDENT_BRIGHT_STEP = 0.05
SNAPSHOT_FIRST_DOY, SNAPSHOT_LAST_DOY = 91, 304
SNAPSHOT_MAX_BRIGHTNESS = 190
TRUE_COLOR_MAX = 0.28
TRUE_COLOR_GAMMA = 0.8
FLAT_SLOPE_DEG = 2.0
MIN_RELIEF_RANGE_M = 1.0
CONTOUR_STEP_SMALL_M, CONTOUR_STEP_LARGE_M, LARGE_RANGE_M = 1.0, 2.0, 10.0
IMAGERY_HEIGHT_PX, IMAGERY_WIDTH_PX, IMAGERY_MARGIN = 640, 900, 2.5
OUTLINE_COLOR = "#ffffff"
HTML_FILE = "field_details.html"
FIGURES_FILE = "field_details_figures.json"

CONFIDENCE_LABELS: dict[str, Label] = {
    "high": Label(en="high", uk="висока"),
    "medium": Label(en="medium", uk="середня"),
    "low": Label(en="low", uk="низька"),
}
EVENT_KINDS: dict[str, tuple[Label, str]] = {
    "appeared": (Label(en="object appeared", uk="з'явився об'єкт"), "#e6550d"),
    "removed": (Label(en="object removed", uk="об'єкт зник"), "#3182bd"),
    "changed": (Label(en="surface changed", uk="змінилась поверхня"), "#756bb1"),
}
STRADDLE_COLOR = "#ffd92f"
DIRECTIONS = [
    "північ",
    "північний схід",
    "схід",
    "південний схід",
    "південь",
    "південний захід",
    "захід",
    "північний захід",
]
SIDE_NAMES = [
    "північна",
    "північно-східна",
    "східна",
    "південно-східна",
    "південна",
    "південно-західна",
    "західна",
    "північно-західна",
]
LETTERS = "АБВГДЕЖЗИКЛМНОПРСТУФ"
COVER_COLORS = {
    "tree_cover": "#006400",
    "shrubland": "#ffbb22",
    "grassland": "#ffff4c",
    "cropland": "#f096ff",
    "built_up": "#fa0000",
    "bare_sparse": "#b4b4b4",
    "snow_ice": "#f0f0f0",
    "water": "#0064c8",
    "wetland": "#0096a0",
    "mangroves": "#00cf75",
    "moss_lichen": "#fae6a0",
}


class FieldDetailsParams(Model):
    start_year: int = Field(2017, ge=2015, le=2100, description="first year of the history (Sentinel-2)")
    end_date: date | None = Field(None, description="last day of the history; today by default")
    max_scene_cloud: float = Field(40, ge=0, le=100, description="skip scenes cloudier than this, %")
    min_clear_share: float = Field(
        0.9, ge=0, le=1, description="a date counts when at least this share of the analysed area is clear"
    )
    max_scenes: int = Field(
        300, ge=20, le=3000, description="at most this many scenes are read, thinned evenly over the period"
    )
    surroundings_m: float = Field(
        60, ge=20, le=500, description="ground around the parcel analysed too: objects on the boundary"
    )
    change_threshold: float = Field(
        0.08,
        ge=0.02,
        le=0.5,
        description="a cell changed when its spectrum moved this far from the baseline (reflectance units)",
    )
    min_object_m2: float = Field(
        300, ge=100, le=10_000, description="smaller groups of changed cells are noise"
    )
    radar: bool = Field(True, description="read Sentinel-1 radar to confirm structures; False: optical only")
    snapshots_per_year: int = Field(3, ge=1, le=12, description="true-colour images per year, spread over it")
    snapshot_buffer_m: float = Field(
        150, ge=0, le=2000, description="ground around the parcel on a snapshot, m"
    )
    relief_buffer_m: float = Field(400, ge=100, le=3000, description="relief read around the parcel, m")


class CoverShare(Model):
    code: str
    label: Label
    share: float


class Elevation(Model):
    min: float
    max: float
    mean: float
    range: float


class Slope(Model):
    mean: float
    p90: float
    max: float


class Relief(Model):
    elevation_m: Elevation = Field(description="inside the parcel")
    around_elevation_m: Elevation = Field(description="in the window of relief_buffer_m around it")
    slope_deg: Slope
    flat: bool
    low_side: str | None = Field(description="Ukrainian name of the direction of the lower part, if any")


class Observations(Model):
    scenes_found: int
    scenes_read: int
    dates_counted: int
    radar_acquisitions: int
    first_date: date | None
    last_date: date | None


class YearComposite(Model):
    year: int
    dates: int = Field(description="clear dates in the composite")
    first_date: date | None
    last_date: date | None
    png_base64: str | None = Field(description="true-colour median of the year, the parcel outlined")


class ChangeEvent(Model):
    number: int
    kind: Literal["appeared", "removed", "changed"]
    kind_label: Label
    color: str
    date_from: date | None = Field(description="last clear image before the change")
    date_to: date | None = Field(description="first clear image that shows the change")
    year_from: int
    year_to: int
    area_m2: float = Field(description="whole object, inside and outside the parcel")
    inside_m2: float
    inside_share: float = Field(description="part of the object inside the parcel")
    parcel_share: float = Field(description="part of the parcel the inside part takes")
    location: str = Field(description="Ukrainian: where in the parcel (or on which side of it)")
    on_boundary: bool
    side: int | None = Field(description="index in metrics.boundary of the side the object lies on")
    brightness_before: float
    brightness_after: float
    surface_before: str
    surface_after: str
    radar_change_db: float | None
    confidence: Literal["high", "medium", "low"]
    confidence_label: Label
    text: str = Field(description="Ukrainian description for the report")
    geometry: dict[str, Any] = Field(description="GeoJSON Polygon/MultiPolygon of the object cells, lng/lat")


class BoundarySide(Model):
    index: int
    letters: str = Field(description="e.g. А–Б, as on a cadastral plan (in the order of the polygon)")
    name: str = Field(description="Ukrainian: західна межа (А–Б)")
    name_locative: str = Field(description="Ukrainian, locative: західній межі (А–Б)")
    direction: str = Field(description="Ukrainian direction the side faces")
    length_m: float
    start: tuple[float, float] = Field(description="lng, lat")
    end: tuple[float, float]
    bbox: tuple[float, float, float, float] = Field(description="lng/lat frame for a picture of the side")
    events: list[int] = Field(description="numbers of the events whose object lies on this side")
    text: str = Field(description="Ukrainian: what happened on this side")


class Snapshot(Model):
    date: date
    scene_id: str
    clear_share: float
    png_base64: str


class ImageOut(Model):
    name: str
    title: str
    description: str
    png_base64: str


class FieldDetailsMetrics(Model):
    land_cover: list[CoverShare]
    relief: Relief | None
    observations: Observations
    years: list[YearComposite]
    events: list[ChangeEvent]
    boundary: list[BoundarySide]
    encroachment_events: list[int] = Field(
        description="numbers of the events whose object lies on the boundary"
    )
    snapshots: list[Snapshot]
    images: list[ImageOut]


class _Scene:
    def __init__(self, item: pystac.Item, bands: FloatArray, clear_share: float) -> None:
        assert item.datetime is not None
        self.item = item
        self.day = item.datetime.date()
        self.bands = bands
        self.clear_share = clear_share
        self.scene_cloud = float(item.properties.get("eo:cloud_cover", 0))

    @property
    def brightness(self) -> FloatArray:
        values: FloatArray = self.bands[list(BRIGHTNESS_BANDS)].mean(axis=0)
        return values


class _Object:
    def __init__(self, cells: BoolArray, year_from: int, year_to: int) -> None:
        self.cells = cells
        self.year_from = year_from
        self.year_to = year_to


def thin(items: list[pystac.Item], limit: int) -> list[pystac.Item]:
    if len(items) <= limit:
        return items
    positions = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in sorted(set(positions.tolist()))]


def to_rgb(bands: FloatArray) -> Rgb:
    stack = np.dstack([bands[i] for i in RGB_BANDS])
    stretched = np.clip(np.nan_to_num(stack) / TRUE_COLOR_MAX, 0, 1) ** TRUE_COLOR_GAMMA
    rgb: Rgb = (stretched * 255).round().astype("uint8")
    return rgb


def composite(scenes: list[_Scene]) -> FloatArray | None:
    if not scenes:
        return None
    season = [s for s in scenes if s.day.month in COMPOSITE_MONTHS]
    chosen = season if len(season) >= MIN_COMPOSITE_DATES else scenes
    with np.errstate(all="ignore"):
        values: FloatArray = np.nanmedian(np.stack([s.bands for s in chosen]), axis=0)
    return values


def spectral_distance(a: FloatArray, b: FloatArray) -> FloatArray:
    with np.errstate(all="ignore"):
        distance: FloatArray = np.sqrt(((a - b) ** 2).sum(axis=0))
    distance[~(np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0))] = np.nan
    return distance


def changed_objects(
    baseline: FloatArray, current: FloatArray, area: BoolArray, *, threshold: float, min_cells: int
) -> list[BoolArray]:
    distance = spectral_distance(baseline, current)
    changed = area & np.isfinite(distance) & (distance > threshold)
    labels, count = ndimage.label(changed)
    objects = []
    for index in range(1, count + 1):
        cells: BoolArray = labels == index
        if int(cells.sum()) >= min_cells:
            objects.append(cells)
    return objects


def change_kind(before: float, after: float) -> str:
    if after - before >= BRIGHT_STEP:
        return "appeared"
    if before - after >= BRIGHT_STEP:
        return "removed"
    return "changed"


def radar_agrees(kind: str, radar_change_db: float | None) -> bool | None:
    if radar_change_db is None or kind == "changed":
        return None
    if kind == "appeared":
        return radar_change_db >= RADAR_STRUCTURE_DB
    return radar_change_db <= -RADAR_STRUCTURE_DB


def confidence_of(cells: int, brightness_step: float, radar_agreement: bool | None) -> str:
    large = cells >= CONFIDENT_CELLS
    clear = abs(brightness_step) >= CONFIDENT_BRIGHT_STEP
    if radar_agreement is True and (large or clear):
        return "high"
    if large and clear and radar_agreement is None:
        return "high"
    if radar_agreement is False and not large:
        return "low"
    if not large and not clear:
        return "low"
    return "medium"


def surface_name(bands: FloatArray) -> str:
    blue, green, red, nir = (float(bands[i]) for i in range(4))
    brightness = (blue + green + red) / 3
    with np.errstate(all="ignore"):
        greenness = (nir - red) / (nir + red) if nir + red > 0 else 0.0
    if greenness >= 0.35:
        return "рослинність (трава, кущі або дерева)"
    if brightness >= 0.18:
        return "світла поверхня (світлий дах або бетон)"
    if brightness < 0.07:
        return "темна поверхня (асфальт або темний дах)"
    if red > green * 1.15:
        return "рудувата поверхня (ґрунт, цегла або руда покрівля)"
    return "сіра поверхня (покриття, дах або ущільнений ґрунт)"


def date_change(
    days: list[date], values: list[float], *, before: float, after: float, year_to: int
) -> tuple[date | None, date | None]:
    if not days or before == after:
        return None, None
    midpoint = (before + after) / 2
    new_side = [(v > midpoint) == (after > before) for v in values]
    for i in range(len(days)):
        if days[i].year < year_to - 1:
            continue
        if all(new_side[i : i + CONFIRM_DATES]) and i + CONFIRM_DATES <= len(days):
            return (days[i - 1] if i > 0 else None), days[i]
    return None, None


def direction_name(north_m: float, east_m: float, names: list[str] = DIRECTIONS) -> str:
    sector = int(((degrees(atan2(east_m, north_m)) + 360 + 22.5) // 45) % 8)
    return names[sector]


def boundary_sides(
    ring: Sequence[tuple[float, float]], *, tolerance_m: float = BOUNDARY_SIMPLIFY_M
) -> list[LineString]:
    simplified = Polygon(ring).simplify(tolerance_m, preserve_topology=True)
    assert isinstance(simplified, Polygon)
    corners = [(c[0], c[1]) for c in simplified.exterior.coords]
    sides = [LineString([a, b]) for a, b in pairwise(corners)]
    merged: list[LineString] = []
    for side in sides:
        if merged and (side.length < MIN_SIDE_M or merged[-1].length < MIN_SIDE_M):
            merged[-1] = LineString([merged[-1].coords[0], side.coords[-1]])
        else:
            merged.append(side)
    if len(merged) > 1 and merged[-1].length < MIN_SIDE_M:
        merged[0] = LineString([merged[-1].coords[0], merged[0].coords[-1]])
        merged.pop()
    return merged


def side_direction(side: LineString, interior: Polygon) -> str:
    (x0, y0), (x1, y1) = side.coords
    dx, dy = x1 - x0, y1 - y0
    normal = np.array([dy, -dx]) / max(np.hypot(dx, dy), 1e-9)
    mid = side.interpolate(0.5, normalized=True)
    probes = [Point(mid.x + sign * normal[0] * 3, mid.y + sign * normal[1] * 3) for sign in (1, -1)]
    outside, inside = interior.distance(probes[0]), interior.distance(probes[1])
    if inside > outside:
        normal = -normal
    elif inside == outside:  # both probes inside (a chord cutting a corner): away from the centroid
        centroid = interior.centroid
        if (mid.x - centroid.x) * normal[0] + (mid.y - centroid.y) * normal[1] < 0:
            normal = -normal
    return direction_name(float(normal[1]), float(normal[0]), SIDE_NAMES)


def event_text(event: ChangeEvent, sides: list[BoundarySide]) -> str:
    when = (
        f"між {uk_date(event.date_from)} і {uk_date(event.date_to)}"
        if event.date_from and event.date_to
        else f"між {event.year_from} і {event.year_to} роками (точну дату за знімками не видно)"
    )
    what = {
        "appeared": "поверхня стала значно світлішою — з'явився новий об'єкт або тверде покриття",
        "removed": "поверхня потемніла — об'єкт або покриття зникло (знесення, демонтаж, заростання)",
        "changed": "спектр поверхні змінився без помітної зміни яскравості — перебудова або інше покриття",
    }[event.kind]
    inside = (
        f", з них у межах ділянки {event.inside_m2:.0f} м² ({event.parcel_share:.1%} ділянки)"
        if event.inside_share < 0.999
        else f" ({event.parcel_share:.1%} ділянки), цілком у межах ділянки"
    )
    parts = [
        f"Об'єкт {event.number}: {what}. Коли: {when}.",
        f"Де: {event.location}. Площа ~{event.area_m2:.0f} м²{inside}.",
        f"Було: {event.surface_before} (яскравість {event.brightness_before:.2f}); "
        f"стало: {event.surface_after} ({event.brightness_after:.2f}).",
    ]
    if event.radar_change_db is not None:
        parts.append(_radar_text(event.kind, event.radar_change_db))
    parts.append(f"Надійність виявлення: {event.confidence_label.uk}.")
    if event.on_boundary and event.side is not None:
        side = sides[event.side]
        parts.append(
            f"Об'єкт лежить на {side.name_locative}: {event.inside_share:.0%} його площі всередині ділянки, "
            f"{1 - event.inside_share:.0%} — за межею. Можливе заходження на ділянку (або з неї) — перевірте "
            "на знімку цієї межі."
        )
    return " ".join(parts)


def sides_with_events(
    sides: list[BoundarySide], events: list[ChangeEvent], years: tuple[int, int], min_object_m2: float
) -> list[BoundarySide]:
    result = []
    for side in sides:
        own = [e for e in events if e.side == side.index and e.on_boundary]
        text = side_text(own, years, min_object_m2)
        result.append(side.model_copy(update={"events": [e.number for e in own], "text": text}))
    return result


def _radar_text(kind: str, change_db: float) -> str:
    if abs(change_db) < RADAR_STRUCTURE_DB:
        meaning = {
            "appeared": (
                "висота об'єктів, схоже, не змінилась: скоріше покриття, майданчик або невисокий об'єкт"
            ),
            "removed": "висота об'єктів, схоже, не змінилась: прибрано покриття або невисокий об'єкт",
            "changed": "висота об'єктів, схоже, не змінилась",
        }[kind]
        return f"Радар Sentinel-1: відбиття майже не змінилось ({change_db:+.1f} дБ) — {meaning}."
    direction = f"зросло на {change_db:.1f} дБ" if change_db > 0 else f"впало на {-change_db:.1f} дБ"
    if kind == "appeared":
        meaning = (
            "підтверджує нову будівлю або споруду"
            if change_db > 0
            else "капітальну споруду не підтверджує: відбиття впало, скоріше змінилось покриття"
        )
    elif kind == "removed":
        meaning = (
            "підтверджує, що споруду прибрано"
            if change_db < 0
            else "знесення не підтверджує: можливо, змінилось лише покриття або складовано матеріали"
        )
    else:
        meaning = "щось збудовано або встановлено" if change_db > 0 else "щось прибрано"
    return f"Радар Sentinel-1: відбиття {direction} — {meaning}."


def side_text(side_events: list[ChangeEvent], years: tuple[int, int], min_object_m2: float) -> str:
    if not side_events:
        return (
            f"За {years[0]}–{years[1]} роки на цій межі змін, що лежать по обидва її боки, не виявлено "
            f"(знімки 10 м: об'єкти менші за ~{min_object_m2:.0f} м² можуть бути не помічені)."
        )
    items = ", ".join(
        f"об'єкт {e.number} ({e.kind_label.uk}, {e.year_to}, усередині {e.inside_m2:.0f} м²)"
        for e in side_events
    )
    return f"На цій межі виявлено {plural_uk(len(side_events), 'зміну', 'зміни', 'змін')}: {items}."


def _land_cover_finding(cover: list[CoverShare]) -> Finding:
    main = cover[0]
    others = [c for c in cover[1:] if c.share >= 0.1]
    text = (
        f"За картою покриву ESA WorldCover 2021 ділянка — переважно {main.label.uk.lower()} "
        f"({main.share:.0%})"
    )
    if others:
        text += ", також " + ", ".join(f"{c.label.uk.lower()} ({c.share:.0%})" for c in others)
    return Finding(code="land_cover", level="info", text=text + ".")


def _relief_finding(relief: Relief, buffer_m: float) -> Finding:
    e, s, a = relief.elevation_m, relief.slope_deg, relief.around_elevation_m
    text = (
        f"Висота ділянки {e.min:.0f}–{e.max:.0f} м над рівнем моря (перепад {e.range:.1f} м), "
        f"ухил у середньому {s.mean:.1f}°, найбільший {s.max:.1f}°"
    )
    if relief.flat:
        text += ": ділянка рівна."
    elif relief.low_side:
        text += f": є ухил, нижча частина — на {relief.low_side}."
    else:
        text += "."
    text += f" Довкола (до {buffer_m:.0f} м) висоти {a.min:.0f}–{a.max:.0f} м."
    return Finding(code="relief", level="info", text=text)


def _observations_finding(o: Observations, min_clear_share: float) -> Finding:
    if not (o.first_date and o.last_date):
        return Finding(
            code="no_clear_dates", level="warning", text="Жодного ясного знімка Sentinel-2 за період."
        )
    radar = (
        f", радар Sentinel-1: {plural_uk(o.radar_acquisitions, 'знімок', 'знімки', 'знімків')}"
        if o.radar_acquisitions
        else ""
    )
    return Finding(
        code="observations",
        level="info",
        text=(
            f"Знімки Sentinel-2 з {uk_date(o.first_date)} по {uk_date(o.last_date)}: "
            f"{plural_uk(o.scenes_read, 'сцена', 'сцени', 'сцен')}, з них "
            f"{plural_uk(o.dates_counted, 'ясна дата', 'ясні дати', 'ясних дат')} "
            f"(не менше {min_clear_share:.0%} площі без хмар){radar}."
        ),
    )


def _events_finding(events: list[ChangeEvent], years: tuple[int, int]) -> Finding:
    inside = [e for e in events if e.inside_share > 1 - STRADDLE_MIN_SHARE]
    nearby = [e for e in events if e.inside_share < STRADDLE_MIN_SHARE]
    if not inside:
        return Finding(
            code="no_changes",
            level="info",
            text=(
                f"За {years[0]}–{years[1]} роки в межах ділянки змін забудови чи покриття, помітних "
                "на знімках "
                "10 м (від ~200 м²), не виявлено."
            ),
        )
    kinds = {kind: sum(1 for e in inside if e.kind == kind) for kind in EVENT_KINDS}
    parts = [f"{EVENT_KINDS[k][0].uk}: {n}" for k, n in kinds.items() if n]
    total = sum(e.inside_m2 for e in inside)
    confident = sum(1 for e in inside if e.confidence == "high")
    nearby_text = (
        f" Ще {plural_uk(len(nearby), 'зміна', 'зміни', 'змін')} — поряд з ділянкою, за її межею."
        if nearby
        else ""
    )
    return Finding(
        code="changes",
        level="notice",
        text=(
            f"За {years[0]}–{years[1]} роки в межах ділянки виявлено "
            f"{plural_uk(len(inside), 'зміну', 'зміни', 'змін')} забудови або покриття ({'; '.join(parts)}), "
            f"разом ~{total:.0f} м²; з високою надійністю — {confident}. Дати й опис — у таблиці змін."
            f"{nearby_text}"
        ),
    )


def _boundary_finding(events: list[ChangeEvent]) -> Finding:
    on_boundary = [e for e in events if e.on_boundary]
    if not on_boundary:
        return Finding(
            code="no_encroachment",
            level="info",
            text=(
                "Об'єктів, що з'явились або зникли по обидва боки межі ділянки, за період не виявлено: ознак "
                "заходження на ділянку за знімками немає."
            ),
        )
    inside = sum(e.inside_m2 for e in on_boundary)
    objects = ("об'єкт", "об'єкти", "об'єктів")
    return Finding(
        code="encroachment",
        level="warning",
        text=(
            f"На межі ділянки виявлено {plural_uk(len(on_boundary), objects[0], objects[1], objects[2])}, "
            f"що {'лежить' if len(on_boundary) == 1 else 'лежать'} по обидва її боки, разом ~{inside:.0f} м² "
            "усередині ділянки: можливе заходження на ділянку. Подробиці — у розділі про межі."
        ),
    )


def field_details_findings(metrics: FieldDetailsMetrics, params: FieldDetailsParams) -> list[Finding]:
    findings: list[Finding] = []
    if metrics.land_cover:
        findings.append(_land_cover_finding(metrics.land_cover))
    if metrics.relief is not None:
        findings.append(_relief_finding(metrics.relief, params.relief_buffer_m))
    findings.append(_observations_finding(metrics.observations, params.min_clear_share))
    if metrics.years:
        years = (metrics.years[0].year, metrics.years[-1].year)
        findings.append(_events_finding(metrics.events, years))
        findings.append(_boundary_finding(metrics.events))
    return findings


def _rings(geometry: BaseGeometry) -> list[tuple[FloatArray, FloatArray]]:
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    rings = []
    for polygon in polygons:
        assert isinstance(polygon, Polygon)
        for ring in [polygon.exterior, *polygon.interiors]:
            x, y = ring.xy
            rings.append((np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")))
    return rings


def _lnglat_lines(geometry: BaseGeometry) -> tuple[list[float | None], list[float | None]]:
    lng: list[float | None] = []
    lat: list[float | None] = []
    for x, y in _rings(geometry):
        lng += [*x.tolist(), None]
        lat += [*y.tolist(), None]
    return lng, lat


def _zoom(bbox: tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = bbox
    width = max(maxx - minx, 1e-4) * IMAGERY_MARGIN
    height = max(maxy - miny, 1e-4) * IMAGERY_MARGIN / cos(radians((miny + maxy) / 2))
    by_width = log2(360 * IMAGERY_WIDTH_PX / (512 * width))
    by_height = log2(360 * IMAGERY_HEIGHT_PX / (512 * height))
    return round(min(by_width, by_height, 19), 2)


def _skip_failed[R](
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


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def _grid_outline(grid: ParcelGrid, geometry: BaseGeometry) -> pictures.Outline:
    t = grid.transform
    return [((x - t.c) / t.a - 0.5, (y - t.f) / t.e - 0.5) for x, y in _rings(geometry)]


class FieldDetailsTool(Tool[FieldDetailsParams, FieldDetailsMetrics]):
    name = "field_details"
    version = "2.0.0"
    title = Label(
        en="Parcel details for a report (imagery, buildings that appeared or were removed, boundary)",
        uk="Відомості про ділянку для звіту (знімки, зміни забудови, межі)",
    )
    maps_title = "Карти ділянки"
    params_model = FieldDetailsParams
    metrics_model = FieldDetailsMetrics

    def run(self, params: FieldDetailsParams, context: RunContext) -> ToolOutput[FieldDetailsMetrics]:
        parcel = context.parcel
        warnings: list[Message] = []
        end = params.end_date or datetime.now(UTC).astimezone().date()
        area = Parcel(geometry=parcel.in_crs("EPSG:4326", params.surroundings_m).iloc[0], name=parcel.name)
        grid = parcel_grid(area, GRID_RESOLUTION_M)
        parcel_shape = parcel.in_crs(grid.crs).iloc[0]
        inside: BoolArray = ~geometry_mask([parcel_shape], out_shape=grid.shape, transform=grid.transform)
        inside &= grid.inside

        cover, cover_items = worldcover.land_cover(parcel)
        land_cover = [CoverShare(code=c.code, label=c.label, share=round(c.share, 3)) for c in cover]
        relief, relief_items, relief_png = self._relief(parcel, params)

        items = sentinel2.least_cloudy_per_day(
            planetary.search(
                sentinel2.COLLECTION,
                area,
                time_range=f"{params.start_year}-01-01/{end.isoformat()}",
                query={"eo:cloud_cover": {"lt": params.max_scene_cloud}},
                max_items=MAX_SEARCH_ITEMS,
            )
        )
        found = len(items)
        items = thin(items, params.max_scenes)
        log.info("sentinel-2: %d scenes found, %d read", found, len(items))
        read = _skip_failed(lambda item: self._read_scene(item, area, grid, params), warnings)
        read_scenes = [s for s in planetary.read_parallel(read, items) if s is not None]
        scenes = sorted(
            (s for s in read_scenes if s.clear_share >= params.min_clear_share), key=lambda s: s.day
        )
        log.info("sentinel-2: %d clear dates", len(scenes))

        radar_years, radar_ids = self._radar(area, grid, params, end) if params.radar else ({}, [])

        composites = {
            year: values
            for year in range(params.start_year, end.year + 1)
            if (values := composite([s for s in scenes if s.day.year == year])) is not None
        }
        sides, side_lines = self._boundary(parcel, grid)
        events = self._events(
            composites=composites,
            scenes=scenes,
            grid=grid,
            inside=inside,
            parcel_shape=parcel_shape,
            side_lines=side_lines,
            sides=sides,
            radar_years=radar_years,
            params=params,
        )
        year_range = (min(composites), max(composites)) if composites else (params.start_year, end.year)
        sides = sides_with_events(sides, events, year_range, params.min_object_m2)

        outline = _grid_outline(grid, parcel_shape)
        years = self._year_composites(scenes, composites, outline, params.start_year, end.year)

        snapshots, snapshot_pictures = self._snapshots(parcel, scenes, params, warnings)
        observations = Observations(
            scenes_found=found,
            scenes_read=len(read_scenes),
            dates_counted=len(scenes),
            radar_acquisitions=len(radar_ids),
            first_date=scenes[0].day if scenes else None,
            last_date=scenes[-1].day if scenes else None,
        )
        if len(composites) < 2:
            warnings.append(
                Message(code="few_years", text="fewer than two years with clear images: no changes")
            )

        images = self._images(
            context=context,
            grid=grid,
            outline=outline,
            composites=composites,
            events=events,
            scenes=scenes,
            snapshot_pictures=snapshot_pictures,
            relief_png=relief_png,
            params=params,
        )
        metrics = FieldDetailsMetrics(
            land_cover=land_cover,
            relief=relief,
            observations=observations,
            years=years,
            events=events,
            boundary=sides,
            encroachment_events=[e.number for e in events if e.on_boundary],
            snapshots=snapshots,
            images=images,
        )
        summary = field_details_findings(metrics, params)
        sources = [
            worldcover.DATA_SOURCE.model_copy(update={"items": cover_items}),
            dem.DATA_SOURCE.model_copy(update={"items": relief_items}),
            sentinel2.DATA_SOURCE.model_copy(update={"items": [s.item.id for s in scenes]}),
        ]
        if params.radar:
            sources.append(sentinel1.DATA_SOURCE.model_copy(update={"items": radar_ids}))
        artifacts = self._interactive(
            context=context,
            grid=grid,
            composites=composites,
            scenes=scenes,
            events=events,
            sides=sides,
            land_cover=land_cover,
            summary=summary,
            sources=sources,
            images=images,
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
    def _year_composites(
        scenes: list[_Scene],
        composites: dict[int, FloatArray],
        outline: pictures.Outline,
        first_year: int,
        last_year: int,
    ) -> list[YearComposite]:
        years: list[YearComposite] = []
        for year in range(first_year, last_year + 1):
            of_year = [s for s in scenes if s.day.year == year]
            png = None
            if year in composites:
                png = pictures.composites_grid(
                    [(str(year), to_rgb(composites[year]))], outline=outline, caption=""
                )
            years.append(
                YearComposite(
                    year=year,
                    dates=len(of_year),
                    first_date=of_year[0].day if of_year else None,
                    last_date=of_year[-1].day if of_year else None,
                    png_base64=_b64(png) if png else None,
                )
            )
        return years

    @staticmethod
    def _read_scene(
        item: pystac.Item, area: Parcel, grid: ParcelGrid, params: FieldDetailsParams
    ) -> _Scene | None:
        scl = grid.put(planetary.read_window(item, "SCL", area, buffer_m=grid.resolution_m))
        covered = grid.inside & np.isfinite(scl) & (scl > 0)
        clear = covered & np.isin(scl, sentinel2.CLEAR_CLASSES) & (scl != 11)
        clear_share = float(clear.sum() / max(grid.cells, 1))
        if clear_share < params.min_clear_share:
            return None
        baseline = sentinel2.processing_baseline(item)
        bands = []
        for asset in BANDS:
            raster = planetary.read_window(item, asset, area, buffer_m=grid.resolution_m)
            values = grid.put(raster.where(raster > 0), Resampling.average)
            bands.append(sentinel2.reflectance(values, baseline))
        stack = np.stack(bands)
        stack[:, ~clear] = np.nan
        return _Scene(item, stack, clear_share)

    @staticmethod
    def _radar(
        area: Parcel, grid: ParcelGrid, params: FieldDetailsParams, end: date
    ) -> tuple[dict[int, FloatArray], list[str]]:
        items = [
            item
            for item in planetary.search(
                sentinel1.COLLECTION,
                area,
                time_range=f"{params.start_year}-01-01/{end.isoformat()}",
                max_items=MAX_SEARCH_ITEMS,
            )
            if item.datetime is not None and item.datetime.month in RADAR_MONTHS
        ]
        chosen: list[pystac.Item] = []
        for year in range(params.start_year, end.year + 1):
            year_items = sorted(
                (i for i in items if i.datetime is not None and i.datetime.year == year),
                key=lambda i: i.datetime or datetime.min.replace(tzinfo=UTC),
            )
            chosen += thin(year_items, RADAR_PER_YEAR)
        warnings: list[Message] = []
        read = _skip_failed(lambda item: sentinel1.scene(item, area, grid), warnings)
        scenes = [s for s in planetary.read_parallel(read, chosen) if s is not None]
        log.info("sentinel-1: %d acquisitions", len(scenes))
        by_year: dict[int, FloatArray] = {}
        for year in range(params.start_year, end.year + 1):
            year_scenes = [s for s in scenes if s.day.year == year]
            if len(year_scenes) < 3:
                continue
            with np.errstate(all="ignore"):
                by_year[year] = 10 * np.log10(np.nanmedian(np.stack([s.vv for s in year_scenes]), axis=0))
        return by_year, [s.item_id for s in scenes]

    @staticmethod
    def _boundary(parcel: Parcel, grid: ParcelGrid) -> tuple[list[BoundarySide], list[LineString]]:
        shape = parcel.in_crs(grid.crs).iloc[0]
        polygon = max(shape.geoms, key=lambda p: p.area) if isinstance(shape, MultiPolygon) else shape
        assert isinstance(polygon, Polygon)
        lines = boundary_sides([(c[0], c[1]) for c in polygon.exterior.coords])
        to_wgs84 = gpd.GeoSeries(lines, crs=grid.crs).to_crs("EPSG:4326")
        sides = []
        for index, (line, wgs) in enumerate(zip(lines, to_wgs84, strict=True)):
            letters = f"{LETTERS[index % len(LETTERS)]}–{LETTERS[(index + 1) % len(LETTERS)]}"
            direction = side_direction(line, polygon)
            padded = (
                gpd.GeoSeries([line.buffer(BOUNDARY_PAD_M)], crs=grid.crs).to_crs("EPSG:4326").total_bounds
            )
            (x0, y0), (x1, y1) = wgs.coords
            sides.append(
                BoundarySide(
                    index=index,
                    letters=letters,
                    name=f"{direction} межа ({letters})",
                    name_locative=f"{direction[:-1]}ій межі ({letters})",
                    direction=direction,
                    length_m=round(float(line.length), 1),
                    start=(round(x0, 7), round(y0, 7)),
                    end=(round(x1, 7), round(y1, 7)),
                    bbox=(
                        round(float(padded[0]), 7),
                        round(float(padded[1]), 7),
                        round(float(padded[2]), 7),
                        round(float(padded[3]), 7),
                    ),
                    events=[],
                    text="",
                )
            )
        return sides, lines

    def _events(
        self,
        *,
        composites: dict[int, FloatArray],
        scenes: list[_Scene],
        grid: ParcelGrid,
        inside: BoolArray,
        parcel_shape: BaseGeometry,
        side_lines: list[LineString],
        sides: list[BoundarySide],
        radar_years: dict[int, FloatArray],
        params: FieldDetailsParams,
    ) -> list[ChangeEvent]:
        years = sorted(composites)
        if len(years) < 2:
            return []
        min_cells = max(round(params.min_object_m2 / grid.resolution_m**2), 1)
        baseline = composites[years[0]].copy()
        baseline_year = np.full(grid.shape, years[0])
        objects: list[_Object] = []
        for previous, year in pairwise(years):
            current = composites[year]
            for cells in changed_objects(
                baseline, current, grid.inside, threshold=params.change_threshold, min_cells=min_cells
            ):
                if not (cells & inside).any():
                    continue
                objects.append(_Object(cells, int(np.median(baseline_year[cells])), year))
            changed = spectral_distance(baseline, current) > params.change_threshold
            baseline[:, changed] = current[:, changed]
            baseline_year[changed] = year
            del previous

        return [
            self._event_of(
                number,
                obj,
                composites=composites,
                scenes=scenes,
                grid=grid,
                inside=inside,
                parcel_shape=parcel_shape,
                side_lines=side_lines,
                sides=sides,
                radar_years=radar_years,
            )
            for number, obj in enumerate(objects, start=1)
        ]

    @staticmethod
    def _event_of(
        number: int,
        obj: _Object,
        *,
        composites: dict[int, FloatArray],
        scenes: list[_Scene],
        grid: ParcelGrid,
        inside: BoolArray,
        parcel_shape: BaseGeometry,
        side_lines: list[LineString],
        sides: list[BoundarySide],
        radar_years: dict[int, FloatArray],
    ) -> ChangeEvent:
        cell_m2 = grid.resolution_m**2
        before = np.nanmean(composites[obj.year_from][:, obj.cells], axis=1)
        after = np.nanmean(composites[obj.year_to][:, obj.cells], axis=1)
        brightness_before = float(before[list(BRIGHTNESS_BANDS)].mean())
        brightness_after = float(after[list(BRIGHTNESS_BANDS)].mean())
        kind = change_kind(brightness_before, brightness_after)
        seen = [s for s in scenes if np.isfinite(s.brightness[obj.cells]).any()]
        date_from, date_to = date_change(
            [s.day for s in seen],
            [float(np.nanmean(s.brightness[obj.cells])) for s in seen],
            before=brightness_before,
            after=brightness_after,
            year_to=obj.year_to,
        )
        inside_cells = int((obj.cells & inside).sum())
        total_cells = int(obj.cells.sum())
        inside_share = inside_cells / total_cells
        on_boundary = STRADDLE_MIN_SHARE <= inside_share <= 1 - STRADDLE_MIN_SHARE
        geometry = shapely.unary_union(
            [
                shapely.geometry.shape(g)
                for g, v in shapes(obj.cells.astype("uint8"), mask=obj.cells, transform=grid.transform)
                if v == 1
            ]
        )
        side_index = int(np.argmin([line.distance(geometry) for line in side_lines])) if side_lines else None
        centroid, object_centroid = parcel_shape.centroid, geometry.centroid
        direction = direction_name(object_centroid.y - centroid.y, object_centroid.x - centroid.x)
        if on_boundary and side_index is not None:
            location = f"на {sides[side_index].name_locative}, з боку: {direction}"
        elif inside_share < STRADDLE_MIN_SHARE:
            location = f"за межею ділянки, поряд з нею ({direction})"
        else:
            location = f"у частині ділянки: {direction}"
        radar_change = None
        if obj.year_from in radar_years and obj.year_to in radar_years:
            delta = radar_years[obj.year_to][obj.cells] - radar_years[obj.year_from][obj.cells]
            if np.isfinite(delta).any():
                radar_change = round(float(np.nanmean(delta)), 1)
        label, color = EVENT_KINDS[kind]
        confidence = confidence_of(
            total_cells, brightness_after - brightness_before, radar_agrees(kind, radar_change)
        )
        event = ChangeEvent(
            number=number,
            kind=kind,
            kind_label=label,
            color=STRADDLE_COLOR if on_boundary else color,
            date_from=date_from,
            date_to=date_to,
            year_from=obj.year_from,
            year_to=obj.year_to,
            area_m2=round(total_cells * cell_m2),
            inside_m2=round(inside_cells * cell_m2),
            inside_share=round(inside_share, 3),
            parcel_share=round(inside_cells / max(int(inside.sum()), 1), 4),
            location=location,
            on_boundary=on_boundary,
            side=side_index if on_boundary else None,
            brightness_before=round(brightness_before, 3),
            brightness_after=round(brightness_after, 3),
            surface_before=surface_name(before),
            surface_after=surface_name(after),
            radar_change_db=radar_change,
            confidence=confidence,
            confidence_label=CONFIDENCE_LABELS[confidence],
            text="",
            geometry=shapely.geometry.mapping(
                gpd.GeoSeries([geometry], crs=grid.crs).to_crs("EPSG:4326").iloc[0]
            ),
        )
        return event.model_copy(update={"text": event_text(event, sides)})

    def _snapshots(
        self, parcel: Parcel, scenes: list[_Scene], params: FieldDetailsParams, warnings: list[Message]
    ) -> tuple[list[Snapshot], list[tuple[date, Rgb, pictures.Outline]]]:
        by_day = {s.day: s for s in scenes}
        chosen: list[date] = []
        edges = np.linspace(SNAPSHOT_FIRST_DOY, SNAPSHOT_LAST_DOY + 1, params.snapshots_per_year + 1)
        for year in sorted({s.day.year for s in scenes}):
            for low, high in pairwise(edges):
                candidates = [
                    s for s in scenes if s.day.year == year and low <= s.day.timetuple().tm_yday < high
                ]
                if candidates:
                    chosen.append(max(candidates, key=lambda s: (s.clear_share, -s.scene_cloud)).day)
        read_rgb = _skip_failed(
            lambda item: self._true_color(item, parcel, params.snapshot_buffer_m), warnings
        )
        images = planetary.read_parallel(read_rgb, [by_day[d].item for d in chosen])
        snapshots: list[Snapshot] = []
        pictures_in: list[tuple[date, Rgb, pictures.Outline]] = []
        for day, image in zip(chosen, images, strict=True):
            if image is None:
                continue
            rgb, outline = image
            if float(np.median(rgb)) > SNAPSHOT_MAX_BRIGHTNESS:
                log.info("snapshot %s skipped: snow or haze", day)
                continue
            png = pictures.snapshots_grid([(day, rgb, outline)], columns=1, caption="")
            scene = by_day[day]
            snapshots.append(
                Snapshot(
                    date=day,
                    scene_id=scene.item.id,
                    clear_share=round(scene.clear_share, 3),
                    png_base64=_b64(png),
                )
            )
            pictures_in.append((day, rgb, outline))
        return snapshots, pictures_in

    @staticmethod
    def _true_color(item: pystac.Item, parcel: Parcel, buffer_m: float) -> tuple[Rgb, pictures.Outline]:
        bands = [
            planetary.read_window(item, asset, parcel, buffer_m=buffer_m) for asset in ("B04", "B03", "B02")
        ]
        baseline = sentinel2.processing_baseline(item)
        stack = np.dstack([sentinel2.reflectance(band.values, baseline) for band in bands])
        stretched = np.clip(stack / TRUE_COLOR_MAX, 0, 1) ** TRUE_COLOR_GAMMA
        rgb: Rgb = (stretched * 255).round().astype("uint8")
        reference = bands[0]
        transform = reference.rio.transform()
        geometry = parcel.in_crs(reference.rio.crs).iloc[0]
        outline = [
            ((x - transform.c) / transform.a - 0.5, (y - transform.f) / transform.e - 0.5)
            for x, y in _rings(geometry)
        ]
        return rgb, outline

    def _relief(
        self, parcel: Parcel, params: FieldDetailsParams
    ) -> tuple[Relief | None, list[str], bytes | None]:
        self._relief_layers: list[interactive.MapLayer] = []
        self._relief_geometry: interactive.GridGeometry | None = None
        try:
            window, items = dem.elevation(parcel, buffer_m=params.relief_buffer_m)
        except LookupError as error:
            log.warning("relief: %s", error)
            return None, [], None
        heights = window.values.astype("float64")
        slope, _aspect = dem.slope_aspect(heights, dem.RESOLUTION_M)
        inside: BoolArray = ~geometry_mask(
            parcel.in_crs(window.rio.crs).geometry, out_shape=window.shape, transform=window.rio.transform()
        ) & np.isfinite(heights)
        if not inside.any():
            return None, items, None
        z, s = heights[inside], slope[inside]
        around = heights[np.isfinite(heights)]
        rows, cols = np.nonzero(inside)
        ys, xs = window.y.values[rows], window.x.values[cols]
        low, high = z <= np.percentile(z, 33), z >= np.percentile(z, 67)
        p90 = float(np.percentile(s, 90))
        flat = p90 < FLAT_SLOPE_DEG
        low_side = None
        if not flat and float(z.max() - z.min()) >= MIN_RELIEF_RANGE_M:
            low_side = direction_name(
                float(ys[low].mean() - ys[high].mean()), float(xs[low].mean() - xs[high].mean())
            )

        def elevation_of(values: FloatArray) -> Elevation:
            return Elevation(
                min=round(float(values.min()), 1),
                max=round(float(values.max()), 1),
                mean=round(float(values.mean()), 1),
                range=round(float(values.max() - values.min()), 1),
            )

        relief = Relief(
            elevation_m=elevation_of(z),
            around_elevation_m=elevation_of(around),
            slope_deg=Slope(mean=round(float(s.mean()), 2), p90=round(p90, 2), max=round(float(s.max()), 2)),
            flat=flat,
            low_side=low_side,
        )
        step = (
            CONTOUR_STEP_LARGE_M if relief.around_elevation_m.range > LARGE_RANGE_M else CONTOUR_STEP_SMALL_M
        )
        shown = np.isfinite(heights)
        self._relief_layers = [
            interactive.MapLayer(
                title=f"Висота (горизонталі через {step:g} м)",
                hover_label="Висота",
                unit="м",
                values=np.where(shown, heights, np.nan),
                colorscale="terrain",
                contour_step=step,
                decimals=1,
            ),
            interactive.MapLayer(
                title="Ухил",
                hover_label="Ухил",
                unit="°",
                values=np.where(shown, slope, np.nan),
                colorscale="magma_r",
                zmin=0,
                zmax=max(3.0, float(np.nanmax(slope))),
                decimals=1,
            ),
        ]
        self._relief_geometry = interactive.grid_geometry(
            window.x.values, window.y.values, window.rio.crs, shown
        )
        x0, x1 = float(window.x.values.min()), float(window.x.values.max())
        y0, y1 = float(window.y.values.min()), float(window.y.values.max())
        half = dem.RESOLUTION_M / 2
        extent = (-half, x1 + half - x0, -half, y1 + half - y0)
        outline = [(x - x0, y - y0) for x, y in _rings(parcel.in_crs(window.rio.crs).iloc[0])]
        png = pictures.relief_maps(
            heights=heights, slope=slope, extent=extent, outline=outline, contour_step_m=step
        )
        return relief, items, png

    def _images(
        self,
        *,
        context: RunContext,
        grid: ParcelGrid,
        outline: pictures.Outline,
        composites: dict[int, FloatArray],
        events: list[ChangeEvent],
        scenes: list[_Scene],
        snapshot_pictures: list[tuple[date, Rgb, pictures.Outline]],
        relief_png: bytes | None,
        params: FieldDetailsParams,
    ) -> list[ImageOut]:
        images: list[ImageOut] = []

        def add(name: str, title: str, description: str, png: bytes) -> None:
            (context.workdir / f"{name}.png").write_bytes(png)
            images.append(ImageOut(name=name, title=title, description=description, png_base64=_b64(png)))

        if composites:
            add(
                "composites",
                "Ділянка по роках (Sentinel-2)",
                "Для кожного року — медіана ясних знімків квітня–жовтня у справжніх кольорах, 10 м; "
                "контур ділянки білим. Так видно, що змінилось від року до року.",
                pictures.composites_grid(
                    [(str(y), to_rgb(c)) for y, c in composites.items()], outline=outline, caption=""
                ),
            )
        if events and composites:
            latest = to_rgb(composites[max(composites)])
            objects = []
            for e in events:
                shape = (
                    gpd.GeoSeries([shapely.geometry.shape(e.geometry)], crs="EPSG:4326")
                    .to_crs(grid.crs)
                    .iloc[0]
                )
                objects.append((e.number, _grid_outline(grid, shape), e.color))
            add(
                "objects",
                "Виявлені зміни на останньому композиті",
                "Контури об'єктів, що з'явились (помаранчевий), зникли (синій), змінили поверхню "
                "(фіолетовий) "
                "або лежать на межі ділянки (жовтий), з номерами з таблиці змін.",
                pictures.objects_map(
                    latest,
                    outline=outline,
                    objects=objects,
                    caption=f"Знімок-композит {max(composites)} року",
                ),
            )
            series = []
            for e in events:
                cells = ~geometry_mask(
                    [
                        gpd.GeoSeries([shapely.geometry.shape(e.geometry)], crs="EPSG:4326")
                        .to_crs(grid.crs)
                        .iloc[0]
                    ],
                    out_shape=grid.shape,
                    transform=grid.transform,
                )
                days = [s.day for s in scenes if np.isfinite(s.brightness[cells]).any()]
                values = [
                    float(np.nanmean(s.brightness[cells]))
                    for s in scenes
                    if np.isfinite(s.brightness[cells]).any()
                ]
                window = (e.date_from, e.date_to) if e.date_from and e.date_to else None
                series.append((e.number, f"{e.kind_label.uk}, {e.year_to}", days, values, e.color, window))
            add(
                "object_series",
                "Яскравість поверхні об'єктів за всі ясні дати",
                "Для кожного об'єкта — середнє відбиття у видимих каналах його клітинок на кожну ясну дату; "
                "затінено вікно дат, коли сталася зміна.",
                pictures.object_series(
                    series=series, title="Яскравість поверхні об'єктів (0 — чорне, 0,3 — дуже світле)"
                ),
            )
        if snapshot_pictures:
            add(
                "snapshots",
                "Знімки Sentinel-2 по роках",
                f"Справжні кольори, 10 м, до {params.snapshots_per_year} на рік; контур ділянки білим.",
                pictures.snapshots_grid(snapshot_pictures, columns=params.snapshots_per_year, caption=""),
            )
        if relief_png is not None:
            add(
                "relief",
                "Рельєф",
                f"Висота з горизонталями і ухил за Copernicus DEM 30 м, "
                f"{params.relief_buffer_m:.0f} м довкола.",
                relief_png,
            )
        return images

    def _interactive(
        self,
        *,
        context: RunContext,
        grid: ParcelGrid,
        composites: dict[int, FloatArray],
        scenes: list[_Scene],
        events: list[ChangeEvent],
        sides: list[BoundarySide],
        land_cover: list[CoverShare],
        summary: list[Finding],
        sources: list[DataSource],
        images: list[ImageOut],
    ) -> list[Artifact]:
        parcel = context.parcel
        figures: dict[str, go.Figure] = {"imagery": self._imagery_figure(parcel, events, sides)}
        if composites:
            figures["composites"] = self._composites_figure(grid, parcel, composites)
        if events:
            figures["events"] = interactive.table_figure(
                "Зміни забудови і покриття",
                ["№", "коли", "що", "де", "площа, м²", "у ділянці, м²", "радар, дБ"],
                [
                    [
                        str(e.number),
                        f"{uk_date(e.date_from)} – {uk_date(e.date_to)}"
                        if e.date_from and e.date_to
                        else f"{e.year_from}–{e.year_to}",
                        e.kind_label.uk,
                        e.location,
                        f"{e.area_m2:.0f}",
                        f"{e.inside_m2:.0f}",
                        f"{e.radar_change_db:+.1f}" if e.radar_change_db is not None else "—",
                    ]
                    for e in events
                ],
            )
            figures["object_series"] = self._series_figure(grid, scenes, events)
        if sides:
            figures["boundary"] = interactive.table_figure(
                "Межі ділянки",
                ["межа", "напрямок", "довжина, м", "що виявлено"],
                [[s.name, s.direction, f"{s.length_m:.0f}", s.text] for s in sides],
                column_widths=[1.2, 1, 0.8, 4],
            )
        if self._relief_layers and self._relief_geometry is not None:
            figures["relief"] = interactive.map_panels(parcel, self._relief_geometry, self._relief_layers)
            figures["relief"].update_layout(title="Рельєф довкола ділянки (Copernicus DEM 30 м)")
        if land_cover:
            figures["land_cover"] = interactive.bar_chart(
                "Покрив землі (ESA WorldCover 2021)",
                [c.label.uk for c in land_cover],
                [c.share for c in land_cover],
                [COVER_COLORS.get(c.code, "#999999") for c in land_cover],
            )
        interactive.write_page(
            context.workdir / HTML_FILE,
            title=f"{parcel.name}: відомості про ділянку",
            subtitle=(
                f"{parcel.area_ha:.2f} га · Sentinel-2, Sentinel-1, ESA WorldCover, Copernicus DEM, "
                "супутникова підкладка"
            ),
            summary=summary,
            figures=list(figures.values()),
            sources=sources,
        )
        interactive.write_figures_json(context.workdir / FIGURES_FILE, figures)
        return [
            Artifact(
                path=HTML_FILE, media_type="text/html", description="Інтерактивна сторінка з усіма графіками."
            ),
            Artifact(path=FIGURES_FILE, media_type="application/json", description="Специфікації Plotly."),
            *[
                Artifact(path=f"{i.name}.png", media_type="image/png", description=i.description)
                for i in images
            ],
        ]

    @staticmethod
    def _imagery_figure(parcel: Parcel, events: list[ChangeEvent], sides: list[BoundarySide]) -> go.Figure:
        figure = go.Figure()
        for e in events:
            lng, lat = _lnglat_lines(shapely.geometry.shape(e.geometry))
            figure.add_trace(
                go.Scattermap(
                    lon=lng,
                    lat=lat,
                    mode="lines",
                    fill="toself",
                    fillcolor=interactive._with_alpha(e.color, 0.45),
                    line={"color": e.color, "width": 1},
                    name=f"{e.number}: {e.kind_label.uk}, {e.year_to}",
                    hovertemplate=e.text.replace(". ", ".<br>") + "<extra></extra>",
                )
            )
        lng, lat = _lnglat_lines(parcel.geometry)
        figure.add_trace(
            go.Scattermap(
                lon=lng,
                lat=lat,
                mode="lines",
                line={"color": OUTLINE_COLOR, "width": 3},
                name="межа ділянки",
                hoverinfo="skip",
            )
        )
        if sides:
            figure.add_trace(
                go.Scattermap(
                    lon=[(s.start[0] + s.end[0]) / 2 for s in sides],
                    lat=[(s.start[1] + s.end[1]) / 2 for s in sides],
                    mode="text",
                    text=[s.letters for s in sides],
                    textfont={"color": "#ffffff", "size": 12},
                    name="сторони межі",
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=[s.name for s in sides],
                )
            )
        minx, miny, maxx, maxy = parcel.bbox
        figure.update_layout(
            title="Ділянка на супутниковому знімку",
            height=IMAGERY_HEIGHT_PX,
            margin={"l": 0, "r": 0, "t": 40, "b": 30},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
            map={
                "style": "white-bg",
                "center": {"lon": (minx + maxx) / 2, "lat": (miny + maxy) / 2},
                "zoom": _zoom(parcel.bbox),
                "layers": interactive.imagery_layers(),
            },
        )
        return figure

    @staticmethod
    def _composites_figure(grid: ParcelGrid, parcel: Parcel, composites: dict[int, FloatArray]) -> go.Figure:
        labels = [str(year) for year in composites]
        frames = [to_rgb(c) for c in composites.values()]
        outline_x: list[float | None] = []
        outline_y: list[float | None] = []
        for x, y in _grid_outline(grid, parcel.in_crs(grid.crs).iloc[0]):
            outline_x += [*x.tolist(), None]
            outline_y += [*y.tolist(), None]
        title = "Ділянка по роках: медіана ясних знімків квітня–жовтня"
        figure = go.Figure(
            [
                go.Image(z=frames[-1], hoverinfo="skip"),
                go.Scatter(
                    x=outline_x,
                    y=outline_y,
                    mode="lines",
                    line={"color": OUTLINE_COLOR, "width": 1.5},
                    hoverinfo="skip",
                ),
            ]
        )
        figure.update_layout(
            title=f"{title}: {labels[-1]}",
            height=560,
            margin={"l": 20, "r": 20, "t": 60, "b": 30},
            showlegend=False,
            xaxis={"visible": False},
            yaxis={"visible": False, "scaleanchor": "x"},
            sliders=[
                {
                    "active": len(labels) - 1,
                    "currentvalue": {"prefix": "Рік: "},
                    "pad": {"t": 30},
                    "steps": [
                        {
                            "label": label,
                            "method": "update",
                            "args": [{"z": [rgb]}, {"title": f"{title}: {label}"}, [0]],
                        }
                        for label, rgb in zip(labels, frames, strict=True)
                    ],
                }
            ],
        )
        return figure

    @staticmethod
    def _series_figure(grid: ParcelGrid, scenes: list[_Scene], events: list[ChangeEvent]) -> go.Figure:
        series = []
        for e in events:
            shape = (
                gpd.GeoSeries([shapely.geometry.shape(e.geometry)], crs="EPSG:4326").to_crs(grid.crs).iloc[0]
            )
            cells = ~geometry_mask([shape], out_shape=grid.shape, transform=grid.transform)
            seen = [s for s in scenes if np.isfinite(s.brightness[cells]).any()]
            series.append(
                interactive.Series(
                    name=f"об'єкт {e.number}",
                    x=[s.day for s in seen],
                    y=[float(np.nanmean(s.brightness[cells])) for s in seen],
                    color=e.color,
                )
            )
        figure = interactive.line_chart(
            "Яскравість поверхні об'єктів за всі ясні дати", series, y_title="відбиття у видимих каналах"
        )
        for e in events:
            if e.date_from and e.date_to:
                figure.add_vrect(
                    x0=e.date_from.isoformat(),
                    x1=e.date_to.isoformat(),
                    fillcolor=e.color,
                    opacity=0.15,
                    line_width=0,
                    annotation_text=str(e.number),
                    annotation_position="top left",
                )
        return figure
