import logging
from collections import Counter
from math import cos, log2, radians
from typing import Any, Literal, cast

import geopandas as gpd
import plotly.graph_objects as go
import shapely
from pydantic import Field
from shapely.geometry import MultiPolygon, Polygon, mapping

from argo_geoai import interactive
from argo_geoai.analysis.outline import straighten
from argo_geoai.parcels import WGS84, Parcel
from argo_geoai.schema import Artifact, Finding, Label, Message, Model
from argo_geoai.sources import field_boundaries
from argo_geoai.sources.field_boundaries import FieldRecord
from argo_geoai.tools.base import RunContext, Tool, ToolOutput
from argo_geoai.tools.vegetation import plural_uk

log = logging.getLogger(__name__)

Relation = Literal["inside", "crosses", "contains", "touches"]

STRAIGHTEN_M = 7.0
MIN_AREA_HA = 1.0
COORDINATE_DECIMALS = 7
RELIABLE_CONFIDENCE = 69
COVERED_SHARE_NOTICE = 0.8
NEGLIGIBLE_SHARE = 0.01
GEOJSON_FILE = "fields.geojson"
HTML_FILE = "search_fields.html"
FIGURES_FILE = "search_fields_figures.json"
TABLE_MAX_ROWS = 300

RELATIONS: dict[Relation, tuple[Label, str]] = {
    "inside": (Label(en="inside the polygon", uk="всередині ділянки"), "#2b8a3e"),
    "crosses": (Label(en="crosses the polygon edge", uk="перетинає межу ділянки"), "#f08c00"),
    "contains": (Label(en="holds the whole polygon", uk="ділянка всередині поля"), "#1971c2"),
    "touches": (Label(en="touches the edge only", uk="лише торкається межі"), "#868e96"),
}
MAP_WIDTH_PX = 700
MAP_HEIGHT_PX = 760


class SearchFieldsParams(Model):
    year: int | None = Field(
        None, ge=2024, description="year of the field map; the latest one in the data by default"
    )
    min_confidence: float = Field(
        0,
        ge=0,
        le=100,
        description=(
            "skip fields with a lower FTW model confidence (0-100; 69 is the FTW reliability filter); "
            "fields without a confidence value (outside the FTW confidence layer) are kept"
        ),
    )
    straighten_m: float = Field(
        STRAIGHTEN_M,
        ge=0,
        le=20,
        description=(
            "straighten the staircase edges of the returned outlines: the outline goes through the "
            "middles of the 10 m steps and drops points closer than this to a straight line, m; "
            "0: the source outline. Areas, overlaps and relations are always of the source outline"
        ),
    )
    min_area_ha: float = Field(
        MIN_AREA_HA,
        ge=0,
        description="skip fields smaller than this (by the source outline), ha; 0: keep every polygon",
    )


class FieldGeometry(Model):
    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list[Any] = Field(description="GeoJSON coordinates, lng/lat (EPSG:4326)")


class FieldProperties(Model):
    area_ha: float = Field(description="area of the whole field")
    overlap_ha: float = Field(description="part of the field inside the polygon")
    overlap_share: float = Field(description="overlap_ha / area_ha")
    relation: Relation
    relation_label: Label
    confidence: float | None = Field(
        description="FTW model confidence 0-100; null outside the FTW confidence layer (not a low value)"
    )
    year: int


class FieldFeature(Model):
    type: Literal["Feature"] = "Feature"
    id: str = Field(description="FTW field id")
    geometry: FieldGeometry
    properties: FieldProperties


class FieldCollection(Model):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[FieldFeature] = Field(description="largest overlap first")


class SearchFieldsMetrics(Model):
    year: int | None = Field(description="year of the field map used; null when no data")
    years_available: list[int] = Field(description="years of the field map around the polygon")
    fields_found: int
    counts: dict[str, int] = Field(description="fields per relation")
    fields_area_ha: float = Field(description="sum of the whole areas of the found fields")
    covered_share: float = Field(description="share of the polygon under the found fields")
    low_confidence: int = Field(description=f"fields with a confidence below {RELIABLE_CONFIDENCE}")
    small_skipped: int = Field(
        0, description="polygons that met the parcel but were smaller than min_area_ha"
    )
    fields: FieldCollection


class SearchFieldsTool(Tool[SearchFieldsParams, SearchFieldsMetrics]):
    name = "search_fields"
    version = "1.2.0"
    title = Label(en="Fields around the polygon", uk="Поля навколо ділянки")
    maps_title = "Знайдені поля"
    params_model = SearchFieldsParams
    metrics_model = SearchFieldsMetrics

    def run(self, params: SearchFieldsParams, context: RunContext) -> ToolOutput[SearchFieldsMetrics]:
        parcel = context.parcel
        records, partitions = field_boundaries.search(parcel.bbox)
        warnings = []
        if not partitions:
            warnings.append(
                Message(
                    code="outside_field_map", text="the polygon is outside the countries of the field map"
                )
            )
        years = sorted({record.year for record in records})
        year = params.year if params.year is not None else (years[-1] if years else None)
        if params.year is not None and params.year not in years and partitions:
            warnings.append(
                Message(
                    code="year_not_available",
                    text=f"no field map of {params.year} here; available: {years or 'none'}",
                )
            )
        selected = [
            record
            for record in records
            if record.year == year
            and (record.confidence is None or record.confidence >= params.min_confidence)
        ]
        metrics = search_metrics(
            parcel,
            selected,
            year=year,
            years=years,
            straighten_m=params.straighten_m,
            min_area_ha=params.min_area_ha,
        )
        summary = search_findings(metrics, covered=bool(partitions), min_area_ha=params.min_area_ha)

        (context.workdir / GEOJSON_FILE).write_text(metrics.fields.model_dump_json() + "\n", encoding="utf-8")
        figures = {"maps": fields_map(parcel, metrics.fields)}
        if metrics.fields.features:
            figures["table"] = fields_table(metrics.fields)
        interactive.write_page(
            context.workdir / HTML_FILE,
            title=f"{self.title.uk}: {parcel.name}",
            subtitle=f"Карта полів FTW {year or '—'}, ділянка {parcel.area_ha:.1f} га",
            summary=summary,
            figures=list(figures.values()),
            sources=[field_boundaries.DATA_SOURCE],
            note=(
                "Межі полів — результат моделі за супутниковими знімками, не кадастр. Підкладка — "
                "супутникові знімки іншої дати, щоб порівняти межі на око."
            ),
        )
        interactive.write_figures_json(context.workdir / FIGURES_FILE, figures)
        return ToolOutput(
            metrics=metrics,
            summary=summary,
            artifacts=[
                Artifact(
                    path=GEOJSON_FILE,
                    media_type="application/geo+json",
                    description="Знайдені поля: GeoJSON FeatureCollection з площею кожного поля.",
                ),
                Artifact(
                    path=HTML_FILE,
                    media_type="text/html",
                    description="Інтерактивна карта знайдених полів на супутниковій підкладці і їх таблиця.",
                ),
                Artifact(
                    path=FIGURES_FILE,
                    media_type="application/json",
                    description="Специфікації графіків Plotly для іншого фронтенду.",
                ),
            ],
            data_sources=[field_boundaries.DATA_SOURCE.model_copy(update={"items": partitions})],
            warnings=warnings,
        )


def relation_of(parcel: Polygon | MultiPolygon, field: Polygon | MultiPolygon) -> Relation | None:
    if not parcel.intersects(field):
        return None
    if parcel.covers(field):
        return "inside"
    if field.covers(parcel):
        return "contains"
    if parcel.touches(field):
        return "touches"
    return "crosses"


def search_metrics(
    parcel: Parcel,
    records: list[FieldRecord],
    *,
    year: int | None,
    years: list[int],
    straighten_m: float = STRAIGHTEN_M,
    min_area_ha: float = MIN_AREA_HA,
) -> SearchFieldsMetrics:
    meeting = [
        (record, relation)
        for record in records
        if (relation := relation_of(parcel.geometry, record.geometry))
    ]
    projected = gpd.GeoSeries([record.geometry for record, _ in meeting], crs=WGS84).to_crs(parcel.utm_crs)
    large = (projected.area >= min_area_ha * 10_000).to_numpy()
    found = [pair for pair, keep in zip(meeting, large, strict=True) if keep]
    shapes = projected[large]
    area = parcel.in_crs(parcel.utm_crs).iloc[0]
    outlines = gpd.GeoSeries(
        [straighten(shape, straighten_m) for shape in shapes], crs=parcel.utm_crs
    ).to_crs(WGS84)
    features = []
    for (record, relation), field, outline in zip(found, shapes, outlines, strict=True):
        field_ha = field.area / 10_000
        overlap_ha = field.intersection(area).area / 10_000
        features.append(
            FieldFeature(
                id=record.id,
                geometry=_geometry(outline),
                properties=FieldProperties(
                    area_ha=round(field_ha, 2),
                    overlap_ha=round(overlap_ha, 2),
                    overlap_share=round(overlap_ha / field_ha, 3) if field_ha else 0.0,
                    relation=relation,
                    relation_label=RELATIONS[relation][0],
                    confidence=None if record.confidence is None else round(record.confidence, 1),
                    year=record.year,
                ),
            )
        )
    features.sort(key=lambda feature: (-feature.properties.overlap_ha, -feature.properties.area_ha))
    covered = shapely.union_all(list(shapes)).intersection(area).area if len(shapes) else 0.0
    counts = Counter(feature.properties.relation for feature in features)
    return SearchFieldsMetrics(
        year=year,
        years_available=years,
        fields_found=len(features),
        counts={relation: counts.get(relation, 0) for relation in RELATIONS},
        fields_area_ha=round(sum(feature.properties.area_ha for feature in features), 2),
        covered_share=round(covered / area.area, 3) if area.area else 0.0,
        low_confidence=sum(
            1
            for feature in features
            if feature.properties.confidence is not None
            and feature.properties.confidence < RELIABLE_CONFIDENCE
        ),
        small_skipped=int((~large).sum()),
        fields=FieldCollection(features=features),
    )


def _geometry(geometry: Polygon | MultiPolygon) -> FieldGeometry:
    data = mapping(geometry)
    return FieldGeometry(type=data["type"], coordinates=cast("list[Any]", _rounded(data["coordinates"])))


def _rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, COORDINATE_DECIMALS)
    if isinstance(value, list | tuple):
        return [_rounded(item) for item in value]
    return value


def search_findings(
    metrics: SearchFieldsMetrics, *, covered: bool, min_area_ha: float = MIN_AREA_HA
) -> list[Finding]:
    if not covered:
        return [
            Finding(
                code="outside_field_map",
                level="warning",
                text="Ділянка поза країнами, для яких підключена карта полів (Україна): поля не шукались.",
            )
        ]
    if metrics.year is None:
        return [Finding(code="no_field_map", level="warning", text="Для цієї ділянки немає карти полів.")]
    if not metrics.fields_found:
        return [
            Finding(
                code="no_fields",
                level="notice",
                text=(
                    f"На карті полів {metrics.year} року жодне поле не перетинає ділянку і не торкається її."
                ),
            )
        ] + ([_small_skipped(metrics.small_skipped, min_area_ha)] if metrics.small_skipped else [])
    parts = [
        f"{RELATIONS[relation][0].uk} — {count}"  # type: ignore[index]
        for relation, count in metrics.counts.items()
        if count
    ]
    findings = [
        Finding(
            code="fields_found",
            level="info",
            text=(
                f"Знайдено {plural_uk(metrics.fields_found, 'поле', 'поля', 'полів')} загальною площею "
                f"{metrics.fields_area_ha:.1f} га (карта полів {metrics.year} року за знімками Sentinel-2): "
                f"{', '.join(parts)}."
            ),
        )
    ]
    if metrics.covered_share < COVERED_SHARE_NOTICE:
        findings.append(
            Finding(
                code="partly_covered",
                level="notice",
                text=(
                    f"Знайдені поля покривають лише {metrics.covered_share:.0%} ділянки; решта, ймовірно, "
                    "межі, дороги, лісосмуги або площа, яку модель не визнала полем."
                ),
            )
        )
    else:
        findings.append(
            Finding(
                code="covered",
                level="info",
                text=f"Знайдені поля покривають {metrics.covered_share:.0%} ділянки.",
            )
        )
    if metrics.small_skipped:
        findings.append(_small_skipped(metrics.small_skipped, min_area_ha))
    if metrics.low_confidence:
        findings.append(
            Finding(
                code="low_confidence",
                level="notice",
                text=(
                    f"У {plural_uk(metrics.low_confidence, 'поля', 'полів', 'полів')} низька довіра моделі "
                    f"(нижче {RELIABLE_CONFIDENCE} зі 100): їхні межі можуть бути неточні."
                ),
            )
        )
    return findings


def _small_skipped(count: int, min_area_ha: float) -> Finding:
    return Finding(
        code="small_skipped",
        level="info",
        text=(
            f"Не показано {plural_uk(count, 'дрібний полігон', 'дрібні полігони', 'дрібних полігонів')} "
            f"(менше {min_area_ha:g} га): ймовірно, клаптики між полями, а не окремі поля."
        ),
    )


def fields_map(parcel: Parcel, fields: FieldCollection) -> go.Figure:
    figure = go.Figure()
    for relation, (label, color) in RELATIONS.items():
        members = [feature for feature in fields.features if feature.properties.relation == relation]
        if not members:
            continue
        lng, lat = _rings([shapely.geometry.shape(f.geometry.model_dump()) for f in members])
        figure.add_trace(
            go.Scattermap(
                lon=lng,
                lat=lat,
                mode="lines",
                fill="toself",
                fillcolor=_with_alpha(color, 0.25),
                line={"color": color, "width": 2},
                name=f"{label.uk} ({len(members)})",
                hoverinfo="skip",
            )
        )
        points = [shapely.geometry.shape(f.geometry.model_dump()).representative_point() for f in members]
        figure.add_trace(
            go.Scattermap(
                lon=[point.x for point in points],
                lat=[point.y for point in points],
                mode="markers",
                marker={"size": 7, "color": color},
                text=[_hover(feature) for feature in members],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
    lng, lat = _rings([parcel.geometry])
    figure.add_trace(
        go.Scattermap(
            lon=lng,
            lat=lat,
            mode="lines",
            line={"color": "#e03131", "width": 3},
            name="ділянка",
            hoverinfo="skip",
        )
    )
    bounds = [parcel.bbox] + [shapely.geometry.shape(f.geometry.model_dump()).bounds for f in fields.features]
    minx, miny = min(b[0] for b in bounds), min(b[1] for b in bounds)
    maxx, maxy = max(b[2] for b in bounds), max(b[3] for b in bounds)
    figure.update_layout(
        height=MAP_HEIGHT_PX,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        map={
            "style": "white-bg",
            "center": {"lon": (minx + maxx) / 2, "lat": (miny + maxy) / 2},
            "zoom": _zoom(minx, miny, maxx, maxy),
            "layers": interactive.imagery_layers(),
        },
    )
    return figure


def fields_table(fields: FieldCollection) -> go.Figure:
    rows = [
        [
            feature.id,
            f"{feature.properties.area_ha:.2f}",
            f"{feature.properties.overlap_ha:.2f}",
            feature.properties.relation_label.uk,
            "—" if feature.properties.confidence is None else f"{feature.properties.confidence:.0f}",
        ]
        for feature in fields.features[:TABLE_MAX_ROWS]
    ]
    title = "Знайдені поля" + (
        f" (перші {TABLE_MAX_ROWS} з {len(fields.features)})" if len(fields.features) > TABLE_MAX_ROWS else ""
    )
    figure = interactive.table_figure(
        title,
        ["id поля", "площа, га", "у межах ділянки, га", "як перетинає", "довіра моделі"],
        rows,
        column_widths=[1.2, 0.8, 1, 1.4, 0.8],
    )
    figure.update_layout(height=120 + 28 * max(len(rows), 1))
    return figure


def _rings(geometries: list[Polygon | MultiPolygon | Any]) -> tuple[list[float | None], list[float | None]]:
    lng: list[float | None] = []
    lat: list[float | None] = []
    for geometry in geometries:
        for polygon in getattr(geometry, "geoms", [geometry]):
            xs, ys = polygon.exterior.xy
            lng += [round(x, COORDINATE_DECIMALS) for x in xs] + [None]
            lat += [round(y, COORDINATE_DECIMALS) for y in ys] + [None]
    return lng, lat


def _hover(feature: FieldFeature) -> str:
    properties = feature.properties
    confidence = "—" if properties.confidence is None else f"{properties.confidence:.0f}"
    return (
        f"поле {feature.id}<br>площа {properties.area_ha:.2f} га"
        f"<br>у межах ділянки {properties.overlap_ha:.2f} га"
        f"<br>{properties.relation_label.uk}<br>довіра моделі {confidence}"
    )


def _with_alpha(color: str, alpha: float) -> str:
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def _zoom(minx: float, miny: float, maxx: float, maxy: float) -> float:
    width = max(maxx - minx, 1e-4)
    height = max(maxy - miny, 1e-4) / cos(radians((miny + maxy) / 2))
    by_width = log2(360 * MAP_WIDTH_PX / (512 * width * 1.1))
    by_height = log2(360 * MAP_HEIGHT_PX / (512 * height * 1.1))
    return round(min(by_width, by_height, 17), 2)
