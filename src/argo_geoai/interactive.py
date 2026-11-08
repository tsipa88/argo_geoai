import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon

from argo_geoai.config import get_settings
from argo_geoai.parcels import Parcel
from argo_geoai.schema import DataSource, Finding

FloatArray = npt.NDArray[np.float64]

PLOTLY_CDN = "cdn"
LEVEL_COLORS = {"info": "#2b6cb0", "notice": "#b7791f", "warning": "#c53030"}
OVERLAY_COLOR = "#d7301f"
NO_DATA_HOVER = "—"
# Plotly ships arrays as base64 typed arrays: rounding does not shrink them, the dtype does
ARRAY_DTYPE = np.float32
HOVER_BUDGET_BYTES = 2_000_000


def categorical_colorscale(colors: Sequence[str]) -> list[list[float | str]]:
    scale: list[list[float | str]] = []
    for i, color in enumerate(colors):
        scale += [[i / len(colors), color], [(i + 1) / len(colors), color]]
    return scale


def mpl_colorscale(name: str, steps: int = 11) -> list[list[float | str]]:
    cmap = mpl.colormaps[name]
    return [[i / (steps - 1), mpl.colors.to_hex(cmap(i / (steps - 1)))] for i in range(steps)]


@dataclass(frozen=True)
class MapLayer:
    title: str
    hover_label: str
    unit: str
    values: FloatArray
    colorscale: str
    zmin: float | None = None
    zmax: float | None = None
    contour_step: float | None = None
    decimals: int = 2
    plotly_colorscale: list[list[float | str]] | None = None
    hover_values: FloatArray | None = None
    overlay: npt.NDArray[np.bool_] | None = None
    ticks: tuple[list[float], list[str]] | None = None


@dataclass(frozen=True)
class GridGeometry:
    x_m: FloatArray
    y_m: FloatArray
    lng: FloatArray
    lat: FloatArray
    rows: slice
    cols: slice
    origin: tuple[float, float]
    crs: object


def grid_geometry(x: FloatArray, y: FloatArray, crs: object, inside: npt.NDArray[np.bool_]) -> GridGeometry:
    rows, cols = np.nonzero(inside)
    row_slice, col_slice = slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)
    xs, ys = x[col_slice], y[row_slice]
    origin = (float(xs.min()), float(ys.min()))
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    grid_x, grid_y = np.meshgrid(xs, ys)
    lng, lat = to_wgs84.transform(grid_x, grid_y)
    return GridGeometry(
        x_m=xs - origin[0],
        y_m=ys - origin[1],
        lng=np.asarray(lng),
        lat=np.asarray(lat),
        rows=row_slice,
        cols=col_slice,
        origin=origin,
        crs=crs,
    )


def _outline(parcel: Parcel, grid: GridGeometry) -> tuple[list[float | None], list[float | None]]:
    shapes = [parcel.in_crs(grid.crs).iloc[0]]  # type: ignore[arg-type]
    if parcel.highlight is not None:
        highlight = gpd.GeoSeries([parcel.highlight], crs="EPSG:4326").to_crs(grid.crs).iloc[0]
        shapes.append(highlight)
    polygons = [p for shape in shapes for p in (shape.geoms if isinstance(shape, MultiPolygon) else [shape])]
    xs: list[float | None] = []
    ys: list[float | None] = []
    for polygon in polygons:
        assert isinstance(polygon, Polygon)
        for ring in [polygon.exterior, *polygon.interiors]:
            ring_x, ring_y = ring.xy
            xs += [v - grid.origin[0] for v in ring_x] + [None]
            ys += [v - grid.origin[1] for v in ring_y] + [None]
    return xs, ys


def _compact(values: FloatArray, decimals: int) -> npt.NDArray[np.float32]:
    compact: npt.NDArray[np.float32] = np.round(values, decimals).astype(ARRAY_DTYPE)
    return compact


def _own_template(layer: MapLayer) -> str:
    if layer.hover_values is not None:
        return f"{layer.title}<extra></extra>"
    return f"{layer.hover_label}: %{{z:.{layer.decimals}f}} {layer.unit}<extra></extra>"


def map_panels(
    parcel: Parcel, grid: GridGeometry, layers: Sequence[MapLayer], *, columns: int = 2
) -> go.Figure:
    count = len(layers)
    rows = -(-count // columns)
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[layer.title for layer in layers],
        horizontal_spacing=0.12,
        vertical_spacing=0.1,
    )
    cropped = [_compact(layer.values[grid.rows, grid.cols], layer.decimals) for layer in layers]
    hover = [
        _compact(
            (layer.hover_values if layer.hover_values is not None else layer.values)[grid.rows, grid.cols],
            layer.decimals,
        )
        for layer in layers
    ]
    cells = grid.lng.size
    full_hover = cells * (2 + count) * count * np.dtype(ARRAY_DTYPE).itemsize <= HOVER_BUDGET_BYTES
    customdata = np.dstack([grid.lng, grid.lat, *hover]).astype(ARRAY_DTYPE) if full_hover else None
    hover_lines = [
        f"{layer.hover_label}: %{{customdata[{i + 2}]:.{layer.decimals}f}} {layer.unit}"
        for i, layer in enumerate(layers)
    ]
    full_template = (
        "<br>".join([*hover_lines, "%{customdata[1]:.5f}, %{customdata[0]:.5f}"]) + "<extra></extra>"
    )
    outline_x, outline_y = _outline(parcel, grid)

    for index, (layer, values) in enumerate(zip(layers, cropped, strict=True)):
        row, col = index // columns + 1, index % columns + 1
        axis = "" if index == 0 else str(index + 1)
        x_domain = figure.layout[f"xaxis{axis}"].domain
        y_domain = figure.layout[f"yaxis{axis}"].domain
        figure.add_trace(
            go.Heatmap(
                x=grid.x_m,
                y=grid.y_m,
                z=values,
                customdata=customdata,
                hovertemplate=full_template if full_hover else _own_template(layer),
                colorscale=layer.plotly_colorscale or mpl_colorscale(layer.colorscale),
                zmin=layer.zmin,
                zmax=layer.zmax,
                colorbar={
                    "title": {"text": layer.unit, "side": "right"},
                    "x": x_domain[1] + 0.01,
                    "y": (y_domain[0] + y_domain[1]) / 2,
                    "len": (y_domain[1] - y_domain[0]) * 0.95,
                    "thickness": 12,
                    **({"tickvals": layer.ticks[0], "ticktext": layer.ticks[1]} if layer.ticks else {}),
                },
            ),
            row=row,
            col=col,
        )
        if layer.contour_step:
            figure.add_trace(
                go.Contour(
                    x=grid.x_m,
                    y=grid.y_m,
                    z=values,
                    contours={
                        "coloring": "none",
                        "start": np.floor(np.nanmin(values)),
                        "end": np.ceil(np.nanmax(values)),
                        "size": layer.contour_step,
                        "showlabels": True,
                        "labelfont": {"size": 9, "color": "#333"},
                    },
                    line={"color": "rgba(0,0,0,0.55)", "width": 0.7},
                    hoverinfo="skip",
                    showscale=False,
                ),
                row=row,
                col=col,
            )
        if layer.overlay is not None and layer.overlay[grid.rows, grid.cols].any():
            figure.add_trace(
                go.Contour(
                    x=grid.x_m,
                    y=grid.y_m,
                    z=layer.overlay[grid.rows, grid.cols].astype(float),
                    contours={"coloring": "none", "start": 0.5, "end": 0.5, "size": 1},
                    line={"color": OVERLAY_COLOR, "width": 1.2},
                    hoverinfo="skip",
                    showscale=False,
                ),
                row=row,
                col=col,
            )
        figure.add_trace(
            go.Scatter(
                x=outline_x,
                y=outline_y,
                mode="lines",
                line={"color": "#222", "width": 1.2},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        bottom_row = index + columns >= count
        figure.update_xaxes(
            matches="x",
            showgrid=False,
            zeroline=False,
            ticksuffix=" м",
            showticklabels=bottom_row,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            matches="y",
            scaleanchor=f"x{axis}",
            showgrid=False,
            zeroline=False,
            ticksuffix=" м",
            showticklabels=col == 1,  # the colour bar of the left panel sits where right labels would be
            row=row,
            col=col,
        )
    figure.update_layout(
        height=430 * rows,
        margin={"l": 40, "r": 40, "t": 60, "b": 30},
        plot_bgcolor="white",
        hoverlabel={"bgcolor": "white"},
        showlegend=False,
    )
    return figure


def date_slider_map(
    parcel: Parcel,
    grid: GridGeometry,
    *,
    title: str,
    frames: dict[str, FloatArray],
    colorscale: str,
    zmin: float,
    zmax: float,
    decimals: int = 2,
    captions: dict[str, str] | None = None,
) -> go.Figure:
    labels = list(frames)
    layers = [_compact(frames[label][grid.rows, grid.cols], decimals) for label in labels]
    coordinates = np.dstack([grid.lng, grid.lat]).astype(ARRAY_DTYPE)
    outline_x, outline_y = _outline(parcel, grid)
    figure = go.Figure(
        [
            go.Heatmap(
                x=grid.x_m,
                y=grid.y_m,
                z=layers[-1],
                customdata=coordinates,
                hovertemplate=(
                    f"%{{z:.{decimals}f}}<br>%{{customdata[1]:.5f}}, %{{customdata[0]:.5f}}<extra></extra>"
                ),
                colorscale=mpl_colorscale(colorscale),
                zmin=zmin,
                zmax=zmax,
                colorbar={"thickness": 12},
            ),
            go.Scatter(
                x=outline_x, y=outline_y, mode="lines", line={"color": "#222", "width": 1.2}, hoverinfo="skip"
            ),
        ]
    )
    figure.update_layout(
        title=_captioned(title, (captions or {}).get(labels[-1]) if labels else None),
        height=560 if not captions else 600,
        margin={"l": 40, "r": 40, "t": 60, "b": 30},
        plot_bgcolor="white",
        showlegend=False,
        xaxis={"showgrid": False, "zeroline": False, "ticksuffix": " м"},
        yaxis={"showgrid": False, "zeroline": False, "ticksuffix": " м", "scaleanchor": "x"},
        sliders=[
            {
                "active": len(labels) - 1,
                "currentvalue": {"prefix": "Дата: "},
                "pad": {"t": 30},
                "steps": [
                    _slider_step(label, layer, title, (captions or {}).get(label))
                    for label, layer in zip(labels, layers, strict=True)
                ],
            }
        ],
    )
    return figure


def _captioned(title: str, caption: str | None) -> str:
    if not caption:
        return title
    return f"{title}<br><sub>{html.escape(caption)}</sub>"


def _slider_step(
    label: str, layer: npt.NDArray[np.float32], title: str, caption: str | None
) -> dict[str, object]:
    if caption is None:
        return {"label": label, "method": "restyle", "args": [{"z": [layer]}, [0]]}
    return {
        "label": label,
        "method": "update",
        "args": [{"z": [layer]}, {"title.text": _captioned(title, caption)}, [0]],
    }


@dataclass(frozen=True)
class Series:
    name: str
    x: Sequence[object]
    y: Sequence[float]
    color: str
    band: tuple[Sequence[float], Sequence[float]] | None = None
    band_name: str = ""


def line_chart(
    title: str, series: Sequence[Series], *, y_title: str, y_range: tuple[float, float] | None = None
) -> go.Figure:
    figure = go.Figure()
    for item in series:
        if item.band is not None:
            low, high = item.band
            figure.add_trace(
                go.Scatter(
                    x=list(item.x),
                    y=list(high),
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=list(item.x),
                    y=list(low),
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=_with_alpha(item.color, 0.25),
                    name=item.band_name,
                    hoverinfo="skip",
                )
            )
        figure.add_trace(
            go.Scatter(
                x=list(item.x),
                y=list(item.y),
                mode="lines+markers",
                name=item.name,
                line={"color": item.color, "width": 2},
                hovertemplate=f"{item.name}: %{{y:.2f}}<br>%{{x}}<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        yaxis={"title": y_title, "range": list(y_range) if y_range else None},
        height=380,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        plot_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.15},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#eee")
    figure.update_yaxes(showgrid=True, gridcolor="#eee")
    return figure


def _with_alpha(color: str, alpha: float) -> str:
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def bar_chart(title: str, labels: Sequence[str], values: Sequence[float], colors: Sequence[str]) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=list(labels),
            y=[v * 100 for v in values],
            marker_color=list(colors),
            text=[f"{v:.0%}" for v in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        yaxis={"title": "% площі поля", "rangemode": "tozero"},
        height=360,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        plot_bgcolor="white",
    )
    return figure


def stacked_bar_chart(
    title: str, x: Sequence[str], series: Sequence[tuple[str, Sequence[float], str]], *, y_title: str
) -> go.Figure:
    figure = go.Figure(
        [
            go.Bar(
                x=list(x),
                y=[v * 100 for v in values],
                name=name,
                marker_color=color,
                hovertemplate=f"{name}: %{{y:.0f}}%<extra></extra>",
            )
            for name, values, color in series
        ]
    )
    figure.update_layout(
        title=title,
        barmode="stack",
        yaxis={"title": y_title, "range": [0, 100]},
        height=380,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        plot_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.15},
    )
    return figure


def rose_chart(title: str, labels: Sequence[str], values: Sequence[float]) -> go.Figure:
    figure = go.Figure(
        go.Barpolar(
            r=[v * 100 for v in values],
            theta=list(labels),
            marker_color="#7fbf7b",
            marker_line_color="#1b7837",
            marker_line_width=1,
            hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        polar={
            "angularaxis": {"rotation": 90, "direction": "clockwise"},
            "radialaxis": {"ticksuffix": "%", "angle": 67.5, "tickangle": 67.5, "tickfont": {"size": 9}},
        },
        height=360,
        margin={"l": 40, "r": 40, "t": 50, "b": 30},
    )
    return figure


MAPS_NOTE = (
    "Карти інтерактивні: наближення колесом або рамкою синхронне для всіх панелей, подвійний клік — "
    "повернутись; підказка показує всі значення пікселя і його координати."
)
DISCLAIMER = (
    "Результати — автоматична оцінка за відкритими супутниковими даними, довідкова. Вона не є кадастровим, "
    "юридичним, агрономічним чи страховим висновком і не замінює обстеження на місці; межі полів — "
    "результат моделі, не кадастр."
)
# legal notices the data licences ask for, by STAC collection (see README "Data sources")
ATTRIBUTIONS: dict[str, str] = {
    "sentinel-2-l2a": "Contains modified Copernicus Sentinel data {year}, processed by ESA.",
    "sentinel-1-rtc": "Contains modified Copernicus Sentinel data {year}, processed by ESA and Microsoft.",
    "cop-dem-glo-30": (
        "Copernicus DEM: © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, "
        "provided under COPERNICUS by the European Union and ESA; all rights reserved."
    ),
    "esa-worldcover": (
        "ESA WorldCover 2021 © ESA WorldCover project 2021, CC BY 4.0; contains modified Copernicus "
        "Sentinel data (2021) processed by the ESA WorldCover consortium."
    ),
    "jrc-gsw": (
        "JRC Global Surface Water: Pekel, Cottam, Gorelick, Belward (2016), Nature 540, 418–422; "
        "European Commission JRC / Google."
    ),
    "ftw-global-field-boundaries": ("Field boundaries: Fields of The World (Source Cooperative), CC BY 4.0."),
}


def attributions(sources: Sequence[DataSource], *, year: int | None = None) -> list[str]:
    """Legal notices for the data behind a page: one per licence, in the order of the sources."""
    year = year or datetime.now(UTC).year
    notes: list[str] = []
    for source in sources:
        text = ATTRIBUTIONS.get(source.collection)
        if text and text.format(year=year) not in notes:
            notes.append(text.format(year=year))
    imagery = get_settings().imagery_attribution if get_settings().imagery_tiles else ""
    if imagery:
        notes.append(f"Підкладка карт: {imagery}.")
    return notes


def imagery_layers() -> list[dict[str, object]]:
    """Plotly map layers with the satellite basemap, or none when the basemap is switched off."""
    settings = get_settings()
    if not settings.imagery_tiles:
        return []
    return [
        {
            "below": "traces",
            "sourcetype": "raster",
            "sourceattribution": settings.imagery_attribution,
            "source": [settings.imagery_tiles],
        }
    ]


TABLE_ROW_HEIGHT_PX = 50


def table_figure(
    title: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    column_widths: Sequence[float] | None = None,
) -> go.Figure:
    columns = [list(column) for column in zip(*rows, strict=True)] if rows else [[] for _ in header]
    figure = go.Figure(
        go.Table(
            columnwidth=list(column_widths) if column_widths else None,
            header={"values": list(header), "align": "left", "fill_color": "#edf2f7", "font": {"size": 13}},
            cells={"values": columns, "align": "left", "height": 26, "font": {"size": 12}},
        )
    )
    figure.update_layout(
        title=title,
        height=120 + TABLE_ROW_HEIGHT_PX * max(len(rows), 1),
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


def write_page(
    path: Path,
    *,
    title: str,
    subtitle: str,
    summary: Sequence[Finding],
    figures: Sequence[go.Figure],
    sources: Sequence[DataSource],
    note: str = MAPS_NOTE,
) -> None:
    findings = "\n".join(
        f'<li><span class="badge" style="background:{LEVEL_COLORS[f.level]}">{f.level}</span> '
        f"{html.escape(f.text)}</li>"
        for f in summary
    )
    charts = "\n".join(
        figure.to_html(
            full_html=False, include_plotlyjs=PLOTLY_CDN if i == 0 else False, config={"locale": "uk"}
        )
        for i, figure in enumerate(figures)
    )
    data = "\n".join(
        f"<li>{html.escape(s.title)} ({s.resolution_m:g} м) — {html.escape(s.provider)}</li>" for s in sources
    )
    notices = "\n".join(f"<li>{html.escape(text)}</li>" for text in attributions(sources))
    path.write_text(
        f"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin: 0; color: #1a202c;
          background: #f7f7f5; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px 48px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .subtitle {{ color: #4a5568; margin-bottom: 20px; }}
  section {{ background: white; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;
             box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
  ul.findings {{ list-style: none; padding: 0; margin: 0; }}
  ul.findings li {{ margin: 8px 0; line-height: 1.45; }}
  .badge {{ color: white; font-size: 11px; border-radius: 4px; padding: 1px 6px; margin-right: 6px;
            text-transform: uppercase; }}
  .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 800px) {{ .row {{ grid-template-columns: 1fr; }} }}
  .note {{ color: #718096; font-size: 13px; }}
</style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <div class="subtitle">{html.escape(subtitle)}</div>
  <section><h2 style="font-size:16px;margin-top:0">Висновки</h2><ul class="findings">{findings}</ul></section>
  <section>{charts}</section>
  <section class="note">Джерела даних:<ul>{data}</ul>
  {html.escape(note)}</section>
  <section class="note"><p>{html.escape(DISCLAIMER)}</p><ul>{notices}</ul></section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_figures_json(path: Path, figures: dict[str, go.Figure]) -> None:
    path.write_text(
        json.dumps(
            {name: json.loads(figure.to_json()) for name, figure in figures.items()}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
