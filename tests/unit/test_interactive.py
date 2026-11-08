import json
from dataclasses import replace

import numpy as np
import pytest
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.affinity import scale

from argo_geoai import interactive
from argo_geoai.config import get_settings
from argo_geoai.schema import DataSource, Finding

UTM = CRS.from_epsg(32635)


@pytest.fixture
def grid(field_slope):
    frame = field_slope.in_crs(UTM)
    minx, miny, maxx, maxy = frame.total_bounds
    x = np.arange(minx - 60, maxx + 60, 30.0)
    y = np.arange(maxy + 60, miny - 60, -30.0)
    transform = from_origin(x[0] - 15, y[0] + 15, 30, 30)
    inside = ~geometry_mask(frame.geometry, out_shape=(len(y), len(x)), transform=transform)
    return x, y, inside


def test_matplotlib_colorscale():
    scale = interactive.mpl_colorscale("RdYlGn", steps=3)

    assert [stop for stop, _ in scale] == [0.0, 0.5, 1.0]
    assert scale[0][1] == "#a50026"
    assert scale[-1][1] == "#006837"


def test_grid_geometry_is_cropped_and_georeferenced(field_slope, grid):
    x, y, inside = grid

    geometry = interactive.grid_geometry(x, y, UTM, inside)

    assert geometry.x_m.min() == 0
    assert geometry.y_m.min() == 0
    min_lng, min_lat, max_lng, max_lat = field_slope.bbox
    assert min_lng - 0.001 < geometry.lng.mean() < max_lng + 0.001
    assert min_lat - 0.001 < geometry.lat.mean() < max_lat + 0.001


def test_map_panels_share_zoom_and_hover_every_layer(field_slope, grid):
    x, y, inside = grid
    geometry = interactive.grid_geometry(x, y, UTM, inside)
    elevation = np.where(inside, 150.0, np.nan)
    layers = [
        interactive.MapLayer("Висота", "Висота", "м", elevation, "terrain", contour_step=2),
        interactive.MapLayer("NDVI", "NDVI", "", np.where(inside, 0.7, np.nan), "RdYlGn", zmin=0, zmax=0.9),
    ]

    figure = interactive.map_panels(field_slope, geometry, layers)

    heatmaps = [t for t in figure.data if t.type == "heatmap"]
    assert len(heatmaps) == 2
    assert [t.type for t in figure.data].count("contour") == 1
    assert "Висота: %{customdata[2]:.2f} м" in heatmaps[0].hovertemplate
    assert "NDVI: %{customdata[3]:.2f}" in heatmaps[0].hovertemplate
    assert figure.layout.xaxis2.matches == "x"
    assert figure.layout.showlegend is False
    outline = next(t for t in figure.data if t.type == "scatter")
    assert None in outline.x


def test_page_and_figures_json(tmp_path):
    figure = interactive.bar_chart("Класи", ["a", "b"], [0.25, 0.75], ["#fff", "#000"])
    summary = [Finding(code="x", level="warning", text="Ухил <5°> & ерозія")]
    source = DataSource(collection="c", title="Copernicus DEM", provider="ESA", resolution_m=30)

    interactive.write_page(
        tmp_path / "page.html",
        title="plot_1",
        subtitle="1 га",
        summary=summary,
        figures=[figure],
        sources=[source],
    )
    interactive.write_figures_json(tmp_path / "figures.json", {"classes": figure})

    page = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert "cdn.plot.ly" in page
    assert "Ухил &lt;5°&gt; &amp; ерозія" in page
    assert "не є кадастровим" in page
    assert "Підкладка карт: Esri" in page
    spec = json.loads((tmp_path / "figures.json").read_text(encoding="utf-8"))
    assert spec["classes"]["data"][0]["type"] == "bar"
    assert "layout" in spec["classes"]


def test_overlay_hover_values_and_category_ticks(field_slope, grid):
    x, y, inside = grid
    geometry = interactive.grid_geometry(x, y, UTM, inside)
    upslope_ha = np.where(inside, 25.0, np.nan)
    zones = np.where(inside, 1.0, np.nan)
    layers = [
        interactive.MapLayer(
            "Водозбір",
            "Водозбір",
            "га",
            np.log10(upslope_ha),
            "Blues",
            hover_values=upslope_ha,
            decimals=1,
            overlay=inside,
        ),
        interactive.MapLayer(
            "Зони",
            "Зона",
            "",
            zones,
            "tab10",
            zmin=0.5,
            zmax=1.5,
            decimals=0,
            plotly_colorscale=interactive.categorical_colorscale(["#1b9e77"]),
            ticks=([1.0], ["1: основна"]),
        ),
    ]

    figure = interactive.map_panels(field_slope, geometry, layers)

    heatmaps = [t for t in figure.data if t.type == "heatmap"]
    assert np.nanmax(np.asarray(heatmaps[0].customdata)[..., 2]) == 25.0
    assert np.nanmax(np.asarray(heatmaps[0].z, dtype=float)) == pytest.approx(1.4)
    overlays = [t for t in figure.data if t.type == "contour" and t.line.color == interactive.OVERLAY_COLOR]
    assert len(overlays) == 1
    assert list(heatmaps[1].colorbar.ticktext) == ["1: основна"]


def test_date_slider_map_swaps_values_per_date(field_slope, grid):
    x, y, inside = grid
    geometry = interactive.grid_geometry(x, y, UTM, inside)
    frames = {"01.05.2026": np.where(inside, 0.2, np.nan), "01.06.2026": np.where(inside, 0.8, np.nan)}

    figure = interactive.date_slider_map(
        field_slope, geometry, title="NDVI", frames=frames, colorscale="RdYlGn", zmin=0, zmax=0.9
    )

    steps = figure.layout.sliders[0].steps
    assert [s.label for s in steps] == list(frames)
    assert np.nanmax(np.asarray(steps[1].args[0]["z"][0], dtype=float)) == pytest.approx(0.8)


def test_line_chart_with_band():
    figure = interactive.line_chart(
        "NDVI",
        [
            interactive.Series(
                "середнє", ["2026-05-01", "2026-06-01"], [0.3, 0.7], "#1b7837", band=([0.2, 0.6], [0.4, 0.8])
            )
        ],
        y_title="NDVI",
        y_range=(0, 0.9),
    )

    assert len(figure.data) == 3
    assert list(figure.layout.yaxis.range) == [0, 0.9]


def test_large_maps_hover_only_their_own_value(field_slope, grid, monkeypatch):
    x, y, inside = grid
    geometry = interactive.grid_geometry(x, y, UTM, inside)
    layers = [
        interactive.MapLayer("NDVI 1", "NDVI 1", "", np.where(inside, 0.5, np.nan), "RdYlGn"),
        interactive.MapLayer("NDVI 2", "NDVI 2", "", np.where(inside, 0.6, np.nan), "RdYlGn"),
    ]
    monkeypatch.setattr(interactive, "HOVER_BUDGET_BYTES", 1)

    figure = interactive.map_panels(field_slope, geometry, layers)

    heatmaps = [t for t in figure.data if t.type == "heatmap"]
    assert heatmaps[0].customdata is None
    assert heatmaps[1].hovertemplate == "NDVI 2: %{z:.2f} <extra></extra>"


def test_highlight_is_drawn_with_the_parcel_outline(field_slope, grid):
    x, y, inside = grid
    geometry = interactive.grid_geometry(x, y, UTM, inside)
    field = scale(field_slope.geometry, 0.5, 0.5)
    frames = {"01.05.2026": np.where(inside, 0.2, np.nan)}

    def outline_breaks(parcel):
        figure = interactive.date_slider_map(
            parcel, geometry, title="NDVI", frames=frames, colorscale="RdYlGn", zmin=0, zmax=0.9
        )
        return list(figure.data[1].x).count(None)

    assert outline_breaks(replace(field_slope, highlight=field)) == outline_breaks(field_slope) + 1


def test_imagery_layers_follow_the_settings(monkeypatch):
    assert [layer["sourcetype"] for layer in interactive.imagery_layers()] == ["raster"]
    assert interactive.imagery_layers()[0]["source"] == [get_settings().imagery_tiles]

    monkeypatch.setenv("ARGO_GEOAI_IMAGERY_TILES", "")
    get_settings.cache_clear()

    assert interactive.imagery_layers() == []


def test_attributions_name_every_licence_once(monkeypatch):
    sources = [
        DataSource(collection="sentinel-2-l2a", title="S2", provider="ESA", resolution_m=10),
        DataSource(collection="sentinel-2-l2a", title="S2 again", provider="ESA", resolution_m=10),
        DataSource(collection="cop-dem-glo-30", title="DEM", provider="ESA", resolution_m=30),
        DataSource(collection="unknown", title="x", provider="y", resolution_m=1),
    ]

    notes = interactive.attributions(sources, year=2026)

    assert notes[0] == "Contains modified Copernicus Sentinel data 2026, processed by ESA."
    assert notes[1].startswith("Copernicus DEM: © DLR e.V.")
    assert notes[2].startswith("Підкладка карт: Esri")
    assert len(notes) == 3

    monkeypatch.setenv("ARGO_GEOAI_IMAGERY_TILES", "")
    get_settings.cache_clear()

    assert len(interactive.attributions(sources, year=2026)) == 2
