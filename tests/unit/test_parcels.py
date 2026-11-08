import json

import pytest
from pydantic import ValidationError
from shapely.geometry import MultiPolygon, Polygon, box, mapping

from argo_geoai.parcels import ParcelError, from_request, normalize
from argo_geoai.request import ParcelRequest, safe_name

SQUARE = mapping(box(28.1, 47.1, 28.11, 47.11))


def parcel(data: object, default_name: str | None = None):
    return from_request(ParcelRequest.from_json(json.dumps(data), default_name=default_name))


def test_field_slope_file(field_slope):
    assert field_slope.name == "field_slope"
    assert field_slope.area_ha == pytest.approx(84.58, abs=0.05)
    assert field_slope.expected_area_ha == pytest.approx(84.58)
    assert field_slope.metadata == {"number": "1", "source": "ftw:41665"}
    assert field_slope.checks() == []


def test_info_uses_lng_lat_order(field_slope):
    info = field_slope.info()

    min_lng, min_lat, max_lng, max_lat = info.bbox
    assert 28 < min_lng < max_lng < 29
    assert 47 < min_lat < max_lat < 48
    assert min_lng < info.centroid[0] < max_lng


def test_feature_with_name_area_and_metadata():
    feature = {
        "type": "Feature",
        "geometry": SQUARE,
        "properties": {"name": "field_two_parts", "area_ha": 81.0, "metadata": {"source": "test", "id": 1}},
    }

    result = parcel(feature)

    assert result.name == "field_two_parts"
    assert result.metadata == {"source": "test", "id": 1}
    assert result.info().metadata == {"source": "test", "id": 1}
    assert result.info().expected_area_ha == 81.0


def test_feature_collection_with_one_feature_and_bom():
    data = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": SQUARE, "properties": {}}],
    }

    request = ParcelRequest.from_json("﻿" + json.dumps(data), default_name="my field.geojson")

    assert request.name == "my_field.geojson"


def test_bare_geometry_gets_a_stable_name():
    first, second = ParcelRequest.from_json(json.dumps(SQUARE)), ParcelRequest.from_json(json.dumps(SQUARE))

    assert first.name.startswith("parcel_")
    assert first.name == second.name


@pytest.mark.parametrize(("value", "expected"), [("plot 1/../x", "plot_1_.._x"), ("///", "parcel")])
def test_safe_name(value, expected):
    assert safe_name(value) == expected


def test_unsafe_name_in_properties_is_rejected():
    with pytest.raises(ValidationError):
        ParcelRequest.from_json(
            json.dumps({"type": "Feature", "geometry": SQUARE, "properties": {"name": "../x"}})
        )


def test_no_features():
    with pytest.raises(ValueError, match="exactly one feature"):
        ParcelRequest.from_json(json.dumps({"type": "FeatureCollection", "features": []}))


@pytest.mark.parametrize(
    "geometry",
    [{"type": "Point", "coordinates": [28.1, 47.1]}, {"type": "Polygon"}],
    ids=["not a polygon", "no coordinates"],
)
def test_unusable_geometry(geometry):
    with pytest.raises(ParcelError):
        parcel(geometry)


def test_self_intersecting_polygon_is_repaired():
    bowtie = {
        "type": "Polygon",
        "coordinates": [[[28.1, 47.1], [28.11, 47.11], [28.11, 47.1], [28.1, 47.11], [28.1, 47.1]]],
    }

    result = parcel(bowtie)

    assert result.repaired
    assert result.geometry.is_valid
    assert "geometry_repaired" in [m.code for m in result.checks()]


def test_area_mismatch_is_reported():
    result = parcel({"type": "Feature", "geometry": SQUARE, "properties": {"area_ha": 200.0}})

    assert [m.code for m in result.checks()] == ["area_mismatch"]


def test_multipolygon_with_two_parts():
    two = MultiPolygon([box(28.65, 45.95, 28.67, 45.96), box(28.68, 45.95, 28.69, 45.96)])

    result = parcel(mapping(two))

    assert result.parts == 2
    assert result.info().parts == 2


def test_single_part_multipolygon_becomes_a_polygon_with_its_holes():
    with_hole = box(28.68, 45.95, 28.69, 45.96).difference(box(28.681, 45.951, 28.682, 45.952))
    assert isinstance(with_hole, Polygon)

    geometry = normalize(MultiPolygon([with_hole]), "test")

    assert isinstance(geometry, Polygon)
    assert len(geometry.interiors) == 1


def test_highlight_is_kept_for_the_maps_only():
    field = mapping(box(28.102, 47.102, 28.105, 47.105))
    result = parcel({"type": "Feature", "geometry": SQUARE, "properties": {"highlight": field}})

    assert isinstance(result.highlight, Polygon)
    assert "highlight" not in result.metadata
    assert result.area_ha == pytest.approx(parcel(SQUARE).area_ha)


def test_broken_highlight_is_refused():
    with pytest.raises(ParcelError, match="highlight"):
        parcel({"type": "Feature", "geometry": SQUARE, "properties": {"highlight": {"type": "Polygon"}}})
