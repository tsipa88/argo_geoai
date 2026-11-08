from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import geopandas as gpd
import shapely
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry

from argo_geoai.request import ParcelRequest
from argo_geoai.schema import Message, ParcelInfo

WGS84 = "EPSG:4326"
AREA_MISMATCH_SHARE = 0.02

Area = Polygon | MultiPolygon


class ParcelError(Exception):
    pass


@dataclass(frozen=True)
class Parcel:
    geometry: Area
    name: str
    expected_area_ha: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    repaired: bool = False
    highlight: Area | None = None

    @cached_property
    def frame(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(geometry=[self.geometry], crs=WGS84)

    @cached_property
    def utm_crs(self) -> CRS:
        crs: CRS = self.frame.estimate_utm_crs()
        return crs

    @property
    def parts(self) -> int:
        return len(self.geometry.geoms) if isinstance(self.geometry, MultiPolygon) else 1

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        minx, miny, maxx, maxy = self.geometry.bounds
        return (minx, miny, maxx, maxy)

    @cached_property
    def area_ha(self) -> float:
        return float(self.frame.to_crs(self.utm_crs).area.iloc[0]) / 10_000

    def in_crs(self, crs: CRS | str, buffer_m: float = 0) -> gpd.GeoSeries:
        geometry = self.frame.geometry
        if buffer_m:
            geometry = geometry.to_crs(self.utm_crs).buffer(buffer_m)
        return geometry.to_crs(crs)

    def checks(self) -> list[Message]:
        messages = []
        expected = self.expected_area_ha
        if expected and abs(self.area_ha - expected) / expected > AREA_MISMATCH_SHARE:
            messages.append(
                Message(
                    code="area_mismatch",
                    text=(
                        f"polygon area {self.area_ha:.2f} ha differs from the expected {expected:.2f} ha "
                        f"by {abs(self.area_ha - expected) / expected:.0%}"
                    ),
                )
            )
        if self.repaired:
            messages.append(
                Message(
                    code="geometry_repaired", text="the geometry was invalid (self-intersecting), repaired"
                )
            )
        return messages

    def info(self) -> ParcelInfo:
        centroid = self.geometry.centroid
        return ParcelInfo(
            name=self.name,
            parts=self.parts,
            area_ha=round(self.area_ha, 2),
            expected_area_ha=self.expected_area_ha,
            bbox=tuple(round(v, 6) for v in self.bbox),
            centroid=(round(centroid.x, 6), round(centroid.y, 6)),
            metadata=self.metadata,
        )


def normalize(geometry: BaseGeometry, origin: str) -> Area:
    if not isinstance(geometry, Polygon | MultiPolygon):
        raise ParcelError(f"{origin}: expected a Polygon or MultiPolygon, got {geometry.geom_type}")
    if not geometry.is_valid:
        polygons = [g for g in getattr(shapely.make_valid(geometry), "geoms", [geometry]) if not g.is_empty]
        geometry = shapely.union_all([g for g in polygons if isinstance(g, Polygon | MultiPolygon)])
        if not isinstance(geometry, Polygon | MultiPolygon) or geometry.is_empty:
            raise ParcelError(f"{origin}: geometry is not valid and cannot be repaired")
    if isinstance(geometry, MultiPolygon) and len(geometry.geoms) == 1:
        return geometry.geoms[0]
    return geometry


def from_request(request: ParcelRequest) -> Parcel:
    try:
        raw = shape(request.geometry)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ParcelError(f"{request.name}: not a GeoJSON geometry ({error})") from error
    highlight = None
    if request.properties.highlight is not None:
        try:
            highlight = normalize(shape(request.properties.highlight), f"{request.name} highlight")
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ParcelError(f"{request.name}: highlight is not a GeoJSON geometry ({error})") from error
    return Parcel(
        geometry=normalize(raw, request.name),
        highlight=highlight,
        name=request.name,
        expected_area_ha=request.properties.area_ha,
        metadata=request.properties.metadata,
        repaired=not raw.is_valid,
    )
