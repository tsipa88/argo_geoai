from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pystac
import xarray as xr
from rasterio.features import geometry_mask
from rioxarray.merge import merge_arrays

from argo_geoai.parcels import Parcel
from argo_geoai.schema import DataSource
from argo_geoai.sources import planetary

COLLECTION = "jrc-gsw"
DATA_SOURCE = DataSource(
    collection=COLLECTION,
    title="JRC Global Surface Water 1984-2020",
    provider="EC JRC / Google / Microsoft Planetary Computer",
    resolution_m=30,
)
YEARS = "1984–2020"
NODATA = 255
FREQUENT_OCCURRENCE_PCT = 25


@dataclass(frozen=True)
class SurfaceWater:
    share_ever: float
    share_frequent: float
    max_occurrence_pct: int
    max_months_per_year: int


def _stats(
    occurrence: npt.NDArray[np.uint8], seasonality: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_]
) -> SurfaceWater:
    valid = mask & (occurrence != NODATA)
    values, months = occurrence[valid], seasonality[valid & (seasonality != NODATA)]
    if values.size == 0:
        return SurfaceWater(share_ever=0.0, share_frequent=0.0, max_occurrence_pct=0, max_months_per_year=0)
    return SurfaceWater(
        share_ever=round(float((values > 0).mean()), 3),
        share_frequent=round(float((values >= FREQUENT_OCCURRENCE_PCT).mean()), 3),
        max_occurrence_pct=int(values.max()),
        max_months_per_year=int(months.max()) if months.size else 0,
    )


def _mosaic(items: list[pystac.Item], asset: str, parcel: Parcel, buffer_m: float) -> xr.DataArray:
    pieces = [planetary.read_window(item, asset, parcel, buffer_m=buffer_m) for item in items]
    return merge_arrays(pieces, nodata=NODATA) if len(pieces) > 1 else pieces[0]


def surface_water(parcel: Parcel, *, nearby_m: float) -> tuple[SurfaceWater, SurfaceWater, list[str]]:
    items = planetary.search(COLLECTION, parcel, buffer_m=nearby_m)
    if not items:
        empty = SurfaceWater(share_ever=0.0, share_frequent=0.0, max_occurrence_pct=0, max_months_per_year=0)
        return empty, empty, []

    occurrence = _mosaic(items, "occurrence", parcel, nearby_m)
    seasonality = _mosaic(items, "seasonality", parcel, nearby_m).rio.reproject_match(occurrence)
    transform, shape = occurrence.rio.transform(), occurrence.shape
    inside = ~geometry_mask(parcel.in_crs(occurrence.rio.crs).geometry, out_shape=shape, transform=transform)
    around = ~geometry_mask(
        parcel.in_crs(occurrence.rio.crs, nearby_m).geometry, out_shape=shape, transform=transform
    )
    return (
        _stats(occurrence.values, seasonality.values, inside),
        _stats(occurrence.values, seasonality.values, around & ~inside),
        [item.id for item in items],
    )
