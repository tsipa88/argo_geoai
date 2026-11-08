from dataclasses import dataclass
from functools import cached_property

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import xarray as xr
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from shapely.geometry.base import BaseGeometry

from argo_geoai.parcels import Parcel

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ParcelGrid:
    crs: CRS
    transform: Affine
    shape: tuple[int, int]
    inside: BoolArray

    @property
    def resolution_m(self) -> float:
        return float(self.transform.a)

    @property
    def cells(self) -> int:
        return int(self.inside.sum())

    @cached_property
    def x(self) -> FloatArray:
        cols = np.arange(self.shape[1])
        values: FloatArray = (self.transform.c + (cols + 0.5) * self.transform.a).astype("float64")
        return values

    @cached_property
    def y(self) -> FloatArray:
        rows = np.arange(self.shape[0])
        values: FloatArray = (self.transform.f + (rows + 0.5) * self.transform.e).astype("float64")
        return values

    def template(self, values: FloatArray | None = None) -> xr.DataArray:
        data = np.full(self.shape, np.nan) if values is None else values
        raster = xr.DataArray(data, coords={"y": self.y, "x": self.x}, dims=("y", "x"))
        result: xr.DataArray = raster.rio.write_crs(self.crs).rio.write_transform(self.transform)
        return result

    def put(self, raster: xr.DataArray, resampling: Resampling = Resampling.nearest) -> FloatArray:
        nodata = raster.rio.nodata
        source = raster.astype("float64")
        if nodata is not None and not np.isnan(nodata):
            source = source.where(raster != nodata)
        source = source.rio.write_nodata(np.nan, encoded=False)
        warped = source.rio.reproject(
            self.crs,
            shape=self.shape,
            transform=self.transform,
            resampling=resampling,
            nodata=np.nan,
        )
        values: FloatArray = warped.values.astype("float64")
        return values


def parcel_grid(parcel: Parcel, resolution_m: float) -> ParcelGrid:
    crs = parcel.utm_crs
    minx, miny, maxx, maxy = parcel.in_crs(crs).total_bounds
    left = np.floor(minx / resolution_m) * resolution_m
    top = np.ceil(maxy / resolution_m) * resolution_m
    cols = int(np.ceil((maxx - left) / resolution_m))
    rows = int(np.ceil((top - miny) / resolution_m))
    transform = Affine(resolution_m, 0, left, 0, -resolution_m, top)
    inside = ~geometry_mask(parcel.in_crs(crs).geometry, out_shape=(rows, cols), transform=transform)
    return ParcelGrid(crs=crs, transform=transform, shape=(rows, cols), inside=inside)


def cells_of(grid: ParcelGrid, geometry: BaseGeometry, *, inner_edge_m: float = 0) -> BoolArray:
    shape = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(grid.crs).iloc[0]
    if inner_edge_m and not shape.buffer(-inner_edge_m).is_empty:
        shape = shape.buffer(-inner_edge_m)
    inside_shape = ~geometry_mask([shape], out_shape=grid.shape, transform=grid.transform)
    cells: BoolArray = inside_shape[grid.inside]
    return cells
