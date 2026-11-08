from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import numpy.typing as npt
import pystac
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import geometry_mask

from argo_geoai.analysis.grid import ParcelGrid
from argo_geoai.parcels import Parcel
from argo_geoai.schema import DataSource
from argo_geoai.sources import planetary

COLLECTION = "sentinel-2-l2a"
DATA_SOURCE = DataSource(
    collection=COLLECTION,
    title="Sentinel-2 Level-2A",
    provider="ESA Copernicus / Microsoft Planetary Computer",
    resolution_m=10,
)

CLEAR_CLASSES = (4, 5, 6, 7, 11)
# processing baseline 04.00+ (from 2022-01-25) stores reflectance with a +1000 offset
OFFSET_BASELINE = 4.0
BOA_OFFSET = 1000
REFLECTANCE_SCALE = 10_000

FloatArray = npt.NDArray[np.float64]


def reflectance(digital_numbers: npt.NDArray[np.number], processing_baseline: float) -> FloatArray:
    values = digital_numbers.astype("float64")
    if processing_baseline >= OFFSET_BASELINE:
        values -= BOA_OFFSET
    return values / REFLECTANCE_SCALE


def ndvi(red: FloatArray, nir: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result: FloatArray = (nir - red) / (nir + red)
    result[~np.isfinite(result)] = np.nan
    return result


def processing_baseline(item: pystac.Item) -> float:
    try:
        return float(item.properties.get("s2:processing_baseline") or 0)
    except ValueError:
        return 0.0


def least_cloudy_per_day(items: list[pystac.Item]) -> list[pystac.Item]:
    by_day: dict[date, pystac.Item] = {}
    for item in items:
        if item.datetime is None:
            continue
        day = item.datetime.date()
        current = by_day.get(day)
        if current is None or item.properties["eo:cloud_cover"] < current.properties["eo:cloud_cover"]:
            by_day[day] = item
    return [by_day[day] for day in sorted(by_day)]


@dataclass(frozen=True)
class SceneNdvi:
    day: date
    item_id: str
    scene_cloud_cover: float
    parcel_pixels: int
    pixel_area_m2: float
    ndvi: xr.DataArray

    @property
    def clear_pixels(self) -> int:
        return int(np.isfinite(self.ndvi.values).sum())

    @property
    def clear_share(self) -> float:
        return self.clear_pixels / self.parcel_pixels if self.parcel_pixels else 0.0

    @property
    def values(self) -> FloatArray:
        data: FloatArray = self.ndvi.values
        return data[np.isfinite(data)]


def scene_ndvi(item: pystac.Item, parcel: Parcel, *, buffer_m: float = 0) -> SceneNdvi | None:
    red = planetary.read_window(item, "B04", parcel, buffer_m=buffer_m)
    nir = planetary.read_window(item, "B08", parcel, buffer_m=buffer_m)
    scl = planetary.read_window(item, "SCL", parcel, buffer_m=buffer_m)
    scl = scl.rio.reproject_match(red, resampling=Resampling.nearest)

    parcel_mask = ~geometry_mask(
        parcel.in_crs(red.rio.crs).geometry, out_shape=red.shape, transform=red.rio.transform()
    )
    covered = parcel_mask & (red.values > 0) & (nir.values > 0)
    if not covered.any():
        return None

    baseline = processing_baseline(item)
    values = ndvi(reflectance(red.values, baseline), reflectance(nir.values, baseline))
    clear = covered & np.isin(scl.values, CLEAR_CLASSES)
    values[~clear] = np.nan

    resolution_x, resolution_y = red.rio.resolution()
    raster = xr.DataArray(values.astype("float32"), coords=red.coords, dims=red.dims)
    raster = raster.rio.write_crs(red.rio.crs).rio.write_nodata(np.nan)
    assert item.datetime is not None
    return SceneNdvi(
        day=item.datetime.date(),
        item_id=item.id,
        scene_cloud_cover=float(item.properties["eo:cloud_cover"]),
        parcel_pixels=int(parcel_mask.sum()),
        pixel_area_m2=abs(resolution_x * resolution_y),
        ndvi=raster,
    )


def normalized_difference(a: FloatArray, b: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result: FloatArray = (a - b) / (a + b)
    result[~np.isfinite(result)] = np.nan
    return result


@dataclass(frozen=True)
class SceneMoisture:
    day: date
    item_id: str
    scene_cloud_cover: float
    parcel_pixels: int
    pixel_area_m2: float
    ndmi: xr.DataArray
    ndvi: xr.DataArray

    @property
    def clear_share(self) -> float:
        clear = int(np.isfinite(self.ndmi.values).sum())
        return clear / self.parcel_pixels if self.parcel_pixels else 0.0


def scene_moisture(item: pystac.Item, parcel: Parcel) -> SceneMoisture | None:
    nir = planetary.read_window(item, "B8A", parcel)
    swir = planetary.read_window(item, "B11", parcel).rio.reproject_match(nir, resampling=Resampling.nearest)
    scl = planetary.read_window(item, "SCL", parcel).rio.reproject_match(nir, resampling=Resampling.nearest)
    red = planetary.read_window(item, "B04", parcel).rio.reproject_match(nir, resampling=Resampling.average)

    parcel_mask = ~geometry_mask(
        parcel.in_crs(nir.rio.crs).geometry, out_shape=nir.shape, transform=nir.rio.transform()
    )
    covered = parcel_mask & (nir.values > 0) & (swir.values > 0) & (red.values > 0)
    if not covered.any():
        return None

    baseline = processing_baseline(item)
    nir_r, swir_r, red_r = (reflectance(band.values, baseline) for band in (nir, swir, red))
    clear = covered & np.isin(scl.values, CLEAR_CLASSES)

    def raster(values: FloatArray) -> xr.DataArray:
        values = np.where(clear, values, np.nan)
        data = xr.DataArray(values.astype("float32"), coords=nir.coords, dims=nir.dims)
        result: xr.DataArray = data.rio.write_crs(nir.rio.crs).rio.write_nodata(np.nan)
        return result

    resolution_x, resolution_y = nir.rio.resolution()
    assert item.datetime is not None
    return SceneMoisture(
        day=item.datetime.date(),
        item_id=item.id,
        scene_cloud_cover=float(item.properties["eo:cloud_cover"]),
        parcel_pixels=int(parcel_mask.sum()),
        pixel_area_m2=abs(resolution_x * resolution_y),
        ndmi=raster(normalized_difference(nir_r, swir_r)),
        ndvi=raster(normalized_difference(nir_r, red_r)),
    )


SNOW_CLASS = 11
MIN_SURFACE_CLEAR_SHARE = 0.3


@dataclass(frozen=True)
class SceneSurface:
    day: date
    item_id: str
    scene_cloud_cover: float
    clear_share: float
    snow_share: float
    ndvi: FloatArray | None
    ndti: FloatArray | None


def scene_surface(item: pystac.Item, parcel: Parcel, grid: ParcelGrid) -> SceneSurface:
    buffer_m = grid.resolution_m
    scl = grid.put(planetary.read_window(item, "SCL", parcel, buffer_m=buffer_m))
    covered = grid.inside & np.isfinite(scl) & (scl > 0)
    clear = covered & np.isin(scl, CLEAR_CLASSES) & (scl != SNOW_CLASS)
    cells = max(grid.cells, 1)
    assert item.datetime is not None
    result = SceneSurface(
        day=item.datetime.date(),
        item_id=item.id,
        scene_cloud_cover=float(item.properties.get("eo:cloud_cover", 0)),
        clear_share=float(clear.sum() / cells),
        snow_share=float((covered & (scl == SNOW_CLASS)).sum() / cells),
        ndvi=None,
        ndti=None,
    )
    if result.clear_share < MIN_SURFACE_CLEAR_SHARE:
        return result

    baseline = processing_baseline(item)

    def band(asset: str, overview_level: int | None = None) -> FloatArray:
        raster = planetary.read_window(item, asset, parcel, buffer_m=buffer_m, overview_level=overview_level)
        values = grid.put(raster.where(raster > 0), Resampling.average)
        return reflectance(values, baseline)

    red = band("B04", overview_level=0)
    nir, swir1, swir2 = band("B8A"), band("B11"), band("B12")
    ndvi_values = normalized_difference(nir, red)
    ndti_values = normalized_difference(swir1, swir2)
    valid = clear & np.isfinite(ndvi_values) & np.isfinite(ndti_values)
    return replace(
        result,
        ndvi=np.where(valid, ndvi_values, np.nan),
        ndti=np.where(valid, ndti_values, np.nan),
    )
