import numpy as np
import numpy.typing as npt
import xarray as xr
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays

from argo_geoai.parcels import Parcel
from argo_geoai.schema import DataSource
from argo_geoai.sources import planetary

COLLECTION = "cop-dem-glo-30"
DATA_SOURCE = DataSource(
    collection=COLLECTION,
    title="Copernicus DEM GLO-30",
    provider="ESA Copernicus / Microsoft Planetary Computer",
    resolution_m=30,
)
RESOLUTION_M = 30
NODATA_BELOW_M = -1000

FloatArray = npt.NDArray[np.float64]


def elevation(parcel: Parcel, *, buffer_m: float) -> tuple[xr.DataArray, list[str]]:
    items = planetary.search(COLLECTION, parcel, buffer_m=buffer_m)
    if not items:
        raise LookupError("no Copernicus DEM tiles cover the parcel")
    pieces = [planetary.read_window(item, "data", parcel, buffer_m=buffer_m) for item in items]
    mosaic = merge_arrays(pieces) if len(pieces) > 1 else pieces[0]
    grid = mosaic.rio.reproject(parcel.utm_crs, resolution=RESOLUTION_M, resampling=Resampling.bilinear)
    return grid.where(grid > NODATA_BELOW_M), [item.id for item in items]


def slope_aspect(heights: FloatArray, resolution_m: float) -> tuple[FloatArray, FloatArray]:
    d_row, d_col = np.gradient(heights.astype("float64"), resolution_m)
    dz_east, dz_north = d_col, -d_row
    slope = np.degrees(np.arctan(np.hypot(dz_east, dz_north)))
    aspect = (np.degrees(np.arctan2(-dz_east, -dz_north)) + 360) % 360
    return slope, aspect
