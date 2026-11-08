import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import geoai
import planetary_computer
import pystac
import rioxarray
import xarray as xr

from argo_geoai.config import get_settings
from argo_geoai.parcels import Parcel

log = logging.getLogger(__name__)

for _name, _value in {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
}.items():
    os.environ.setdefault(_name, _value)

READ_WORKERS = 8


def search(
    collection: str,
    parcel: Parcel,
    *,
    time_range: str | None = None,
    query: dict[str, Any] | None = None,
    buffer_m: float = 0,
    max_items: int = 500,
) -> list[pystac.Item]:
    bbox = parcel.in_crs("EPSG:4326", buffer_m).total_bounds.tolist()
    items: list[pystac.Item] = geoai.pc_stac_search(
        collection,
        bbox=bbox,
        time_range=time_range,
        query=query,
        max_items=max_items,
        quiet=True,
        endpoint=get_settings().stac_url,
    )
    log.info("%s: %d items", collection, len(items))
    return items


def read_window(
    item: pystac.Item,
    asset: str,
    parcel: Parcel,
    *,
    buffer_m: float = 0,
    overview_level: int | None = None,
) -> xr.DataArray:
    href = planetary_computer.sign(item.assets[asset].href)
    # not geoai.read_pc_item_asset: its global rasterio lock would read scenes one by one
    opened = rioxarray.open_rasterio(href, lock=False, overview_level=overview_level)
    assert isinstance(opened, xr.DataArray)
    try:
        data = opened.squeeze(drop=True)
        minx, miny, maxx, maxy = parcel.in_crs(data.rio.crs, buffer_m).total_bounds
        window: xr.DataArray = data.rio.clip_box(minx, miny, maxx, maxy).load()
    finally:
        # close the remote file now: left to garbage collection it is closed at interpreter
        # shutdown, which prints "Error in sys.excepthook" to stderr
        opened.close()
    return window


def read_parallel[T, R](read: Callable[[T], R], items: Iterable[T]) -> list[R]:
    with ThreadPoolExecutor(READ_WORKERS) as pool:
        return list(pool.map(read, items))
