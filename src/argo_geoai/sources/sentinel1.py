from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import numpy.typing as npt
import pystac
from rasterio.enums import Resampling

from argo_geoai.analysis.grid import ParcelGrid
from argo_geoai.parcels import Parcel
from argo_geoai.schema import DataSource
from argo_geoai.sources import planetary

COLLECTION = "sentinel-1-rtc"
DATA_SOURCE = DataSource(
    collection=COLLECTION,
    title="Sentinel-1 RTC (радар)",
    provider="ESA Copernicus / Microsoft Planetary Computer",
    resolution_m=20,
)
OVERVIEW_LEVEL = 0
MIN_COVERED_SHARE = 0.8

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RadarScene:
    acquired: datetime
    item_id: str
    orbit: str
    vv: FloatArray
    vh: FloatArray

    @property
    def day(self) -> date:
        return self.acquired.date()


def orbit_key(item: pystac.Item) -> str:
    properties = item.properties
    return f"{properties.get('sat:relative_orbit')}-{properties.get('sat:orbit_state', '')}"


def scene(item: pystac.Item, parcel: Parcel, grid: ParcelGrid) -> RadarScene | None:
    bands = {
        name: grid.put(
            planetary.read_window(
                item, name, parcel, buffer_m=grid.resolution_m, overview_level=OVERVIEW_LEVEL
            ),
            Resampling.average,
        )
        for name in ("vv", "vh")
    }
    for values in bands.values():
        values[~grid.inside | (values <= 0)] = np.nan
    covered = np.isfinite(bands["vv"]) & np.isfinite(bands["vh"])
    if covered.sum() < MIN_COVERED_SHARE * grid.cells:
        return None
    assert item.datetime is not None
    return RadarScene(
        acquired=item.datetime,
        item_id=item.id,
        orbit=orbit_key(item),
        vv=np.where(covered, bands["vv"], np.nan),
        vh=np.where(covered, bands["vh"], np.nan),
    )
