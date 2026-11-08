from dataclasses import dataclass

import numpy as np
from rasterio.features import geometry_mask

from argo_geoai.parcels import Parcel
from argo_geoai.schema import DataSource, Label
from argo_geoai.sources import planetary

COLLECTION = "esa-worldcover"
DATA_SOURCE = DataSource(
    collection=COLLECTION,
    title="ESA WorldCover 2021",
    provider="ESA / Microsoft Planetary Computer",
    resolution_m=10,
)

CLASSES: dict[int, tuple[str, Label]] = {
    10: ("tree_cover", Label(en="Tree cover", uk="Дерева")),
    20: ("shrubland", Label(en="Shrubland", uk="Чагарники")),
    30: ("grassland", Label(en="Grassland", uk="Трав'яний покрив")),
    40: ("cropland", Label(en="Cropland", uk="Рілля")),
    50: ("built_up", Label(en="Built-up", uk="Забудова")),
    60: ("bare_sparse", Label(en="Bare / sparse vegetation", uk="Відкритий ґрунт / рідка рослинність")),
    70: ("snow_ice", Label(en="Snow and ice", uk="Сніг і лід")),
    80: ("water", Label(en="Permanent water", uk="Водойми")),
    90: ("wetland", Label(en="Herbaceous wetland", uk="Водно-болотні угіддя")),
    95: ("mangroves", Label(en="Mangroves", uk="Мангри")),
    100: ("moss_lichen", Label(en="Moss and lichen", uk="Мохи й лишайники")),
}


@dataclass(frozen=True)
class CoverShare:
    code: str
    label: Label
    share: float


def land_cover(parcel: Parcel) -> tuple[list[CoverShare], list[str]]:
    items = planetary.search(COLLECTION, parcel)
    if not items:
        return [], []

    counts: dict[int, int] = {}
    for item in items:
        cover = planetary.read_window(item, "map", parcel)
        inside = ~geometry_mask(
            parcel.in_crs(cover.rio.crs).geometry, out_shape=cover.shape, transform=cover.rio.transform()
        )
        classes, pixel_counts = np.unique(cover.values[inside & (cover.values > 0)], return_counts=True)
        for value, count in zip(classes, pixel_counts, strict=True):
            counts[int(value)] = counts.get(int(value), 0) + int(count)

    total = sum(counts.values())
    shares = [
        CoverShare(
            code=CLASSES.get(value, (str(value), Label(en=str(value), uk=str(value))))[0],
            label=CLASSES.get(value, (str(value), Label(en=str(value), uk=str(value))))[1],
            share=count / total,
        )
        for value, count in sorted(counts.items(), key=lambda pair: -pair[1])
    ]
    return shares, [item.id for item in items]
