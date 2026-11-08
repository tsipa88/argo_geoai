from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import xarray as xr
from rasterio.enums import Resampling
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

MAX_ZONES = 4
MIN_ZONE_SHARE = 0.05
MIN_SILHOUETTE = 0.35
MIN_CURVE_GAP = 0.2
MIN_PIXEL_DATES = 0.5
SILHOUETTE_SAMPLE = 3000
SEED = 0


@dataclass(frozen=True)
class NdviStack:
    dates: list[date]
    values: FloatArray
    rows: IntArray
    cols: IntArray
    template: xr.DataArray
    pixel_area_m2: float


@dataclass(frozen=True)
class Zoning:
    labels: IntArray
    count: int
    silhouette: float | None


def stack(rasters: dict[date, xr.DataArray]) -> NdviStack:
    dates = sorted(rasters)
    reference = max(dates, key=lambda d: int(np.isfinite(rasters[d].values).sum()))
    template = rasters[reference]
    layers = [
        rasters[d].values
        if rasters[d].shape == template.shape and np.array_equal(rasters[d].x, template.x)
        else rasters[d].rio.reproject_match(template, resampling=Resampling.nearest).values
        for d in dates
    ]
    cube = np.stack(layers, axis=-1).astype("float64")
    seen = np.isfinite(cube).sum(axis=-1)
    rows, cols = np.nonzero(seen >= max(1, MIN_PIXEL_DATES * len(dates)))
    resolution_x, resolution_y = template.rio.resolution()
    return NdviStack(
        dates=dates,
        values=cube[rows, cols],
        rows=rows.astype("int64"),
        cols=cols.astype("int64"),
        template=template,
        pixel_area_m2=abs(resolution_x * resolution_y),
    )


def _filled(values: FloatArray) -> FloatArray:
    medians = np.nanmedian(values, axis=0)
    return np.where(np.isfinite(values), values, medians)


def _acceptable(labels: IntArray, centers: FloatArray, count: int) -> bool:
    shares = np.bincount(labels, minlength=count) / labels.size
    if shares.min() < MIN_ZONE_SHARE:
        return False
    gaps = [np.abs(centers[a] - centers[b]).max() for a in range(count) for b in range(a + 1, count)]
    return bool(min(gaps) >= MIN_CURVE_GAP)


def find_zones(values: FloatArray, max_zones: int = MAX_ZONES) -> Zoning:
    pixels = values.shape[0]
    single = Zoning(labels=np.zeros(pixels, dtype="int64"), count=1, silhouette=None)
    if pixels < 20 or values.shape[1] < 2:
        return single

    data = _filled(values)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(pixels, size=min(pixels, SILHOUETTE_SAMPLE), replace=False)

    best: Zoning = single
    for count in range(2, max_zones + 1):
        model = KMeans(n_clusters=count, n_init=10, random_state=SEED).fit(data)
        labels = model.labels_.astype("int64")
        if len(np.unique(labels[sample])) < 2:
            continue
        score = float(silhouette_score(data[sample], labels[sample]))
        if score < MIN_SILHOUETTE or not _acceptable(labels, model.cluster_centers_, count):
            continue
        if best.silhouette is None or score > best.silhouette:
            best = Zoning(labels=labels, count=count, silhouette=round(score, 3))

    order = np.argsort(-np.bincount(best.labels, minlength=best.count))
    relabel = np.empty_like(order)
    relabel[order] = np.arange(best.count)
    return Zoning(labels=relabel[best.labels].astype("int64"), count=best.count, silhouette=best.silhouette)


def zone_raster(ndvi_stack: NdviStack, zoning: Zoning) -> FloatArray:
    raster = np.full(ndvi_stack.template.shape, np.nan)
    raster[ndvi_stack.rows, ndvi_stack.cols] = zoning.labels + 1
    return raster


MIN_PATCH_HA = 0.5


@dataclass(frozen=True)
class Patches:
    count: int
    largest_share: float


def patches(raster: FloatArray, zone_number: int, pixel_area_m2: float) -> Patches:
    labeled, _ = ndimage.label(raster == zone_number, structure=np.ones((3, 3)))
    sizes = np.bincount(labeled.ravel())[1:]
    if sizes.size == 0:
        return Patches(count=0, largest_share=0.0)
    min_pixels = MIN_PATCH_HA * 10_000 / pixel_area_m2
    return Patches(
        count=int((sizes >= min_pixels).sum()) or 1, largest_share=round(float(sizes.max() / sizes.sum()), 3)
    )
