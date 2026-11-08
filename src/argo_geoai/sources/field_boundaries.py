import fcntl
import logging
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import shapely
from shapely.geometry import MultiPolygon, Polygon

from argo_geoai.config import get_settings
from argo_geoai.schema import DataSource

log = logging.getLogger(__name__)

BASE_URL = "https://data.source.coop/ftw/global-data/predictions/vectors/alpha/results-by-admin-conf"
DATA_SOURCE = DataSource(
    collection="ftw-global-field-boundaries",
    title="Fields of The World — global field boundaries (PRUE, Sentinel-2)",
    provider="Fields of The World / Source Cooperative, CC-BY-4.0",
    resolution_m=10,
)
COLUMNS = ["id", "geometry", "bbox", "determination:datetime", "confidence"]
DOWNLOAD_CHUNK = 8 << 20
# Source Cooperative answers 403 to Python's default User-Agent ("Python-urllib/3.x")
USER_AGENT = "argo-geoai"


@dataclass(frozen=True)
class Partition:
    code: str
    path: str
    bbox: tuple[float, float, float, float]


PARTITIONS = [
    Partition(
        code="UA",
        path="admin:country_code=UA/Ukraine.parquet",
        bbox=(22.138880839225628, 44.3936563933728, 40.24777242196722, 52.3952805418396),
    ),
]


@dataclass(frozen=True)
class FieldRecord:
    id: str
    geometry: Polygon | MultiPolygon
    year: int
    confidence: float | None


def search(bbox: tuple[float, float, float, float]) -> tuple[list[FieldRecord], list[str]]:
    records: list[FieldRecord] = []
    used = []
    for partition in PARTITIONS:
        if not _boxes_meet(partition.bbox, bbox):
            continue
        records.extend(read_fields(country_file(partition), bbox))
        used.append(partition.path)
    return records, used


# no spatial index in the file: a window read over HTTP touches ~8 row groups of ~32 MB, so it is kept locally
def country_file(partition: Partition) -> Path:
    folder = get_settings().cache_dir / "ftw"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{partition.code}.parquet"
    if target.is_file():
        return target
    with (folder / f"{partition.code}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.is_file():
            _download(f"{BASE_URL}/{partition.path}", target)
    return target


def _download(url: str, target: Path) -> None:
    log.info("downloading %s to %s (once; a country is ~2 GB)", url, target)
    partial = target.with_suffix(".download")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        expected = int(response.headers.get("Content-Length", -1))
        shutil.copyfileobj(response, out, DOWNLOAD_CHUNK)
    size = partial.stat().st_size
    if expected >= 0 and size != expected:
        partial.unlink()
        raise OSError(f"{url}: download incomplete ({size} of {expected} bytes)")
    partial.rename(target)
    log.info("downloaded %s: %.0f MB", target, target.stat().st_size / 1e6)


def read_fields(path: Path, bbox: tuple[float, float, float, float]) -> list[FieldRecord]:
    file = pq.ParquetFile(path)
    meta = file.metadata
    columns = {meta.schema.column(i).path: i for i in range(meta.num_columns)}
    records = []
    for group in range(meta.num_row_groups):
        if not _group_may_meet(meta.row_group(group), columns, bbox):
            continue
        boxes = file.read_row_group(group, columns=["bbox"]).column("bbox").combine_chunks()
        xmin, ymin, xmax, ymax = (boxes.field(name).to_numpy() for name in ("xmin", "ymin", "xmax", "ymax"))
        hit = (xmin <= bbox[2]) & (xmax >= bbox[0]) & (ymin <= bbox[3]) & (ymax >= bbox[1])
        if not hit.any():
            continue
        table = file.read_row_group(group, columns=COLUMNS).filter(pa.array(hit))
        records.extend(_records(table))
    return records


def _records(table: pa.Table) -> list[FieldRecord]:
    geometries = shapely.from_wkb(table.column("geometry").to_numpy(zero_copy_only=False))
    years = pc.year(table.column("determination:datetime")).to_numpy(zero_copy_only=False)
    confidence = table.column("confidence").to_pylist()
    ids = table.column("id").to_pylist()
    records = []
    for field_id, raw, year, value in zip(ids, geometries, years, confidence, strict=True):
        geometry = raw if raw.is_valid else shapely.make_valid(raw)
        polygons = [part for part in getattr(geometry, "geoms", [geometry]) if isinstance(part, Polygon)]
        if not polygons:
            continue
        records.append(
            FieldRecord(
                id=str(field_id),
                geometry=polygons[0] if len(polygons) == 1 else MultiPolygon(polygons),
                year=int(year),
                confidence=None if value is None or np.isnan(value) else float(value),
            )
        )
    return records


def _group_may_meet(
    group: pq.RowGroupMetaData, columns: dict[str, int], bbox: tuple[float, float, float, float]
) -> bool:
    bounds = {}
    for name, side in (("xmin", "min"), ("ymin", "min"), ("xmax", "max"), ("ymax", "max")):
        statistics = group.column(columns[f"bbox.{name}"]).statistics
        if statistics is None or not statistics.has_min_max:
            return True
        bounds[name] = statistics.min if side == "min" else statistics.max
    return _boxes_meet((bounds["xmin"], bounds["ymin"], bounds["xmax"], bounds["ymax"]), bbox)


def _boxes_meet(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]
