import numpy as np
import numpy.typing as npt
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

CELL_M = 10.0
CORNER_CELLS = 1.5
ON_SOURCE_CELLS = 1.5
EDGE_CELLS = 3.0
CUT_RUN_CELLS = 2.0
MEET_CELLS = 2.5
CORNER_TURN_DEG = (30.0, 150.0)


def straighten(
    geometry: Polygon | MultiPolygon, tolerance_m: float, *, cell_m: float = CELL_M
) -> Polygon | MultiPolygon:
    if tolerance_m <= 0:
        return geometry
    corner = CORNER_CELLS * cell_m
    parts: list[Polygon] = []
    for polygon in getattr(geometry, "geoms", [geometry]):
        shell = _through_steps(np.asarray(polygon.exterior.coords), cell_m, corner)
        holes = [_through_steps(np.asarray(ring.coords), cell_m, corner) for ring in polygon.interiors]
        rings = [ring for ring in holes if len(ring) >= 4]
        simple = Polygon(shell, rings).simplify(tolerance_m, preserve_topology=True)
        parts.extend(
            _sharp(part, polygon.boundary, cell_m)
            for part in getattr(simple, "geoms", [simple])
            if isinstance(part, Polygon)
        )
    result: Polygon | MultiPolygon | BaseGeometry = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    if not result.is_valid:
        result = shapely.make_valid(result)
    polygons = [part for part in getattr(result, "geoms", [result]) if isinstance(part, Polygon)]
    if not polygons or result.is_empty:
        return geometry
    return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)


def _through_steps(ring: npt.NDArray[np.float64], cell: float, corner: float) -> npt.NDArray[np.float64]:
    points = ring[:-1]
    following = np.roll(points, -1, axis=0)
    lengths = np.hypot(*(following - points).T)
    kept = []
    for i in range(len(points)):
        if lengths[i - 1] > corner and lengths[i] > corner:
            kept.append(points[i])
        pieces = max(1, int(np.ceil(lengths[i] / cell - 1e-9)))
        share = (np.arange(pieces) + 0.5) / pieces
        kept.extend(points[i] + share[:, None] * (following[i] - points[i]))
    if len(kept) < 3:
        return ring
    return np.vstack([*kept, kept[0]])


def _sharp(polygon: Polygon, source: BaseGeometry, cell: float) -> Polygon:
    sharp = Polygon(
        _sharpen(np.asarray(polygon.exterior.coords), source, cell),
        [_sharpen(np.asarray(ring.coords), source, cell) for ring in polygon.interiors],
    )
    return sharp if sharp.is_valid and not sharp.is_empty else polygon


def _sharpen(ring: npt.NDArray[np.float64], source: BaseGeometry, cell: float) -> npt.NDArray[np.float64]:
    points = ring[:-1]
    edge = EDGE_CELLS * cell
    lengths = np.hypot(*(np.roll(points, -1, axis=0) - points).T)
    long_segments = np.flatnonzero(lengths >= edge)
    if len(points) < 5 or not len(long_segments):
        return ring
    # start the ring with a long segment: no cut corner then runs over the end of the arrays
    points = np.roll(points, -int(long_segments[0]), axis=0)
    vectors = np.roll(points, -1, axis=0) - points
    lengths = np.hypot(*vectors.T)
    count = len(points)
    kept = [points[0]]
    i = 1
    while i < count:
        run_end = _cut_corner_end(lengths, i, cell)
        if run_end is not None:
            after = (run_end + 1) % count
            meet = _meet(
                start=points[i],
                incoming=vectors[i - 1],
                end=points[after],
                outgoing=vectors[after],
                source=source,
                cell=cell,
            )
            if meet is not None:
                kept.append(meet)
                if after == 0:
                    kept[0] = meet
                    kept.pop()
                i = run_end + 2
                continue
        kept.append(points[i])
        i += 1
    if len(kept) < 3:
        return ring
    return np.vstack([*kept, kept[0]])


def _cut_corner_end(lengths: npt.NDArray[np.float64], start: int, cell: float) -> int | None:
    count, edge = len(lengths), EDGE_CELLS * cell
    if lengths[start - 1] < edge or lengths[start] >= edge:
        return None
    end, total = start, lengths[start]
    while end + 1 < count and lengths[end + 1] < edge:
        end += 1
        total += lengths[end]
    if total > CUT_RUN_CELLS * cell:
        return None
    return end if lengths[(end + 1) % count] >= edge else None


def _meet(
    *,
    start: npt.NDArray[np.float64],
    incoming: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    outgoing: npt.NDArray[np.float64],
    source: BaseGeometry,
    cell: float,
) -> npt.NDArray[np.float64] | None:
    u, v = incoming / np.hypot(*incoming), outgoing / np.hypot(*outgoing)
    turn = np.degrees(np.arccos(np.clip(u @ v, -1, 1)))
    if not CORNER_TURN_DEG[0] <= turn <= CORNER_TURN_DEG[1]:
        return None
    t, _ = np.linalg.solve(np.column_stack([u, -v]), end - start)
    meet: npt.NDArray[np.float64] = start + t * u
    limit = MEET_CELLS * cell
    if np.hypot(*(meet - start)) > limit or np.hypot(*(meet - end)) > limit:
        return None
    if source.distance(Point(meet)) > ON_SOURCE_CELLS * cell:
        return None
    return meet
