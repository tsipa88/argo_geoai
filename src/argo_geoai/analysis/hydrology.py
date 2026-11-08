import heapq

import numpy as np
import numpy.typing as npt
from scipy import ndimage

FloatArray = npt.NDArray[np.float64]

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# contour length factors of Quinn et al.: cardinal 0.5, diagonal 0.354 cell widths
CONTOUR_FACTOR = {(dr, dc): 0.354 if dr and dc else 0.5 for dr, dc in NEIGHBOURS}
FLOW_EXPONENT = 1.1
FILL_EPSILON_M = 1e-4  # tiny rise across filled flats so they still drain toward an outlet
MIN_TAN_SLOPE = 0.001


def _outlets(valid: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    border = np.zeros_like(valid)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    near_missing = ndimage.binary_dilation(~valid, structure=np.ones((3, 3), dtype=bool))
    result: npt.NDArray[np.bool_] = valid & (border | near_missing)
    return result


def fill_depressions(heights: FloatArray, epsilon: float = FILL_EPSILON_M) -> FloatArray:
    filled = heights.astype("float64").copy()
    rows, cols = filled.shape
    valid = np.isfinite(filled)
    closed = ~valid
    heap: list[tuple[float, int, int]] = []
    for r, c in zip(*np.nonzero(_outlets(valid)), strict=True):
        heap.append((float(filled[r, c]), int(r), int(c)))
        closed[r, c] = True
    heapq.heapify(heap)

    while heap:
        level, r, c = heapq.heappop(heap)
        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not closed[nr, nc]:
                closed[nr, nc] = True
                if filled[nr, nc] <= level:
                    filled[nr, nc] = level + epsilon
                heapq.heappush(heap, (float(filled[nr, nc]), nr, nc))
    return filled


def flow_accumulation(
    filled: FloatArray, cell_size: float, exponent: float = FLOW_EXPONENT
) -> tuple[FloatArray, FloatArray]:
    rows, cols = filled.shape
    valid = np.isfinite(filled)
    area = np.where(valid, cell_size**2, 0.0)
    edge_area = np.where(_outlets(valid) & _window_border(valid.shape), cell_size**2, 0.0)

    order = np.argsort(np.where(valid, filled, -np.inf), axis=None)[::-1]
    for index in order:
        r, c = divmod(int(index), cols)
        if not valid[r, c]:
            continue
        level = filled[r, c]
        targets, weights = [], []
        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and valid[nr, nc] and filled[nr, nc] < level:
                distance = cell_size * (1.4142135623730951 if dr and dc else 1.0)
                weights.append(((level - filled[nr, nc]) / distance) ** exponent * CONTOUR_FACTOR[(dr, dc)])
                targets.append((nr, nc))
        total = sum(weights)
        if not total:
            continue
        for (nr, nc), weight in zip(targets, weights, strict=True):
            area[nr, nc] += area[r, c] * weight / total
            edge_area[nr, nc] += edge_area[r, c] * weight / total

    with np.errstate(divide="ignore", invalid="ignore"):
        edge_share = np.where(valid, edge_area / area, np.nan)
    return np.where(valid, area, np.nan), edge_share


def _window_border(shape: tuple[int, ...]) -> npt.NDArray[np.bool_]:
    border = np.zeros(shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return border


def wetness_index(area_m2: FloatArray, slope_deg: FloatArray, cell_size: float) -> FloatArray:
    specific_area = area_m2 / cell_size
    tan_slope = np.maximum(np.tan(np.radians(slope_deg)), MIN_TAN_SLOPE)
    result: FloatArray = np.log(specific_area / tan_slope)
    return result
