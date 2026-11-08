from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import pairwise
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

EventCode = Literal["tillage", "emergence", "green_cut", "harvest"]
Source = Literal["sentinel-1", "sentinel-2"]

SNOW_OR_WATER_NDVI = 0.05
NO_PLANTS_NDVI = 0.3
BARE_NDVI = 0.25
EMERGED_NDVI = 0.3
GREEN_NDVI = 0.5
TILLAGE_NDTI_DROP = 0.04
TILLED_NDTI = 0.06
GREEN_CUT_NDVI_DROP = 0.3
GREEN_CUT_MAX_GAP_DAYS = 15
# a green cut stays low: when the next clear image within this many days has won back at least this part
# of the drop, the low image was haze or a cloud shadow the mask missed. A later image says nothing (mown
# grass grows back in 2-3 weeks), the cut is kept then.
GREEN_CUT_RECOVERY_DAYS = 10
GREEN_CUT_RECOVERED_SHARE = 0.5
MAX_OPTICAL_GAP_DAYS = 45
# field work and emergence happen March-November: in winter NDVI and NDTI change with snow, frost and thaw
# (and a winter crop greening up after the snow is not an emergence)
FIELD_WORK_MONTHS = range(3, 12)
EMERGENCE_MONTHS = range(4, 12)

RADAR_STEP_OBSERVATIONS = 4
HARVEST_RATIO_DROP_DB = 1.5
HARVEST_FIELD_DROP_DB = 1.0
HARVEST_CELL_DAYS = 21
HARVEST_SEARCH_DAYS = 150
HARVEST_MONTHS = range(6, 13)  # June-December: a winter crop browning under frost is not a harvest
HARVEST_WINDOW_QUANTILES = (0.15, 0.85)

MERGE_GAP_DAYS = 14
MAIN_PART_SHARE = 0.25
MIN_EVENT_SHARE = 0.15


@dataclass(frozen=True)
class OpticalObservation:
    day: date
    ndvi: FloatArray
    ndti: FloatArray


@dataclass(frozen=True)
class RadarObservation:
    acquired: datetime
    orbit: str
    ratio_db: FloatArray


@dataclass(frozen=True)
class Detection:
    code: EventCode
    date_from: date
    date_to: date
    cells: BoolArray
    source: Source


@dataclass(frozen=True)
class Event:
    code: EventCode
    date_from: date
    date_to: date
    share: float
    source: Source
    cells: BoolArray | None = None


def recovered(
    before: OpticalObservation, after: OpticalObservation, nxt: OpticalObservation | None
) -> BoolArray:
    if nxt is None or (nxt.day - after.day).days > GREEN_CUT_RECOVERY_DAYS:
        return np.zeros(after.ndvi.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        back: BoolArray = (nxt.ndvi - after.ndvi) >= GREEN_CUT_RECOVERED_SHARE * (before.ndvi - after.ndvi)
    return back & np.isfinite(nxt.ndvi)


def optical_detections(observations: list[OpticalObservation]) -> list[Detection]:
    detections: list[Detection] = []
    ordered = sorted(observations, key=lambda o: o.day)
    for index, (before, after) in enumerate(pairwise(ordered)):
        gap = (after.day - before.day).days
        if gap > MAX_OPTICAL_GAP_DAYS or after.day.month not in FIELD_WORK_MONTHS:
            continue
        seen = np.isfinite(before.ndvi) & np.isfinite(after.ndvi)
        ground = seen & (before.ndvi >= SNOW_OR_WATER_NDVI) & (after.ndvi >= SNOW_OR_WATER_NDVI)
        no_plants = ground & (before.ndvi < NO_PLANTS_NDVI) & (after.ndvi < NO_PLANTS_NDVI)
        with np.errstate(invalid="ignore"):
            cells = {
                "tillage": no_plants
                & (after.ndti - before.ndti <= -TILLAGE_NDTI_DROP)
                & (after.ndti < TILLED_NDTI),
                "emergence": ground
                & (before.ndvi < BARE_NDVI)
                & (after.ndvi >= EMERGED_NDVI)
                & (after.day.month in EMERGENCE_MONTHS),
                "green_cut": seen
                & (before.ndvi >= GREEN_NDVI)
                & (before.ndvi - after.ndvi >= GREEN_CUT_NDVI_DROP)
                & (gap <= GREEN_CUT_MAX_GAP_DAYS)
                & ~recovered(before, after, ordered[index + 2] if index + 2 < len(ordered) else None),
            }
        for code, changed in cells.items():
            if changed.any():
                detections.append(
                    Detection(
                        code=code,  # type: ignore[arg-type]
                        date_from=before.day,
                        date_to=after.day,
                        cells=changed,
                        source="sentinel-2",
                    )
                )
    return detections


def normalized_ratio(observations: list[RadarObservation]) -> tuple[list[RadarObservation], FloatArray]:
    ordered = sorted(observations, key=lambda o: o.acquired)
    ratio = np.stack([o.ratio_db for o in ordered]).astype("float64")
    orbits = np.array([o.orbit for o in ordered])
    for orbit in set(orbits):
        rows = orbits == orbit
        with np.errstate(all="ignore"):
            ratio[rows] -= np.nanmedian(ratio[rows], axis=0)
    return ordered, ratio


def radar_steps(ratio: FloatArray, window: int = RADAR_STEP_OBSERVATIONS) -> FloatArray:
    steps = np.full(ratio.shape, np.nan)
    finite = np.isfinite(ratio)
    values = np.where(finite, ratio, 0.0)
    total = np.vstack([np.zeros((1, ratio.shape[1])), np.cumsum(values, axis=0)])
    count = np.vstack([np.zeros((1, ratio.shape[1])), np.cumsum(finite, axis=0)])
    for i in range(window, ratio.shape[0] - window + 1):
        after_n = count[i + window] - count[i]
        before_n = count[i] - count[i - window]
        with np.errstate(all="ignore"):
            after = (total[i + window] - total[i]) / after_n
            before = (total[i] - total[i - window]) / before_n
        steps[i] = np.where((after_n > 0) & (before_n > 0), after - before, np.nan)
    return steps


def crop_ends(observations: list[OpticalObservation]) -> list[tuple[date, BoolArray]]:
    ordered = sorted(observations, key=lambda o: o.day)
    ends: dict[date, BoolArray] = {}
    cells = ordered[0].ndvi.shape if ordered else (0,)
    last_green = np.full(cells, -1)
    for index, observation in enumerate(ordered):
        with np.errstate(invalid="ignore"):
            green = observation.ndvi >= GREEN_NDVI
            gone = (observation.ndvi < NO_PLANTS_NDVI) & (last_green >= 0)
        if observation.day.month not in HARVEST_MONTHS:
            last_green[green] = index
            continue
        for green_index in np.unique(last_green[gone]):
            day = ordered[int(green_index)].day
            mask = gone & (last_green == green_index)
            ends[day] = ends.get(day, np.zeros(cells, dtype=bool)) | mask
        last_green[gone] = -1
        last_green[green] = index
    return sorted(ends.items())


def first_tilled(observations: list[OpticalObservation], cells: BoolArray, after: date) -> date:
    for observation in sorted(observations, key=lambda o: o.day):
        if observation.day <= after or not cells.any():
            continue
        with np.errstate(all="ignore"):
            ndvi = float(np.nanmedian(observation.ndvi[cells]))
            ndti = float(np.nanmedian(observation.ndti[cells]))
        if ndvi < NO_PLANTS_NDVI and ndti < TILLED_NDTI:
            return observation.day
    return date.max


def harvest_detections(radar: list[RadarObservation], optical: list[OpticalObservation]) -> list[Detection]:
    if len(radar) < 2 * RADAR_STEP_OBSERVATIONS:
        return []
    ordered, ratio = normalized_ratio(radar)
    steps = radar_steps(ratio)
    times = np.array([o.acquired.date() for o in ordered])
    months = np.array([t.month in HARVEST_MONTHS for t in times])
    detections: list[Detection] = []
    for green_day, cells in crop_ends(optical):
        until = min(green_day + timedelta(days=HARVEST_SEARCH_DAYS), first_tilled(optical, cells, green_day))
        window = months & (times > green_day) & (times <= until)
        with np.errstate(all="ignore"):
            field = radar_steps(np.nanmedian(ratio[:, cells], axis=1)[:, None])[:, 0]
        field = np.where(window & np.isfinite(field), field, np.inf)
        center = int(field.argmin())
        if field[center] > -HARVEST_FIELD_DROP_DB:
            continue
        near = np.array([abs((t - times[center]).days) <= HARVEST_CELL_DAYS for t in times]) & window
        rows = np.flatnonzero(near)
        sub = np.where(np.isfinite(steps[rows]), steps[rows], np.inf)
        best = sub.argmin(axis=0)
        drop = sub[best, np.arange(sub.shape[1])]
        harvested = cells & (drop <= -HARVEST_RATIO_DROP_DB)
        if not harvested.any():
            continue
        first, last = np.quantile(rows[best[harvested]], HARVEST_WINDOW_QUANTILES, method="nearest")
        detections.append(
            Detection(
                code="harvest",
                date_from=times[max(int(first) - 1, 0)],
                date_to=times[int(last)],
                cells=harvested,
                source="sentinel-1",
            )
        )
    return detections


def merge(detections: list[Detection], cells: int) -> list[Event]:
    events: list[Event] = []
    for code in sorted({d.code for d in detections}):
        group: list[Detection] = []
        for detection in sorted((d for d in detections if d.code == code), key=lambda d: d.date_to):
            if group and (detection.date_to - max(g.date_to for g in group)).days > MERGE_GAP_DAYS:
                events.extend(_event(group, cells))
                group = []
            group.append(detection)
        if group:
            events.extend(_event(group, cells))
    return sorted(events, key=lambda e: (e.date_to, e.date_from))


def _event(group: list[Detection], cells: int) -> list[Event]:
    covered = np.logical_or.reduce([d.cells for d in group])
    share = float(covered.sum() / cells) if cells else 0.0
    if share < MIN_EVENT_SHARE:
        return []
    largest = max(int(d.cells.sum()) for d in group)
    kept = [d for d in group if d.cells.sum() >= MAIN_PART_SHARE * largest]
    return [
        Event(
            code=group[0].code,
            date_from=min(d.date_from for d in kept),
            date_to=max(d.date_to for d in kept),
            share=round(share, 3),
            source=group[0].source,
            cells=covered,
        )
    ]
