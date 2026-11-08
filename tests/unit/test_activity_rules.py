from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from argo_geoai.analysis import activity
from argo_geoai.tools.activity import (
    EVENT_LABELS,
    MAX_PERIOD_DAYS,
    TOUCHED_BAND_COLOR,
    UNTOUCHED_BAND_COLOR,
    ActivityEvent,
    ActivityMetrics,
    ActivityParams,
    ActivityTool,
    OpticalDate,
    activity_findings,
    ground_state,
    highlight_event_text,
    highlight_finding,
    highlight_state_text,
    season_peaks,
    timeline,
    without_doubled_harvests,
)

CELLS = 100


def optical(
    day: date, ndvi: float | list[float], ndti: float | list[float] = 0.1
) -> activity.OpticalObservation:
    return activity.OpticalObservation(
        day=day,
        ndvi=np.broadcast_to(np.asarray(ndvi, dtype="float64"), (CELLS,)).copy(),
        ndti=np.broadcast_to(np.asarray(ndti, dtype="float64"), (CELLS,)).copy(),
    )


def radar_series(start: date, ratios: list[float], *, every_days: int = 3) -> list[activity.RadarObservation]:
    return [
        activity.RadarObservation(
            acquired=datetime(start.year, start.month, start.day, 16, tzinfo=UTC)
            + timedelta(days=i * every_days),
            orbit="58-ascending",
            ratio_db=np.full(CELLS, value),
        )
        for i, value in enumerate(ratios)
    ]


def codes(detections: list[activity.Detection]) -> list[str]:
    return [d.code for d in detections]


def test_residue_worked_into_bare_soil_is_tillage():
    before = optical(date(2026, 4, 24), 0.14, 0.09)
    after = optical(date(2026, 5, 4), 0.14, 0.012)

    (detection,) = activity.optical_detections([before, after])

    assert detection.code == "tillage"
    assert (detection.date_from, detection.date_to) == (date(2026, 4, 24), date(2026, 5, 4))
    assert detection.cells.all()


def test_snow_is_not_tillage():
    before = optical(date(2025, 12, 25), 0.11, 0.03)
    after = optical(date(2026, 1, 19), -0.03, -0.03)

    assert activity.optical_detections([before, after]) == []


def test_residue_that_stays_is_not_tillage():
    assert (
        activity.optical_detections(
            [optical(date(2026, 3, 10), 0.18, 0.18), optical(date(2026, 3, 20), 0.2, 0.16)]
        )
        == []
    )


def test_bare_field_turning_green_is_emergence():
    detections = activity.optical_detections(
        [optical(date(2026, 5, 11), 0.14), optical(date(2026, 6, 13), 0.35)]
    )

    assert codes(detections) == ["emergence"]


def test_canopy_gone_within_days_is_a_green_cut_but_slow_ripening_is_not():
    cut = activity.optical_detections([optical(date(2025, 6, 1), 0.8), optical(date(2025, 6, 9), 0.3)])
    ripening = activity.optical_detections(
        [optical(date(2025, 9, 26), 0.62), optical(date(2025, 10, 26), 0.27)]
    )

    assert codes(cut) == ["green_cut"]
    assert ripening == []


def test_dates_too_far_apart_say_nothing():
    assert (
        activity.optical_detections([optical(date(2026, 1, 1), 0.14), optical(date(2026, 3, 1), 0.4)]) == []
    )


def test_radar_steps_find_the_fall_of_the_ratio():
    ratio = np.array([[-6.0]] * 6 + [[-10.0]] * 6)

    steps = activity.radar_steps(ratio, window=3)

    assert np.nanargmin(steps[:, 0]) == 6
    assert steps[6, 0] == pytest.approx(-4.0)


def test_crop_end_is_the_last_green_date_before_the_field_is_without_plants():
    observations = [
        optical(date(2026, 8, 2), 0.75),
        optical(date(2026, 9, 8), 0.53),
        optical(date(2026, 9, 16), 0.38),
        optical(date(2026, 9, 28), 0.22),
    ]

    ((day, cells),) = activity.crop_ends(observations)

    assert day == date(2026, 9, 8)
    assert cells.all()


def test_harvest_is_dated_by_the_radar_after_the_canopy():
    observations = [optical(date(2026, 8, 29), 0.67), optical(date(2026, 9, 28), 0.22)]
    radar = radar_series(date(2026, 8, 30), [-6.0] * 7 + [-10.0] * 7)

    (detection,) = activity.harvest_detections(radar, observations)

    assert detection.code == "harvest"
    assert (detection.date_from, detection.date_to) == (date(2026, 9, 17), date(2026, 9, 20))
    assert detection.cells.all()


def test_no_harvest_without_a_crop_seen_green():
    observations = [optical(date(2026, 8, 29), 0.25), optical(date(2026, 9, 28), 0.22)]
    radar = radar_series(date(2026, 8, 30), [-6.0] * 7 + [-10.0] * 7)

    assert activity.harvest_detections(radar, observations) == []


def test_orbit_levels_are_removed_before_steps():
    radar = [
        activity.RadarObservation(
            acquired=datetime(2026, 9, 1, tzinfo=UTC) + timedelta(days=i),
            orbit="a" if i % 2 else "b",
            ratio_db=np.full(CELLS, -6.0 if i % 2 else -9.0),
        )
        for i in range(12)
    ]

    _, ratio = activity.normalized_ratio(radar)

    assert np.allclose(ratio, 0)


def detection(code: str, day_from: date, day_to: date, share: float) -> activity.Detection:
    cells = np.zeros(CELLS, dtype=bool)
    cells[: int(share * CELLS)] = True
    return activity.Detection(code=code, date_from=day_from, date_to=day_to, cells=cells, source="sentinel-2")  # type: ignore[arg-type]


def test_detections_close_in_time_are_one_event():
    parts = [
        detection("harvest", date(2026, 9, 17), date(2026, 9, 20), 0.3),
        detection("harvest", date(2026, 9, 26), date(2026, 9, 29), 0.3),
    ]
    parts[1].cells[:] = False
    parts[1].cells[30:60] = True

    (event,) = activity.merge(parts, CELLS)

    assert (event.date_from, event.date_to) == (date(2026, 9, 17), date(2026, 9, 29))
    assert event.share == pytest.approx(0.6)


def test_small_or_distant_detections_stay_apart_or_drop():
    events = activity.merge(
        [
            detection("tillage", date(2025, 4, 4), date(2025, 4, 14), 0.3),
            detection("tillage", date(2025, 11, 15), date(2025, 12, 25), 0.7),
            detection("tillage", date(2026, 3, 22), date(2026, 3, 25), 0.05),
        ],
        CELLS,
    )

    assert [(e.date_to, e.share) for e in events] == [(date(2025, 4, 14), 0.3), (date(2025, 12, 25), 0.7)]


def test_season_peak_only_when_the_field_was_green():
    peaks = season_peaks(
        [optical(date(2025, 8, 2), 0.84), optical(date(2025, 8, 14), 0.83), optical(date(2026, 3, 1), 0.2)]
    )

    assert [day for day, _ in peaks] == [date(2025, 8, 2)]
    assert peaks[0][1] == pytest.approx(0.84)


def metrics(events: list[ActivityEvent]) -> ActivityMetrics:
    return ActivityMetrics(
        period_start=date(2024, 10, 11),
        period_end=date(2026, 10, 11),
        optical_scenes=200,
        optical_dates=90,
        radar_acquisitions=400,
        events=events,
        counts={e.code: sum(1 for x in events if x.code == e.code) for e in events},
        timeline=[],
        history=[],
    )


def event(code: str, day_from: date, day_to: date, share: float | None = 0.9) -> ActivityEvent:
    return ActivityEvent(
        code=code,
        label=EVENT_LABELS[code],
        date_from=day_from,
        date_to=day_to,
        share=share,
        area_ha=None if share is None else share * 200,
        source="sentinel-1" if code == "harvest" else "sentinel-2",
        text="",
    )


def test_findings_count_events_and_name_the_latest():
    findings = activity_findings(
        metrics(
            [
                event("emergence", date(2026, 5, 11), date(2026, 6, 13)),
                event("peak", date(2026, 8, 2), date(2026, 8, 2), None),
                event("harvest", date(2026, 9, 17), date(2026, 10, 1)),
            ]
        )
    )

    assert [f.code for f in findings] == ["events_found", "latest_event", "dates_are_windows"]
    assert "2 події" in findings[0].text
    assert "збирання врожаю — 1" in findings[0].text
    assert findings[1].text == "Остання: збирання врожаю 17.09.2026–01.10.2026 на 90% поля."


def test_findings_without_events_say_so():
    (finding,) = activity_findings(metrics([event("peak", date(2026, 8, 2), date(2026, 8, 2), None)]))

    assert finding.code == "no_events"


def test_start_date_sets_the_period():
    params = ActivityParams(start_date=date(2026, 4, 15))

    assert params.period(date(2026, 10, 15)) == (date(2026, 4, 15), date(2026, 10, 15))
    assert ActivityParams(months=6).period(date(2026, 10, 15))[0] == date(2026, 4, 12)


def test_start_date_after_end_or_in_the_future_is_refused():
    with pytest.raises(ValueError, match="before end_date"):
        ActivityParams(start_date=date(2026, 5, 1), end_date=date(2026, 4, 1))
    with pytest.raises(ValueError, match="future"):
        ActivityParams(start_date=datetime.now(UTC).date() + timedelta(days=2))


def test_a_very_old_start_date_is_cut_to_the_longest_period():
    start, end = ActivityParams(start_date=date(2015, 1, 1)).period(date(2026, 10, 15))

    assert (end - start).days == MAX_PERIOD_DAYS


def record(ndvi: float, ndti: float, snow: float = 0.0) -> OpticalDate:
    return OpticalDate(
        date=date(2026, 5, 4),
        scene_id="s",
        clear_share=1.0,
        snow_share=snow,
        counted=True,
        ndvi_median=ndvi,
        ndti_median=ndti,
    )


@pytest.mark.parametrize(
    ("ndvi", "ndti", "snow", "code"),
    [
        (0.8, 0.3, 0.0, "dense_crop"),
        (0.45, 0.15, 0.0, "growing_crop"),
        (0.3, 0.1, 0.0, "sparse_green"),
        (0.2, 0.15, 0.0, "residue"),
        (0.14, 0.012, 0.0, "bare_soil"),
        (0.15, 0.07, 0.0, "no_plants"),
        (-0.03, -0.03, 0.9, "snow"),
    ],
)
def test_ground_state_of_a_clear_image(ndvi, ndti, snow, code):
    assert ground_state(record(ndvi, ndti, snow))[0] == code


def test_timeline_puts_an_event_before_the_image_that_shows_it():
    harvest = event("harvest", date(2026, 9, 19), date(2026, 10, 2))
    rows = timeline([harvest], [record(0.2, 0.1).model_copy(update={"date": date(2026, 10, 2)})])

    assert [(r.date, r.kind) for r in rows] == [
        (date(2026, 10, 2), "event"),
        (date(2026, 10, 2), "observation"),
    ]
    assert rows[0].text.startswith(
        "Збирання врожаю на 90% поля (180.0 га), зміна між 19.09.2026 і 02.10.2026."
    )
    assert rows[1].text.startswith("Знімок: Рослин немає, на полі рослинні рештки")


def test_a_radar_harvest_overlapping_a_green_cut_is_not_doubled():
    def merged(code: str, day_from: date, day_to: date) -> activity.Event:
        return activity.Event(code=code, date_from=day_from, date_to=day_to, share=0.2, source="sentinel-2")  # type: ignore[arg-type]

    events = [
        merged("harvest", date(2026, 7, 20), date(2026, 7, 26)),
        merged("harvest", date(2026, 8, 21), date(2026, 8, 27)),
        merged("green_cut", date(2026, 8, 22), date(2026, 8, 29)),
    ]

    assert [(e.code, e.date_to) for e in without_doubled_harvests(events)] == [
        ("harvest", date(2026, 7, 26)),
        ("green_cut", date(2026, 8, 29)),
    ]


def land_event(share: float | None, code: str = "tillage") -> ActivityEvent:
    return event(code, date(2026, 9, 5), date(2026, 10, 3)).model_copy(
        update={
            "highlight_share": share,
            "highlight_area_ha": None if share is None else round(share * 70, 1),
        }
    )


@pytest.mark.parametrize(
    ("share", "tolerance", "text", "touched"),
    [
        (0.0, 0.05, "Не зачеплена.", False),
        (0.04, 0.05, "Не зачеплена.", False),
        (0.04, 0.03, "Частково: ~4% ділянки (2.8 га).", True),
        (0.4, 0.05, "Частково: ~40% ділянки (28.0 га).", True),
        (0.8, 0.05, "Ділянку теж: ~80% (56.0 га).", True),
    ],
)
def test_land_column_of_an_event_follows_the_tolerance_of_the_caller(share, tolerance, text, touched):
    assert highlight_event_text(land_event(share), tolerance=tolerance) == (text, touched)


def land_record(
    field: tuple[float, float], land: tuple[float, float] | None, seen: float = 1.0
) -> OpticalDate:
    return record(*field).model_copy(
        update={
            "highlight_clear_share": seen,
            "highlight_ndvi_median": land[0] if land else None,
            "highlight_ndti_median": land[1] if land else None,
        }
    )


def test_land_column_of_an_image():
    assert highlight_state_text(land_record((0.2, 0.15), (0.21, 0.14))) == "Так само, як поле."
    assert highlight_state_text(land_record((0.75, 0.3), (0.2, 0.15))).startswith(
        "Інакше, ніж поле: рослин немає, на полі рослинні рештки"
    )
    assert highlight_state_text(land_record((0.75, 0.3), None, seen=0.2)) == "Ділянку не видно (хмари)."


def test_timeline_has_a_land_column_only_with_a_highlight():
    plain = timeline([event("harvest", date(2026, 9, 19), date(2026, 10, 2))], [record(0.2, 0.1)])
    with_land = timeline(
        [land_event(0.4)], [land_record((0.2, 0.1), (0.2, 0.1))], tolerance=0.05, small_highlight=True
    )

    assert all(r.highlight_text is None for r in plain)
    assert [(r.highlight_text, r.highlight_touched) for r in with_land] == [
        ("Так само, як поле.", False),
        ("Частково: ~40% ділянки (28.0 га). Ділянка мала, оцінка приблизна.", True),
    ]


def test_land_finding_says_whether_the_land_was_touched():
    untouched = highlight_finding([land_event(0.01), land_event(0.0, "harvest")], tolerance=0.05)
    touched = highlight_finding([land_event(0.01), land_event(0.4, "harvest")], tolerance=0.05)

    assert untouched is not None
    assert untouched.text == (
        "На ділянці за період змін не видно (похибка до 5%), хоча на полі навколо були 2 події."
    )
    assert touched is not None
    assert touched.level == "warning"
    assert touched.text == "Ділянку зачепили (1 раз): збирання врожаю 05.09.2026–03.10.2026 (~40%)."
    assert highlight_finding([event("harvest", date(2026, 9, 1), date(2026, 9, 9))], tolerance=0.05) is None


def test_field_and_land_chart_paints_touched_events_red():
    harvest = land_event(0.0, "harvest")
    tillage = event("tillage", date(2026, 9, 20), date(2026, 9, 28)).model_copy(
        update={"highlight_share": 0.4, "highlight_area_ha": 28.0}
    )
    history = [land_record((0.8, 0.3), (0.82, 0.3)), land_record((0.2, 0.1), (0.8, 0.3))]
    history[1] = history[1].model_copy(update={"date": date(2026, 10, 5)})
    data = metrics([harvest, tillage]).model_copy(
        update={"history": history, "timeline": timeline([harvest, tillage], history, tolerance=0.05)}
    )
    touched = {(e.date, e.code) for e in data.timeline if e.kind == "event" and e.highlight_touched}

    figure = ActivityTool._field_and_land(data, touched)

    assert [trace.name for trace in figure.data] == ["Поле (масиви і їхні ділянки)", "Ця ділянка"]
    colors = sorted(shape.fillcolor for shape in figure.layout.shapes)
    assert colors == sorted([TOUCHED_BAND_COLOR, UNTOUCHED_BAND_COLOR])


def test_a_dip_that_recovers_on_the_next_image_is_not_a_green_cut():
    observations = [
        optical(date(2026, 7, 25), 0.79),
        optical(date(2026, 7, 30), 0.47),
        optical(date(2026, 8, 4), 0.70),
    ]

    assert "green_cut" not in codes(activity.optical_detections(observations))


def test_a_cut_that_stays_low_is_a_green_cut():
    observations = [
        optical(date(2025, 6, 1), 0.8),
        optical(date(2025, 6, 9), 0.3),
        optical(date(2025, 6, 14), 0.32),
    ]

    assert codes(activity.optical_detections(observations)) == ["green_cut"]


def test_regrowth_seen_only_weeks_later_does_not_hide_a_cut():
    observations = [
        optical(date(2025, 6, 1), 0.8),
        optical(date(2025, 6, 9), 0.3),
        optical(date(2025, 7, 2), 0.75),
    ]

    assert "green_cut" in codes(activity.optical_detections(observations))
