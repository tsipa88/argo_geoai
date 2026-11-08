from datetime import UTC, date, datetime

import numpy as np
import pystac
from shapely.geometry import Polygon

from argo_geoai.schema import Label
from argo_geoai.tools.field_details import (
    CONFIDENCE_LABELS,
    EVENT_KINDS,
    BoundarySide,
    ChangeEvent,
    CoverShare,
    Elevation,
    FieldDetailsMetrics,
    FieldDetailsParams,
    Observations,
    Relief,
    Slope,
    YearComposite,
    boundary_sides,
    change_kind,
    changed_objects,
    composite,
    confidence_of,
    date_change,
    direction_name,
    event_text,
    field_details_findings,
    radar_agrees,
    side_direction,
    sides_with_events,
    surface_name,
    thin,
)


def item(day: str) -> pystac.Item:
    return pystac.Item(
        id=day,
        geometry=None,
        bbox=None,
        datetime=datetime.fromisoformat(f"{day}T10:00:00").replace(tzinfo=UTC),
        properties={},
    )


def test_thin_keeps_the_ends_and_spreads_evenly():
    items = [item(f"2020-01-{d:02d}") for d in range(1, 11)]

    kept = thin(items, 4)

    assert [i.id for i in kept] == ["2020-01-01", "2020-01-04", "2020-01-07", "2020-01-10"]
    assert thin(items, 20) == items


def test_changed_objects_groups_cells_and_drops_small_groups():
    area = np.ones((4, 6), dtype=bool)
    baseline = np.full((5, 4, 6), 0.1)
    current = baseline.copy()
    current[:, 0:2, 0:2] += 0.1
    current[:, 3, 5] += 0.1

    objects = changed_objects(baseline, current, area, threshold=0.08, min_cells=2)

    assert len(objects) == 1
    assert objects[0].sum() == 4
    assert objects[0][0:2, 0:2].all()


def test_change_kind_and_surface_names():
    assert change_kind(0.10, 0.20) == "appeared"
    assert change_kind(0.20, 0.10) == "removed"
    assert change_kind(0.10, 0.11) == "changed"
    assert surface_name(np.array([0.25, 0.25, 0.25, 0.3, 0.2])).startswith("світла")
    assert surface_name(np.array([0.04, 0.05, 0.05, 0.08, 0.1])).startswith("темна")
    assert surface_name(np.array([0.05, 0.08, 0.12, 0.4, 0.2])).startswith("рослинність")
    assert surface_name(np.array([0.08, 0.09, 0.12, 0.14, 0.2])).startswith("рудувата")
    assert surface_name(np.array([0.10, 0.10, 0.10, 0.12, 0.2])).startswith("сіра")


def test_date_change_finds_the_window_of_two_confirmed_dates():
    days = [date(2020, 5, 1), date(2020, 8, 1), date(2021, 4, 1), date(2021, 5, 1), date(2021, 6, 1)]
    values = [0.10, 0.11, 0.10, 0.21, 0.22]

    assert date_change(days, values, before=0.1, after=0.2, year_to=2021) == (
        date(2021, 4, 1),
        date(2021, 5, 1),
    )
    assert date_change(days, [0.10, 0.11, 0.10, 0.21, 0.10], before=0.1, after=0.2, year_to=2021) == (
        None,
        None,
    )
    assert date_change([], [], before=0.1, after=0.2, year_to=2021) == (None, None)


def test_composite_prefers_the_season_dates():
    class Scene:
        def __init__(self, day, value):
            self.day = day
            self.bands = np.full((5, 2, 2), value)

    scenes = [
        Scene(date(2020, 1, 10), 0.5),
        Scene(date(2020, 5, 1), 0.1),
        Scene(date(2020, 6, 1), 0.1),
        Scene(date(2020, 7, 1), 0.12),
    ]

    values = composite(scenes)  # type: ignore[arg-type]

    assert values is not None
    assert values[0, 0, 0] == 0.1
    assert composite([]) is None


def test_direction_names():
    assert direction_name(100, 0) == "північ"
    assert direction_name(-100, 5) == "південь"
    assert direction_name(0, -100) == "захід"
    assert direction_name(70, 70) == "північний схід"


def test_boundary_sides_of_a_rectangle_with_a_notch():
    ring = [(0, 0), (100, 0), (100, 50), (0, 50), (0, 30), (1, 29), (0, 28), (0, 0)]

    sides = boundary_sides(ring, tolerance_m=4)

    assert len(sides) == 4
    polygon = Polygon(ring)
    assert side_direction(sides[0], polygon) == "південна"
    assert side_direction(sides[1], polygon) == "східна"
    assert side_direction(sides[2], polygon) == "північна"
    assert side_direction(sides[3], polygon) == "західна"


def test_boundary_sides_join_short_pieces_to_the_side_before():
    ring = [(0, 0), (100, 0), (100, 50), (0, 50), (0, 30), (10, 30), (10, 25), (0, 25), (0, 0)]

    sides = boundary_sides(ring, tolerance_m=1)

    assert all(side.length >= 20 for side in sides)
    assert len(sides) == 5
    polygon = Polygon(ring)
    assert [side_direction(side, polygon) for side in sides] == [
        "південна",
        "східна",
        "північна",
        "західна",
        "західна",
    ]


def side(index: int = 0) -> BoundarySide:
    return BoundarySide(
        index=index,
        letters="А–Б",
        name="південна межа (А–Б)",
        name_locative="південній межі (А–Б)",
        direction="південна",
        length_m=100,
        start=(28.0, 47.0),
        end=(28.001, 47.0),
        bbox=(28.0, 47.0, 28.001, 47.0005),
        events=[],
        text="",
    )


def event(**overrides) -> ChangeEvent:
    base = {
        "number": 1,
        "kind": "appeared",
        "kind_label": EVENT_KINDS["appeared"][0],
        "color": EVENT_KINDS["appeared"][1],
        "date_from": date(2021, 4, 1),
        "date_to": date(2021, 5, 1),
        "year_from": 2020,
        "year_to": 2021,
        "area_m2": 600,
        "inside_m2": 600,
        "inside_share": 1.0,
        "parcel_share": 0.023,
        "location": "у частині ділянки: північний захід",
        "on_boundary": False,
        "side": None,
        "brightness_before": 0.09,
        "brightness_after": 0.21,
        "surface_before": "сіра поверхня (покриття, дах або ущільнений ґрунт)",
        "surface_after": "світла поверхня (світлий дах або бетон)",
        "radar_change_db": 4.2,
        "confidence": "high",
        "confidence_label": CONFIDENCE_LABELS["high"],
        "text": "",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[28, 47], [28.001, 47], [28.001, 47.001], [28, 47]]],
        },
    }
    return ChangeEvent(**{**base, **overrides})


def test_event_text_describes_a_new_building_inside_the_parcel():
    text = event_text(event(), [side()])

    assert text == (
        "Об'єкт 1: поверхня стала значно світлішою — з'явився новий об'єкт або тверде покриття. "
        "Коли: між 01.04.2021 і 01.05.2021. Де: у частині ділянки: північний захід. Площа ~600 м² "
        "(2.3% ділянки), цілком у межах ділянки. Було: сіра поверхня (покриття, дах або ущільнений ґрунт) "
        "(яскравість 0.09); стало: світла поверхня (світлий дах або бетон) (0.21). Радар Sentinel-1: "
        "відбиття зросло на 4.2 дБ — підтверджує нову будівлю або споруду. Надійність виявлення: висока."
    )


def test_event_text_of_an_object_on_the_boundary_warns_about_encroachment():
    text = event_text(
        event(on_boundary=True, side=0, inside_m2=200, inside_share=0.33, area_m2=600, radar_change_db=None),
        [side()],
    )

    assert "з них у межах ділянки 200 м²" in text
    assert "Радар" not in text
    assert text.endswith(
        "Надійність виявлення: висока. Об'єкт лежить на південній межі (А–Б): 33% його площі всередині "
        "ділянки, 67% — за межею. Можливе заходження на ділянку (або з неї) — перевірте на знімку цієї межі."
    )


def test_radar_text_never_contradicts_the_optical_kind():
    assert "не підтверджує" in event_text(event(radar_change_db=-5.5), [side()])
    assert "підтверджує, що споруду прибрано" in event_text(
        event(kind="removed", radar_change_db=-5.5), [side()]
    )
    assert "знесення не підтверджує" in event_text(event(kind="removed", radar_change_db=3.0), [side()])
    assert "висота об'єктів, схоже, не змінилась" in event_text(event(radar_change_db=0.5), [side()])


def test_confidence_rules():
    assert confidence_of(2, 0.02, None) == "low"
    assert confidence_of(2, 0.06, False) == "low"
    assert confidence_of(6, 0.06, None) == "high"
    assert confidence_of(3, 0.06, True) == "high"
    assert confidence_of(6, 0.02, None) == "medium"
    assert radar_agrees("appeared", 3.0) is True
    assert radar_agrees("removed", 3.0) is False
    assert radar_agrees("changed", 3.0) is None
    assert radar_agrees("appeared", None) is None


def metrics(**overrides) -> FieldDetailsMetrics:
    base = {
        "land_cover": [
            CoverShare(code="built_up", label=Label(en="Built-up", uk="Забудова"), share=0.85),
            CoverShare(code="grassland", label=Label(en="Grassland", uk="Трав'яний покрив"), share=0.15),
        ],
        "relief": Relief(
            elevation_m=Elevation(min=260.0, max=262.0, mean=261.0, range=2.0),
            around_elevation_m=Elevation(min=250.0, max=270.0, mean=260.0, range=20.0),
            slope_deg=Slope(mean=0.5, p90=1.0, max=1.5),
            flat=True,
            low_side=None,
        ),
        "observations": Observations(
            scenes_found=300,
            scenes_read=250,
            dates_counted=120,
            radar_acquisitions=80,
            first_date=date(2017, 6, 1),
            last_date=date(2024, 6, 1),
        ),
        "years": [
            YearComposite(
                year=2017, dates=12, first_date=date(2017, 4, 1), last_date=date(2017, 10, 1), png_base64=None
            ),
            YearComposite(
                year=2024, dates=10, first_date=date(2024, 4, 1), last_date=date(2024, 6, 1), png_base64=None
            ),
        ],
        "events": [event(), event(number=2, on_boundary=True, side=0, inside_m2=200, inside_share=0.33)],
        "boundary": [side()],
        "encroachment_events": [2],
        "snapshots": [],
        "images": [],
    }
    return FieldDetailsMetrics(**{**base, **overrides})


def test_findings_name_the_changes_and_the_boundary():
    texts = {f.code: f.text for f in field_details_findings(metrics(), FieldDetailsParams())}

    assert texts["land_cover"] == (
        "За картою покриву ESA WorldCover 2021 ділянка — переважно забудова (85%), "
        "також трав'яний покрив (15%)."
    )
    assert texts["observations"] == (
        "Знімки Sentinel-2 з 01.06.2017 по 01.06.2024: 250 сцен, з них 120 ясних дат "
        "(не менше 90% площі без хмар), радар Sentinel-1: 80 знімків."
    )
    assert texts["changes"] == (
        "За 2017–2024 роки в межах ділянки виявлено 1 зміну забудови або покриття (з'явився об'єкт: 1), "
        "разом ~600 м²; з високою надійністю — 1. Дати й опис — у таблиці змін."
    )
    assert texts["encroachment"] == (
        "На межі ділянки виявлено 1 об'єкт, що лежить по обидва її боки, разом ~200 м² усередині ділянки: "
        "можливе заходження на ділянку. Подробиці — у розділі про межі."
    )


def test_findings_without_changes():
    texts = {
        f.code: f.text
        for f in field_details_findings(metrics(events=[], encroachment_events=[]), FieldDetailsParams())
    }

    assert texts["no_changes"].startswith("За 2017–2024 роки в межах ділянки змін забудови чи покриття")
    assert texts["no_encroachment"].startswith("Об'єктів, що з'явились або зникли по обидва боки межі")


def test_findings_without_data():
    empty = metrics(
        land_cover=[],
        relief=None,
        observations=Observations(
            scenes_found=0,
            scenes_read=0,
            dates_counted=0,
            radar_acquisitions=0,
            first_date=None,
            last_date=None,
        ),
        years=[],
        events=[],
        boundary=[],
        encroachment_events=[],
    )

    findings = field_details_findings(empty, FieldDetailsParams())

    assert [f.code for f in findings] == ["no_clear_dates"]
    assert findings[0].level == "warning"


def test_sides_with_events_lists_the_objects_on_each_side():
    sides = [side(0), side(1)]
    events = [event(number=1, on_boundary=True, side=1, inside_m2=200, inside_share=0.4), event(number=2)]

    result = sides_with_events(sides, events, (2017, 2026), 300)

    assert result[0].events == []
    assert result[0].text == (
        "За 2017–2026 роки на цій межі змін, що лежать по обидва її боки, не виявлено "
        "(знімки 10 м: об'єкти менші за ~300 м² можуть бути не помічені)."
    )
    assert result[1].events == [1]
    assert (
        result[1].text == "На цій межі виявлено 1 зміну: об'єкт 1 (з'явився об'єкт, 2021, усередині 200 м²)."
    )
