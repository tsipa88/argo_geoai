from datetime import date

import numpy as np
import pytest
import xarray as xr

from argo_geoai.sources.sentinel2 import SceneMoisture
from argo_geoai.tools.moisture import (
    DryZone,
    MoistureDate,
    MoistureMetrics,
    MoistureTool,
    NdmiStats,
    Offset,
    closed_canopy_ndmi,
    moisture_findings,
    ndmi_residual,
    percent,
    phases,
    pixel_peaks,
    stress_shares,
)


def record(
    day: date,
    *,
    ndvi: float,
    vegetated: float = 0.9,
    dense: float = 0.8,
    stress: tuple[float, float, float] | None = None,
):
    shares = (
        stress_shares(
            np.array(
                [-0.1] * int(stress[0] * 100) + [0.1] * int(stress[1] * 100) + [0.3] * int(stress[2] * 100)
            )
        )
        if stress
        else []
    )
    return MoistureDate(
        date=day,
        scene_id=f"S2_{day:%Y%m%d}",
        scene_cloud_cover=1,
        clear_share=1,
        counted=True,
        vegetated_share=vegetated,
        dense_share=dense,
        ndvi_mean=ndvi,
        ndmi=NdmiStats(mean=0.2, median=0.2, p10=0.1, p90=0.3),
        phase="growing" if stress else None,
        stress=shares,
        stressed_share=round(stress[0] + stress[1], 3) if stress else None,
    )


def metrics(history: list[MoistureDate], **overrides: object) -> MoistureMetrics:
    assessed = [h for h in history if h.phase == "growing"]
    values: dict[str, object] = {
        "period_start": date(2026, 4, 1),
        "period_end": date(2026, 10, 1),
        "scenes_found": 30,
        "dates_total": len(history),
        "dates_counted": len(history),
        "dates_assessed": len(assessed),
        "ndvi_peak": date(2026, 6, 20),
        "worst": max(assessed, key=lambda h: h.stressed_share or 0, default=None),
        "latest": assessed[-1] if assessed else None,
        "dry_zone": None,
        "mixed_crops": False,
        "history": history,
    }
    return MoistureMetrics.model_validate(values | overrides)


def test_stress_classes_of_green_pixels():
    shares = {s.code: s.share for s in stress_shares(np.array([-0.05, 0.05, 0.15, 0.25, 0.4]))}

    assert shares == {"strong": 0.4, "moderate": 0.2, "none": 0.4}


def test_phases_bare_growing_and_ripening():
    history = [
        record(date(2026, 4, 1), ndvi=0.25, vegetated=0.1, dense=0),
        record(date(2026, 5, 1), ndvi=0.45, vegetated=0.6, dense=0.1),
        record(date(2026, 5, 20), ndvi=0.6),
        record(date(2026, 6, 20), ndvi=0.85),
        record(date(2026, 7, 10), ndvi=0.75),
        record(date(2026, 8, 5), ndvi=0.45),
    ]

    peak, phase_of = phases(history)

    assert peak == date(2026, 6, 20)
    assert list(phase_of.values()) == ["bare", "sparse", "growing", "growing", "growing", "ripening"]


def test_strong_stress_is_a_warning():
    history = [record(date(2026, 7, 1), ndvi=0.7, stress=(0.3, 0.4, 0.3))]

    findings = {f.code: f for f in moisture_findings(metrics(history))}

    assert findings["water_stress_strong"].level == "warning"
    assert "01.07.2026" in findings["water_stress_strong"].text
    assert "30%" in findings["water_stress_strong"].text
    assert findings["green_but_dry"].level == "notice"


def test_moderate_stress_is_a_notice():
    history = [record(date(2026, 6, 1), ndvi=0.5, stress=(0.05, 0.25, 0.7))]

    findings = {f.code: f for f in moisture_findings(metrics(history))}

    assert findings["water_stress_moderate"].level == "notice"
    assert "green_but_dry" not in findings


def test_no_stress():
    history = [
        record(date(2026, 5, 1), ndvi=0.6, stress=(0, 0.05, 0.95)),
        record(date(2026, 6, 1), ndvi=0.8, stress=(0, 0.02, 0.98)),
    ]

    findings = {f.code: f for f in moisture_findings(metrics(history))}

    assert findings["no_water_stress"].level == "info"
    assert findings["latest_moisture"].text.startswith("Остання оцінена дата 01.06.2026")


def test_without_green_dates_nothing_is_concluded():
    history = [record(date(2026, 4, 1), ndvi=0.2, vegetated=0.05)]

    findings = moisture_findings(metrics(history, ndvi_peak=None))

    assert [f.code for f in findings] == ["no_green_dates"]


def test_ripening_dates_are_explained():
    history = [
        record(date(2026, 6, 20), ndvi=0.85, stress=(0, 0.1, 0.9)),
        record(date(2026, 8, 5), ndvi=0.4).model_copy(update={"phase": "ripening"}),
    ]

    findings = {f.code: f for f in moisture_findings(metrics(history))}

    assert "дати після піку (1) не оцінюються" in findings["ripening_dates"].text


@pytest.mark.parametrize(
    ("zone", "code", "words"),
    [
        (
            DryZone(share=0.12, offset=Offset(north_m=-300, east_m=-300), dates_used=6),
            "dry_zone",
            "південному заході",
        ),
        (
            DryZone(share=0.12, offset=Offset(north_m=20, east_m=10), dates_used=6),
            "dry_zone",
            "розкидані по полю",
        ),
        (DryZone(share=0.02, offset=None, dates_used=6), "no_dry_zone", "змінюються від дати до дати"),
    ],
)
def test_dry_zone(zone, code, words):
    history = [record(date(2026, 6, 1), ndvi=0.8, stress=(0, 0.1, 0.9))]

    findings = {f.code: f for f in moisture_findings(metrics(history, dry_zone=zone))}

    assert words in findings[code].text


def test_dry_zone_is_not_concluded_on_mixed_crops():
    history = [record(date(2026, 6, 1), ndvi=0.8, stress=(0, 0.1, 0.9))]
    zone = DryZone(share=0.15, offset=Offset(north_m=200, east_m=200), dates_used=4)

    codes = [f.code for f in moisture_findings(metrics(history, dry_zone=zone, mixed_crops=True))]

    assert "dry_zone_mixed_crops" in codes
    assert "dry_zone" not in codes


def test_percent_never_says_zero_for_a_small_share():
    assert percent(0.004) == "менш ніж 1%"
    assert percent(0) == "0%"
    assert percent(0.25) == "25%"


def scene(day: date, ndmi: np.ndarray, ndvi: np.ndarray) -> SceneMoisture:
    coords = {"y": 5_350_000 - 20 * np.arange(4), "x": 300_000 + 20 * np.arange(4)}

    def raster(values: np.ndarray) -> xr.DataArray:
        array = xr.DataArray(values.astype("float32"), coords=coords, dims=("y", "x"))
        located: xr.DataArray = array.rio.write_crs("EPSG:32635")
        return located

    return SceneMoisture(
        day=day,
        item_id="S2",
        scene_cloud_cover=0,
        parcel_pixels=16,
        pixel_area_m2=400,
        ndmi=raster(ndmi),
        ndvi=raster(ndvi),
    )


NDVI = np.tile(np.array([0.5, 0.6, 0.7, 0.8]), (4, 1))
HEALTHY = 0.6 * NDVI - 0.15


def test_ndmi_residual_removes_the_canopy_effect():
    residual = ndmi_residual(scene(date(2026, 6, 1), HEALTHY, NDVI))

    assert np.allclose(residual.values, 0, atol=1e-6)


def test_dry_zone_is_where_leaves_are_drier_than_their_greenness_explains():
    dry_corner = HEALTHY.copy()
    dry_corner[:2, :2] -= 0.15
    scenes = {date(2026, 6, d): scene(date(2026, 6, d), dry_corner + 0.01 * d, NDVI) for d in (1, 10, 20)}

    zone, frequency = MoistureTool()._dry_zone(scenes)

    assert zone is not None
    assert zone.share == 0.25
    assert zone.offset is not None
    assert zone.offset.north_m > 0
    assert zone.offset.east_m < 0
    assert frequency is not None
    assert float(frequency.values[0, 0]) == 1.0


def test_sparser_part_is_not_a_dry_zone():
    scenes = {date(2026, 6, d): scene(date(2026, 6, d), HEALTHY, NDVI) for d in (1, 10, 20)}

    zone, _ = MoistureTool()._dry_zone(scenes)

    assert zone is not None
    assert zone.share == 0


def test_no_dry_zone_with_too_few_dates():
    scenes = {date(2026, 6, 1): scene(date(2026, 6, 1), HEALTHY, NDVI)}

    assert MoistureTool()._dry_zone(scenes) == (None, None)


def test_a_ripening_crop_next_to_a_growing_one_is_not_stress():
    may = np.where(np.arange(4) < 2, 0.9, 0.5) * np.ones((4, 1))
    june = np.where(np.arange(4) < 2, 0.75, 0.8) * np.ones((4, 1))
    june_ndmi = np.where(np.arange(4) < 2, 0.05, 0.3) * np.ones((4, 1))
    scenes = {
        date(2026, 5, 20): scene(date(2026, 5, 20), 0.6 * may - 0.15, may),
        date(2026, 6, 25): scene(date(2026, 6, 25), june_ndmi, june),
    }

    canopy = closed_canopy_ndmi(scenes[date(2026, 6, 25)], pixel_peaks(scenes))

    assert canopy.ripening == 0.5
    assert canopy.developing == 0
    assert np.allclose(canopy.ndmi, 0.3)


def test_a_canopy_still_closing_is_not_stress():
    june = np.full((4, 4), 0.77)
    july = np.full((4, 4), 0.85)
    scenes = {
        date(2026, 6, 25): scene(date(2026, 6, 25), np.full((4, 4), 0.14), june),
        date(2026, 6, 30): scene(date(2026, 6, 30), np.full((4, 4), 0.3), july),
    }

    canopy = closed_canopy_ndmi(scenes[date(2026, 6, 25)], pixel_peaks(scenes))

    assert canopy.developing == 1.0
    assert canopy.ndmi.size == 0
