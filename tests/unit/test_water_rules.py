from datetime import date

import pytest

from argo_geoai.tools.common import Correlation
from argo_geoai.tools.water import (
    WETNESS_CLASSES,
    NdviWetness,
    Offset,
    OpenWater,
    WaterMetrics,
    Wetness,
    WetnessClassShare,
    locative_direction,
    wetness_findings,
)

NO_WATER = OpenWater(share_ever=0, share_frequent=0, max_occurrence_pct=0, max_months_per_year=0)


def metrics(**overrides: object) -> WaterMetrics:
    """The runoff picture of tests/data/field_ravines.geojson: inflow from beyond the field."""
    values: dict[str, object] = {
        "pixels": 817,
        "area_ha": 73.53,
        "flat": False,
        "twi": Wetness(mean=8.23, median=7.82, p10=6.91, p90=10.27, max=14.25),
        "wetness_classes": [
            WetnessClassShare(code=code, label=label, min_twi=None, max_twi=None, share=share)
            for (code, _, _, label), share in zip(WETNESS_CLASSES, (0.543, 0.424, 0.032, 0.001), strict=True)
        ],
        "wet_share": 0.033,
        "wet_part_offset": Offset(north_m=-52, east_m=182),
        "max_upslope_ha": 98.7,
        "flow_lines_share": 0.086,
        "catchment_truncated_share": 0,
        "open_water_on_parcel": NO_WATER,
        "open_water_nearby": NO_WATER,
        "ndvi_relation": [],
        "ndvi_mixed_crops": False,
    }
    values.update(overrides)
    return WaterMetrics.model_validate(values)


def relation(rho: float, dry: float, wet: float, p_value: float = 1e-9) -> NdviWetness:
    return NdviWetness(
        date=date(2026, 8, 2),
        reason="vegetation_peak",
        scene_id="S2A_x",
        pixels=800,
        vs_twi=Correlation(rho=rho, p_value=p_value),
        mean_by_wetness_third={"dry": dry, "middle": (dry + wet) / 2, "wet": wet},
    )


def by_code(found):
    return {f.code: f for f in found}


def test_ravine_field_like_field_ravines():
    found = by_code(wetness_findings(metrics()))

    assert found["wet_share"].level == "info"
    assert "мало: 3%" in found["wet_share"].text
    assert "99 га" in found["inflow"].text
    assert "більшої за саме поле (74 га)" in found["inflow"].text
    assert "9% поля лежить на лініях концентрованого стоку" in found["flow_lines"].text
    assert "відкритої води на полі не фіксувалось" in found["open_water"].text
    assert "flat_field" not in found


def test_wet_part_is_named_when_the_wet_cells_sit_apart():
    found = by_code(
        wetness_findings(metrics(wet_share=0.07, wet_part_offset=Offset(north_m=-400, east_m=50)))
    )

    assert found["wet_part"].text == "Вологіші ділянки зосереджені переважно на півдні поля."


@pytest.mark.parametrize(
    ("share", "level"), [(0.004, "info"), (0.05, "info"), (0.15, "notice"), (0.35, "warning")]
)
def test_wet_share_levels(share, level):
    assert by_code(wetness_findings(metrics(wet_share=share)))["wet_share"].level == level


def test_flat_field_gets_a_caveat_and_no_flow_lines():
    found = by_code(wetness_findings(metrics(flat=True)))

    assert found["flat_field"].level == "notice"
    assert "flow_lines" not in found


def test_no_inflow_when_the_catchment_is_inside_the_field():
    assert "inflow" not in by_code(wetness_findings(metrics(max_upslope_ha=7)))


def test_open_water_on_the_parcel():
    water = OpenWater(share_ever=0.04, share_frequent=0.01, max_occurrence_pct=46, max_months_per_year=7)

    found = by_code(wetness_findings(metrics(open_water_on_parcel=water)))

    assert found["open_water"].level == "warning"
    assert "4% площі" in found["open_water"].text
    assert "46% місяців" in found["open_water"].text


def test_open_water_only_nearby():
    nearby = OpenWater(share_ever=0.2, share_frequent=0.1, max_occurrence_pct=90, max_months_per_year=12)

    text = by_code(wetness_findings(metrics(open_water_nearby=nearby)))["open_water"].text

    assert "не фіксувалось" in text
    assert "Поруч" in text


def test_ndvi_wetness_texts():
    weak_wet = wetness_findings(metrics(ndvi_relation=[relation(-0.25, dry=0.68, wet=0.59)]))[-1]
    tiny = wetness_findings(metrics(ndvi_relation=[relation(-0.4, dry=0.85, wet=0.83)]))[-1]

    assert "вологіші ділянки слабші" in weak_wet.text
    assert "найвологіша третина 0.59" in weak_wet.text
    assert "практично не вплинула" in tiny.text


def test_no_ndvi_wetness_conclusions_with_several_crops():
    found = by_code(
        wetness_findings(metrics(ndvi_relation=[relation(-0.3, 0.7, 0.5)], ndvi_mixed_crops=True))
    )

    assert "ndvi_vs_wetness" not in found
    assert "кілька культур" in found["ndvi_vs_wetness_mixed_crops"].text


@pytest.mark.parametrize(
    ("north", "east", "expected"), [(500, 0, "півночі"), (-300, -300, "південному заході"), (0, 400, "сході")]
)
def test_locative_direction(north, east, expected):
    assert locative_direction(north, east) == expected


def test_tiny_wet_share_is_not_called_zero_percent():
    text = by_code(wetness_findings(metrics(wet_share=0.004)))["wet_share"].text

    assert text == "Місць, де збирається вода, на полі практично немає."


def test_flat_field_draws_no_ndvi_wetness_conclusions():
    found = by_code(wetness_findings(metrics(flat=True, ndvi_relation=[relation(0.2, dry=0.64, wet=0.69)])))

    assert "ndvi_vs_wetness" not in found
    assert "не оцінюємо" in found["flat_field"].text
