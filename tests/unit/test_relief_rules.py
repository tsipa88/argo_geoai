from datetime import date

import pytest

from argo_geoai.tools.common import Correlation, describe_strength
from argo_geoai.tools.relief import (
    ASPECTS,
    SLOPE_CLASSES,
    ClassNdvi,
    ClassShare,
    Elevation,
    NdviRelation,
    Offset,
    ReliefMetrics,
    Slope,
    SlopeClassShare,
    direction_uk,
    relief_findings,
)


def metrics(
    slope_shares=(0.085, 0.501, 0.204, 0.143, 0.067),
    aspects=(("e", 0.353), ("w", 0.222), ("ne", 0.148), ("sw", 0.099)),
    offset=(-215, 86),
    relations=(),
) -> ReliefMetrics:
    """The relief of tests/data/field_slope.geojson (Copernicus DEM, season 2026)."""
    labels = {code: label for code, label, _ in ASPECTS}
    return ReliefMetrics(
        pixels=942,
        area_ha=84.78,
        elevation_m=Elevation(
            min=363.7, max=392.7, mean=382.0, p5=372.7, p95=390.1, range=29.0, range_p5_p95=17.4
        ),
        slope_deg=Slope(mean=3.27, median=2.71, p90=6.5, max=10.27),
        slope_classes=[
            SlopeClassShare(code=code, label=label, min_deg=lo, max_deg=hi, share=share)
            for (code, lo, hi, label), share in zip(SLOPE_CLASSES, slope_shares, strict=True)
        ],
        aspect_of_slopes=[ClassShare(code=code, label=labels[code], share=share) for code, share in aspects],
        low_third_offset_from_high_third=Offset(north_m=offset[0], east_m=offset[1]),
        ndvi_relation=list(relations),
    )


def relation(rho: float, p_value: float = 1e-6, slope_rho: float = 0.0) -> NdviRelation:
    return NdviRelation(
        date=date(2026, 7, 30),
        reason="vegetation_peak",
        scene_id="S2B_x",
        pixels=942,
        vs_elevation=Correlation(rho=rho, p_value=p_value),
        vs_slope=Correlation(rho=slope_rho, p_value=0.5),
        mean_by_elevation_third={"low": 0.79, "middle": 0.78, "high": 0.75},
        mean_by_slope_class=[],
    )


def by_code(findings):
    return {f.code: f for f in findings}


def test_field_slope():
    findings = by_code(relief_findings(metrics()))

    assert findings["elevation_range"].text == "Перепад висот на полі 29 м (364–393 м над рівнем моря)."
    assert findings["erosion_risk"].level == "notice"
    assert findings["erosion_risk"].text.startswith("21% поля на ухилах 5° і більше")
    assert findings["aspect"].text == "Схили переважно звернені на схід і північний схід (50%)."
    assert "на південь від верхньої" in findings["low_part"].text


def test_cool_slopes():
    finding = by_code(relief_findings(metrics(aspects=(("ne", 0.58), ("n", 0.2)))))["aspect"]

    assert "довше тримають вологу" in finding.text


@pytest.mark.parametrize(
    ("shares", "code", "level"),
    [
        ((0.5, 0.4, 0.1, 0.0, 0.0), "mostly_flat", "info"),
        ((0.2, 0.4, 0.25, 0.1, 0.05), "erosion_risk", "notice"),
        ((0.0, 0.1, 0.3, 0.3, 0.3), "erosion_risk", "warning"),
    ],
)
def test_erosion_levels(shares, code, level):
    findings = by_code(relief_findings(metrics(slope_shares=shares)))

    assert findings[code].level == level


def test_warm_slopes():
    finding = by_code(relief_findings(metrics(aspects=(("s", 0.5), ("sw", 0.2)))))["aspect"]

    assert "краще прогріваються" in finding.text


def test_low_part_is_skipped_when_the_field_is_compact():
    assert "low_part" not in by_code(relief_findings(metrics(offset=(40, 30))))


def test_ndvi_relation_texts():
    weak = relief_findings(metrics(relations=[relation(-0.24)]))[-1]
    none = relief_findings(metrics(relations=[relation(-0.05)]))[-1]
    insignificant = relief_findings(metrics(relations=[relation(-0.4, p_value=0.2)]))[-1]

    assert "слабкий зв'язок NDVI з висотою — вищі ділянки слабші" in weak.text
    assert none.text == "30.07.2026: NDVI з висотою не пов'язаний."
    assert insignificant.text == none.text


@pytest.mark.parametrize(
    ("rho", "p_value", "expected"),
    [
        (0.05, 0.001, None),
        (-0.2, 0.001, "слабкий"),
        (0.35, 0.001, "помірний"),
        (-0.7, 0.001, "сильний"),
        (0.7, 0.2, None),
    ],
)
def test_describe_strength(rho, p_value, expected):
    assert describe_strength(rho, p_value) == expected


@pytest.mark.parametrize(
    ("north", "east", "expected"),
    [(500, 0, "північ"), (300, 300, "північний схід"), (0, -500, "захід"), (-400, 10, "південь")],
)
def test_direction_uk(north, east, expected):
    assert direction_uk(north, east) == expected


def test_flat_field_has_no_aspect_and_no_slope_relation():
    flat = metrics(slope_shares=(0.884, 0.115, 0.001, 0.0, 0.0), aspects=(("sw", 0.37), ("w", 0.19)))
    flat = flat.model_copy(
        update={
            "slope_deg": Slope(mean=0.56, median=0.47, p90=1.06, max=3.6),
            "ndvi_relation": [
                relation(0.2, slope_rho=-0.35).model_copy(
                    update={"vs_slope": Correlation(rho=-0.35, p_value=1e-9)}
                )
            ],
        }
    )

    findings = by_code(relief_findings(flat))

    assert "aspect" not in findings
    assert "mostly_flat" not in findings
    assert "практично рівне" in findings["flat_field"].text
    assert "З ухилом" not in findings["ndvi_vs_relief"].text
    assert "вищі ділянки сильніші" in findings["ndvi_vs_relief"].text


def test_tiny_ndvi_difference_is_not_a_relation():
    dense = relation(-0.38).model_copy(
        update={"mean_by_elevation_third": {"low": 0.852, "middle": 0.832, "high": 0.826}}
    )

    text = relief_findings(metrics(relations=[dense]))[-1].text

    assert "зв'язок" not in text
    assert "0.026" in text
    assert "практично не вплинув" in text


def test_tiny_difference_between_slope_classes_is_not_a_relation():
    by_slope = [
        ClassNdvi(code="gentle", label=SLOPE_CLASSES[1][3], ndvi_mean=0.80),
        ClassNdvi(code="steep", label=SLOPE_CLASSES[4][3], ndvi_mean=0.79),
    ]
    tiny = relation(-0.24).model_copy(
        update={"vs_slope": Correlation(rho=-0.3, p_value=1e-9), "mean_by_slope_class": by_slope}
    )

    text = relief_findings(metrics(relations=[tiny]))[-1].text

    assert "З ухилом" not in text


def test_two_neighbouring_aspects_are_named_together():
    found = by_code(
        relief_findings(metrics(aspects=(("w", 0.279), ("sw", 0.278), ("s", 0.135), ("nw", 0.1))))
    )

    assert found["aspect"].text.startswith("Схили переважно звернені на захід і південний захід (56%).")


def test_scattered_aspects():
    found = by_code(relief_findings(metrics(aspects=(("w", 0.3), ("e", 0.25), ("n", 0.2), ("s", 0.15)))))

    assert found["aspect"].text == "Схили звернені в різні боки, найбільше — на захід (30%)."


def test_no_ndvi_relief_conclusions_on_a_field_with_several_crops():
    mixed = metrics(relations=[relation(-0.24)]).model_copy(update={"ndvi_mixed_crops": True})

    found = by_code(relief_findings(mixed))

    assert "ndvi_vs_relief" not in found
    assert "кілька культур" in found["ndvi_vs_relief_mixed_crops"].text
