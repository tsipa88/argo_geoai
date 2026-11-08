from datetime import date

import numpy as np
import pytest

from argo_geoai.schema import Label
from argo_geoai.tools.vegetation import (
    Gap,
    LandCoverShare,
    NdviStats,
    SeasonPoint,
    VegetationMetrics,
    season_findings,
    stats,
)


def point(day: date, mean: float, spread: float = 0.1) -> SeasonPoint:
    return SeasonPoint(
        date=day,
        ndvi=NdviStats(mean=mean, median=mean, p10=mean - spread / 2, p90=mean + spread / 2, spread=spread),
    )


def metrics(**overrides: object) -> VegetationMetrics:
    """The 2026 season of tests/data/field_slope.geojson: one late crop, moderate peak."""
    peak = point(date(2026, 7, 30), 0.672, 0.231)
    values: dict[str, object] = {
        "period_start": date(2026, 3, 30),
        "period_end": date(2026, 10, 2),
        "scenes_found": 36,
        "dates_total": 30,
        "dates_counted": 24,
        "peak": peak,
        "max_spread": point(date(2026, 8, 9), 0.624, 0.245),
        "latest": point(date(2026, 10, 1), 0.428, 0.195),
        "change_peak_to_latest": -0.244,
        "longest_gap": Gap(start=date(2026, 4, 6), end=date(2026, 5, 4), days=28),
        "history": [],
        "land_cover": [LandCoverShare(code="cropland", label=Label(en="Cropland", uk="Рілля"), share=0.993)],
    }
    values.update(overrides)
    return VegetationMetrics.model_validate(values)


def by_code(findings):
    return {f.code: f for f in findings}


def test_full_season_field_slope():
    findings = by_code(season_findings(metrics()))

    assert list(findings) == [
        "land_cover_main",
        "clear_dates",
        "peak",
        "uneven_at_peak",
        "latest",
        "data_gap",
    ]
    assert findings["land_cover_main"].text == "Основний тип покриву ділянки — рілля (99%)."
    assert "30.07.2026" in findings["peak"].text
    assert "помірно" in findings["peak"].text
    assert "згасає" in findings["latest"].text
    assert "28 днів" in findings["data_gap"].text


@pytest.mark.parametrize(
    ("peak_mean", "level", "word"),
    [(0.8, "info", "густа"), (0.6, "notice", "помірно"), (0.3, "warning", "слабкий")],
)
def test_peak_verdict(peak_mean, level, word):
    finding = by_code(season_findings(metrics(peak=point(date(2026, 7, 30), peak_mean))))["peak"]

    assert finding.level == level
    assert word in finding.text


def test_even_field_and_short_gaps_produce_no_notices():
    findings = by_code(
        season_findings(
            metrics(
                peak=point(date(2026, 6, 30), 0.8, 0.05),
                max_spread=point(date(2026, 7, 1), 0.7, 0.1),
                longest_gap=Gap(start=date(2026, 6, 1), end=date(2026, 6, 10), days=9),
            )
        )
    )

    assert {"uneven_at_peak", "max_spread", "data_gap"}.isdisjoint(findings)


def test_no_clear_dates_is_a_warning():
    findings = season_findings(metrics(peak=None, max_spread=None, latest=None, dates_counted=0))

    assert [f.code for f in findings] == ["land_cover_main", "no_clear_dates"]
    assert findings[-1].level == "warning"


def test_stats():
    result = stats(np.linspace(0, 1, 101))

    assert (result.mean, result.median, result.p10, result.p90, result.spread) == (0.5, 0.5, 0.1, 0.9, 0.8)
