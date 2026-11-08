import logging
from dataclasses import asdict
from datetime import date
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import Field
from rasterio.features import geometry_mask
from scipy.stats import spearmanr

from argo_geoai import render
from argo_geoai.analysis import hydrology
from argo_geoai.schema import Artifact, Finding, Label, Message, Model
from argo_geoai.sources import dem, sentinel2, surface_water
from argo_geoai.tools.base import RunContext, Tool, ToolOutput
from argo_geoai.tools.common import (
    FLAT_FIELD_P90_DEG,
    MIN_NDVI_DIFFERENCE,
    Correlation,
    NdviDateReason,
    describe_strength,
    has_mixed_crops,
    ndvi_dates,
    ndvi_on_grid,
    vegetation_metrics_for,
)

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

WETNESS_CLASSES: list[tuple[str, float, float, Label]] = [
    ("dry", -np.inf, 8, Label(en="dry", uk="сухі (вершини, схили)")),
    ("normal", 8, 11, Label(en="normal", uk="нормальні")),
    ("moist", 11, 14, Label(en="moist", uk="зволожені")),
    ("wet", 14, np.inf, Label(en="wet", uk="перезволожені")),
]
WET_TWI = 11
WET_SHARE_WARNING = 0.3
WET_SHARE_NOTICE = 0.1
NEGLIGIBLE_SHARE = 0.01
FLOW_LINE_HA = 10
FLOW_LINE_SHARE_NOTICE = 0.01
NDVI_BUFFER_M = 120
TRUNCATED_EDGE_SHARE = 0.1
TRUNCATED_CELL_SHARE = 0.05
WET_PART_MIN_OFFSET_M = 100
LOCATIVE = [
    "півночі",
    "північному сході",
    "сході",
    "південному сході",
    "півдні",
    "південному заході",
    "заході",
    "північному заході",
]


def locative_direction(north_m: float, east_m: float) -> str:
    angle = (np.degrees(np.arctan2(east_m, north_m)) + 360) % 360
    return LOCATIVE[int((angle + 22.5) // 45) % 8]


class WaterParams(Model):
    ndvi_dates: list[date] = Field(
        default_factory=list,
        description=(
            "dates to compare NDVI with wetness; by default peak and max-spread of the latest vegetation run"
        ),
    )
    catchment_buffer_m: float = Field(
        1500, ge=300, le=5000, description="relief read around the parcel to route water coming from upslope"
    )
    nearby_water_m: float = Field(
        200, ge=0, le=2000, description="ring around the parcel checked for open water"
    )


class Wetness(Model):
    mean: float
    median: float
    p10: float
    p90: float
    max: float


class WetnessClassShare(Model):
    code: str
    label: Label
    min_twi: float | None
    max_twi: float | None
    share: float


class Offset(Model):
    north_m: float
    east_m: float


class OpenWater(Model):
    years: str = surface_water.YEARS
    share_ever: float = Field(description="share of the area where open water was seen at least once")
    share_frequent: float = Field(
        description=(
            f"share where water was seen in >= {surface_water.FREQUENT_OCCURRENCE_PCT} % "
            "of the observed months"
        )
    )
    max_occurrence_pct: int
    max_months_per_year: int


class NdviWetness(Model):
    date: date
    reason: NdviDateReason
    scene_id: str
    pixels: int
    vs_twi: Correlation
    mean_by_wetness_third: dict[str, float] = Field(
        description="NDVI mean in the driest, middle and wettest third"
    )


class WaterMetrics(Model):
    pixels: int
    area_ha: float
    flat: bool = Field(
        description=f"90 % of the parcel below {FLAT_FIELD_P90_DEG} degrees: wetness is indicative"
    )
    twi: Wetness
    wetness_classes: list[WetnessClassShare]
    wet_share: float = Field(description=f"share with TWI >= {WET_TWI}")
    wet_part_offset: Offset | None = Field(description="from the parcel centre to the wet cells' centre")
    max_upslope_ha: float = Field(description="largest area draining through a point of the parcel")
    flow_lines_share: float = Field(description=f"share draining >= {FLOW_LINE_HA} ha (concentrated runoff)")
    catchment_truncated_share: float
    open_water_on_parcel: OpenWater
    open_water_nearby: OpenWater
    ndvi_relation: list[NdviWetness]
    ndvi_mixed_crops: bool


def wet_share_finding(wet: float) -> Finding:
    level: Literal["info", "notice", "warning"]
    if wet >= WET_SHARE_WARNING:
        level = "warning"
        text = f"{wet:.0%} поля — місця, де збирається вода: ризик застою вологи після злив і навесні."
    elif wet >= WET_SHARE_NOTICE:
        level, text = "notice", f"{wet:.0%} поля — місця, де збирається вода (пониззя, улоговини)."
    elif wet >= NEGLIGIBLE_SHARE:
        level, text = "info", f"Місць, де збирається вода, мало: {wet:.0%} поля."
    else:
        level, text = "info", "Місць, де збирається вода, на полі практично немає."
    return Finding(code="wet_share", level=level, text=text)


def open_water_finding(water: OpenWater, nearby: OpenWater) -> Finding:
    if water.share_ever > 0:
        return Finding(
            code="open_water",
            level="warning" if water.share_frequent > 0 else "notice",
            text=(
                f"За {water.years} роки відкрита вода на полі фіксувалась на {water.share_ever:.0%} площі "
                f"(у найвологішому місці — у {water.max_occurrence_pct}% місяців спостережень)."
            ),
        )
    text = f"За {water.years} роки відкритої води на полі не фіксувалось (супутниковий архів JRC)."
    if nearby.share_ever > 0:
        text += " Поруч є місця, де вода стояла, — водойма або заплава."
    return Finding(code="open_water", level="info", text=text)


def wetness_findings(metrics: WaterMetrics) -> list[Finding]:
    findings: list[Finding] = []
    if metrics.flat:
        findings.append(
            Finding(
                code="flat_field",
                level="notice",
                text=(
                    "Поле рівне, а рельєф 30 м не бачить мікропониззя: зони вологості на ньому орієнтовні — "
                    "реальний застій води залежить від дрібних западин, тому зв'язок NDVI з ними не оцінюємо."
                ),
            )
        )

    wet = metrics.wet_share
    findings.append(wet_share_finding(wet))

    offset = metrics.wet_part_offset
    concentrated = offset is not None and np.hypot(offset.north_m, offset.east_m) >= WET_PART_MIN_OFFSET_M
    if offset is not None and concentrated and WET_SHARE_NOTICE / 2 <= wet <= 0.6:
        findings.append(
            Finding(
                code="wet_part",
                level="info",
                text=(
                    "Вологіші ділянки зосереджені переважно на "
                    f"{locative_direction(offset.north_m, offset.east_m)} поля."
                ),
            )
        )

    if metrics.max_upslope_ha > metrics.area_ha:
        findings.append(
            Finding(
                code="inflow",
                level="notice",
                text=(
                    f"Через поле проходить стік з площі до {metrics.max_upslope_ha:.0f} га — "
                    "більшої за саме поле "
                    f"({metrics.area_ha:.0f} га): вода надходить з-за його меж."
                ),
            )
        )
    if metrics.flow_lines_share >= FLOW_LINE_SHARE_NOTICE and not metrics.flat:
        findings.append(
            Finding(
                code="flow_lines",
                level="notice",
                text=(
                    f"{metrics.flow_lines_share:.0%} поля лежить на лініях концентрованого стоку "
                    f"(водозбір понад {FLOW_LINE_HA} га): там можливі розмиви і промоїни."
                ),
            )
        )

    findings.append(open_water_finding(metrics.open_water_on_parcel, metrics.open_water_nearby))

    if metrics.flat:
        pass
    elif metrics.ndvi_mixed_crops and metrics.ndvi_relation:
        findings.append(
            Finding(
                code="ndvi_vs_wetness_mixed_crops",
                level="info",
                text=(
                    "На полі кілька культур (див. зони у vegetation), тому зв'язок NDVI з вологістю "
                    "порівнював би культури — висновків про нього не робимо."
                ),
            )
        )
    else:
        findings += [
            Finding(code="ndvi_vs_wetness", level="info", text=ndvi_wetness_text(relation))
            for relation in metrics.ndvi_relation
        ]
    return findings


def ndvi_wetness_text(relation: NdviWetness) -> str:
    thirds = relation.mean_by_wetness_third
    day = relation.date.strftime("%d.%m.%Y")
    strength = describe_strength(relation.vs_twi.rho, relation.vs_twi.p_value)
    difference = abs(thirds["wet"] - thirds["dry"])
    if strength is None:
        return f"{day}: NDVI з вологістю рельєфу не пов'язаний."
    if difference < MIN_NDVI_DIFFERENCE:
        return (
            f"{day}: різниця NDVI між найвологішою і найсухішою третиною поля лише {difference:.3f} — "
            "вологість рельєфу на рослинність практично не вплинула."
        )
    side = "вологіші ділянки слабші" if relation.vs_twi.rho < 0 else "вологіші ділянки сильніші"
    return (
        f"{day}: {strength} зв'язок NDVI з вологістю — {side} "
        f"(найвологіша третина {thirds['wet']:.2f}, найсухіша {thirds['dry']:.2f})."
    )


class WaterTool(Tool[WaterParams, WaterMetrics]):
    name = "water"
    version = "1.0.0"
    title = Label(en="Water: wetness, runoff, open water", uk="Вода: вологість, стік, відкрита вода")
    params_model = WaterParams
    metrics_model = WaterMetrics

    def run(self, params: WaterParams, context: RunContext) -> ToolOutput[WaterMetrics]:
        parcel = context.parcel
        grid, dem_items = dem.elevation(parcel, buffer_m=params.catchment_buffer_m)
        heights = grid.values.astype("float64")
        slope, _ = dem.slope_aspect(heights, dem.RESOLUTION_M)
        area, edge_share = hydrology.flow_accumulation(hydrology.fill_depressions(heights), dem.RESOLUTION_M)
        twi = hydrology.wetness_index(area, slope, dem.RESOLUTION_M)
        inside: BoolArray = ~geometry_mask(
            parcel.in_crs(grid.rio.crs).geometry, out_shape=grid.shape, transform=grid.rio.transform()
        ) & np.isfinite(twi)

        warnings: list[Message] = []
        values, upslope_ha = twi[inside], area[inside] / 10_000
        truncated = float((edge_share[inside] >= TRUNCATED_EDGE_SHARE).mean())
        if truncated > TRUNCATED_CELL_SHARE:
            warnings.append(
                Message(
                    code="catchment_truncated",
                    text=(
                        f"{truncated:.0%} of the parcel gets water from beyond the "
                        f"{params.catchment_buffer_m:.0f} m "
                        "window: wetness is underestimated there, increase catchment_buffer_m"
                    ),
                )
            )

        on_parcel, nearby, water_items = surface_water.surface_water(parcel, nearby_m=params.nearby_water_m)
        vegetation_metrics = vegetation_metrics_for(context, warnings)
        relations, ndvi_maps = self._ndvi_relations(
            params=params,
            context=context,
            grid=grid,
            twi=twi,
            inside=inside,
            vegetation_metrics=vegetation_metrics,
            warnings=warnings,
        )

        metrics = WaterMetrics(
            pixels=int(inside.sum()),
            area_ha=round(inside.sum() * dem.RESOLUTION_M**2 / 10_000, 2),
            flat=bool(np.percentile(slope[inside], 90) < FLAT_FIELD_P90_DEG),
            twi=Wetness(
                mean=round(float(values.mean()), 2),
                median=round(float(np.median(values)), 2),
                p10=round(float(np.percentile(values, 10)), 2),
                p90=round(float(np.percentile(values, 90)), 2),
                max=round(float(values.max()), 2),
            ),
            wetness_classes=[
                WetnessClassShare(
                    code=code,
                    label=label,
                    min_twi=None if np.isinf(lo) else lo,
                    max_twi=None if np.isinf(hi) else hi,
                    share=round(float(((values >= lo) & (values < hi)).mean()), 3),
                )
                for code, lo, hi, label in WETNESS_CLASSES
            ],
            wet_share=round(float((values >= WET_TWI).mean()), 3),
            wet_part_offset=self._wet_offset(grid, twi, inside),
            max_upslope_ha=round(float(upslope_ha.max()), 1),
            flow_lines_share=round(float((upslope_ha >= FLOW_LINE_HA).mean()), 3),
            catchment_truncated_share=round(truncated, 3),
            open_water_on_parcel=OpenWater(**asdict(on_parcel)),
            open_water_nearby=OpenWater(**asdict(nearby)),
            ndvi_relation=relations,
            ndvi_mixed_crops=has_mixed_crops(vegetation_metrics),
        )

        path = context.workdir / "water.png"
        render.water_panels(
            path,
            f"{parcel.name}: вологість рельєфу (TWI) і стік",
            twi=np.where(inside, twi, np.nan),
            flow_lines=inside & (area / 10_000 >= FLOW_LINE_HA),
            ndvi_maps=ndvi_maps,
        )
        sources = [
            dem.DATA_SOURCE.model_copy(update={"items": dem_items}),
            surface_water.DATA_SOURCE.model_copy(update={"items": water_items}),
        ]
        if relations:
            sources.append(
                sentinel2.DATA_SOURCE.model_copy(update={"items": [r.scene_id for r in relations]})
            )
        return ToolOutput(
            metrics=metrics,
            summary=wetness_findings(metrics),
            artifacts=[
                Artifact(
                    path=path.name,
                    media_type="image/png",
                    description=(
                        "Індекс вологості рельєфу (TWI) з лініями концентрованого стоку "
                        "і NDVI на тих самих датах."
                    ),
                )
            ],
            data_sources=sources,
            warnings=warnings,
            partial=any(w.code == "ndvi_scene_missing" for w in warnings),
        )

    def _wet_offset(self, grid: xr.DataArray, twi: FloatArray, inside: BoolArray) -> Offset | None:
        wet = inside & (twi >= WET_TWI)
        if not wet.any():
            return None
        ys, xs = np.nonzero(inside)
        wys, wxs = np.nonzero(wet)
        y, x = grid.y.values, grid.x.values
        return Offset(
            north_m=round(float(y[wys].mean() - y[ys].mean())),
            east_m=round(float(x[wxs].mean() - x[xs].mean())),
        )

    def _ndvi_relations(
        self,
        *,
        params: WaterParams,
        context: RunContext,
        grid: xr.DataArray,
        twi: FloatArray,
        inside: BoolArray,
        vegetation_metrics: dict[str, Any],
        warnings: list[Message],
    ) -> tuple[list[NdviWetness], dict[str, FloatArray]]:
        relations: list[NdviWetness] = []
        maps: dict[str, FloatArray] = {}
        for day, reason in ndvi_dates(params.ndvi_dates, vegetation_metrics, warnings):
            found = ndvi_on_grid(day, context.parcel, grid, buffer_m=NDVI_BUFFER_M, warnings=warnings)
            if found is None:
                continue
            ndvi, scene_id = found
            ok = inside & np.isfinite(ndvi)
            n, w = ndvi[ok], twi[ok]
            rho, p_value = spearmanr(w, n)
            dry, wet = np.percentile(w, [33, 67])
            relations.append(
                NdviWetness(
                    date=day,
                    reason=reason,
                    scene_id=scene_id,
                    pixels=int(ok.sum()),
                    vs_twi=Correlation(rho=round(float(rho), 3), p_value=float(f"{p_value:.2g}")),
                    mean_by_wetness_third={
                        "dry": round(float(n[w <= dry].mean()), 3),
                        "middle": round(float(n[(w > dry) & (w < wet)].mean()), 3),
                        "wet": round(float(n[w >= wet].mean()), 3),
                    },
                )
            )
            maps[day.strftime("%d.%m.%Y")] = np.where(inside, ndvi, np.nan)
        return relations, maps
