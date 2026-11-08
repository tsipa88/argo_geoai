import logging
from datetime import date
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import Field
from rasterio.features import geometry_mask
from scipy.stats import spearmanr

from argo_geoai import render
from argo_geoai.schema import Artifact, Finding, Label, Message, Model
from argo_geoai.sources import dem, sentinel2
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

SLOPE_CLASSES: list[tuple[str, float, float, Label]] = [
    ("flat", 0, 1, Label(en="flat", uk="рівнина")),
    ("gentle", 1, 3, Label(en="gentle", uk="пологий")),
    ("moderate", 3, 5, Label(en="moderate", uk="помірний")),
    ("strong", 5, 7, Label(en="strong", uk="сильний")),
    ("steep", 7, 90, Label(en="steep", uk="крутий")),
]
EROSION_RISK_SLOPE_DEG = 5

ASPECTS: list[tuple[str, Label, str]] = [
    ("n", Label(en="N", uk="Пн"), "північ"),
    ("ne", Label(en="NE", uk="ПнСх"), "північний схід"),
    ("e", Label(en="E", uk="Сх"), "схід"),
    ("se", Label(en="SE", uk="ПдСх"), "південний схід"),
    ("s", Label(en="S", uk="Пд"), "південь"),
    ("sw", Label(en="SW", uk="ПдЗх"), "південний захід"),
    ("w", Label(en="W", uk="Зх"), "захід"),
    ("nw", Label(en="NW", uk="ПнЗх"), "північний захід"),
]
COLD_ASPECTS = {"n", "ne", "nw"}
WARM_ASPECTS = {"s", "se", "sw"}
MIN_SLOPE_FOR_ASPECT_DEG = 1
MOSTLY_ASPECT_SHARE = 0.4


class ReliefParams(Model):
    ndvi_dates: list[date] = Field(
        default_factory=list,
        description=(
            "dates to compare NDVI with relief; by default peak and max-spread of the latest vegetation run"
        ),
    )
    buffer_m: float = Field(
        120, ge=0, le=1000, description="margin read around the parcel so edge slopes are whole"
    )


class Elevation(Model):
    min: float
    max: float
    mean: float
    p5: float
    p95: float
    range: float
    range_p5_p95: float


class Slope(Model):
    mean: float
    median: float
    p90: float
    max: float


class ClassShare(Model):
    code: str
    label: Label
    share: float


class SlopeClassShare(ClassShare):
    min_deg: float
    max_deg: float


class ClassNdvi(Model):
    code: str
    label: Label
    ndvi_mean: float


class Offset(Model):
    north_m: float
    east_m: float


class NdviRelation(Model):
    date: date
    reason: NdviDateReason
    scene_id: str
    pixels: int
    vs_elevation: Correlation
    vs_slope: Correlation
    mean_by_elevation_third: dict[Literal["low", "middle", "high"], float]
    mean_by_slope_class: list[ClassNdvi]


class ReliefMetrics(Model):
    pixels: int
    area_ha: float
    elevation_m: Elevation
    slope_deg: Slope
    slope_classes: list[SlopeClassShare]
    aspect_of_slopes: list[ClassShare] = Field(description=f"slopes of {MIN_SLOPE_FOR_ASPECT_DEG}° and more")
    low_third_offset_from_high_third: Offset
    ndvi_relation: list[NdviRelation]
    ndvi_mixed_crops: bool = Field(
        False,
        description="vegetation found zones of different crops: NDVI vs relief would compare crops",
    )


def direction_uk(north_m: float, east_m: float) -> str:
    angle = (np.degrees(np.arctan2(east_m, north_m)) + 360) % 360
    return ASPECTS[int((angle + 22.5) // 45) % 8][2]


def relief_findings(metrics: ReliefMetrics) -> list[Finding]:
    e, s = metrics.elevation_m, metrics.slope_deg
    findings = [
        Finding(
            code="elevation_range",
            level="info",
            text=f"Перепад висот на полі {e.range:.0f} м ({e.min:.0f}–{e.max:.0f} м над рівнем моря).",
        )
    ]

    risky = sum(c.share for c in metrics.slope_classes if c.min_deg >= EROSION_RISK_SLOPE_DEG)
    if risky >= 0.3:
        findings.append(
            Finding(
                code="erosion_risk",
                level="warning",
                text=(
                    f"{risky:.0%} поля на ухилах {EROSION_RISK_SLOPE_DEG}° і більше — "
                    "ерозійно небезпечні землі: "
                    "доречний обробіток поперек схилу й обережність із просапними культурами."
                ),
            )
        )
    elif risky >= 0.1:
        findings.append(
            Finding(
                code="erosion_risk",
                level="notice",
                text=(
                    f"{risky:.0%} поля на ухилах {EROSION_RISK_SLOPE_DEG}° і більше — "
                    "є ерозійно небезпечні ділянки."
                ),
            )
        )
    elif s.p90 >= FLAT_FIELD_P90_DEG:
        findings.append(
            Finding(
                code="mostly_flat",
                level="info",
                text=f"Поле переважно рівне або пологе: середній ухил {s.mean:.1f}°.",
            )
        )
    findings.append(
        Finding(
            code="slope",
            level="info",
            text=f"Середній ухил {s.mean:.1f}°, у 10% поля крутіше {s.p90:.1f}°, максимум {s.max:.1f}°.",
        )
    )

    flat = s.p90 < FLAT_FIELD_P90_DEG
    if flat:
        findings.append(
            Finding(
                code="flat_field",
                level="info",
                text=(
                    f"Поле практично рівне: у 90% площі ухил менший за {s.p90:.1f}°, "
                    "тож експозиція схилів і зв'язок з ухилом тут значення не мають."
                ),
            )
        )
    elif metrics.aspect_of_slopes:
        cold = sum(a.share for a in metrics.aspect_of_slopes if a.code in COLD_ASPECTS)
        warm = sum(a.share for a in metrics.aspect_of_slopes if a.code in WARM_ASPECTS)
        note = ""
        if cold >= 0.6:
            note = " Такі схили прогріваються повільніше і довше тримають вологу."
        elif warm >= 0.6:
            note = " Такі схили краще прогріваються і швидше пересихають."
        findings.append(
            Finding(code="aspect", level="info", text=aspect_text(metrics.aspect_of_slopes) + note)
        )

    offset = metrics.low_third_offset_from_high_third
    if np.hypot(offset.north_m, offset.east_m) >= 100:
        findings.append(
            Finding(
                code="low_part",
                level="info",
                text=(
                    "Нижня частина поля розташована на "
                    f"{direction_uk(offset.north_m, offset.east_m)} від верхньої."
                ),
            )
        )

    if metrics.ndvi_mixed_crops and metrics.ndvi_relation:
        findings.append(
            Finding(
                code="ndvi_vs_relief_mixed_crops",
                level="info",
                text=(
                    "На полі кілька культур (див. зони у vegetation), тому зв'язок NDVI з рельєфом "
                    "порівнював би культури, а не вплив рельєфу — висновків про нього не робимо."
                ),
            )
        )
    else:
        findings += [
            Finding(code="ndvi_vs_relief", level="info", text=ndvi_relation_text(relation, flat=flat))
            for relation in metrics.ndvi_relation
        ]
    return findings


def ndvi_relation_text(relation: NdviRelation, *, flat: bool) -> str:
    thirds = relation.mean_by_elevation_third
    strength = describe_strength(relation.vs_elevation.rho, relation.vs_elevation.p_value)
    day = relation.date.strftime("%d.%m.%Y")
    difference = abs(thirds["low"] - thirds["high"])
    if strength is None:
        text = f"{day}: NDVI з висотою не пов'язаний."
    elif difference < MIN_NDVI_DIFFERENCE:
        text = (
            f"{day}: різниця NDVI між нижньою і верхньою третиною поля лише {difference:.3f} — "
            "рельєф на рослинність практично не вплинув."
        )
    else:
        side = "вищі ділянки слабші" if relation.vs_elevation.rho < 0 else "вищі ділянки сильніші"
        text = (
            f"{day}: {strength} зв'язок NDVI з висотою — {side} "
            f"(нижня третина {thirds['low']:.2f}, верхня {thirds['high']:.2f})."
        )
    slope_strength = describe_strength(relation.vs_slope.rho, relation.vs_slope.p_value)
    slope_means = [c.ndvi_mean for c in relation.mean_by_slope_class]
    slope_difference = max(slope_means) - min(slope_means) if len(slope_means) > 1 else 0.0
    if slope_strength is not None and not flat and slope_difference >= MIN_NDVI_DIFFERENCE:
        side = "крутіші ділянки слабші" if relation.vs_slope.rho < 0 else "крутіші ділянки сильніші"
        text += f" З ухилом — {slope_strength} зв'язок: {side}."
    return text


def aspect_text(aspects: list[ClassShare]) -> str:
    by_code = {a.code: a.share for a in aspects}
    codes = [code for code, _, _ in ASPECTS]
    full = {code: name for code, _, name in ASPECTS}
    main = aspects[0]
    if main.share >= MOSTLY_ASPECT_SHARE:
        return f"Схили переважно звернені на {full[main.code]} ({main.share:.0%})."

    index = codes.index(main.code)
    neighbour = max((codes[(index - 1) % 8], codes[(index + 1) % 8]), key=lambda code: by_code.get(code, 0))
    together = main.share + by_code.get(neighbour, 0)
    if together >= MOSTLY_ASPECT_SHARE:
        return f"Схили переважно звернені на {full[main.code]} і {full[neighbour]} ({together:.0%})."
    return f"Схили звернені в різні боки, найбільше — на {full[main.code]} ({main.share:.0%})."


def share(mask: BoolArray, inside: BoolArray) -> float:
    return round(float(mask[inside].sum() / inside.sum()), 3)


class ReliefTool(Tool[ReliefParams, ReliefMetrics]):
    name = "relief"
    version = "1.2.0"
    title = Label(en="Relief (elevation, slope, aspect)", uk="Рельєф (висота, ухил, експозиція)")
    params_model = ReliefParams
    metrics_model = ReliefMetrics

    def run(self, params: ReliefParams, context: RunContext) -> ToolOutput[ReliefMetrics]:
        parcel = context.parcel
        grid, dem_items = dem.elevation(parcel, buffer_m=params.buffer_m)
        heights = grid.values.astype("float64")
        slope, aspect = dem.slope_aspect(heights, dem.RESOLUTION_M)
        inside: BoolArray = ~geometry_mask(
            parcel.in_crs(grid.rio.crs).geometry, out_shape=grid.shape, transform=grid.rio.transform()
        ) & np.isfinite(heights)

        z, s = heights[inside], slope[inside]
        p5, p95 = np.percentile(z, [5, 95])
        sloped = inside & (slope >= MIN_SLOPE_FOR_ASPECT_DEG)
        sectors = (np.nan_to_num((aspect + 22.5) // 45) % 8).astype(int)
        aspects = (
            [
                ClassShare(code=code, label=label, share=share(sectors == k, sloped))
                for k, (code, label, _) in enumerate(ASPECTS)
            ]
            if sloped.any()
            else []
        )

        rows, cols = np.nonzero(inside)
        ys, xs = grid.y.values[rows], grid.x.values[cols]
        low, high = z <= np.percentile(z, 33), z >= np.percentile(z, 67)

        warnings: list[Message] = []
        vegetation_metrics = vegetation_metrics_for(context, warnings)
        mixed_crops = has_mixed_crops(vegetation_metrics)
        relations, ndvi_maps = self._ndvi_relations(
            vegetation_metrics=vegetation_metrics,
            params=params,
            context=context,
            grid=grid,
            heights=heights,
            slope=slope,
            inside=inside,
            warnings=warnings,
        )

        metrics = ReliefMetrics(
            pixels=int(inside.sum()),
            area_ha=round(inside.sum() * dem.RESOLUTION_M**2 / 10_000, 2),
            elevation_m=Elevation(
                min=round(float(z.min()), 1),
                max=round(float(z.max()), 1),
                mean=round(float(z.mean()), 1),
                p5=round(float(p5), 1),
                p95=round(float(p95), 1),
                range=round(float(z.max() - z.min()), 1),
                range_p5_p95=round(float(p95 - p5), 1),
            ),
            slope_deg=Slope(
                mean=round(float(s.mean()), 2),
                median=round(float(np.median(s)), 2),
                p90=round(float(np.percentile(s, 90)), 2),
                max=round(float(s.max()), 2),
            ),
            slope_classes=[
                SlopeClassShare(
                    code=code,
                    label=label,
                    min_deg=lo,
                    max_deg=hi,
                    share=share((slope >= lo) & (slope < hi), inside),
                )
                for code, lo, hi, label in SLOPE_CLASSES
            ],
            aspect_of_slopes=sorted(aspects, key=lambda a: -a.share),
            low_third_offset_from_high_third=Offset(
                north_m=round(float(ys[low].mean() - ys[high].mean())),
                east_m=round(float(xs[low].mean() - xs[high].mean())),
            ),
            ndvi_relation=relations,
            ndvi_mixed_crops=mixed_crops,
        )

        path = context.workdir / "relief.png"
        render.relief_panels(
            path,
            f"{parcel.name}: рельєф (Copernicus DEM 30 м) і NDVI",
            elevation=np.where(inside, heights, np.nan),
            slope=np.where(inside, slope, np.nan),
            ndvi_maps=ndvi_maps,
        )
        sources = [dem.DATA_SOURCE.model_copy(update={"items": dem_items})]
        if relations:
            sources.append(
                sentinel2.DATA_SOURCE.model_copy(update={"items": [r.scene_id for r in relations]})
            )
        return ToolOutput(
            metrics=metrics,
            summary=relief_findings(metrics),
            artifacts=[
                Artifact(
                    path=path.name,
                    media_type="image/png",
                    description=(
                        "Висота з горизонталями (крок 2 м), ухил і NDVI на тій самій 30-метровій сітці."
                    ),
                )
            ],
            data_sources=sources,
            warnings=warnings,
            partial=any(w.code == "ndvi_scene_missing" for w in warnings),
        )

    def _ndvi_relations(
        self,
        *,
        vegetation_metrics: dict[str, Any],
        params: ReliefParams,
        context: RunContext,
        grid: xr.DataArray,
        heights: FloatArray,
        slope: FloatArray,
        inside: BoolArray,
        warnings: list[Message],
    ) -> tuple[list[NdviRelation], dict[str, FloatArray]]:
        relations: list[NdviRelation] = []
        maps: dict[str, FloatArray] = {}
        for day, reason in ndvi_dates(params.ndvi_dates, vegetation_metrics, warnings):
            found = ndvi_on_grid(day, context.parcel, grid, buffer_m=params.buffer_m, warnings=warnings)
            if found is None:
                continue
            ndvi, scene_id = found
            ok = inside & np.isfinite(ndvi)
            n, e, sl = ndvi[ok], heights[ok], slope[ok]
            rho_e, p_e = spearmanr(e, n)
            rho_s, p_s = spearmanr(sl, n)
            low, high = np.percentile(e, [33, 67])
            relations.append(
                NdviRelation(
                    date=day,
                    reason=reason,
                    scene_id=scene_id,
                    pixels=int(ok.sum()),
                    vs_elevation=Correlation(rho=round(float(rho_e), 3), p_value=float(f"{p_e:.2g}")),
                    vs_slope=Correlation(rho=round(float(rho_s), 3), p_value=float(f"{p_s:.2g}")),
                    mean_by_elevation_third={
                        "low": round(float(n[e <= low].mean()), 3),
                        "middle": round(float(n[(e > low) & (e < high)].mean()), 3),
                        "high": round(float(n[e >= high].mean()), 3),
                    },
                    mean_by_slope_class=[
                        ClassNdvi(
                            code=code,
                            label=label,
                            ndvi_mean=round(float(n[(sl >= lo) & (sl < hi)].mean()), 3),
                        )
                        for code, lo, hi, label in SLOPE_CLASSES
                        if ((sl >= lo) & (sl < hi)).sum() >= 10
                    ],
                )
            )
            maps[day.strftime("%d.%m.%Y")] = np.where(inside, ndvi, np.nan)
        return relations, maps
