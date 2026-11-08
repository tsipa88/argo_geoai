from collections.abc import Sequence
from datetime import date
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

NDVI_RANGE = (0.0, 0.9)
DPI = 110


def ndvi_map(path: Path, title: str, ndvi: xr.DataArray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    image = ax.imshow(ndvi.values, cmap="RdYlGn", vmin=NDVI_RANGE[0], vmax=NDVI_RANGE[1])
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, label="NDVI", shrink=0.8)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def ndvi_timeseries(
    path: Path,
    title: str,
    *,
    days: Sequence[date],
    mean: Sequence[float],
    p10: Sequence[float],
    p90: Sequence[float],
) -> None:
    x = np.array(days, dtype="datetime64[D]")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(x, p10, p90, color="#7fbf7b", alpha=0.35, label="10–90% поля")
    ax.plot(x, mean, marker="o", color="#1b7837", label="середнє NDVI")
    ax.set_ylim(*NDVI_RANGE)
    ax.set_ylabel("NDVI")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def relief_panels(
    path: Path,
    title: str,
    *,
    elevation: npt.NDArray[np.float64],
    slope: npt.NDArray[np.float64],
    ndvi_maps: dict[str, npt.NDArray[np.float64]],
    contour_step_m: float = 2,
) -> None:
    panels: list[tuple[str, npt.NDArray[np.float64], str, float | None, float | None]] = [
        ("Висота, м", elevation, "terrain", None, None),
        ("Ухил, °", slope, "magma_r", 0, max(3.0, float(np.nanmax(slope)))),
    ]
    panels += [(f"NDVI {label}", values, "RdYlGn", *NDVI_RANGE) for label, values in ndvi_maps.items()]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.2))
    for ax, (panel_title, values, cmap, vmin, vmax) in zip(np.atleast_1d(axes), panels, strict=True):
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        if values is elevation and np.isfinite(values).any():
            levels = np.arange(
                np.floor(np.nanmin(values)), np.ceil(np.nanmax(values)) + contour_step_m, contour_step_m
            )
            ax.contour(values, levels=levels, colors="k", linewidths=0.4)
        ax.set_title(panel_title)
        ax.axis("off")
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle(title, y=0.98)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


ZONE_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]


def zone_map(path: Path, title: str, zones: npt.NDArray[np.float64], labels: Sequence[str]) -> None:
    colors = ZONE_COLORS[: len(labels)]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(zones, cmap=ListedColormap(colors), vmin=0.5, vmax=len(labels) + 0.5, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    ax.legend(
        handles=[Patch(color=c, label=label) for c, label in zip(colors, labels, strict=True)],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def zone_timeseries(
    path: Path,
    title: str,
    *,
    curves: Sequence[tuple[Sequence[date], Sequence[float]]],
    labels: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for (days, values), label, color in zip(curves, labels, ZONE_COLORS, strict=False):
        ax.plot(np.array(days, dtype="datetime64[D]"), values, marker="o", color=color, label=label)
    ax.set_ylim(*NDVI_RANGE)
    ax.set_ylabel("NDVI")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def water_panels(
    path: Path,
    title: str,
    *,
    twi: npt.NDArray[np.float64],
    flow_lines: npt.NDArray[np.bool_],
    ndvi_maps: dict[str, npt.NDArray[np.float64]],
) -> None:
    rows, cols = np.nonzero(np.isfinite(twi))
    crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    panels: list[tuple[str, npt.NDArray[np.float64], str, float | None, float | None]] = [
        ("Індекс вологості (TWI)", twi[crop], "YlGnBu", 5.0, 15.0)
    ]
    panels += [(f"NDVI {label}", values[crop], "RdYlGn", *NDVI_RANGE) for label, values in ndvi_maps.items()]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.2))
    for ax, (panel_title, values, cmap, vmin, vmax) in zip(np.atleast_1d(axes), panels, strict=True):
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        if panel_title.startswith("Індекс") and flow_lines[crop].any():
            ax.contour(flow_lines[crop].astype(float), levels=[0.5], colors="#d7301f", linewidths=0.8)
        ax.set_title(panel_title)
        ax.axis("off")
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle(title + " (червоним — лінії стоку з водозбором понад 10 га)", y=0.98)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
