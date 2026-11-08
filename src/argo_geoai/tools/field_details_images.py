import io
from collections.abc import Sequence
from datetime import date

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

FloatArray = npt.NDArray[np.float64]
Rgb = npt.NDArray[np.uint8]
Outline = list[tuple[FloatArray, FloatArray]]

DPI = 120
OUTLINE_COLOR = "white"
OUTLINE_WIDTH = 1.4


def _png(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _draw_outline(
    ax: Axes, outline: Outline, color: str = OUTLINE_COLOR, width: float = OUTLINE_WIDTH
) -> None:
    for x, y in outline:
        ax.plot(x, y, color=color, linewidth=width)


def snapshots_grid(snapshots: Sequence[tuple[date, Rgb, Outline]], *, columns: int, caption: str) -> bytes:
    count = max(len(snapshots), 1)
    rows = -(-count // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.6 * rows + 0.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (day, rgb, outline) in zip(axes.flat, snapshots, strict=False):
        ax.imshow(rgb, interpolation="nearest")
        _draw_outline(ax, outline)
        ax.set_title(day.strftime("%d.%m.%Y"), fontsize=9)
    if caption:
        fig.suptitle(caption, fontsize=10)
    fig.tight_layout()
    return _png(fig)


def composites_grid(composites: Sequence[tuple[str, Rgb]], *, outline: Outline, caption: str) -> bytes:
    columns = min(len(composites), 5) or 1
    rows = -(-max(len(composites), 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.7 * rows + 0.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (label, rgb) in zip(axes.flat, composites, strict=False):
        ax.imshow(rgb, interpolation="nearest")
        _draw_outline(ax, outline)
        ax.set_title(label, fontsize=9)
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout()
    return _png(fig)


def objects_map(
    rgb: Rgb,
    *,
    outline: Outline,
    objects: Sequence[tuple[int, Outline, str]],
    caption: str,
) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 8 * rgb.shape[0] / max(rgb.shape[1], 1) + 0.6))
    ax.imshow(rgb, interpolation="nearest")
    _draw_outline(ax, outline)
    for number, rings, color in objects:
        _draw_outline(ax, rings, color=color, width=1.6)
        xs = np.concatenate([x for x, _ in rings])
        ys = np.concatenate([y for _, y in rings])
        ax.text(
            float(xs.mean()),
            float(ys.mean()),
            str(number),
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.2", "facecolor": color, "edgecolor": "none"},
        )
    ax.axis("off")
    ax.set_title(caption, fontsize=10)
    return _png(fig)


def object_series(
    *,
    series: Sequence[tuple[int, str, Sequence[date], Sequence[float], str, tuple[date, date] | None]],
    title: str,
) -> bytes:
    columns = min(len(series), 3) or 1
    rows = -(-max(len(series), 1) // columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 2.2 * rows + 0.5), squeeze=False, sharex=True
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, (number, label, days, values, color, window) in zip(axes.flat, series, strict=False):
        ax.axis("on")
        x = np.array(days, dtype="datetime64[D]")
        ax.plot(x, values, marker=".", markersize=2.5, linewidth=0.8, color=color)
        if window is not None:
            x0 = mdates.date2num(window[0])  # type: ignore[no-untyped-call]
            x1 = mdates.date2num(window[1])  # type: ignore[no-untyped-call]
            ax.axvspan(x0, x1, color=color, alpha=0.25)
        ax.set_title(f"{number}: {label}", fontsize=8)
        ax.set_ylim(0, 0.35)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _png(fig)


def relief_maps(
    *,
    heights: FloatArray,
    slope: FloatArray,
    extent: tuple[float, float, float, float],
    outline: Outline,
    contour_step_m: float,
) -> bytes:
    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(10, 4.6))
    image = ax_h.imshow(heights, cmap="terrain", extent=extent)
    if np.isfinite(heights).any():
        levels = np.arange(
            np.floor(np.nanmin(heights)), np.ceil(np.nanmax(heights)) + contour_step_m, contour_step_m
        )
        contours = ax_h.contour(
            heights, levels=levels, colors="k", linewidths=0.4, extent=extent, origin="upper"
        )
        ax_h.clabel(contours, fontsize=6, fmt="%g")
    _draw_outline(ax_h, outline)
    ax_h.set_title(f"Висота, м (горизонталі через {contour_step_m:g} м)", fontsize=10)
    fig.colorbar(image, ax=ax_h, shrink=0.8)
    image = ax_s.imshow(slope, cmap="magma_r", vmin=0, vmax=max(3.0, float(np.nanmax(slope))), extent=extent)
    _draw_outline(ax_s, outline)
    ax_s.set_title("Ухил, °", fontsize=10)
    fig.colorbar(image, ax=ax_s, shrink=0.8)
    for ax in (ax_h, ax_s):
        ax.set_xlabel("м")
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _png(fig)
