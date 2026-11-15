#!/usr/bin/env python3
"""Create NASA-style SAR interpretation figures from the paired NISAR GCOV files.

Products are display-stretched false-colour or greyscale renderings of actual
frequency-B backscatter. They are visual aids, not optical photographs and not
a flood classification.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCE = PROJECT / "data" / "source" / "New"
FIG = HERE / "Fig"
FIG.mkdir(exist_ok=True)

files = sorted(SOURCE.glob("NISAR_L2_PR_GCOV_*.h5"))
if len(files) != 2:
    raise SystemExit(f"Expected exactly two paired GCOV H5 files in {SOURCE}; found {len(files)}.")

before_file, after_file = files
with h5py.File(before_file, "r") as before, h5py.File(after_file, "r") as after:
    before_grid = before["science/LSAR/GCOV/grids/frequencyB"]
    after_grid = after["science/LSAR/GCOV/grids/frequencyB"]
    hh_before = before_grid["HHHH"][:]
    hh_after = after_grid["HHHH"][:]
    hv_before = before_grid["HVHV"][:]
    hv_after = after_grid["HVHV"][:]
    valid = (before_grid["mask"][:] == 1) & (after_grid["mask"][:] == 1)
    looks_before = before_grid["numberOfLooks"][:]
    looks_after = after_grid["numberOfLooks"][:]
    x = before_grid["xCoordinates"][:]
    y = before_grid["yCoordinates"][:]

with (HERE / "frequencyB_hotspot_summary.json").open(encoding="utf-8") as source:
    summary = json.load(source)
with (HERE / "frequencyB_hotspot_points.csv").open(encoding="utf-8", newline="") as source:
    hotspot_points = list(csv.DictReader(source))

threshold = float(summary["positive_q999_threshold_db"])
hotspot = np.zeros(valid.shape, dtype=bool)
for point in hotspot_points:
    hotspot[int(point["b_row"]), int(point["b_col"])] = True


def to_db(array: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Transform power in-place to dB and hide invalid pixels."""
    array[array <= 0] = np.nan
    np.log10(array, out=array)
    array *= np.float32(10.0)
    array[~good] = np.nan
    return array


def limits(array: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float]:
    values = array[np.isfinite(array)]
    if values.size > 2_000_000:
        values = values[:: max(1, values.size // 2_000_000)]
    return tuple(np.percentile(values, [low, high]))


def stretch(array: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((array - low) / (high - low), 0, 1)


def decimate(array: np.ndarray, step: int = 4) -> np.ndarray:
    return array[::step, ::step]


def add_note(fig: plt.Figure, text: str) -> None:
    """Keep provenance in README; avoid overlap with axis labels in exported PNGs."""
    _ = fig, text


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def map_axes(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#172b3a")
    if subtitle:
        ax.set_title(subtitle, loc="right", fontsize=9, color="#596d7c")
    ax.set_xlabel("Easting (km, EPSG:32644)")
    ax.set_ylabel("Northing (km, EPSG:32644)")
    ax.set_facecolor("#101820")


extent = [x.min() / 1000, x.max() / 1000, y.min() / 1000, y.max() / 1000]
hh_before = to_db(hh_before, valid)
hh_after = to_db(hh_after, valid)
hv_before = to_db(hv_before, valid)
hv_after = to_db(hv_after, valid)
hh_delta = hh_after - hh_before
hv_delta = hv_after - hv_before
hh_hv_ratio = hh_after - hv_after
looks_delta = looks_after - looks_before
looks_delta[~valid] = np.nan
selection_valid = valid & np.isfinite(hh_delta) & np.isfinite(hv_delta)
selection_support = ndimage.binary_erosion(
    selection_valid,
    structure=np.ones((3, 3), dtype=bool),
    border_value=0,
)

hh_limits = limits(np.concatenate((hh_before[::8, ::8].ravel(), hh_after[::8, ::8].ravel())))
hv_limits = limits(np.concatenate((hv_before[::8, ::8].ravel(), hv_after[::8, ::8].ravel())))
ratio_limits = limits(hh_hv_ratio)
delta_limit = max(abs(np.nanpercentile(hh_delta, 2)), abs(np.nanpercentile(hh_delta, 98)))
hv_delta_limit = max(abs(np.nanpercentile(hv_delta, 2)), abs(np.nanpercentile(hv_delta, 98)))


def scalar_map(
    name: str,
    array: np.ndarray,
    title: str,
    colorbar: str,
    cmap: str,
    value_limits: tuple[float, float],
    note: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.4), constrained_layout=True)
    image = ax.imshow(
        decimate(array),
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=value_limits[0],
        vmax=value_limits[1],
        interpolation="nearest",
    )
    map_axes(ax, title, "frequency B · valid pixels only")
    cb = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.015)
    cb.set_label(colorbar)
    add_note(fig, note)
    save(fig, name)


scalar_map(
    "01_hhhh_before_full.png",
    hh_before,
    "HHHH backscatter — before scene",
    "Backscatter power (dB; 2–98% display stretch)",
    "gray",
    hh_limits,
    "NISAR L2 GCOV · 14 Aug 2026 23:47 UTC · amplitude/power rendered as false greyscale.",
)
scalar_map(
    "02_hhhh_after_full.png",
    hh_after,
    "HHHH backscatter — after scene",
    "Backscatter power (dB; shared display stretch)",
    "gray",
    hh_limits,
    "NISAR L2 GCOV · 26 Aug 2026 23:47 UTC · same stretch as Figure 01 for visual comparison.",
)
scalar_map(
    "03_hvhv_before_full.png",
    hv_before,
    "HVHV cross-polarised backscatter — before scene",
    "Backscatter power (dB; 2–98% display stretch)",
    "magma",
    hv_limits,
    "NISAR L2 GCOV · 14 Aug 2026 23:47 UTC · false colour encodes radar-return strength.",
)
scalar_map(
    "04_hvhv_after_full.png",
    hv_after,
    "HVHV cross-polarised backscatter — after scene",
    "Backscatter power (dB; shared display stretch)",
    "magma",
    hv_limits,
    "NISAR L2 GCOV · 26 Aug 2026 23:47 UTC · same stretch as Figure 03 for visual comparison.",
)
scalar_map(
    "05_hhhh_change_full.png",
    hh_delta,
    "Bitemporal HHHH change",
    "After − before (dB)",
    "RdBu_r",
    (-delta_limit, delta_limit),
    "Warm colours: stronger post-scene HHHH return. Cool colours: weaker post-scene return. Not a flood map.",
)
scalar_map(
    "06_hvhv_change_full.png",
    hv_delta,
    "Bitemporal HVHV change",
    "After − before (dB)",
    "PuOr_r",
    (-hv_delta_limit, hv_delta_limit),
    "Cross-polarised change gives an independent view of surface/volume-scattering response.",
)
scalar_map(
    "07_hhhh_hvhv_ratio_after.png",
    hh_hv_ratio,
    "HHHH / HVHV contrast — after scene",
    "HHHH − HVHV (dB)",
    "viridis",
    ratio_limits,
    "A polarimetric contrast image: values are a logarithmic co- versus cross-polarisation ratio.",
)


# Bitemporal RGB: post HH in red, pre HH in green, post HV in blue.
rgb_temporal = np.dstack(
    (
        stretch(decimate(hh_after), *hh_limits),
        stretch(decimate(hh_before), *hh_limits),
        stretch(decimate(hv_after), *hv_limits),
    )
)
fig, ax = plt.subplots(figsize=(11.5, 8.4), constrained_layout=True)
ax.imshow(rgb_temporal, extent=extent, origin="upper", interpolation="nearest")
map_axes(ax, "Bitemporal false-colour composite", "R = post HHHH · G = pre HHHH · B = post HVHV")
add_note(fig, "Colour is an analytical display assignment, not natural colour. Red/green dominance helps reveal temporal HHHH change.")
save(fig, "08_bitemporal_false_colour_full.png")

# Polarimetric RGB: post HH, post HV, HH/HV contrast.
ratio_low, ratio_high = ratio_limits
rgb_polar = np.dstack(
    (
        stretch(decimate(hh_after), *hh_limits),
        stretch(decimate(hv_after), *hv_limits),
        stretch(decimate(hh_hv_ratio), ratio_low, ratio_high),
    )
)
fig, ax = plt.subplots(figsize=(11.5, 8.4), constrained_layout=True)
ax.imshow(rgb_polar, extent=extent, origin="upper", interpolation="nearest")
map_axes(ax, "Polarimetric false-colour composite", "R = HHHH · G = HVHV · B = HHHH/HVHV")
add_note(fig, "False-colour polarimetric display; channel stretches are independent and intended for pattern interpretation.")
save(fig, "09_polarimetric_false_colour_after.png")

# Candidate-change overlay on the post-scene intensity image.
fig, ax = plt.subplots(figsize=(11.5, 8.4), constrained_layout=True)
base = ax.imshow(decimate(hh_after), extent=extent, origin="upper", cmap="gray", vmin=hh_limits[0], vmax=hh_limits[1])
candidate = np.where(
    decimate(selection_support) & (decimate(hh_delta) >= threshold),
    1.0,
    np.nan,
)
ax.imshow(candidate, extent=extent, origin="upper", cmap=ListedColormap(["#ff3b30"]), alpha=0.82, interpolation="nearest")
hot_view = decimate(hotspot.astype(float))
ax.contour(hot_view, levels=[0.5], colors=["#00e5ff"], linewidths=1.3, extent=extent, origin="upper")
map_axes(ax, "Extreme positive HHHH-change candidates", f"threshold ≥ +{threshold:.2f} dB")
add_note(fig, "Red: all valid frequency-B pixels above the q99.9 positive-change threshold. Cyan: selected largest 8-connected hotspot.")
save(fig, "10_extreme_change_candidates.png")

r0, r1, c0, c1 = summary["hotspot_bbox_b_row_col"]
margin = 150
rs = slice(max(0, r0 - margin), min(valid.shape[0], r1 + margin + 1))
cs = slice(max(0, c0 - margin), min(valid.shape[1], c1 + margin + 1))
crop_extent = [x[cs.start] / 1000, x[cs.stop - 1] / 1000, y[rs.stop - 1] / 1000, y[rs.start] / 1000]

fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.5), constrained_layout=True)
for ax, data, title in [
    (axes[0], hh_before[rs, cs], "Before · 14 Aug 2026"),
    (axes[1], hh_after[rs, cs], "After · 26 Aug 2026"),
]:
    image = ax.imshow(data, extent=crop_extent, origin="upper", cmap="gray", vmin=hh_limits[0], vmax=hh_limits[1], interpolation="nearest")
    ax.contour(hotspot[rs, cs], levels=[0.5], colors=["#ffcc00"], linewidths=1.4, extent=crop_extent, origin="upper")
    map_axes(ax, title, "HHHH backscatter")
fig.colorbar(image, ax=axes, shrink=0.78, pad=0.015, label="Backscatter power (dB)")
add_note(fig, "Yellow outline: 90-cell selected hotspot. Both panels use an identical display stretch.")
save(fig, "11_hotspot_before_after_zoom.png")

crop_temporal = np.dstack(
    (
        stretch(hh_after[rs, cs], *hh_limits),
        stretch(hh_before[rs, cs], *hh_limits),
        stretch(hv_after[rs, cs], *hv_limits),
    )
)
fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.5), constrained_layout=True)
change = axes[0].imshow(
    hh_delta[rs, cs],
    extent=crop_extent,
    origin="upper",
    cmap="RdBu_r",
    vmin=-delta_limit,
    vmax=delta_limit,
    interpolation="nearest",
)
axes[1].imshow(crop_temporal, extent=crop_extent, origin="upper", interpolation="nearest")
for ax, title in [
    (axes[0], "HHHH change · after − before"),
    (axes[1], "False-colour temporal composite"),
]:
    ax.contour(hotspot[rs, cs], levels=[0.5], colors=["#101820"], linewidths=1.3, extent=crop_extent, origin="upper")
    map_axes(ax, title, "black outline = selected hotspot")
fig.colorbar(change, ax=axes[0], shrink=0.78, pad=0.015, label="HHHH change (dB)")
add_note(fig, "Right panel: R = post HHHH, G = pre HHHH, B = post HVHV. Colours are analytical, not natural colour.")
save(fig, "12_hotspot_change_and_false_colour.png")

fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.5), constrained_layout=True)
mask_joint = np.where(valid[rs, cs], 1.0, np.nan)
quality = axes[0].imshow(mask_joint, extent=crop_extent, origin="upper", cmap=ListedColormap(["#3bb273"]), vmin=0, vmax=1, interpolation="nearest")
looks = axes[1].imshow(
    looks_delta[rs, cs],
    extent=crop_extent,
    origin="upper",
    cmap="RdBu_r",
    vmin=-0.003,
    vmax=0.003,
    interpolation="nearest",
)
for ax, title in [(axes[0], "Joint-valid mask"), (axes[1], "Number-of-looks change")]:
    ax.contour(hotspot[rs, cs], levels=[0.5], colors=["#101820"], linewidths=1.3, extent=crop_extent, origin="upper")
    map_axes(ax, title, "black outline = selected hotspot")
fig.colorbar(looks, ax=axes[1], shrink=0.78, pad=0.015, label="After − before looks")
add_note(fig, "Hotspot cells are valid in both scenes. Look-count change is near zero at the selected component.")
save(fig, "13_hotspot_quality_layers.png")

(FIG / "README.md").write_text(
    """# SAR interpretation figures

These figures are generated from the two local NISAR L2 GCOV frequency-B rasters.
They use display stretches and false-colour channel assignments similar to common
SAR interpretation products. They are not optical imagery, and none of them is a
flood-inundation classification.

The presentation approach follows NASA Earthdata SAR image-interpretation guidance:
https://www.earthdata.nasa.gov/learn/earth-observation-data-basics/sar/image-interpretation
Each false-colour PNG states its exact channel assignment in the title.

- 01–04: matched before/after HHHH and HVHV intensity views.
- 05–07: bitemporal change and polarimetric contrast.
- 08–09: analytical false-colour composites.
- 10: extreme positive-change candidates and selected hotspot.
- 11–13: hotspot zoom and quality layers.

Regenerate with: python3 make_sar_interpretation_figures.py
""",
    encoding="utf-8",
)
print(f"Wrote 13 data-derived SAR figures to {FIG}")
