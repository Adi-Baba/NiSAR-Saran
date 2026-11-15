#!/usr/bin/env python3
"""Generate additional defensible NISAR GCOV figures requested for the Saran audit.

This script adds frequency-A, cross-frequency, multiscale, quality, threshold,
component, profile, and story-tile products. It uses only local paired H5 inputs
and existing hotspot metadata; no figure is a flood-extent classification.
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
TILES = FIG / "story_tiles_bitemporal"
FIG.mkdir(exist_ok=True)
TILES.mkdir(exist_ok=True)

files = sorted(SOURCE.glob("NISAR_L2_PR_GCOV_*.h5"))
if len(files) != 2:
    raise SystemExit(f"Expected two paired GCOV H5 files; found {len(files)}.")
before_file, after_file = files
with (HERE / "frequencyB_hotspot_summary.json").open(encoding="utf-8") as stream:
    summary = json.load(stream)
with (HERE / "frequencyB_hotspot_points.csv").open(encoding="utf-8", newline="") as stream:
    hotspot_points = list(csv.DictReader(stream))


def to_db(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    values[values <= 0] = np.nan
    np.log10(values, out=values)
    values *= np.float32(10.0)
    values[~valid] = np.nan
    return values


def finite_limits(values: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float]:
    flat = values[np.isfinite(values)]
    if flat.size > 1_500_000:
        flat = flat[:: max(1, flat.size // 1_500_000)]
    return tuple(np.percentile(flat, [low, high]))


def stretch(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((values - low) / (high - low), 0, 1)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def decorate(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#172b3a")
    if subtitle:
        ax.set_title(subtitle, loc="right", fontsize=8, color="#596d7c")
    ax.set_xlabel("Easting (km, EPSG:32644)")
    ax.set_ylabel("Northing (km, EPSG:32644)")
    ax.set_facecolor("#101820")


with h5py.File(before_file, "r") as before, h5py.File(after_file, "r") as after:
    b0 = before["science/LSAR/GCOV/grids/frequencyB"]
    b1 = after["science/LSAR/GCOV/grids/frequencyB"]
    b_hh0 = b0["HHHH"][:]
    b_hh1 = b1["HHHH"][:]
    b_hv0 = b0["HVHV"][:]
    b_hv1 = b1["HVHV"][:]
    b_valid = (b0["mask"][:] == 1) & (b1["mask"][:] == 1)
    b_looks0 = b0["numberOfLooks"][:]
    b_looks1 = b1["numberOfLooks"][:]
    b_factor0 = b0["rtcGammaToSigmaFactor"][::4, ::4]
    b_factor1 = b1["rtcGammaToSigmaFactor"][::4, ::4]
    bx = b0["xCoordinates"][:]
    by = b0["yCoordinates"][:]

    # A is 8x finer than B. A stride 32 and B stride 4 give the same ~320 m
    # display grid without loading the full 1.2-billion-pixel A grids.
    a0 = before["science/LSAR/GCOV/grids/frequencyA"]
    a1 = after["science/LSAR/GCOV/grids/frequencyA"]
    a_hh0 = a0["HHHH"][::32, ::32]
    a_hh1 = a1["HHHH"][::32, ::32]
    a_hv0 = a0["HVHV"][::32, ::32]
    a_hv1 = a1["HVHV"][::32, ::32]
    a_valid = (a0["mask"][::32, ::32] == 1) & (a1["mask"][::32, ::32] == 1)
    a_looks0 = a0["numberOfLooks"][::32, ::32]
    a_looks1 = a1["numberOfLooks"][::32, ::32]
    a_factor0 = a0["rtcGammaToSigmaFactor"][::32, ::32]
    a_factor1 = a1["rtcGammaToSigmaFactor"][::32, ::32]
    ax = a0["xCoordinates"][::32]
    ay = a0["yCoordinates"][::32]

b_hh0 = to_db(b_hh0, b_valid)
b_hh1 = to_db(b_hh1, b_valid)
b_hv0 = to_db(b_hv0, b_valid)
b_hv1 = to_db(b_hv1, b_valid)
b_hh_delta = b_hh1 - b_hh0
b_hv_delta = b_hv1 - b_hv0
b_ratio0 = b_hh0 - b_hv0
b_ratio1 = b_hh1 - b_hv1
b_ratio_delta = b_ratio1 - b_ratio0
b_looks_delta = b_looks1 - b_looks0
b_looks_delta[~b_valid] = np.nan

# Keep every threshold/component figure on the exact support used by
# hotspot_overlay.py: both channels finite on both dates, joint-valid mask,
# one-pixel 8-neighbour erosion, and positive HHHH change only.
selection_valid = b_valid & np.isfinite(b_hh_delta) & np.isfinite(b_hv_delta)
selection_support = ndimage.binary_erosion(
    selection_valid,
    structure=np.ones((3, 3), dtype=bool),
    border_value=0,
)
selection_positive = b_hh_delta[selection_support & (b_hh_delta > 0)]
canonical_q999 = float(summary["positive_q999_threshold_db"])

a_hh0 = to_db(a_hh0, a_valid)
a_hh1 = to_db(a_hh1, a_valid)
a_hv0 = to_db(a_hv0, a_valid)
a_hv1 = to_db(a_hv1, a_valid)
a_hh_delta = a_hh1 - a_hh0
a_hv_delta = a_hv1 - a_hv0
a_looks_delta = a_looks1 - a_looks0
a_looks_delta[~a_valid] = np.nan

b_extent = [bx.min() / 1000, bx.max() / 1000, by.min() / 1000, by.max() / 1000]
a_extent = [ax.min() / 1000, ax.max() / 1000, ay.min() / 1000, ay.max() / 1000]
b_hh_limits = finite_limits(np.concatenate((b_hh0[::8, ::8].ravel(), b_hh1[::8, ::8].ravel())))
b_hv_limits = finite_limits(np.concatenate((b_hv0[::8, ::8].ravel(), b_hv1[::8, ::8].ravel())))
a_hh_limits = finite_limits(np.concatenate((a_hh0.ravel(), a_hh1.ravel())))
a_hv_limits = finite_limits(np.concatenate((a_hv0.ravel(), a_hv1.ravel())))
delta_limit = max(abs(np.nanpercentile(np.concatenate((b_hh_delta[::8, ::8].ravel(), a_hh_delta.ravel())), 2)), abs(np.nanpercentile(np.concatenate((b_hh_delta[::8, ::8].ravel(), a_hh_delta.ravel())), 98)))
ratio_limits = finite_limits(np.concatenate((b_ratio0[::8, ::8].ravel(), b_ratio1[::8, ::8].ravel())))
ratio_delta_limit = max(abs(np.nanpercentile(b_ratio_delta, 2)), abs(np.nanpercentile(b_ratio_delta, 98)))

hotspot = np.zeros(b_valid.shape, dtype=bool)
for point in hotspot_points:
    hotspot[int(point["b_row"]), int(point["b_col"])] = True

# 14–15: frequency-A matched before/after/change panels.
for name, before_data, after_data, delta_data, channel, cmap, limits_for_channel in [
    ("14_frequencyA_hhhh_before_after_change.png", a_hh0, a_hh1, a_hh_delta, "HHHH", "gray", a_hh_limits),
    ("15_frequencyA_hvhv_before_after_change.png", a_hv0, a_hv1, a_hv_delta, "HVHV", "magma", a_hv_limits),
]:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), constrained_layout=True)
    images = [
        axes[0].imshow(before_data, extent=a_extent, origin="upper", cmap=cmap, vmin=limits_for_channel[0], vmax=limits_for_channel[1], interpolation="nearest"),
        axes[1].imshow(after_data, extent=a_extent, origin="upper", cmap=cmap, vmin=limits_for_channel[0], vmax=limits_for_channel[1], interpolation="nearest"),
        axes[2].imshow(delta_data, extent=a_extent, origin="upper", cmap="RdBu_r", vmin=-delta_limit, vmax=delta_limit, interpolation="nearest"),
    ]
    for panel, title in zip(axes, ["Before · 14 Aug", "After · 26 Aug", "After − before"]):
        decorate(panel, title, f"frequency A · {channel}")
    fig.colorbar(images[0], ax=axes[:2], shrink=0.78, pad=0.015, label="Backscatter power (dB)")
    fig.colorbar(images[2], ax=axes[2], shrink=0.78, pad=0.015, label="Change (dB)")
    fig.suptitle(f"Frequency-A {channel}: matched visual comparison", x=0.01, ha="left", fontsize=15, fontweight="bold", color="#172b3a")
    save(fig, name)

# 16: A versus B on matched 320 m sampling lattice plus a descriptive scatter.
b_hh_delta_320 = b_hh_delta[::4, ::4]
b_valid_320 = b_valid[::4, ::4]
joint = b_valid_320 & a_valid & np.isfinite(b_hh_delta_320) & np.isfinite(a_hh_delta)
sample = joint[::4, ::4]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.9), constrained_layout=True)
image_b = axes[0].imshow(b_hh_delta_320, extent=b_extent, origin="upper", cmap="RdBu_r", vmin=-delta_limit, vmax=delta_limit, interpolation="nearest")
image_a = axes[1].imshow(a_hh_delta, extent=a_extent, origin="upper", cmap="RdBu_r", vmin=-delta_limit, vmax=delta_limit, interpolation="nearest")
for panel, title in zip(axes[:2], ["Frequency B HHHH Δ", "Frequency A HHHH Δ"]):
    decorate(panel, title, "matched ~320 m display lattice")
fig.colorbar(image_b, ax=axes[:2], shrink=0.78, pad=0.015, label="After − before (dB)")
x_values = b_hh_delta_320[::4, ::4][sample]
y_values = a_hh_delta[::4, ::4][sample]
axes[2].hexbin(x_values, y_values, gridsize=55, mincnt=1, cmap="magma")
fit = np.polyfit(x_values, y_values, 1)
line = np.linspace(np.percentile(x_values, 1), np.percentile(x_values, 99), 100)
axes[2].plot(line, np.polyval(fit, line), color="white", lw=1.4)
axes[2].set_xlabel("Frequency-B HHHH Δ (dB)")
axes[2].set_ylabel("Frequency-A HHHH Δ (dB)")
axes[2].set_title("Cross-frequency agreement", loc="left", fontsize=12, fontweight="bold", color="#172b3a")
axes[2].grid(alpha=0.25)
axes[2].text(0.04, 0.95, f"Pearson r = {np.corrcoef(x_values, y_values)[0, 1]:.2f}\nn = {x_values.size:,}", transform=axes[2].transAxes, va="top", color="white")
fig.suptitle("A/B cross-frequency comparison", x=0.01, ha="left", fontsize=15, fontweight="bold", color="#172b3a")
save(fig, "16_frequencyA_frequencyB_comparison.png")

# 17: HH/HV ratio before, after, and temporal change.
fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), constrained_layout=True)
images = [
    axes[0].imshow(b_ratio0[::4, ::4], extent=b_extent, origin="upper", cmap="viridis", vmin=ratio_limits[0], vmax=ratio_limits[1], interpolation="nearest"),
    axes[1].imshow(b_ratio1[::4, ::4], extent=b_extent, origin="upper", cmap="viridis", vmin=ratio_limits[0], vmax=ratio_limits[1], interpolation="nearest"),
    axes[2].imshow(b_ratio_delta[::4, ::4], extent=b_extent, origin="upper", cmap="PuOr_r", vmin=-ratio_delta_limit, vmax=ratio_delta_limit, interpolation="nearest"),
]
for panel, title in zip(axes, ["HH/HV before", "HH/HV after", "HH/HV change"]):
    decorate(panel, title, "frequency B · dB")
fig.colorbar(images[0], ax=axes[:2], shrink=0.78, pad=0.015, label="HHHH − HVHV (dB)")
fig.colorbar(images[2], ax=axes[2], shrink=0.78, pad=0.015, label="After − before (dB)")
fig.suptitle("Co- / cross-polarisation ratio: before, after, and change", x=0.01, ha="left", fontsize=15, fontweight="bold", color="#172b3a")
save(fig, "17_hh_hv_ratio_before_after_change.png")

# 18–20: multiscale hotspot views (radius values, so displayed windows are ~2, 10, and 50 km).
r0, r1, c0, c1 = summary["hotspot_bbox_b_row_col"]
centre_r, centre_c = (r0 + r1) // 2, (c0 + c1) // 2
for radius_km, filename in [
    (1, "18_hotspot_multiscale_1km_radius.png"),
    (5, "19_hotspot_multiscale_5km_radius.png"),
    (25, "20_hotspot_multiscale_25km_radius.png"),
]:
    radius_px = max(1, round(radius_km * 1000 / 80))
    rs = slice(max(0, centre_r - radius_px), min(b_valid.shape[0], centre_r + radius_px + 1))
    cs = slice(max(0, centre_c - radius_px), min(b_valid.shape[1], centre_c + radius_px + 1))
    crop_extent = [bx[cs.start] / 1000, bx[cs.stop - 1] / 1000, by[rs.stop - 1] / 1000, by[rs.start] / 1000]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.9), constrained_layout=True)
    images = [
        axes[0].imshow(b_hh0[rs, cs], extent=crop_extent, origin="upper", cmap="gray", vmin=b_hh_limits[0], vmax=b_hh_limits[1], interpolation="nearest"),
        axes[1].imshow(b_hh1[rs, cs], extent=crop_extent, origin="upper", cmap="gray", vmin=b_hh_limits[0], vmax=b_hh_limits[1], interpolation="nearest"),
        axes[2].imshow(b_hh_delta[rs, cs], extent=crop_extent, origin="upper", cmap="RdBu_r", vmin=-delta_limit, vmax=delta_limit, interpolation="nearest"),
    ]
    for panel, title in zip(axes, ["Before HHHH", "After HHHH", "HHHH change"]):
        panel.contour(hotspot[rs, cs], levels=[0.5], colors=["#ffd60a" if title != "HHHH change" else "#101820"], linewidths=1.1, extent=crop_extent, origin="upper")
        decorate(panel, title, f"±{radius_km} km from hotspot centre")
    fig.colorbar(images[0], ax=axes[:2], shrink=0.78, pad=0.015, label="Backscatter power (dB)")
    fig.colorbar(images[2], ax=axes[2], shrink=0.78, pad=0.015, label="Change (dB)")
    fig.suptitle(f"Multiscale hotspot context: ±{radius_km} km", x=0.01, ha="left", fontsize=15, fontweight="bold", color="#172b3a")
    save(fig, filename)

# 21: B quality layers: mask, number of looks, look change, and RTC gamma-to-sigma factor.
b_mask_320 = b_valid[::4, ::4]
b_looks0_320 = b_looks0[::4, ::4]
b_looks_delta_320 = b_looks_delta[::4, ::4]
factor_mean = (b_factor0 + b_factor1) / 2
factor_limits = finite_limits(factor_mean)
look_limits = finite_limits(b_looks0_320)
look_delta_limit = max(abs(np.nanpercentile(b_looks_delta_320, 2)), abs(np.nanpercentile(b_looks_delta_320, 98)))
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
quality_panels = [
    (b_mask_320, "Joint valid mask", ListedColormap(["#e8edf2", "#35a273"]), (0, 1), "valid = 1"),
    (b_looks0_320, "Number of looks · before", "cividis", look_limits, "looks"),
    (b_looks_delta_320, "Number of looks change", "RdBu_r", (-look_delta_limit, look_delta_limit), "after − before"),
    (factor_mean, "Mean RTC gamma-to-sigma factor", "viridis", factor_limits, "factor"),
]
for axis, (data, title, cmap, limits_now, cbar_label) in zip(axes.flat, quality_panels):
    image = axis.imshow(data, extent=b_extent, origin="upper", cmap=cmap, vmin=limits_now[0], vmax=limits_now[1], interpolation="nearest")
    decorate(axis, title, "frequency B")
    fig.colorbar(image, ax=axis, shrink=0.78, pad=0.015, label=cbar_label)
fig.suptitle("Frequency-B quality and radiometric layers", x=0.01, ha="left", fontsize=15, fontweight="bold", color="#172b3a")
save(fig, "21_frequencyB_quality_radiometric_layers.png")

# 22: hotspot directional profiles, distributions, and cross-frequency point scatter.
window = 65
rs = slice(max(0, centre_r - window), min(b_valid.shape[0], centre_r + window + 1))
cs = slice(max(0, centre_c - window), min(b_valid.shape[1], centre_c + window + 1))
horizontal_before = np.nanmean(b_hh0[centre_r - 3:centre_r + 4, cs], axis=0)
horizontal_after = np.nanmean(b_hh1[centre_r - 3:centre_r + 4, cs], axis=0)
vertical_before = np.nanmean(b_hh0[rs, centre_c - 3:centre_c + 4], axis=1)
vertical_after = np.nanmean(b_hh1[rs, centre_c - 3:centre_c + 4], axis=1)
dist_x = (bx[cs] - bx[centre_c]) / 1000
dist_y = (by[rs] - by[centre_r]) / 1000
point_b = np.array([float(row["b_hhhh_delta_db"]) for row in hotspot_points])
point_a = np.array([float(row["a_hhhh_delta_db"]) for row in hotspot_points])
crop_background = b_hh_delta[rs, cs]
q999_cut = canonical_q999
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
axes[0, 0].plot(dist_x, horizontal_before, color="#247ba0", label="before")
axes[0, 0].plot(dist_x, horizontal_after, color="#e76f51", label="after")
axes[0, 0].axvline(0, color="#172b3a", lw=1, ls="--")
axes[0, 0].set(title="East–west profile", xlabel="Distance from hotspot centre (km)", ylabel="HHHH backscatter (dB)")
axes[0, 0].legend(frameon=False)
axes[0, 1].plot(dist_y, vertical_before, color="#247ba0", label="before")
axes[0, 1].plot(dist_y, vertical_after, color="#e76f51", label="after")
axes[0, 1].axvline(0, color="#172b3a", lw=1, ls="--")
axes[0, 1].set(title="North–south profile", xlabel="Distance from hotspot centre (km)", ylabel="HHHH backscatter (dB)")
axes[0, 1].legend(frameon=False)
axes[1, 0].hist(crop_background[np.isfinite(crop_background)], bins=45, density=True, color="#b7c6d1", label="local ±5 km background")
axes[1, 0].hist(point_b, bins=16, density=True, color="#e76f51", alpha=0.82, label="90-cell hotspot")
axes[1, 0].axvline(q999_cut, color="#172b3a", lw=1.3, ls="--", label=f"q99.9 = {q999_cut:.2f} dB")
axes[1, 0].set(title="Hotspot versus local-change distribution", xlabel="B HHHH change (dB)", ylabel="Density")
axes[1, 0].legend(frameon=False, fontsize=8)
axes[1, 1].scatter(point_b, point_a, s=44, color="#9b5de5", alpha=0.82, edgecolor="white", linewidth=0.5)
fit = np.polyfit(point_b, point_a, 1)
line = np.linspace(point_b.min(), point_b.max(), 100)
axes[1, 1].plot(line, np.polyval(fit, line), color="#172b3a")
axes[1, 1].set(title="Hotspot A/B point comparison", xlabel="B HHHH change (dB)", ylabel="A HHHH change (dB)")
axes[1, 1].text(0.04, 0.95, f"r = {np.corrcoef(point_b, point_a)[0, 1]:.2f}\nn = 90", transform=axes[1, 1].transAxes, va="top")
for axis in axes.flat:
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
fig.suptitle("Hotspot profiles, distributions, and per-pixel agreement", x=0.01, ha="left", fontsize=15, fontweight="bold", color="#172b3a")
save(fig, "22_hotspot_profiles_histograms_scatter.png")

# 23: three high-tail thresholds.
quantiles = [99.0, 99.5, 99.9]
thresholds = np.array(
    [
        np.percentile(selection_positive, quantiles[0]),
        np.percentile(selection_positive, quantiles[1]),
        canonical_q999,
    ]
)
fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), constrained_layout=True)
after_320 = b_hh1[::4, ::4]
for axis, percentile, cut in zip(axes, quantiles, thresholds):
    axis.imshow(after_320, extent=b_extent, origin="upper", cmap="gray", vmin=b_hh_limits[0], vmax=b_hh_limits[1], interpolation="nearest")
    high = np.where(selection_support[::4, ::4] & (b_hh_delta[::4, ::4] >= cut), 1.0, np.nan)
    axis.imshow(high, extent=b_extent, origin="upper", cmap=ListedColormap(["#ff3b30"]), alpha=0.85, interpolation="nearest")
    axis.contour(hotspot[::4, ::4], levels=[0.5], colors=["#00e5ff"], linewidths=1.2, extent=b_extent, origin="upper")
    decorate(axis, f"q{percentile:g} · +{cut:.2f} dB", "red = above threshold · cyan = selected hotspot")
fig.suptitle("Positive-change threshold sensitivity", x=0.01, ha="left", fontsize=15, fontweight="bold", color="#172b3a")
save(fig, "23_threshold_sensitivity_q990_q995_q999.png")

# 24: q99.9 components and their size distribution.
q999 = selection_support & (b_hh_delta >= canonical_q999)
labels, component_count = ndimage.label(q999, structure=np.ones((3, 3), dtype=np.uint8))
sizes = np.bincount(labels.ravel())
sizes[0] = 0
largest_label = int(np.argmax(sizes))
largest = labels == largest_label
if not np.array_equal(largest, hotspot):
    raise RuntimeError(
        "figure selection diverged from canonical hotspot: "
        f"largest={int(largest.sum())}, expected={int(hotspot.sum())}"
    )
fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
axes[0].imshow(b_hh1[::4, ::4], extent=b_extent, origin="upper", cmap="gray", vmin=b_hh_limits[0], vmax=b_hh_limits[1], interpolation="nearest")
components_view = np.where(labels[::4, ::4] > 0, 1.0, np.nan)
largest_view = np.where(largest[::4, ::4], 1.0, np.nan)
axes[0].imshow(components_view, extent=b_extent, origin="upper", cmap=ListedColormap(["#f4a261"]), alpha=0.78, interpolation="nearest")
axes[0].imshow(largest_view, extent=b_extent, origin="upper", cmap=ListedColormap(["#e63946"]), alpha=0.92, interpolation="nearest")
decorate(axes[0], "q99.9 connected components", f"{component_count:,} components · red = largest")
top_sizes = np.sort(sizes[sizes > 0])[::-1][:20]
axes[1].bar(np.arange(1, len(top_sizes) + 1), top_sizes, color="#e76f51")
axes[1].set_yscale("log")
axes[1].set(title="Largest q99.9 components", xlabel="Component rank", ylabel="Pixels (log scale)")
axes[1].grid(axis="y", alpha=0.25)
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].text(0.98, 0.95, f"largest = {top_sizes[0]:,} pixels\nselected hotspot = {hotspot.sum()} pixels", transform=axes[1].transAxes, ha="right", va="top")
fig.suptitle("Component selection at the +q99.9 HHHH threshold", x=0.01, ha="left", fontsize=15, fontweight="bold", color="#172b3a")
save(fig, "24_connected_components_q999.png")

# 25: web/story-map image tiles built from actual bitemporal RGB raster.
rgb = np.dstack(
    (
        stretch(b_hh1[::4, ::4], *b_hh_limits),
        stretch(b_hh0[::4, ::4], *b_hh_limits),
        stretch(b_hv1[::4, ::4], *b_hv_limits),
    )
)
height, width, _ = rgb.shape
tile_manifest = [
    "# Bitemporal story-map tiles",
    "",
    "Source: frequency-B NISAR GCOV, R = post HHHH, G = pre HHHH, B = post HVHV.",
    "Tiles are display imagery only and have no standalone flood classification.",
    "",
    "| tile | pixel rows | pixel cols |",
    "|---|---:|---:|",
]
for row in range(3):
    for col in range(3):
        y0, y1 = row * height // 3, (row + 1) * height // 3
        x0, x1 = col * width // 3, (col + 1) * width // 3
        tile_name = f"tile_r{row}_c{col}.png"
        plt.imsave(TILES / tile_name, np.nan_to_num(rgb[y0:y1, x0:x1], nan=0.0))
        tile_manifest.append(f"| {tile_name} | {y0}:{y1} | {x0}:{x1} |")
(TILES / "README.md").write_text("\n".join(tile_manifest) + "\n", encoding="utf-8")

(FIG / "ADVANCED_README.md").write_text(
    """# Additional defensible NISAR figure set

- 14–15: frequency-A HHHH and HVHV before/after/change.
- 16: frequency-A versus frequency-B HHHH comparison on a matched ~320 m display lattice.
- 17: frequency-B HH/HV ratio before/after/change.
- 18–20: hotspot-centred views at ±1, ±5, and ±25 km.
- 21: mask, number-of-looks, and RTC gamma-to-sigma layers.
- 22: profiles, histograms, and A/B hotspot scatter.
- 23: q99.0, q99.5, q99.9 threshold sensitivity.
- 24: q99.9 connected components and ranked component sizes.
- story_tiles_bitemporal: nine web/story-map PNG tiles, plus channel mapping.
- high_resolution_panels: 49 separate PNGs, rendered directly from the source rasters.

All outputs are visualisations of the paired GCOV observations and quality layers.
They do not independently confirm flood inundation.
""",
    encoding="utf-8",
)
print(f"Wrote advanced figures 14–24 and nine story tiles to {FIG}")
