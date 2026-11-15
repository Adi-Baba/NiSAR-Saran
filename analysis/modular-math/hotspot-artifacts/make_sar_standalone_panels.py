#!/usr/bin/env python3
"""Export former multi-panel SAR figures as separate source-derived PNGs.

The existing collages stay in Fig/ as overviews. This script writes every
constituent panel directly from paired NISAR GCOV rasters, so a future collage
never needs to crop or upscale a previous one.
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
ROOT = HERE / "Fig" / "high_resolution_panels"
ROOT.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE.glob("NISAR_L2_PR_GCOV_*.h5"))
if len(files) != 2:
    raise SystemExit(f"Expected two paired GCOV H5 files; found {len(files)}.")
before_file, after_file = files

with (HERE / "frequencyB_hotspot_summary.json").open(encoding="utf-8") as stream:
    summary = json.load(stream)
with (HERE / "frequencyB_hotspot_points.csv").open(encoding="utf-8", newline="") as stream:
    hotspot_points = list(csv.DictReader(stream))


def to_db(power: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = power.astype(np.float32, copy=False)
    result[result <= 0] = np.nan
    np.log10(result, out=result)
    result *= np.float32(10.0)
    result[~valid] = np.nan
    return result


def limits(values: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size > 2_000_000:
        finite = finite[:: max(1, finite.size // 2_000_000)]
    return tuple(float(value) for value in np.percentile(finite, [low, high]))


def symmetric_limit(values: np.ndarray) -> float:
    return float(max(abs(np.nanpercentile(values, 2)), abs(np.nanpercentile(values, 98))))


def stretch(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def map_extent(x: np.ndarray, y: np.ndarray) -> list[float]:
    return [float(x.min() / 1000), float(x.max() / 1000), float(y.min() / 1000), float(y.max() / 1000)]


def extent_for_crop(x: np.ndarray, y: np.ndarray, rows: slice, cols: slice) -> list[float]:
    return [float(x[cols.start] / 1000), float(x[cols.stop - 1] / 1000), float(y[rows.stop - 1] / 1000), float(y[rows.start] / 1000)]


def style_map(axis: plt.Axes, title: str, subtitle: str) -> None:
    axis.set_title(title, loc="left", fontsize=18, fontweight="bold", color="#172b3a")
    axis.set_title(subtitle, loc="right", fontsize=12, color="#596d7c")
    axis.set_xlabel("Easting (km, EPSG:32644)")
    axis.set_ylabel("Northing (km, EPSG:32644)")
    axis.set_facecolor("#101820")


def export_map(
    folder: str,
    filename: str,
    values: np.ndarray,
    image_extent: list[float],
    title: str,
    subtitle: str,
    cmap: object,
    value_limits: tuple[float, float],
    colorbar_label: str,
    overlay: tuple[np.ndarray, object, float] | None = None,
    contour: tuple[np.ndarray, str, float] | None = None,
) -> None:
    path = ROOT / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(13.5, 11.0), constrained_layout=True)
    image = axis.imshow(values, extent=image_extent, origin="upper", cmap=cmap, vmin=value_limits[0], vmax=value_limits[1], interpolation="nearest")
    if overlay is not None:
        data, overlay_cmap, alpha = overlay
        axis.imshow(data, extent=image_extent, origin="upper", cmap=overlay_cmap, alpha=alpha, interpolation="nearest")
    if contour is not None:
        data, colour, width = contour
        axis.contour(data, levels=[0.5], colors=[colour], linewidths=width, extent=image_extent, origin="upper")
    style_map(axis, title, subtitle)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.84, pad=0.018)
    colorbar.set_label(colorbar_label)
    figure.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def export_rgb(
    folder: str,
    filename: str,
    rgb: np.ndarray,
    image_extent: list[float],
    title: str,
    subtitle: str,
    contour: tuple[np.ndarray, str, float] | None = None,
) -> None:
    path = ROOT / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(13.5, 11.0), constrained_layout=True)
    axis.imshow(rgb, extent=image_extent, origin="upper", interpolation="nearest")
    if contour is not None:
        data, colour, width = contour
        axis.contour(data, levels=[0.5], colors=[colour], linewidths=width, extent=image_extent, origin="upper")
    style_map(axis, title, subtitle)
    figure.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def export_chart(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(figure)


# Frequency-B source arrays are already on their native 80 m grid. H5 inputs
# are never opened for writing.
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
    bx = b0["xCoordinates"][:]
    by = b0["yCoordinates"][:]

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
selection_valid = b_valid & np.isfinite(b_hh_delta) & np.isfinite(b_hv_delta)
selection_support = ndimage.binary_erosion(
    selection_valid,
    structure=np.ones((3, 3), dtype=bool),
    border_value=0,
)
selection_positive = b_hh_delta[selection_support & (b_hh_delta > 0)]
canonical_q999 = float(summary["positive_q999_threshold_db"])
b_extent = map_extent(bx, by)
b_hh_limits = limits(np.concatenate((b_hh0[::8, ::8].ravel(), b_hh1[::8, ::8].ravel())))
b_hv_limits = limits(np.concatenate((b_hv0[::8, ::8].ravel(), b_hv1[::8, ::8].ravel())))
b_ratio_limits = limits(np.concatenate((b_ratio0[::8, ::8].ravel(), b_ratio1[::8, ::8].ravel())))
b_delta_limit = symmetric_limit(b_hh_delta)
b_hv_delta_limit = symmetric_limit(b_hv_delta)
b_ratio_delta_limit = symmetric_limit(b_ratio_delta)

hotspot = np.zeros(b_valid.shape, dtype=bool)
for point in hotspot_points:
    hotspot[int(point["b_row"]), int(point["b_col"])] = True

# Six panels formerly bundled in Figures 11–13.
r0, r1, c0, c1 = summary["hotspot_bbox_b_row_col"]
centre_r, centre_c = (r0 + r1) // 2, (c0 + c1) // 2
context_rows = slice(max(0, r0 - 150), min(b_valid.shape[0], r1 + 151))
context_cols = slice(max(0, c0 - 150), min(b_valid.shape[1], c1 + 151))
context_extent = extent_for_crop(bx, by, context_rows, context_cols)
context_hotspot = hotspot[context_rows, context_cols]
export_map("01_hotspot_overview", "01_before_hhhh.png", b_hh0[context_rows, context_cols], context_extent, "Hotspot context · before HHHH", "14 Aug 2026 · frequency B", "gray", b_hh_limits, "Backscatter power (dB)", contour=(context_hotspot, "#ffcc00", 1.8))
export_map("01_hotspot_overview", "02_after_hhhh.png", b_hh1[context_rows, context_cols], context_extent, "Hotspot context · after HHHH", "26 Aug 2026 · frequency B", "gray", b_hh_limits, "Backscatter power (dB)", contour=(context_hotspot, "#ffcc00", 1.8))
export_map("01_hotspot_overview", "03_hhhh_change.png", b_hh_delta[context_rows, context_cols], context_extent, "Hotspot context · HHHH change", "after − before · frequency B", "RdBu_r", (-b_delta_limit, b_delta_limit), "Change (dB)", contour=(context_hotspot, "#101820", 1.8))
context_rgb = np.dstack((stretch(b_hh1[context_rows, context_cols], *b_hh_limits), stretch(b_hh0[context_rows, context_cols], *b_hh_limits), stretch(b_hv1[context_rows, context_cols], *b_hv_limits)))
export_rgb("01_hotspot_overview", "04_bitemporal_false_colour.png", context_rgb, context_extent, "Hotspot context · bitemporal false colour", "R = post HHHH · G = pre HHHH · B = post HVHV", contour=(context_hotspot, "#101820", 1.8))
export_map("01_hotspot_overview", "05_joint_valid_mask.png", np.where(b_valid[context_rows, context_cols], 1.0, np.nan), context_extent, "Hotspot context · joint-valid mask", "valid in both scenes", ListedColormap(["#3bb273"]), (0, 1), "Valid = 1", contour=(context_hotspot, "#101820", 1.8))
export_map("01_hotspot_overview", "06_number_of_looks_change.png", b_looks_delta[context_rows, context_cols], context_extent, "Hotspot context · number-of-looks change", "after − before", "RdBu_r", (-0.003, 0.003), "Change in looks", contour=(context_hotspot, "#101820", 1.8))

# The three former Figure-17 maps now have independent, native-resolution PNGs.
for filename, values, title, cmap, display_limits, label in [
    ("01_hh_minus_hv_before.png", b_ratio0, "Frequency-B HH/HV contrast · before", "viridis", b_ratio_limits, "HHHH − HVHV (dB)"),
    ("02_hh_minus_hv_after.png", b_ratio1, "Frequency-B HH/HV contrast · after", "viridis", b_ratio_limits, "HHHH − HVHV (dB)"),
    ("03_hh_minus_hv_change.png", b_ratio_delta, "Frequency-B HH/HV contrast · change", "PuOr_r", (-b_ratio_delta_limit, b_ratio_delta_limit), "After − before (dB)"),
]:
    export_map("03_frequency_b_polarimetry", filename, values, b_extent, title, "frequency B · native 80 m grid", cmap, display_limits, label)

# Before/after/change are exported independently at every requested scale.
for radius_km in (1, 5, 25):
    radius_px = max(1, round(radius_km * 1000 / 80))
    rows = slice(max(0, centre_r - radius_px), min(b_valid.shape[0], centre_r + radius_px + 1))
    cols = slice(max(0, centre_c - radius_px), min(b_valid.shape[1], centre_c + radius_px + 1))
    image_extent = extent_for_crop(bx, by, rows, cols)
    local_hotspot = hotspot[rows, cols]
    folder = f"05_hotspot_multiscale/{radius_km:02d}km_radius"
    export_map(folder, "01_before_hhhh.png", b_hh0[rows, cols], image_extent, "Before HHHH", f"frequency B · ±{radius_km} km from hotspot", "gray", b_hh_limits, "Backscatter power (dB)", contour=(local_hotspot, "#ffd60a", 1.6))
    export_map(folder, "02_after_hhhh.png", b_hh1[rows, cols], image_extent, "After HHHH", f"frequency B · ±{radius_km} km from hotspot", "gray", b_hh_limits, "Backscatter power (dB)", contour=(local_hotspot, "#ffd60a", 1.6))
    export_map(folder, "03_hhhh_change.png", b_hh_delta[rows, cols], image_extent, "HHHH change", f"frequency B · ±{radius_km} km from hotspot", "RdBu_r", (-b_delta_limit, b_delta_limit), "After − before (dB)", contour=(local_hotspot, "#101820", 1.6))

# Former Figure-21 quality panels are source-resolution maps.
with h5py.File(before_file, "r") as before, h5py.File(after_file, "r") as after:
    b_factor0 = before["science/LSAR/GCOV/grids/frequencyB/rtcGammaToSigmaFactor"][:]
    b_factor1 = after["science/LSAR/GCOV/grids/frequencyB/rtcGammaToSigmaFactor"][:]
factor_mean = (b_factor0 + b_factor1) / 2
look_limits = limits(b_looks0)
look_delta_limit = symmetric_limit(b_looks_delta)
factor_limits = limits(factor_mean)
export_map("06_quality_radiometry", "01_joint_valid_mask.png", b_valid.astype(np.float32), b_extent, "Joint valid-data mask — Frequency B (both dates)", "frequency B · both dates · 49.3% valid", ListedColormap(["#e8edf2", "#35a273"]), (0, 1), "Valid in both scenes", contour=None)
export_map("06_quality_radiometry", "02_number_of_looks_before.png", b_looks0, b_extent, "Number of looks · before", "frequency B · 14 Aug", "cividis", look_limits, "Looks")
export_map("06_quality_radiometry", "03_number_of_looks_change.png", b_looks_delta, b_extent, "Number of looks change", "frequency B · after − before", "RdBu_r", (-look_delta_limit, look_delta_limit), "Change in looks")
export_map("06_quality_radiometry", "04_mean_rtc_gamma_to_sigma_factor.png", factor_mean, b_extent, "Mean RTC gamma-to-sigma factor", "frequency B · mean across scenes", "viridis", factor_limits, "RTC factor")
del b_factor0, b_factor1, factor_mean

# Four independent analytic panels formerly shown together in Figure 22.
profile_rows = slice(max(0, centre_r - 65), min(b_valid.shape[0], centre_r + 66))
profile_cols = slice(max(0, centre_c - 65), min(b_valid.shape[1], centre_c + 66))
horizontal_before = np.nanmean(b_hh0[centre_r - 3 : centre_r + 4, profile_cols], axis=0)
horizontal_after = np.nanmean(b_hh1[centre_r - 3 : centre_r + 4, profile_cols], axis=0)
vertical_before = np.nanmean(b_hh0[profile_rows, centre_c - 3 : centre_c + 4], axis=1)
vertical_after = np.nanmean(b_hh1[profile_rows, centre_c - 3 : centre_c + 4], axis=1)
profile_dir = ROOT / "07_profiles_distributions"
for filename, distance, before_values, after_values, title, label in [
    ("01_east_west_profile.png", (bx[profile_cols] - bx[centre_c]) / 1000, horizontal_before, horizontal_after, "East–west HHHH profile through hotspot", "Distance from hotspot centre (km)"),
    ("02_north_south_profile.png", (by[profile_rows] - by[centre_r]) / 1000, vertical_before, vertical_after, "North–south HHHH profile through hotspot", "Distance from hotspot centre (km)"),
]:
    figure, axis = plt.subplots(figsize=(13.5, 8.5), constrained_layout=True)
    axis.plot(distance, before_values, color="#247ba0", lw=2.2, label="Before · 14 Aug")
    axis.plot(distance, after_values, color="#e76f51", lw=2.2, label="After · 26 Aug")
    axis.axvline(0, color="#172b3a", lw=1.1, ls="--")
    axis.set(title=title, xlabel=label, ylabel="HHHH backscatter (dB)")
    axis.legend(frameon=False, fontsize=13)
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    export_chart(profile_dir / filename, figure)

point_b = np.array([float(row["b_hhhh_delta_db"]) for row in hotspot_points])
point_a = np.array([float(row["a_hhhh_delta_db"]) for row in hotspot_points])
q999_cut = canonical_q999
figure, axis = plt.subplots(figsize=(13.5, 8.5), constrained_layout=True)
local_delta = b_hh_delta[profile_rows, profile_cols]
axis.hist(local_delta[np.isfinite(local_delta)], bins=45, density=True, color="#b7c6d1", label="Local ±5 km background")
axis.hist(point_b, bins=16, density=True, color="#e76f51", alpha=0.82, label="90-cell hotspot")
axis.axvline(q999_cut, color="#172b3a", lw=1.4, ls="--", label=f"q99.9 = {q999_cut:.2f} dB")
axis.set(title="Hotspot versus local-change distribution", xlabel="Frequency-B HHHH change (dB)", ylabel="Density")
axis.legend(frameon=False, fontsize=13)
axis.grid(alpha=0.25)
axis.spines[["top", "right"]].set_visible(False)
export_chart(profile_dir / "03_hotspot_vs_background_distribution.png", figure)
figure, axis = plt.subplots(figsize=(13.5, 9.0), constrained_layout=True)
axis.scatter(point_b, point_a, s=64, color="#9b5de5", alpha=0.82, edgecolor="white", linewidth=0.6)
fit = np.polyfit(point_b, point_a, 1)
fit_x = np.linspace(point_b.min(), point_b.max(), 100)
axis.plot(fit_x, np.polyval(fit, fit_x), color="#172b3a", lw=1.8)
axis.set(title="Hotspot A/B point agreement", xlabel="Frequency-B HHHH change (dB)", ylabel="Frequency-A HHHH change (dB)")
axis.text(0.04, 0.95, f"Pearson r = {np.corrcoef(point_b, point_a)[0, 1]:.2f}\nn = {point_b.size}", transform=axis.transAxes, va="top")
axis.grid(alpha=0.25)
axis.spines[["top", "right"]].set_visible(False)
export_chart(profile_dir / "04_hotspot_frequencyA_frequencyB_scatter.png", figure)

# Individual threshold and connected-component panels replace Figures 23–24.
quantiles = (99.0, 99.5, 99.9)
thresholds = np.array(
    [
        np.percentile(selection_positive, quantiles[0]),
        np.percentile(selection_positive, quantiles[1]),
        canonical_q999,
    ]
)
for percentile, cut in zip(quantiles, thresholds):
    candidate = np.where(selection_support & (b_hh_delta >= cut), 1.0, np.nan)
    export_map("08_thresholds_components", f"q{str(percentile).replace('.', '_')}_positive_change.png", b_hh1, b_extent, f"Positive HHHH change at q{percentile:g}", f"red ≥ +{cut:.2f} dB · cyan = selected hotspot", "gray", b_hh_limits, "Post-scene HHHH power (dB)", overlay=(candidate, ListedColormap(["#ff3b30"]), 0.84), contour=(hotspot, "#00e5ff", 1.4))

q999 = selection_support & (b_hh_delta >= canonical_q999)
labels, component_count = ndimage.label(q999, structure=np.ones((3, 3), dtype=np.uint8))
component_sizes = np.bincount(labels.ravel())
component_sizes[0] = 0
largest = labels == int(np.argmax(component_sizes))
if not np.array_equal(largest, hotspot):
    raise RuntimeError(
        "standalone figure selection diverged from canonical hotspot: "
        f"largest={int(largest.sum())}, expected={int(hotspot.sum())}"
    )
components = np.where(labels > 0, 1.0, np.nan)
export_map("08_thresholds_components", "q999_connected_components.png", b_hh1, b_extent, "q99.9 connected components", f"{component_count:,} components · orange = candidate", "gray", b_hh_limits, "Post-scene HHHH power (dB)", overlay=(components, ListedColormap(["#f4a261"]), 0.78))
top_sizes = np.sort(component_sizes[component_sizes > 0])[::-1][:20]
figure, axis = plt.subplots(figsize=(13.5, 8.5), constrained_layout=True)
axis.bar(np.arange(1, len(top_sizes) + 1), top_sizes, color="#e76f51")
axis.set_yscale("log")
axis.set(title="Largest q99.9 connected components", xlabel="Component rank", ylabel="Pixels (log scale)")
axis.text(0.98, 0.95, f"largest = {top_sizes[0]:,} pixels\nselected hotspot = {hotspot.sum()} pixels", transform=axis.transAxes, ha="right", va="top")
axis.grid(axis="y", alpha=0.25)
axis.spines[["top", "right"]].set_visible(False)
export_chart(ROOT / "08_thresholds_components" / "largest_component_sizes.png", figure)
del labels, components, component_sizes

# Source-grid RGB story-map tiles rather than slices of a figure PNG.
tiles = ROOT / "09_story_tiles_bitemporal"
tiles.mkdir(parents=True, exist_ok=True)
rgb = np.dstack((stretch(b_hh1, *b_hh_limits), stretch(b_hh0, *b_hh_limits), stretch(b_hv1, *b_hv_limits)))
rgb_u8 = np.nan_to_num(rgb * 255, nan=0).astype(np.uint8)
manifest = ["# Source-resolution bitemporal story-map tiles", "", "R = post HHHH; G = pre HHHH; B = post HVHV. Native frequency-B 80 m display.", "", "| tile | source rows | source columns |", "|---|---:|---:|"]
height, width, _ = rgb_u8.shape
for row in range(3):
    for col in range(3):
        y0, y1 = row * height // 3, (row + 1) * height // 3
        x0, x1 = col * width // 3, (col + 1) * width // 3
        name = f"tile_r{row}_c{col}.png"
        plt.imsave(tiles / name, rgb_u8[y0:y1, x0:x1])
        manifest.append(f"| {name} | {y0}:{y1} | {x0}:{x1} |")
(tiles / "README.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")

# Keep only the B-change data required for the A/B comparison before sampling
# frequency A. This avoids an unnecessarily large peak memory footprint.
del b_hh0, b_hh1, b_hv0, b_hv1, b_hv_delta
del b_ratio0, b_ratio1, b_ratio_delta, b_looks0, b_looks1, b_looks_delta
del rgb, rgb_u8

# Frequency-A has native 10 m cells. Sampling every eighth source cell gives a
# 4,293 × 4,365 lattice: the same 80 m display density as native frequency B,
# not the coarser 320 m lattice used in the original overview collage.
with h5py.File(before_file, "r") as before, h5py.File(after_file, "r") as after:
    a0 = before["science/LSAR/GCOV/grids/frequencyA"]
    a1 = after["science/LSAR/GCOV/grids/frequencyA"]
    a_valid = (a0["mask"][::8, ::8] == 1) & (a1["mask"][::8, ::8] == 1)
    a_hh0 = a0["HHHH"][::8, ::8]
    a_hh1 = a1["HHHH"][::8, ::8]
    a_hv0 = a0["HVHV"][::8, ::8]
    a_hv1 = a1["HVHV"][::8, ::8]
    ax_coords = a0["xCoordinates"][::8]
    ay_coords = a0["yCoordinates"][::8]

a_hh0 = to_db(a_hh0, a_valid)
a_hh1 = to_db(a_hh1, a_valid)
a_hv0 = to_db(a_hv0, a_valid)
a_hv1 = to_db(a_hv1, a_valid)
a_hh_delta = a_hh1 - a_hh0
a_hv_delta = a_hv1 - a_hv0
a_extent = map_extent(ax_coords, ay_coords)
a_hh_limits = limits(np.concatenate((a_hh0[::4, ::4].ravel(), a_hh1[::4, ::4].ravel())))
a_hv_limits = limits(np.concatenate((a_hv0[::4, ::4].ravel(), a_hv1[::4, ::4].ravel())))
shared_delta_limit = max(b_delta_limit, symmetric_limit(a_hh_delta), symmetric_limit(a_hv_delta))

for filename, values, title, cmap, display_limits, label in [
    ("01_hhhh_before_14aug.png", a_hh0, "Frequency-A HHHH · before", "gray", a_hh_limits, "Backscatter power (dB)"),
    ("02_hhhh_after_26aug.png", a_hh1, "Frequency-A HHHH · after", "gray", a_hh_limits, "Backscatter power (dB)"),
    ("03_hhhh_change.png", a_hh_delta, "Frequency-A HHHH · change", "RdBu_r", (-shared_delta_limit, shared_delta_limit), "After − before (dB)"),
    ("04_hvhv_before_14aug.png", a_hv0, "Frequency-A HVHV · before", "magma", a_hv_limits, "Backscatter power (dB)"),
    ("05_hvhv_after_26aug.png", a_hv1, "Frequency-A HVHV · after", "magma", a_hv_limits, "Backscatter power (dB)"),
    ("06_hvhv_change.png", a_hv_delta, "Frequency-A HVHV · change", "RdBu_r", (-shared_delta_limit, shared_delta_limit), "After − before (dB)"),
]:
    export_map("02_frequency_a", filename, values, a_extent, title, "80 m source-sampled display lattice", cmap, display_limits, label)

# The A/B comparison now contains three independent assets rather than a
# three-column layout. Both maps use matching 80 m grids and colour limits.
export_map("04_cross_frequency", "01_frequencyB_hhhh_change.png", b_hh_delta, b_extent, "Frequency-B HHHH change", "native 80 m grid", "RdBu_r", (-shared_delta_limit, shared_delta_limit), "After − before (dB)")
export_map("04_cross_frequency", "02_frequencyA_hhhh_change.png", a_hh_delta, a_extent, "Frequency-A HHHH change", "80 m source-sampled display lattice", "RdBu_r", (-shared_delta_limit, shared_delta_limit), "After − before (dB)")
joint = b_valid & a_valid & np.isfinite(b_hh_delta) & np.isfinite(a_hh_delta)
sample = joint[::4, ::4]
sample_b = b_hh_delta[::4, ::4][sample]
sample_a = a_hh_delta[::4, ::4][sample]
figure, axis = plt.subplots(figsize=(13.5, 10.0), constrained_layout=True)
hexbin = axis.hexbin(sample_b, sample_a, gridsize=75, mincnt=1, cmap="magma")
fit = np.polyfit(sample_b, sample_a, 1)
fit_x = np.linspace(np.percentile(sample_b, 1), np.percentile(sample_b, 99), 100)
axis.plot(fit_x, np.polyval(fit, fit_x), color="white", lw=1.8)
axis.set(title="Frequency-A/B HHHH-change agreement", xlabel="Frequency-B HHHH change (dB)", ylabel="Frequency-A HHHH change (dB)")
axis.text(0.04, 0.95, f"Pearson r = {np.corrcoef(sample_b, sample_a)[0, 1]:.2f}\nn = {sample_b.size:,}", transform=axis.transAxes, va="top", color="white")
axis.grid(alpha=0.22)
figure.colorbar(hexbin, ax=axis, shrink=0.84, pad=0.018, label="Sample count")
export_chart(ROOT / "04_cross_frequency" / "03_frequencyA_frequencyB_hexbin.png", figure)

(ROOT / "README.md").write_text(
    """# High-resolution standalone SAR panels

Every former multi-panel overview is exported here as a separate PNG rendered
directly from the paired NISAR L2 GCOV source arrays. No panel is cropped from
or upscaled from a collage. The overview collages remain in the parent `Fig/`
directory only for quick visual comparison.

- `01_hotspot_overview`: the six separate panels previously bundled in Figures 11–13.
- `02_frequency_a`: HHHH/HVHV before, after, and change at an 80 m sampled A grid.
- `03_frequency_b_polarimetry`: individual HH/HV ratio before, after, and change.
- `04_cross_frequency`: B change, A change, and A/B agreement hexbin separately.
- `05_hotspot_multiscale`: independent before/after/change maps at ±1, ±5, and ±25 km.
- `06_quality_radiometry`: mask, number-of-looks, looks-change, and RTC layers.
- `07_profiles_distributions`: each profile, distribution, and hotspot scatter separately.
- `08_thresholds_components`: threshold maps, component map, and component-size plot.
- `09_story_tiles_bitemporal`: nine direct-source RGB story-map tiles.

Frequency-B uses its native 80 m raster. Frequency-A maps sample every eighth
10 m source cell to the same 80 m lattice; they are not enlarged from the old
320 m overview-mosaic display. These are SAR change and quality products, not
flood-inundation classifications.
""",
    encoding="utf-8",
)
print(f"Wrote high-resolution standalone panels to {ROOT}")
