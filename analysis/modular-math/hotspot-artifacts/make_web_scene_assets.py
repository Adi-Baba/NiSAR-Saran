#!/usr/bin/env python3
"""Render clean scene rasters for the Saran audit.

Unlike analytical figure exports, these assets omit axes and colour bars. Their
legend and interpretation stay in the interface. They still derive directly
from the paired frequency-B GCOV arrays and retain the selected hotspot outline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCE = PROJECT / "data" / "source" / "New"
ASSETS = HERE / "web-assets"
ASSETS.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE.glob("NISAR_L2_PR_GCOV_*.h5"))
if len(files) != 2:
    raise SystemExit(f"Expected two paired GCOV H5 files; found {len(files)}.")
before_file, after_file = files
with (HERE / "frequencyB_hotspot_summary.json").open(encoding="utf-8") as stream:
    summary = json.load(stream)
with (HERE / "frequencyB_hotspot_points.csv").open(encoding="utf-8", newline="") as stream:
    points = list(csv.DictReader(stream))


def to_db(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    values[values <= 0] = np.nan
    np.log10(values, out=values)
    values *= np.float32(10.0)
    values[~valid] = np.nan
    return values


def display_limits(values: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    return tuple(float(value) for value in np.percentile(finite[:: max(1, finite.size // 2_000_000)], [low, high]))


def colourize(values: np.ndarray, cmap: str, low: float, high: float) -> np.ndarray:
    normalised = np.clip((values - low) / (high - low), 0, 1)
    rgb = mpl.colormaps[cmap](np.nan_to_num(normalised, nan=0.0))[..., :3]
    rgb[~np.isfinite(values)] = (0.025, 0.052, 0.063)
    return rgb


with h5py.File(before_file, "r") as before, h5py.File(after_file, "r") as after:
    b0 = before["science/LSAR/GCOV/grids/frequencyB"]
    b1 = after["science/LSAR/GCOV/grids/frequencyB"]
    hh0 = b0["HHHH"][:]
    hh1 = b1["HHHH"][:]
    hv1 = b1["HVHV"][:]
    valid = (b0["mask"][:] == 1) & (b1["mask"][:] == 1)

hh0 = to_db(hh0, valid)
hh1 = to_db(hh1, valid)
hv1 = to_db(hv1, valid)
change = hh1 - hh0
hh_limits = display_limits(np.concatenate((hh0[::8, ::8].ravel(), hh1[::8, ::8].ravel())))
hv_limits = display_limits(hv1[::8, ::8])
change_limit = max(abs(np.nanpercentile(change, 2)), abs(np.nanpercentile(change, 98)))

hotspot = np.zeros(valid.shape, dtype=bool)
for point in points:
    hotspot[int(point["b_row"]), int(point["b_col"])] = True
r0, r1, c0, c1 = summary["hotspot_bbox_b_row_col"]
rows = slice(max(0, r0 - 150), min(valid.shape[0], r1 + 151))
cols = slice(max(0, c0 - 150), min(valid.shape[1], c1 + 151))
outline = ndimage.binary_dilation(hotspot[rows, cols], structure=np.ones((3, 3))) & ~hotspot[rows, cols]

before_rgb = colourize(hh0[rows, cols], "cividis", *hh_limits)
after_rgb = colourize(hh1[rows, cols], "cividis", *hh_limits)
change_rgb = colourize(change[rows, cols], "RdBu_r", -change_limit, change_limit)
temporal_rgb = np.dstack((
    np.clip((hh1[rows, cols] - hh_limits[0]) / (hh_limits[1] - hh_limits[0]), 0, 1),
    np.clip((hh0[rows, cols] - hh_limits[0]) / (hh_limits[1] - hh_limits[0]), 0, 1),
    np.clip((hv1[rows, cols] - hv_limits[0]) / (hv_limits[1] - hv_limits[0]), 0, 1),
))
for image in (before_rgb, after_rgb, change_rgb, temporal_rgb):
    image[outline] = (0.35, 0.61, 1.0)  # NASA mission blue #5A9CFF — high contrast over SAR greys

for name, image in [
    ("scene_before_radar_colour.png", before_rgb),
    ("scene_after_radar_colour.png", after_rgb),
    ("scene_hhhh_change_clean.png", change_rgb),
    ("scene_bitemporal_rgb_clean.png", temporal_rgb),
]:
    plt.imsave(ASSETS / name, image)

print(f"Wrote immersive scene assets to {ASSETS}")
