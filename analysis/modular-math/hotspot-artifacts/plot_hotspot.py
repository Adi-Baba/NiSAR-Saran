#!/usr/bin/env python3
"""Make a compact, coordinate-labelled hotspot comparison figure."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "frequencyB_hotspot_points.csv").open()))
lon = np.array([float(r["lon"]) for r in rows]); lat = np.array([float(r["lat"]) for r in rows])
layers = [
    ("B HHHH Δ (dB)", "b_hhhh_delta_db"),
    ("B HVHV Δ (dB)", "b_hvhv_delta_db"),
    ("A HHHH Δ (dB)", "a_hhhh_delta_db"),
    ("B number-of-looks Δ", None),
]
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
for ax, (title, key) in zip(axes.flat, layers):
    if key is None:
        z = np.array([float(r["b_nlooks_after"]) - float(r["b_nlooks_before"]) for r in rows])
    else:
        z = np.array([float(r[key]) for r in rows])
    sc = ax.scatter(lon, lat, c=z, s=22, cmap="coolwarm", edgecolors="none")
    ax.scatter([lon.mean()], [lat.mean()], marker="+", c="black", s=100, linewidths=1.5)
    ax.set_title(title)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=.25)
    cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.04); cb.ax.set_ylabel("value", rotation=270, va="bottom")
fig.suptitle("NISAR frequency-B HHHH hotspot — 14 Aug → 26 Aug 2026\nWGS84 coordinates; + centroid")
fig.savefig(HERE / "frequencyB_hotspot_comparison.png", dpi=180)
print(HERE / "frequencyB_hotspot_comparison.png")
