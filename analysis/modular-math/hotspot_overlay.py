#!/usr/bin/env python3
"""Read-only geolocation and cross-channel audit for the Saran NISAR pair.

This script deliberately works from the two source H5 files and writes only
small, auditable summary/point/map artifacts.  It does not modify either H5.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "source" / "New"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "hotspot-artifacts"


def read_grid(path: Path, freq: str, names: list[str]) -> dict[str, np.ndarray | int]:
    out: dict[str, np.ndarray | int] = {}
    with h5py.File(path, "r") as f:
        g = f[f"science/LSAR/GCOV/grids/{freq}"]
        for name in names:
            if name == "projection_epsg":
                out[name] = int(g["projection"].attrs["epsg_code"])
            else:
                out[name] = g[name][...]
    return out


def delta_db(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    out = np.full(before.shape, np.nan, dtype=np.float32)
    good = np.isfinite(before) & np.isfinite(after) & (before > 0) & (after > 0)
    out[good] = (10.0 * np.log10(after[good] / before[good])).astype(np.float32)
    return out


def utm44n_to_lonlat(easting: np.ndarray | float, northing: np.ndarray | float):
    """WGS84 inverse UTM, zone 44N; avoids a runtime GIS dependency."""
    x = np.asarray(easting, dtype=float) - 500000.0
    y = np.asarray(northing, dtype=float)
    a = 6378137.0
    ecc_sq = 0.0066943799901413165
    ecc_prime_sq = ecc_sq / (1.0 - ecc_sq)
    k0 = 0.9996
    m = y / k0
    mu = m / (a * (1 - ecc_sq / 4 - 3 * ecc_sq**2 / 64 - 5 * ecc_sq**3 / 256))
    e1 = (1 - np.sqrt(1 - ecc_sq)) / (1 + np.sqrt(1 - ecc_sq))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    fp = mu + j1 * np.sin(2 * mu) + j2 * np.sin(4 * mu) + j3 * np.sin(6 * mu) + j4 * np.sin(8 * mu)
    sin_fp = np.sin(fp)
    cos_fp = np.cos(fp)
    tan_fp = np.tan(fp)
    c1 = ecc_prime_sq * cos_fp**2
    t1 = tan_fp**2
    n1 = a / np.sqrt(1 - ecc_sq * sin_fp**2)
    r1 = a * (1 - ecc_sq) / (1 - ecc_sq * sin_fp**2) ** 1.5
    d = x / (n1 * k0)
    lat = fp - (n1 * tan_fp / r1) * (d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ecc_prime_sq) * d**4 / 24 + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ecc_prime_sq - 3 * c1**2) * d**6 / 720)
    lon = np.deg2rad(81.0) + (d - (1 + 2 * t1 + c1) * d**3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ecc_prime_sq + 24 * t1**2) * d**5 / 120) / cos_fp
    return np.rad2deg(lon), np.rad2deg(lat)


def scalar(v) -> float:
    return float(v) if np.isfinite(v) else float("nan")


def safe_stats(x: np.ndarray) -> dict[str, float]:
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "p05": math.nan, "p95": math.nan, "min": math.nan, "max": math.nan}
    return {"n": int(y.size), "mean": float(y.mean()), "median": float(np.median(y)), "p05": float(np.quantile(y, .05)), "p95": float(np.quantile(y, .95)), "min": float(y.min()), "max": float(y.max())}


def main(input_dir: Path, output_dir: Path) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("NISAR_L2_PR_GCOV_*.h5"))
    if len(files) != 2:
        raise SystemExit(f"expected exactly two source H5 files in {input_dir}, found {len(files)}")
    before, after = files
    names = ["HHHH", "HVHV", "mask", "numberOfLooks", "xCoordinates", "yCoordinates"]
    b0 = read_grid(before, "frequencyB", names)
    b1 = read_grid(after, "frequencyB", names)
    d_h = delta_db(b0["HHHH"], b1["HHHH"])
    d_v = delta_db(b0["HVHV"], b1["HVHV"])
    valid = (b0["mask"] == 1) & (b1["mask"] == 1) & np.isfinite(d_h) & np.isfinite(d_v)
    eroded = ndimage.binary_erosion(valid, structure=np.ones((3, 3), dtype=bool), border_value=0)
    positive = d_h[eroded & (d_h > 0)]
    threshold = float(np.quantile(positive, 0.999))
    candidates = eroded & np.isfinite(d_h) & (d_h >= threshold)
    labels, count = ndimage.label(candidates, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndimage.find_objects(labels)
    comps = []
    for label, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        rows, cols = np.where(labels[sl] == label)
        area = int(rows.size)
        if area == 0:
            continue
        rr = rows + sl[0].start
        cc = cols + sl[1].start
        vals = d_h[rr, cc]
        comps.append((area, float(np.nanmean(vals)), float(np.nanmedian(vals)), label, rr, cc))
    if not comps:
        raise SystemExit("no positive q99.9 hotspot component")
    # Stable, intentionally conservative rule: largest contiguous component;
    # this avoids selecting isolated near-zero denominator spikes.
    comps.sort(key=lambda z: (-z[0], -z[1], -z[2], z[3]))
    area, mean_h, median_h, label, rr, cc = comps[0]
    hotspot = labels == label
    rmin, rmax, cmin, cmax = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    x = np.asarray(b0["xCoordinates"])[cc]
    y = np.asarray(b0["yCoordinates"])[rr]
    lon, lat = utm44n_to_lonlat(x, y)
    cx, cy = float(np.mean(x)), float(np.mean(y))
    clon, clat = utm44n_to_lonlat(cx, cy)
    clon, clat = float(clon), float(clat)

    # Map B coordinate centres to nearest A grid and read only the small A window.
    a0 = read_grid(before, "frequencyA", ["xCoordinates", "yCoordinates"])
    xa = np.asarray(a0["xCoordinates"]); ya = np.asarray(a0["yCoordinates"])
    # x is ascending; y is descending in this product.  Use explicit nearest
    # lookup so the A/B comparison cannot silently flip the northing axis.
    ai = np.array([int(np.argmin(np.abs(xa - v))) for v in x], dtype=int)
    aj = np.array([int(np.argmin(np.abs(ya - v))) for v in y], dtype=int)
    aminx, amaxx = int(ai.min()), int(ai.max())
    aminy, amaxy = int(aj.min()), int(aj.max())
    pad = 2
    xs = slice(max(0, aminx - pad), min(len(xa), amaxx + pad + 1))
    ys = slice(max(0, aminy - pad), min(len(ya), amaxy + pad + 1))
    with h5py.File(before, "r") as f0, h5py.File(after, "r") as f1:
        g0 = f0["science/LSAR/GCOV/grids/frequencyA"]; g1 = f1["science/LSAR/GCOV/grids/frequencyA"]
        ah0 = g0["HHHH"][ys, xs]; ah1 = g1["HHHH"][ys, xs]
        am0 = g0["mask"][ys, xs]; am1 = g1["mask"][ys, xs]
        an0 = g0["numberOfLooks"][ys, xs]; an1 = g1["numberOfLooks"][ys, xs]
    ai2 = ai - xs.start; aj2 = aj - ys.start
    a_h = delta_db(ah0, ah1)[aj2, ai2]
    a_mask0 = am0[aj2, ai2]; a_mask1 = am1[aj2, ai2]
    a_n0 = an0[aj2, ai2]; a_n1 = an1[aj2, ai2]

    # Point table: one row per contiguous hotspot pixel.
    points = output_dir / "frequencyB_hotspot_points.csv"
    with points.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["b_row", "b_col", "easting_m", "northing_m", "lon", "lat", "b_hhhh_delta_db", "b_hvhv_delta_db", "b_hhhh_before", "b_hhhh_after", "b_hvhv_before", "b_hvhv_after", "b_mask_before", "b_mask_after", "b_nlooks_before", "b_nlooks_after", "a_hhhh_delta_db", "a_mask_before", "a_mask_after", "a_nlooks_before", "a_nlooks_after"])
        for i, j in zip(rr, cc):
            px, py = float(b0["xCoordinates"][j]), float(b0["yCoordinates"][i])
            plon, plat = utm44n_to_lonlat(px, py)
            k = np.flatnonzero((rr == i) & (cc == j))[0]
            w.writerow([int(i), int(j), px, py, float(plon), float(plat), float(d_h[i, j]), float(d_v[i, j]), float(b0["HHHH"][i, j]), float(b1["HHHH"][i, j]), float(b0["HVHV"][i, j]), float(b1["HVHV"][i, j]), int(b0["mask"][i, j]), int(b1["mask"][i, j]), float(b0["numberOfLooks"][i, j]), float(b1["numberOfLooks"][i, j]), float(a_h[k]), int(a_mask0[k]), int(a_mask1[k]), float(a_n0[k]), float(a_n1[k])])

    # GeoJSON points for direct GIS/EO-browser import.
    features = []
    for i, j in zip(rr, cc):
        px, py = float(b0["xCoordinates"][j]), float(b0["yCoordinates"][i])
        plon, plat = utm44n_to_lonlat(px, py)
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(plon), float(plat)]}, "properties": {"b_row": int(i), "b_col": int(j), "b_hhhh_delta_db": float(d_h[i, j]), "b_hvhv_delta_db": float(d_v[i, j])}})
    geo = {"type": "FeatureCollection", "name": "frequencyB_HHHH_q99p9_largest_component", "crs_note": "Coordinates WGS84 lon/lat; source grid EPSG:32644", "features": features}
    (output_dir / "frequencyB_hotspot.geojson").write_text(json.dumps(geo, indent=2))

    stat_rows = []
    def add(prefix, arr):
        for k, v in safe_stats(arr).items(): stat_rows.append((prefix + "_" + k, v))
    add("B_HHHH_delta_db", d_h[hotspot]); add("B_HVHV_delta_db", d_v[hotspot]); add("A_HHHH_delta_db", a_h)
    add("B_HHHH_before", b0["HHHH"][hotspot]); add("B_HHHH_after", b1["HHHH"][hotspot])
    add("B_HVHV_before", b0["HVHV"][hotspot]); add("B_HVHV_after", b1["HVHV"][hotspot])
    add("B_nlooks_before", b0["numberOfLooks"][hotspot]); add("B_nlooks_after", b1["numberOfLooks"][hotspot]); add("A_nlooks_before", a_n0); add("A_nlooks_after", a_n1)
    add("A_mask_before", a_mask0); add("A_mask_after", a_mask1); add("B_mask_before", b0["mask"][hotspot]); add("B_mask_after", b1["mask"][hotspot])
    summary = {
        "source_before": before.name, "source_after": after.name, "frequency": "B", "projection_epsg": 32644,
        "rule": "valid(mask==1 both dates) + 8-neighbour erosion; positive HHHH delta q99.9; largest 8-connected component",
        "joint_valid_pixels": int(valid.sum()), "eroded_valid_pixels": int(eroded.sum()), "positive_q999_threshold_db": threshold,
        "candidate_components": int(count), "hotspot_pixels": area, "hotspot_bbox_b_row_col": [rmin, rmax, cmin, cmax],
        "hotspot_centroid_utm_easting_m": cx, "hotspot_centroid_utm_northing_m": cy, "hotspot_centroid_lon": clon, "hotspot_centroid_lat": clat,
        "hotspot_lat_range": [float(np.min(lat)), float(np.max(lat))], "hotspot_lon_range": [float(np.min(lon)), float(np.max(lon))],
        "stats": dict(stat_rows), "comparison_notes": {
            "B_HVHV": "same B hotspot pixels; independent cross-pol channel",
            "A_HHHH": "nearest A-grid centres for same UTM coordinates; small window read only",
            "masks": "B and A mask values at same hotspot points",
            "number_of_looks": "before/after values at same hotspot points",
        },
    }
    (output_dir / "frequencyB_hotspot_summary.json").write_text(json.dumps(summary, indent=2))
    with (output_dir / "frequencyB_hotspot_summary.tsv").open("w") as f:
        f.write("metric\tvalue\n")
        for k, v in summary.items():
            if isinstance(v, dict):
                for sk, sv in v.items(): f.write(f"{k}.{sk}\t{json.dumps(sv)}\n")
            else: f.write(f"{k}\t{json.dumps(v)}\n")
    print(json.dumps({"out": str(output_dir), "threshold_db": threshold, "hotspot_pixels": area, "centroid": [clon, clat], "bbox": [rmin, rmax, cmin, cmax], "B_HHHH": safe_stats(d_h[hotspot]), "B_HVHV": safe_stats(d_v[hotspot]), "A_HHHH": safe_stats(a_h)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="directory containing exactly two NISAR L2 GCOV H5 files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="directory for generated audit artifacts")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
