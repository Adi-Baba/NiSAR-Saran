#!/usr/bin/env python3
"""Calculate area overlap after a flood polygon is downloaded.

Usage:
  python compute_overlap.py /path/to/flood_polygon.geojson

Dependencies: geopandas, shapely, pyogrio (or Fiona).
"""
import json, sys
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
if len(sys.argv) != 2:
    raise SystemExit("usage: compute_overlap.py FLOOD_POLYGON.(geojson|shp|gpkg)")

flood_path = Path(sys.argv[1])
hotspot = gpd.read_file(HERE / "frequencyB_hotspot_footprint.geojson").to_crs(32644)
flood = gpd.read_file(flood_path)
if flood.crs is None:
    raise SystemExit("flood layer has no CRS; assign its source CRS before calculating overlap")
flood = flood.to_crs(32644)
hotspot_geom = hotspot.geometry.union_all()
flood_geom = flood.geometry.union_all()
intersection = hotspot_geom.intersection(flood_geom)
union = hotspot_geom.union(flood_geom)
hs_area = hotspot_geom.area
fl_area = flood_geom.area
inter_area = intersection.area
result = {
    "hotspot_area_m2": float(hs_area),
    "flood_polygon_area_m2": float(fl_area),
    "intersection_area_m2": float(inter_area),
    "hotspot_coverage_fraction": float(inter_area / hs_area) if hs_area else None,
    "flood_coverage_fraction": float(inter_area / fl_area) if fl_area else None,
    "iou": float(inter_area / union.area) if union.area else None,
    "hotspot_geojson": str(HERE / "frequencyB_hotspot_footprint.geojson"),
    "flood_polygon": str(flood_path),
    "equal_area_crs": "EPSG:32644 (UTM zone 44N; metres)",
}
(HERE / "hotspot_flood_overlap.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
