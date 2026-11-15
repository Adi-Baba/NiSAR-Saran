#!/usr/bin/env python3
"""Turn the 80-m hotspot pixels into GIS polygons (read-only source CSV)."""
import csv, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hotspot_overlay import utm44n_to_lonlat

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "frequencyB_hotspot_points.csv").open()))
features = []
for r in rows:
    x = float(r["easting_m"]); y = float(r["northing_m"])
    corners = [(x-40,y-40),(x+40,y-40),(x+40,y+40),(x-40,y+40),(x-40,y-40)]
    ring = []
    for xx, yy in corners:
        lon, lat = utm44n_to_lonlat(xx, yy)
        ring.append([float(lon), float(lat)])
    features.append({"type":"Feature", "geometry":{"type":"Polygon","coordinates":[ring]}, "properties":{"b_row":int(r["b_row"]),"b_col":int(r["b_col"]),"b_hhhh_delta_db":float(r["b_hhhh_delta_db"])}})
out = {"type":"FeatureCollection", "name":"frequencyB_HHHH_hotspot_80m_cells", "crs_note":"WGS84 lon/lat; each polygon is one 80 m x 80 m EPSG:32644 source cell", "features":features}
(HERE / "frequencyB_hotspot_footprint.geojson").write_text(json.dumps(out, indent=2))
print(HERE / "frequencyB_hotspot_footprint.geojson")
