# Saran NISAR frequency-B HHHH hotspot audit

Pair: 14 Aug 2026 23:47 UTC → 26 Aug 2026 23:47 UTC, Track 26 / frame 15, ascending, EPSG:32644 (WGS84 / UTM 44N).

## Result

The hotspot is the largest 8-connected component above the positive HHHH q99.9 threshold after requiring `mask == 1` in both scenes and an 8-neighbour erosion. It contains 90 B-grid pixels (80 m grid).

| quantity | result |
|---|---:|
| threshold | +7.743874 dB |
| centroid | 25.981710°N, 84.478561°E |
| footprint | 25.976466–25.988137°N, 84.474155–84.483016°E |
| B HHHH change | mean +9.2851 dB; median +9.0285 dB |
| B HVHV change, same pixels | mean +3.0004 dB; median +3.1263 dB |
| A HHHH change, nearest A-grid centres | mean +9.3925 dB; median +9.6734 dB |
| B number-of-looks | 34.61216 → 34.61302 (Δ +0.00086) |
| A number-of-looks | 4.32446 → 4.32456 (Δ +0.00011) |
| B/A masks at hotspot | all 90 mapped points equal 1 on both dates |

The B HHHH rise is therefore not isolated to one channel: HVHV rises too, while number-of-looks is effectively unchanged. A HHHH at the same coordinates also rises. This is evidence for a real scene/processing change at the mapped pixels, not proof of flooding by itself.

## External overlay / footprint check

The centroid lies in the Saran/Gandak plain. The verified local-data audit is at [`../../ancillary/official_august_2026_data_audit.md`](../../ancillary/official_august_2026_data_audit.md). Its direct observations are:

- Official IMD district daily rainfall on 26 August: **Saran 0.4 mm** and **Bhojpur 0.1 mm**. The currently downloaded IMD file begins on 19 August, so it has no local values for the 14 August baseline scene.
- The NWDP Bihar manual-discharge CSV ends on 31 May 2025. It is historical context only; it must not be presented as an Aug-2026 CWC record.
- Bihar FMISC's date-matched **Gandak barrage** PDFs provide 40 hourly total-discharge values from 26–29 August: **88,300–110,600 cusec**. The first published reading after the post-scene acquisition (27 August 05:17 IST) is **88,300 cusec at 06:00 IST**. This is a barrage measurement, not a spatial flood map for the Saran footprint.

The low same-day local rainfall makes a local-rainfall-only explanation weak. The barrage data supports hydrologic context, but neither it nor rainfall proves inundation at the 90 pixels. Earlier scalar Nepal-event, rainfall, and discharge statements are not used as validation unless their original raw products are downloaded and spatially/time-matched.

**Conclusion:** retain the label **candidate flood/water-related change — not independently confirmed**. The decisive next validation is a local Sentinel-1/2 or NRSC/F​​MISC flood polygon covering Saran on 26–29 August, followed by intersection-over-union with the GeoJSON footprint.

## Files

- `frequencyB_hotspot.geojson` — WGS84 point features for GIS/EO-browser.
- `frequencyB_hotspot_footprint.geojson` — 90 individual 80 m × 80 m source-cell polygons; use this for area overlap.
- `compute_overlap.py` — calculates intersection, hotspot coverage, flood coverage, and IoU once a flood polygon is supplied.
- `frequencyB_hotspot_points.csv` — per-pixel B HHHH/HVHV, A HHHH, masks, looks, and coordinates.
- `frequencyB_hotspot_comparison.png` — coordinate-labelled four-panel comparison.
- `frequencyB_hotspot_summary.json` / `.tsv` — machine-readable summary.
- `august_external_evidence.tsv` — legacy research lead ledger; it is not a substitute for downloaded, date-matched source data.
- `august_context_timeseries.tsv` / `august_context_overlay.png` — legacy contextual visualization; do not use it as validation for this footprint.
- `../../ancillary/official_august_2026_data_audit.md` — reproducible audit of the official IMD, Bihar manual-discharge, and FMISC raw inputs.

All source H5 files are read-only inputs; no `nisar-zeta` files are touched.
