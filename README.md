# NiSAR-Saran

Evidence-first audit of a localized NISAR L2 GCOV backscatter change near Saran, Bihar. The project identifies a compact **candidate surface change** between 14 and 26 August 2026; it does **not** claim a confirmed flood.

## What was the anomaly?

The anomaly is a localized, unusually large radar-backscatter increase between the two NISAR observations. The analysis found 90 connected Frequency-B cells on an 80 m grid near Saran: HHHH increased by an average of **+9.29 dB**, HVHV increased by **+3.00 dB**, and the same location also increased in Frequency-A HHHH. Number-of-looks and masks stayed effectively unchanged. This supports a real mapped scene change, but does not identify its cause. It could involve water, wet soil, vegetation, geometry, or another surface/processing effect.

Start with the [hotspot audit report](analysis/modular-math/hotspot-artifacts/hotspot_overlay_report.md), then inspect the [90-cell footprint](analysis/modular-math/hotspot-artifacts/frequencyB_hotspot_footprint.geojson), [machine-readable summary](analysis/modular-math/hotspot-artifacts/frequencyB_hotspot_summary.json), and [submission presentation](deliverables/saran-shift-clickable-news-png-by-team-azarsatru.pptx).

## Project summary

We developed a reproducible radar-change workflow for a matched NISAR L2 GCOV pair. It applies joint validity masking, one-pixel erosion, HHHH backscatter change in dB, a scene-relative q99.9 threshold, and 8-connected-component selection. The largest component is a 90-cell candidate near 84.4786°E, 25.9817°N, with mean Frequency-B HHHH change of +9.29 dB. Quality, sensitivity, cross-polarization, and cross-frequency checks are kept separate from causal interpretation. This matters because SAR change alone can result from water, vegetation, wet soil, geometry, or processing effects; the workflow makes those limits visible instead of converting change directly into a flood claim.

## Data used

The two original L2 GCOV H5 granules are intentionally excluded from Git because together they are about 13.7 GB and require source access. Put verified copies in `data/source/New/`; exact filenames, timestamps, byte counts, and SHA-256 hashes are in [data/manifest.tsv](data/manifest.tsv).

| Role | UTC acquisition start | NISAR product |
| --- | --- | --- |
| Before | 2026-08-14 23:47:04 | `NISAR_L2_PR_GCOV_028_026_A_015_4005_DHDH_A_20260814T234704_20260814T234738_P05023_N_F_J_001.h5` |
| After | 2026-08-26 23:47:04 | `NISAR_L2_PR_GCOV_029_026_A_015_4005_DHDH_A_20260826T234704_20260826T234738_P05023_N_F_J_001.h5` |

Both are NISAR L2 GCOV, Track 26 / Frame 15, ascending, with Frequency B analyzed in EPSG:32644. Context-only source notes are in [data/ancillary/README.md](data/ancillary/README.md). They are not independent flood confirmation.

## Reproduce core audit

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Obtain exact source granules through authorized NASA Earthdata/ASF access,
# put them in data/source/New/, then verify against data/manifest.tsv.
bash scripts/run_hotspot_audit.sh
```

Optional custom locations keep every run portable:

```bash
bash scripts/run_hotspot_audit.sh /path/to/granules /path/to/output
```

Expected core output: 90-cell candidate, q99.9 threshold about 7.743874 dB, and 3,238 initial connected components. Generated outputs include `frequencyB_hotspot_summary.json`, point CSV, and GeoJSON in `analysis/modular-math/hotspot-artifacts/` by default.

## Repository modules

- [Data](data/README.md) — primary GCOV manifest and contextual source material.
- [Analysis](analysis/README.md) — ancillary, quantitative, and supporting-audit modules.
- [Core hotspot audit](analysis/modular-math/hotspot_overlay.py) — paired-GCOV candidate extraction.
- [Artifacts](analysis/modular-math/hotspot-artifacts/README.md) — figures, points, summaries, and reports.
- [Rust workspace](analysis/modular-math/rust-workspace/) — tested statistical modules.
- [Run scripts](scripts/README.md) — portable audit entry point.
- [Deliverables](deliverables/README.md) — final presentation.

## Scope and limitations

This repository contains the method, inputs, hashes, and generated evidence. It is not a claim of flood detection. A fully clean external reproduction still requires downloading the exact source granules and testing in a fresh environment. No license has been selected yet.
