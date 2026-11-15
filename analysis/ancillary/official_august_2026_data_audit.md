# Official raw-data audit: Aug 2026 NISAR pair

Pair: 14 Aug 2026 23:47 UTC to 26 Aug 2026 23:47 UTC. The mapped hotspot is in Saran, not Arrah/Bhojpur.

## IMD district rainfall

- Source: https://nwdp.nwic.gov.in/dataset/9380580f-9bf0-4708-9773-0cea3cbc595a/resource/8752174f-1d17-4aaf-8058-2eb396f50157/download/rainfall_districtwise_daily_imd.csv
- Local raw file: `data/ancillary/imd/rainfall_districtwise_daily_imd.csv`
- SHA-256: `cf1d2fbb5477ac665785f0f08562b7027be4e90eacf1fd76888d18e365435a9b`
- Rows available for Saran/Bhojpur: 2026-08-19 to 2026-11-05.
- The raw download has no Saran/Bhojpur observations for 14–18 Aug 2026; do not fabricate a baseline rainfall value.
- 26 Aug observed daily rainfall: Bhojpur 0.1 mm, Saran 0.4 mm.
- Derived verbatim subset: `imd_saran_bhojpur_2026-08-19_2026-08-31.csv` (26 rows).

## Manual discharge

- Source: https://nwdp.nwic.gov.in/dataset/dba020d8-5bfa-44af-88a4-021150cd6847/resource/71036da3-88ce-4970-8fdd-0456351db972/download/bihar_sw_daily_discharge_manual.csv
- Local raw file: `data/ancillary/bihar-surface-water/bihar_sw_daily_discharge_manual.csv`
- SHA-256: `d3b62d510b7b26d1d6cf05f6524b8f3f0ab6810cfae9acca7d25515390a759db`
- Available dates: 2020-06-15 to 2025-05-31.
- This is Bihar Water Department manual surface-water discharge, **not CWC**. It has no 2026 rows and cannot validate this NISAR pair.

## Date-matched Gandak barrage reports

- Source index: https://www.fmiscwrdbihar.gov.in/gate/Gatestatus.aspx?g=gandak&y=2026
- Raw source PDFs: `data/ancillary/fmisc-gandak-2026/`.
- Extracted range: 2026-08-26 to 2026-08-29; 40 hourly values.
- Total discharge range in those reports: 88300 to 110600 cusec.
- Nearest published value after the post-scene timestamp (27 Aug 05:17 IST): 88300 cusec at 27 Aug 06:00 IST.
- This is a Gandak barrage measurement, not a CWC manual series and not a spatial flood-confirmation layer for the Saran hotspot.

## Result

The verified local IMD readings make a same-day local-rainfall explanation weak, but they neither prove nor disprove flooding. A date-matched Gandak gauge/barrage record or a local flood extent remains necessary to validate the candidate surface-change hotspot.
