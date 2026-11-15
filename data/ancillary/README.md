# Official ancillary inputs

Raw downloads stay here, separate from NISAR H5 input files.

| input | owner / product | intended use | Aug 2026 suitability |
|---|---|---|---|
| `imd/rainfall_districtwise_daily_imd.csv` | NWDP / IMD observed district-wise daily rainfall | observed rainfall at Saran and Bhojpur | usable from 19 Aug 2026 in the current download; no rows for 14–18 Aug |
| `bihar-surface-water/bihar_sw_daily_discharge_manual.csv` | NWDP / Bihar Water Department manual surface-water discharge | historical station context, including Valmikinagar/Gandak | **not usable** for the Aug 2026 pair; current file ends 31 May 2025 |
| `fmisc-gandak-2026/*.pdf` | Bihar FMISC Gandak barrage reports | date-matched barrage discharge near the post-scene | usable 26–29 Aug 2026; a barrage measurement, not a map of the Saran hotspot |

Run `./fetch_official_nwdp.sh` to fetch missing files. Existing raw files are retained by default; use `--force` only when deliberately refreshing them. Then run `python3 ../../analysis/ancillary/extract_august_2026_context.py` to recreate the auditable derived rainfall subset and coverage audit.

The Bihar surface-water file must not be called a CWC gauge series. The general NWDP CWC manual daily-discharge catalog was checked on 6 Nov 2026 and did not expose a Bihar resource.

The FMISC reports were acquired from the [Gandak 2026 report index](https://www.fmiscwrdbihar.gov.in/gate/Gatestatus.aspx?g=gandak&y=2026). Recreate their tabular extraction with `python3 ../../analysis/ancillary/extract_fmisc_gandak_barrage.py`.
