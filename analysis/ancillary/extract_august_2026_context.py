#!/usr/bin/env python3
"""Extract auditable local context from the official NWDP raw files.

No interpolation and no flood classification are performed here. The script
only keeps rows actually present in the downloaded source data and reports
their temporal coverage.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import date, datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
RAW_IMD = PROJECT / "data/ancillary/imd/rainfall_districtwise_daily_imd.csv"
RAW_DISCHARGE = (
    PROJECT
    / "data/ancillary/bihar-surface-water/bihar_sw_daily_discharge_manual.csv"
)
OUTPUT = Path(__file__).with_name("imd_saran_bhojpur_2026-08-19_2026-08-31.csv")
AUDIT = Path(__file__).with_name("official_august_2026_data_audit.md")
FMISC = Path(__file__).with_name("fmisc_gandak_barrage_2026-08-26_2026-08-29.csv")

IMD_URL = (
    "https://nwdp.nwic.gov.in/dataset/9380580f-9bf0-4708-9773-"
    "0cea3cbc595a/resource/8752174f-1d17-4aaf-8058-2eb396f50157/"
    "download/rainfall_districtwise_daily_imd.csv"
)
DISCHARGE_URL = (
    "https://nwdp.nwic.gov.in/dataset/dba020d8-5bfa-44af-88a4-021150cd6847/"
    "resource/71036da3-88ce-4970-8fdd-0456351db972/download/"
    "bihar_sw_daily_discharge_manual.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def require(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run data/ancillary/fetch_official_nwdp.sh first.")


require(RAW_IMD)
require(RAW_DISCHARGE)

imd_rows = read_csv(RAW_IMD)
selected = [
    row
    for row in imd_rows
    if row["State"].strip().upper() == "BIHAR"
    and row["District"].strip().upper() in {"SARAN", "BHOJPUR"}
    and date(2026, 8, 19) <= datetime.strptime(row["Date"].strip(), "%Y-%m-%d").date()
    <= date(2026, 8, 31)
]
selected.sort(key=lambda row: (row["District"], row["Date"]))

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=imd_rows[0].keys())
    writer.writeheader()
    writer.writerows(selected)

all_imd_dates = [
    datetime.strptime(row["Date"].strip(), "%Y-%m-%d").date()
    for row in imd_rows
    if row["State"].strip().upper() == "BIHAR"
    and row["District"].strip().upper() in {"SARAN", "BHOJPUR"}
]
if not all_imd_dates:
    raise SystemExit("No Saran/Bhojpur IMD rows found; cannot build audit.")
discharge_rows = read_csv(RAW_DISCHARGE)
discharge_dates = [
    datetime.strptime(row["Date"].strip(), "%d-%m-%Y").date()
    for row in discharge_rows
]
if not discharge_dates:
    raise SystemExit("No discharge dates found; cannot build audit.")
if not FMISC.is_file():
    raise SystemExit(f"Missing {FMISC}. Run extract_fmisc_gandak_barrage.py first.")
fmisc_rows = read_csv(FMISC)
post_scene = next(
    row
    for row in fmisc_rows
    if row["date_ist"] == "2026-08-27" and row["time_ist"] == "06:00:00"
)
fmisc_totals = [float(row["total_discharge_cusec"]) for row in fmisc_rows]
aug26 = [row for row in selected if row["Date"].strip() == "2026-08-26"]
aug26_values = ", ".join(
    f"{row['District'].title()} {row['Daily Actual'].strip()} mm"
    for row in sorted(aug26, key=lambda row: row["District"])
)

AUDIT.write_text(
    "# Official raw-data audit: Aug 2026 NISAR pair\n\n"
    "Pair: 14 Aug 2026 23:47 UTC to 26 Aug 2026 23:47 UTC. "
    "The mapped hotspot is in Saran, not Arrah/Bhojpur.\n\n"
    "## IMD district rainfall\n\n"
    f"- Source: {IMD_URL}\n"
    f"- Local raw file: `{RAW_IMD.relative_to(PROJECT)}`\n"
    f"- SHA-256: `{sha256(RAW_IMD)}`\n"
    f"- Rows available for Saran/Bhojpur: {min(all_imd_dates)} to {max(all_imd_dates)}.\n"
    "- The raw download has no Saran/Bhojpur observations for 14–18 Aug 2026; "
    "do not fabricate a baseline rainfall value.\n"
    f"- 26 Aug observed daily rainfall: {aug26_values}.\n"
    f"- Derived verbatim subset: `{OUTPUT.name}` ({len(selected)} rows).\n\n"
    "## Manual discharge\n\n"
    f"- Source: {DISCHARGE_URL}\n"
    f"- Local raw file: `{RAW_DISCHARGE.relative_to(PROJECT)}`\n"
    f"- SHA-256: `{sha256(RAW_DISCHARGE)}`\n"
    f"- Available dates: {min(discharge_dates)} to {max(discharge_dates)}.\n"
    "- This is Bihar Water Department manual surface-water discharge, **not CWC**. "
    "It has no 2026 rows and cannot validate this NISAR pair.\n\n"
    "## Date-matched Gandak barrage reports\n\n"
    "- Source index: https://www.fmiscwrdbihar.gov.in/gate/Gatestatus.aspx?g=gandak&y=2026\n"
    "- Raw source PDFs: `data/ancillary/fmisc-gandak-2026/`.\n"
    f"- Extracted range: {min(row['date_ist'] for row in fmisc_rows)} to "
    f"{max(row['date_ist'] for row in fmisc_rows)}; {len(fmisc_rows)} hourly values.\n"
    f"- Total discharge range in those reports: {min(fmisc_totals):g} to "
    f"{max(fmisc_totals):g} cusec.\n"
    "- Nearest published value after the post-scene timestamp (27 Aug 05:17 IST): "
    f"{post_scene['total_discharge_cusec']} cusec at 27 Aug 06:00 IST.\n"
    "- This is a Gandak barrage measurement, not a CWC manual series and not a "
    "spatial flood-confirmation layer for the Saran hotspot.\n\n"
    "## Result\n\n"
    "The verified local IMD readings make a same-day local-rainfall explanation "
    "weak, but they neither prove nor disprove flooding. A date-matched Gandak "
    "gauge/barrage record or a local flood extent remains necessary to validate "
    "the candidate surface-change hotspot.\n",
    encoding="utf-8",
)

print(f"wrote {OUTPUT}")
print(f"wrote {AUDIT}")
