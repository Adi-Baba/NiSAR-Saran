#!/usr/bin/env python3
"""Extract total discharge values from downloaded FMISC Gandak barrage PDFs.

The reports are source PDFs. `pdftotext -layout` is used only to create a
machine-readable copy of the final, explicitly headed Total Discharge (Cusec)
column; it does not infer missing values.
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
INPUT = PROJECT / "data/ancillary/fmisc-gandak-2026"
OUTPUT = Path(__file__).with_name("fmisc_gandak_barrage_2026-08-26_2026-08-29.csv")
PROVENANCE = Path(__file__).with_name(
    "fmisc_gandak_barrage_2026-08-26_2026-08-29_provenance.tsv"
)
INDEX_URL = "https://www.fmiscwrdbihar.gov.in/gate/Gatestatus.aspx?g=gandak&y=2026"
ROW = re.compile(r"^\s*(\d{2}-\d{2}-\d{4})\s+(\d{2}:\d{2}:\d{2})\s+(.*)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def report_url(pdf: Path) -> str:
    words = pdf.stem.replace("_", " ")
    return (
        "https://www.fmiscwrdbihar.gov.in/gate/gate//GATE2026/GANDAK/AUG-26/"
        f"{words.replace(' ', '%20')}.PDF"
    )


pdfs = sorted(INPUT.glob("*_AUG_2026_AT_*.pdf"))
if len(pdfs) != 8:
    raise SystemExit(f"Expected eight source PDFs in {INPUT}; found {len(pdfs)}.")

observations: list[dict[str, str]] = []
with PROVENANCE.open("w", newline="", encoding="utf-8") as provenance:
    writer = csv.DictWriter(
        provenance,
        fieldnames=["source_pdf", "source_url", "sha256", "extraction_method"],
        delimiter="\t",
    )
    writer.writeheader()
    for pdf in pdfs:
        text = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        writer.writerow(
            {
                "source_pdf": pdf.name,
                "source_url": report_url(pdf),
                "sha256": sha256(pdf),
                "extraction_method": "pdftotext -layout; final numeric token of each dated row",
            }
        )
        for line in text.splitlines():
            match = ROW.match(line)
            if not match:
                continue
            day, time_ist, rest = match.groups()
            values = rest.split()
            # The Total Discharge (Cusec) column is the final value in every
            # source row. Other columns vary in count, so they are deliberately
            # not parsed as stable fields.
            try:
                total_cusec = float(values[-1])
            except (IndexError, ValueError) as error:
                raise ValueError(f"Cannot read total discharge in {pdf.name}: {line}") from error
            observations.append(
                {
                    "date_ist": "-".join(reversed(day.split("-"))),
                    "time_ist": time_ist,
                    "total_discharge_cusec": f"{total_cusec:g}",
                    "source_pdf": pdf.name,
                    "source_url": report_url(pdf),
                }
            )

observations.sort(key=lambda row: (row["date_ist"], row["time_ist"]))
if len(observations) != 40:
    raise SystemExit(f"Expected 40 hourly observations; extracted {len(observations)}.")

with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=observations[0].keys())
    writer.writeheader()
    writer.writerows(observations)

print(f"wrote {OUTPUT}")
print(f"wrote {PROVENANCE}")
print(f"source index: {INDEX_URL}")
