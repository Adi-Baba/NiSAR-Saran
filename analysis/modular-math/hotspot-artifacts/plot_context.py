#!/usr/bin/env python3
"""Plot the dated hydro/imagery context around the NISAR pair."""
from pathlib import Path
import csv
from datetime import datetime
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "august_context_timeseries.tsv").open(), delimiter="\t"))
dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in rows]
lo = [float(r["gandak_inflow_min_cumecs"]) if r["gandak_inflow_min_cumecs"] != "NA" else float("nan") for r in rows]
hi = [float(r["gandak_inflow_max_cumecs"]) if r["gandak_inflow_max_cumecs"] != "NA" else float("nan") for r in rows]
fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
ax.fill_between(dates, lo, hi, color="#4c78a8", alpha=.25, label="Gandak barrage inflow range (cumecs)")
ax.plot(dates, lo, "o-", color="#4c78a8", ms=4, label="min")
ax.plot(dates, hi, "o-", color="#1f4e79", ms=4, label="max")
ax.axvline(datetime(2026, 8, 14, 23, 47), color="#2ca02c", ls="--", label="NISAR before")
ax.axvline(datetime(2026, 8, 26, 23, 47), color="#d62728", ls="--", label="NISAR after")
ax.scatter([datetime(2026, 8, 26)], [114.5 * 35], marker="*", s=130, color="#e6550d", zorder=5, label="DHM max rainfall 114.5 mm (scaled ×35)")
ax.set_title("August 2026 context overlay — NISAR pair, Gandak inflow, rainfall, imagery")
ax.set_ylabel("Gandak barrage inflow (cumecs)\n(rainfall star scaled only for visibility)")
ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8)
fig.autofmt_xdate(); fig.savefig(HERE / "august_context_overlay.png", dpi=180)
print(HERE / "august_context_overlay.png")
