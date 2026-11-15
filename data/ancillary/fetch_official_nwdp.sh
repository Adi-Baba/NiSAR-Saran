#!/usr/bin/env bash
# Fetch public NWDP source files used by the NISAR Saran/Gandak validation.
# Existing files are protected by default. Re-run with --force to replace them.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
force=false
if [[ "${1:-}" == "--force" ]]; then
  force=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--force]" >&2
  exit 2
fi

fetch() {
  local url="$1" destination="$2"
  if [[ -e "$destination" && "$force" != true ]]; then
    echo "exists; keeping $destination (use --force to replace)"
    return
  fi
  local temporary="${destination}.part"
  trap 'rm -f "$temporary"' RETURN
  curl --fail --location --retry 3 --connect-timeout 15 --max-time 180 \
    "$url" --output "$temporary"
  mv "$temporary" "$destination"
  trap - RETURN
  sha256sum "$destination"
}

mkdir -p "$root_dir/imd" "$root_dir/bihar-surface-water"

# IMD observed district-wise daily rainfall. This source currently includes
# Saran and Bhojpur records beginning on 2026-08-19.
fetch \
  "https://nwdp.nwic.gov.in/dataset/9380580f-9bf0-4708-9773-0cea3cbc595a/resource/8752174f-1d17-4aaf-8058-2eb396f50157/download/rainfall_districtwise_daily_imd.csv" \
  "$root_dir/imd/rainfall_districtwise_daily_imd.csv"

# This is Bihar Water Department manual surface-water discharge, not the
# separate CWC package. Its current public file ends 2025-05-31, so it is
# historical context only for the Aug-2026 NISAR pair.
fetch \
  "https://nwdp.nwic.gov.in/dataset/dba020d8-5bfa-44af-88a4-021150cd6847/resource/71036da3-88ce-4970-8fdd-0456351db972/download/bihar_sw_daily_discharge_manual.csv" \
  "$root_dir/bihar-surface-water/bihar_sw_daily_discharge_manual.csv"
