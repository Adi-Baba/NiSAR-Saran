#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_DIR="${1:-$PROJECT_ROOT/data/source/New}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/analysis/modular-math/hotspot-artifacts}"

"$PYTHON_BIN" "$PROJECT_ROOT/analysis/modular-math/hotspot_overlay.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"
