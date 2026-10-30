#!/usr/bin/env bash
set -euo pipefail

# Resolve from this script rather than from the caller's current directory so both
# relative input/output paths in the Python scripts refer to the repository root.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# The OSMnx run writes resumable rows and must finish before statistics can read them.
/usr/bin/time -v uv run python analysis/pnas_v2/calculate_treeness_osmnx.py
# Statistics reads the completed OSMnx CSV and then writes the three revision outputs.
uv run python analysis/pnas_v2/calculate_revision_statistics.py
# Print the prepared summary only after both production steps succeed.
cat analysis/data/pnas-v2-summary.txt
