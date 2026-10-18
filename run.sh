#!/usr/bin/env bash
set -Eeuo pipefail

# One documented entry point for the CRS-only reproduction path.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_dir/scripts/run_revised_crs_all.sh" "$@"
