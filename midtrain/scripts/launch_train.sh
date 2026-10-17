#!/usr/bin/env bash
# Entry point kept for existing job scripts; the logic lives in launch_stage.py.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "${here}/launch_stage.py" "$@"
