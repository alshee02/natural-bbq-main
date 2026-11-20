#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Physical GPU 1 is the default. Override with, for example, GPU_ID=0.
export GPU_ID="${GPU_ID:-1}"

exec "${ROOT}/rerun_matched_body_eval.sh" open
