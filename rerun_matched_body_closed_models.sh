#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# OPENAI_API_KEY and ANTHROPIC_API_KEY must already be exported.
# The shared runner submits all four proprietary models through Batch APIs
# and polls until their changed condition-B predictions are merged.
exec "${ROOT}/rerun_matched_body_eval.sh" api
