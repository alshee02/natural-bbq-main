#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

OUTPUT_DIR="${PROJECT_ROOT}/data/processed/rewrite_output_bon"
BON_INPUTS="${PROJECT_ROOT}/data/processed/bbq_rewrite_inputs_bon_vfull.parquet"
INITIAL_INPUT="${OUTPUT_DIR}/bbq_bon_vfull_sbert_filtered.parquet"
FALLBACK_INPUT="${OUTPUT_DIR}/bbq_bon_vfull_fallback_sbert.parquet"
INITIAL_SCORED="${OUTPUT_DIR}/bbq_bon_vfull_style_scored_matched_body.parquet"
FALLBACK_SCORED="${OUTPUT_DIR}/bbq_bon_vfull_fallback_style_scored_matched_body.parquet"
OLD_SELECTED="${OUTPUT_DIR}/bbq_bon_vfull_selected.parquet"
NEW_SELECTED="${OUTPUT_DIR}/bbq_bon_vfull_selected_matched_body.parquet"
CHANGED_SELECTED="${OUTPUT_DIR}/bbq_bon_vfull_selected_matched_body_changed.parquet"
COMPARISON_JSON="${OUTPUT_DIR}/bbq_bon_vfull_selected_matched_body.comparison.json"
LOG_FILE="${OUTPUT_DIR}/recompute_matched_body_style.log"

THREADS="${THREADS:-$(nproc)}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-128}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-2048}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-8192}"
GPU_ID="${GPU_ID:-1}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

trap 'echo "[error] Failed at line ${LINENO}. See ${LOG_FILE}."' ERR

for required_file in "${PYTHON}" "${BON_INPUTS}" "${INITIAL_INPUT}" "${FALLBACK_INPUT}"; do
  if [[ ! -e "${required_file}" ]]; then
    echo "[error] Missing required file: ${required_file}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ "${DEVICE}" == cuda* ]]; then
  if ! GPU_NAME="$(${PYTHON} -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))')"; then
    echo "[error] Physical GPU ${GPU_ID} is not available to PyTorch." >&2
    echo "[error] Check: CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} -c 'import torch; print(torch.cuda.is_available())'" >&2
    exit 1
  fi
else
  GPU_NAME="not used"
fi

export OPENBLAS_NUM_THREADS="${THREADS}"
export OMP_NUM_THREADS="${THREADS}"

echo "============================================================"
echo "Matched-body StyleDistance recomputation"
echo "============================================================"
echo "Project root:          ${PROJECT_ROOT}"
echo "Physical GPU:          ${GPU_ID}"
echo "CUDA_VISIBLE_DEVICES:  ${CUDA_VISIBLE_DEVICES}"
echo "CUDA logical device:   ${DEVICE} (physical GPU ${GPU_ID})"
echo "GPU name:              ${GPU_NAME}"
echo "CPU/BLAS threads:      ${THREADS}"
echo "Encode batch size:     ${ENCODE_BATCH_SIZE}"
echo "Query block size:      ${QUERY_BATCH_SIZE}"
echo "Reference block size:  ${REFERENCE_BATCH_SIZE}"
echo "Log:                   ${LOG_FILE}"
echo
echo "Original body: original_context + original_question"
echo "Natural body:  natural_question"
echo "Answer options are excluded from both sides."
echo

cd "${PROJECT_ROOT}"

"${PYTHON}" dataset/bbq/scripts/style_distance_check.py \
  --input "${INITIAL_INPUT}" \
  --output "${INITIAL_SCORED}" \
  --retry-input "${FALLBACK_INPUT}" \
  --retry-output "${FALLBACK_SCORED}" \
  --mode maxref \
  --device "${DEVICE}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE}" \
  --query-batch-size "${QUERY_BATCH_SIZE}" \
  --reference-batch-size "${REFERENCE_BATCH_SIZE}"

"${PYTHON}" dataset/bbq/scripts/select_best_candidate_bon.py \
  --input "${INITIAL_SCORED}" \
  --retry-input "${FALLBACK_SCORED}" \
  --bon-inputs "${BON_INPUTS}" \
  --output "${NEW_SELECTED}" \
  --overwrite

OLD_SELECTED="${OLD_SELECTED}" \
NEW_SELECTED="${NEW_SELECTED}" \
CHANGED_SELECTED="${CHANGED_SELECTED}" \
COMPARISON_JSON="${COMPARISON_JSON}" \
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

old_path = Path(os.environ["OLD_SELECTED"])
new_path = Path(os.environ["NEW_SELECTED"])
changed_path = Path(os.environ["CHANGED_SELECTED"])
comparison_path = Path(os.environ["COMPARISON_JSON"])

new = pd.read_parquet(new_path)
if not old_path.exists():
    new.to_parquet(changed_path, index=False)
    comparison = {
        "old_selected_found": False,
        "n_new_selected": int(len(new)),
        "n_requiring_evaluation": int(len(new)),
    }
else:
    old = pd.read_parquet(old_path)
    key = "rewrite_id" if "rewrite_id" in old.columns and "rewrite_id" in new.columns else "natural_question"
    old_keys = old[["bbq_item_id", key]].rename(columns={key: "old_candidate"})
    new_keys = new[["bbq_item_id", key]].rename(columns={key: "new_candidate"})
    merged = old_keys.merge(new_keys, on="bbq_item_id", how="outer", indicator=True)

    changed_mask = (merged["_merge"] != "both") | (
        merged["old_candidate"] != merged["new_candidate"]
    )
    changed_ids = set(merged.loc[changed_mask, "bbq_item_id"])
    changed = new[new["bbq_item_id"].isin(changed_ids)].copy()
    changed.to_parquet(changed_path, index=False)

    n_same = int((~changed_mask).sum())
    n_changed = int(changed_mask.sum())
    comparison = {
        "old_selected_found": True,
        "comparison_key": key,
        "n_old_selected": int(len(old)),
        "n_new_selected": int(len(new)),
        "n_unchanged": n_same,
        "n_changed_or_added": n_changed,
        "changed_rate_over_new": round(n_changed / len(new), 6),
        "n_requiring_evaluation": int(len(changed)),
    }

comparison_path.write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("\n============================================================")
print("Selection comparison")
print("============================================================")
print(json.dumps(comparison, indent=2, sort_keys=True))
print(f"Changed selected rows: {changed_path}")
print(f"Comparison report:     {comparison_path}")
PY

echo
echo "Matched-body recomputation completed."
echo "Selected dataset: ${NEW_SELECTED}"
echo "Summary:          ${NEW_SELECTED%.parquet}.summary.json"
echo "Changed subset:   ${CHANGED_SELECTED}"
