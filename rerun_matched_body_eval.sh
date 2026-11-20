#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
PHASE="${1:-all}"
GPU_ID="${GPU_ID:-1}"

OLD_ROOT="${ROOT}/data/processed/bias_eval_v2"
NEW_ROOT="${ROOT}/data/processed/bias_eval_v2_matched_body"
SELECTED="${ROOT}/data/processed/rewrite_output_bon/bbq_bon_vfull_selected_matched_body.parquet"
OLD_PROMPTS_A="${OLD_ROOT}/prompts/prompts_A_original_mcq.parquet"
OLD_PROMPTS_B="${OLD_ROOT}/prompts/prompts_B_natural_mcq.parquet"
NEW_PROMPTS_A="${NEW_ROOT}/prompts/prompts_A_original_mcq.parquet"
NEW_PROMPTS_B="${NEW_ROOT}/prompts/prompts_B_natural_mcq.parquet"
PRED_DIR="${NEW_ROOT}/raw_predictions"
LOG_DIR="${NEW_ROOT}/logs"

OPEN_MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507|qwen3_4b_instruct_2507|128"
  "google/gemma-3-4b-it|gemma_3_4b_it|128"
  "meta-llama/Llama-3.1-8B-Instruct|llama_3.1_8b_instruct|64"
  "mistralai/Mistral-7B-Instruct-v0.3|mistral_7b_instruct_v0_3|16"
)

API_MODELS=(
  "gpt-4.1|gpt_4_1"
  "gpt-4.1-mini|gpt_4_1_mini"
  "claude-sonnet-5|claude_sonnet_5"
  "claude-haiku-4-5-20251001|claude_haiku_4_5_20251001"
)

MODEL_DIRS=(
  qwen3_4b_instruct_2507
  gemma_3_4b_it
  llama_3.1_8b_instruct
  mistral_7b_instruct_v0_3
  gpt_4_1
  gpt_4_1_mini
  claude_sonnet_5
  claude_haiku_4_5_20251001
)

usage() {
  cat <<'EOF'
Usage: ./rerun_matched_body_eval.sh [phase]

Phases:
  all       Prepare, rerun four open models, rerun four API models, validate,
            and recompute aggregate metrics. This is the default.
  prepare   Build matched-body B prompts and initialize a new output tree.
  open      Run only the four open-weight models on physical GPU 1.
  api       Run only the four proprietary models with Batch API polling.
  finalize  Validate all eight merged outputs and recompute metrics.

Environment overrides:
  GPU_ID=1              Physical GPU for open-weight models (default: 1)
  OPENAI_API_KEY=...    Required for GPT-4.1 and GPT-4.1-mini
  ANTHROPIC_API_KEY=... Required for Claude Sonnet 5 and Claude Haiku 4.5
EOF
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] Missing required file: $1" >&2
    exit 1
  fi
}

copy_if_missing() {
  local source="$1"
  local destination="$2"
  require_file "${source}"
  if [[ ! -f "${destination}" ]]; then
    mkdir -p "$(dirname "${destination}")"
    cp "${source}" "${destination}"
  fi
}

prepare_outputs() {
  require_file "${PYTHON}"
  require_file "${SELECTED}"
  require_file "${OLD_PROMPTS_A}"
  require_file "${OLD_PROMPTS_B}"
  mkdir -p "${NEW_ROOT}/prompts" "${PRED_DIR}" "${LOG_DIR}"

  echo "[prepare] Building the full matched-body condition-B prompt file ..."
  "${PYTHON}" dataset/bbq/scripts/build_bias_eval_prompts.py \
    --input "${SELECTED}" \
    --out-dir "${NEW_ROOT}/prompts" \
    --conditions B_natural_mcq \
    --overwrite

  # Condition A is invariant because candidate reselection changes only the
  # naturalized body. Keep the archived A prompts and predictions unchanged.
  cp "${OLD_PROMPTS_A}" "${NEW_PROMPTS_A}"

  echo "[prepare] Initializing isolated prediction files from archived outputs ..."
  for model_dir in "${MODEL_DIRS[@]}"; do
    old_model_dir="${OLD_ROOT}/raw_predictions/${model_dir}"
    new_model_dir="${PRED_DIR}/${model_dir}"
    mkdir -p "${new_model_dir}"
    copy_if_missing \
      "${old_model_dir}/predictions_A.parquet" \
      "${new_model_dir}/predictions_A.parquet"
    copy_if_missing \
      "${old_model_dir}/predictions_B.parquet" \
      "${new_model_dir}/predictions_B.parquet"
    if [[ -f "${old_model_dir}/predictions_A.manifest.json" ]]; then
      copy_if_missing \
        "${old_model_dir}/predictions_A.manifest.json" \
        "${new_model_dir}/predictions_A.manifest.json"
    fi
    if [[ -f "${old_model_dir}/predictions_B.manifest.json" ]]; then
      copy_if_missing \
        "${old_model_dir}/predictions_B.manifest.json" \
        "${new_model_dir}/predictions_B.manifest.json"
    fi
  done

  OLD_PROMPTS_B="${OLD_PROMPTS_B}" \
  NEW_PROMPTS_B="${NEW_PROMPTS_B}" \
  "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

old = pd.read_parquet(Path(os.environ["OLD_PROMPTS_B"]), columns=["bbq_item_id", "prompt_hash"])
new = pd.read_parquet(Path(os.environ["NEW_PROMPTS_B"]), columns=["bbq_item_id", "prompt_hash"])
merged = old.merge(new, on="bbq_item_id", suffixes=("_old", "_new"), validate="one_to_one")
n_changed = int((merged["prompt_hash_old"] != merged["prompt_hash_new"]).sum())
if len(old) != 58_044 or len(new) != 58_044 or len(merged) != 58_044:
    raise SystemExit(
        f"Prompt coverage mismatch: old={len(old)}, new={len(new)}, paired={len(merged)}"
    )
if n_changed == 0:
    raise SystemExit("No changed B prompt hashes were detected")
print(f"[prepare] Verified {len(new):,} prompts: {n_changed:,} changed, {len(new)-n_changed:,} reusable.")
PY
}

check_cuda() {
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  if ! gpu_name="$(${PYTHON} -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))')"; then
    echo "[error] Physical GPU ${GPU_ID} is unavailable to PyTorch." >&2
    exit 1
  fi
  echo "[open] Physical GPU ${GPU_ID} -> logical cuda:0 (${gpu_name})"
}

run_open_models() {
  prepare_outputs
  check_cuda
  export TOKENIZERS_PARALLELISM=false

  local model model_dir batch_size
  for entry in "${OPEN_MODELS[@]}"; do
    IFS='|' read -r model model_dir batch_size <<< "${entry}"
    echo
    echo "============================================================"
    echo "[open] ${model}"
    echo "============================================================"
    "${PYTHON}" dataset/bbq/scripts/run_bias_eval.py \
      --input "${NEW_PROMPTS_B}" \
      --output "${PRED_DIR}/${model_dir}/predictions_B.parquet" \
      --model "${model}" \
      --batch-size "${batch_size}" \
      --max-new-tokens 16 \
      --temperature 0.0 \
      --top-p 1.0 \
      --resume
  done
}

check_api_keys() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[error] OPENAI_API_KEY is required for the API phase." >&2
    exit 1
  fi
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[error] ANTHROPIC_API_KEY is required for the API phase." >&2
    exit 1
  fi
}

run_api_models() {
  check_api_keys
  prepare_outputs

  local pids=()
  local names=()
  local model model_dir log_path
  for entry in "${API_MODELS[@]}"; do
    IFS='|' read -r model model_dir <<< "${entry}"
    log_path="${LOG_DIR}/${model_dir}_B.log"
    echo "[api] Starting ${model}; log: ${log_path}"
    (
      "${PYTHON}" dataset/bbq/scripts/run_bias_eval_api.py \
        --input "${NEW_PROMPTS_B}" \
        --output "${PRED_DIR}/${model_dir}/predictions_B.parquet" \
        --model "${model}" \
        --max-new-tokens 16 \
        --temperature 0.0 \
        --resume \
        --batch \
        --batch-poll
    ) >"${log_path}" 2>&1 &
    pids+=("$!")
    names+=("${model}")
  done

  local failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[api] Completed: ${names[$i]}"
    else
      echo "[error] Failed: ${names[$i]} (see ${LOG_DIR})" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    exit 1
  fi
}

validate_outputs() {
  MODEL_DIRS="${MODEL_DIRS[*]}" \
  PRED_DIR="${PRED_DIR}" \
  NEW_PROMPTS_B="${NEW_PROMPTS_B}" \
  "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

pred_dir = Path(os.environ["PRED_DIR"])
model_dirs = os.environ["MODEL_DIRS"].split()
prompts = pd.read_parquet(
    Path(os.environ["NEW_PROMPTS_B"]),
    columns=["bbq_item_id", "prompt_hash"],
).rename(columns={"prompt_hash": "expected_prompt_hash"})

errors = []
for model_dir in model_dirs:
    model_path = pred_dir / model_dir
    a_path = model_path / "predictions_A.parquet"
    b_path = model_path / "predictions_B.parquet"
    if not a_path.exists() or not b_path.exists():
        errors.append(f"{model_dir}: missing A or B predictions")
        continue
    a = pd.read_parquet(a_path, columns=["bbq_item_id", "inference_ok"])
    b = pd.read_parquet(
        b_path, columns=["bbq_item_id", "prompt_hash", "inference_ok"]
    )
    paired = prompts.merge(b, on="bbq_item_id", how="left", validate="one_to_one")
    mismatch = int(
        (paired["expected_prompt_hash"] != paired["prompt_hash"]).sum()
    )
    failures = int((~b["inference_ok"]).sum())
    if len(a) != 58_044 or len(b) != 58_044 or mismatch or failures:
        errors.append(
            f"{model_dir}: A={len(a)}, B={len(b)}, "
            f"hash_mismatch={mismatch}, inference_fail={failures}"
        )
    else:
        print(f"[validate] {model_dir}: 58,044 A/B rows, hashes aligned, no failures")

if errors:
    raise SystemExit("Validation failed:\n  " + "\n  ".join(errors))
PY
}

finalize_outputs() {
  validate_outputs
  echo "[finalize] Computing A/B metrics for the eight paper models ..."
  "${PYTHON}" dataset/bbq/scripts/compute_bias_metrics.py \
    --predictions-dir "${PRED_DIR}" \
    --output-dir "${NEW_ROOT}/metrics" \
    --models "${MODEL_DIRS[@]}"
  echo "[finalize] Metrics: ${NEW_ROOT}/metrics"
}

cd "${ROOT}"

case "${PHASE}" in
  all)
    check_api_keys
    run_open_models
    run_api_models
    finalize_outputs
    ;;
  prepare)
    prepare_outputs
    ;;
  open)
    run_open_models
    ;;
  api)
    run_api_models
    ;;
  finalize)
    finalize_outputs
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[error] Unknown phase: ${PHASE}" >&2
    usage >&2
    exit 2
    ;;
esac

echo
echo "Completed phase: ${PHASE}"
echo "Matched-body evaluation root: ${NEW_ROOT}"
