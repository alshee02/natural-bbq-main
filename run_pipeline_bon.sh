#!/usr/bin/env bash
# Best-of-N rewrite pipeline (BON)
#
# Strategy: generate N candidates per BBQ item (one per reference),
# filter semantically invalid candidates, then select the candidate
# with the highest style score on the unused reference bank.
# Final evaluation uses the independent heldout cluster bank.
#
# Three-bank separation:
#   Generation : reference_bank           (1 ref per candidate)
#   Selection  : reference_bank_unused    (unused_delta_style)
#   Evaluation : wildchat_heldout         (heldout_delta_style)
#
# Rewrite uses OpenAI Batch API (Steps 2–4, ~50% cheaper than standard API):
#   Step 2: prepare  — inputs parquet → batch JSONL
#   Step 3: submit   — upload JSONL, create batch → batch_meta.json
#   Step 4: collect  — poll until done, download results → parquet
#
# Fallback retry policy (Steps 8–11):
#   Items where ALL N candidates fail the filter pipeline are retried once
#   through a second Batch API request, then filtered and scored identically.
#   Items that still have 0 valid candidates → final_fail
#   (saved explicitly to *_selected_final_failed.parquet; never silently dropped).
#
# Usage:
#   bash run_pipeline_bon.sh

set -euo pipefail

RUN_VERSION="vfull"
N_CANDIDATES=7      # references per item
MODEL="gpt-5.4-mini"
STYLE_MODE="maxref"   # style similarity mode: "mean" or "maxref"

PYTHON=".venv/bin/python"
BBQ_SCRIPTS="dataset/bbq/scripts"
REWRITE_OUTPUT="data/processed/rewrite_output_bon"
HELDOUT_CLUSTERS="data/processed/wildchat_heldout_by_cluster"
UNUSED_REFS="data/processed/reference_bank_unused_by_cluster"

BON_INPUTS="data/processed/bbq_rewrite_inputs_bon_${RUN_VERSION}.parquet"
EXCLUDE_IDS="data/processed/bbq_rewrite_inputs_bon_v10_bon.parquet"  # prompt development set (132 items)
BATCH_JSONL="${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_batch.jsonl"
BATCH_META="${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_batch_meta.json"
REWRITE_OUTPUTS="${REWRITE_OUTPUT}/bbq_bon_outputs_${RUN_VERSION}.parquet"

mkdir -p "${REWRITE_OUTPUT}"

echo "============================================================"
echo " BON Pipeline  |  version=${RUN_VERSION}"
echo " items=ALL  candidates=${N_CANDIDATES}"
echo " model=${MODEL}  api=batch"
echo "============================================================"


# ---------------------------------------------------------------------------
# Step 1: Build BON rewrite inputs (1 reference per candidate row)
# ---------------------------------------------------------------------------
echo ""
echo "[1/12] Building BON rewrite inputs..."
EXCLUDE_IDS_ARG=""
[ -n "${EXCLUDE_IDS}" ] && EXCLUDE_IDS_ARG="--exclude-ids ${EXCLUDE_IDS}"
$PYTHON $BBQ_SCRIPTS/build_rewrite_inputs_bon_v2.py \
    --output       ${BON_INPUTS} \
    --n-candidates ${N_CANDIDATES} \
    ${EXCLUDE_IDS_ARG}


# ---------------------------------------------------------------------------
# Step 2: Prepare Batch API JSONL
# ---------------------------------------------------------------------------
echo ""
echo "[2/12] Preparing batch JSONL..."
$PYTHON $BBQ_SCRIPTS/prepare_batch_rewrite_bon.py \
    --input    ${BON_INPUTS} \
    --output   ${BATCH_JSONL} \
    --model    ${MODEL} \
    --overwrite


# ---------------------------------------------------------------------------
# Step 3: Submit batch to OpenAI Batch API
# ---------------------------------------------------------------------------
echo ""
echo "[3/12] Submitting batch..."
$PYTHON $BBQ_SCRIPTS/submit_batch_rewrite_bon.py \
    --input      ${BATCH_JSONL} \
    --batch-meta ${BATCH_META} \
    --overwrite


# ---------------------------------------------------------------------------
# Step 4: Poll and collect batch results (blocks until batch completes)
# ---------------------------------------------------------------------------
echo ""
echo "[4/12] Collecting batch results (polling every 60s)..."
$PYTHON $BBQ_SCRIPTS/collect_batch_rewrite_bon.py \
    --batch-meta ${BATCH_META} \
    --inputs     ${BON_INPUTS} \
    --output     ${REWRITE_OUTPUTS} \
    --overwrite


# ---------------------------------------------------------------------------
# Step 5: Hard checks
# ---------------------------------------------------------------------------
echo ""
echo "[5/12] Hard checks..."
$PYTHON $BBQ_SCRIPTS/hard_checks.py \
    --input  ${REWRITE_OUTPUTS} \
    --output ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_hard_checks.parquet


# ---------------------------------------------------------------------------
# Step 6: SBERT content filter
# ---------------------------------------------------------------------------
echo ""
echo "[6/12] SBERT content filter..."
$PYTHON $BBQ_SCRIPTS/sbert_content_filter.py \
    --input  ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_hard_checks.parquet \
    --output ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_sbert_filtered.parquet


# ---------------------------------------------------------------------------
# Step 7: Style distance — unused (selection) + heldout (eval)
# ---------------------------------------------------------------------------
echo ""
echo "[7/12] Style distance (unused + heldout)..."
$PYTHON $BBQ_SCRIPTS/style_distance_check.py \
    --input                ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_sbert_filtered.parquet \
    --output               ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_style_scored.parquet \
    --unused-refs-dir      ${UNUSED_REFS} \
    --heldout-clusters-dir ${HELDOUT_CLUSTERS} \
    --mode                 ${STYLE_MODE}


# ---------------------------------------------------------------------------
# Step 8: Fallback retry — identify missing items, build JSONL, submit batch
# ---------------------------------------------------------------------------
echo ""
echo "[8/12] Fallback retry: identifying missing items and submitting batch..."
FALLBACK_BATCH_JSONL="${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_fallback_batch.jsonl"
FALLBACK_BATCH_META="${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_fallback_batch_meta.json"
FALLBACK_CANDIDATES="${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_fallback_candidates.parquet"
$PYTHON $BBQ_SCRIPTS/bon_fallback_retry.py \
    --bon-inputs   ${BON_INPUTS} \
    --hard-checked ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_hard_checks.parquet \
    --style-scored ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_style_scored.parquet \
    --batch-jsonl  ${FALLBACK_BATCH_JSONL} \
    --batch-meta   ${FALLBACK_BATCH_META} \
    --candidates   ${FALLBACK_CANDIDATES} \
    --model        ${MODEL} \
    --overwrite


# ---------------------------------------------------------------------------
# Step 9: Collect fallback batch results + re-run hard checks
# ---------------------------------------------------------------------------
echo ""
echo "[9/12] Collecting fallback batch results (polling every 60s)..."
FALLBACK_RETRIED="${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_fallback_retried.parquet"
$PYTHON $BBQ_SCRIPTS/collect_batch_fallback_bon.py \
    --batch-meta  ${FALLBACK_BATCH_META} \
    --candidates  ${FALLBACK_CANDIDATES} \
    --output      ${FALLBACK_RETRIED} \
    --overwrite


# ---------------------------------------------------------------------------
# Steps 10–11: Filter fallback retry candidates (only if any were retried)
# ---------------------------------------------------------------------------
FALLBACK_ROWS=$($PYTHON -c "import pandas as pd; print(len(pd.read_parquet('${FALLBACK_RETRIED}')))")

RETRY_STYLE_SCORED_ARG=""
if [ "${FALLBACK_ROWS}" -gt 0 ]; then
    echo ""
    echo "[10/12] SBERT content filter (fallback retry candidates)..."
    $PYTHON $BBQ_SCRIPTS/sbert_content_filter.py \
        --input  ${FALLBACK_RETRIED} \
        --output ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_fallback_sbert.parquet

    echo ""
    echo "[11/12] Style distance (fallback retry candidates)..."
    $PYTHON $BBQ_SCRIPTS/style_distance_check.py \
        --input                ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_fallback_sbert.parquet \
        --output               ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_fallback_style_scored.parquet \
        --unused-refs-dir      ${UNUSED_REFS} \
        --heldout-clusters-dir ${HELDOUT_CLUSTERS} \
        --mode                 ${STYLE_MODE}

    RETRY_STYLE_SCORED_ARG="--retry-input ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_fallback_style_scored.parquet"
else
    echo ""
    echo "[10/12] Skipping SBERT filter (no fallback retry candidates)."
    echo "[11/12] Skipping style distance (no fallback retry candidates)."
fi


# ---------------------------------------------------------------------------
# Step 12: Select best candidate per item (by unused_delta_style)
#   Combines initial + fallback retry valid candidates.
#   Final missing items saved to *_selected_final_failed.parquet.
# ---------------------------------------------------------------------------
echo ""
echo "[12/12] Selecting best-of-${N_CANDIDATES} candidates..."
$PYTHON $BBQ_SCRIPTS/select_best_candidate_bon.py \
    --input      ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_style_scored.parquet \
    --output     ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_selected.parquet \
    --bon-inputs ${BON_INPUTS} \
    ${RETRY_STYLE_SCORED_ARG} \
    --overwrite


echo ""
echo "============================================================"
echo " BON Pipeline complete  |  version=${RUN_VERSION}"
echo " Selected:     ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_selected.parquet"
echo " Final failed: ${REWRITE_OUTPUT}/bbq_bon_${RUN_VERSION}_selected_final_failed.parquet"
echo "============================================================"
