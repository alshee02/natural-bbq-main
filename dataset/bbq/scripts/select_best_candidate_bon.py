"""Select the best-of-N candidate per BBQ item.

Input: output of style_distance_check.py run on all semantically valid
       candidates (already filtered by hard_checks + SBERT).

Selection criterion (unused bank, independent of final evaluation):
  For each BBQ item, pick the candidate with the highest
  ``unused_delta_style`` score.

Final evaluation metrics are read from ``heldout_delta_style`` of the
selected candidates.  The heldout bank was NOT used during selection,
so evaluation results are uncontaminated.

Three-bank separation
---------------------
  Generation  : reference_bank          (1 ref per candidate, from build_rewrite_inputs_bon_v2.py)
  Selection   : reference_bank_unused   (unused_delta_style computed by style_distance_check.py)
  Evaluation  : wildchat_heldout        (heldout_delta_style computed by style_distance_check.py)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BON_INPUTS = PROJECT_ROOT / "data/processed/bbq_rewrite_inputs_bon.parquet"


def parse_candidate_index(rewrite_id: str) -> int:
    """Extract candidate_index from 'bon_{bbq_item_id}_c{idx:02d}'."""
    return int(rewrite_id.split("_c")[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select best-of-N candidate per BBQ item using unused-bank "
            "style score (selection bank, independent of heldout evaluation)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Output of style_distance_check.py: must contain "
            "unused_delta_style and heldout_delta_style columns."
        ),
    )
    parser.add_argument(
        "--retry-input",
        type=Path,
        default=None,
        help=(
            "Output of style_distance_check.py run on fallback-retry candidates "
            "(bon_fallback_retry.py → sbert_content_filter.py → style_distance_check.py). "
            "When provided, these candidates are combined with --input before selection."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bon-inputs",
        type=Path,
        default=DEFAULT_BON_INPUTS,
        help="Original BON inputs parquet (used for coverage reporting).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists. Use --overwrite to replace it."
        )

    # ------------------------------------------------------------------
    # Load initial scored candidates
    # ------------------------------------------------------------------
    df_initial = pd.read_parquet(args.input)

    required = {"bbq_item_id", "unused_delta_style", "heldout_delta_style",
                "heldout_style_sim_original", "heldout_style_sim_natural",
                "heldout_style_check_status", "target_cluster_id"}
    missing = required - set(df_initial.columns)
    if missing:
        raise ValueError(
            f"Input is missing required columns: {sorted(missing)}\n"
            "Run style_distance_check.py with both --heldout-clusters-dir "
            "and --unused-refs-dir before calling this script."
        )

    if "candidate_stage" not in df_initial.columns:
        df_initial["candidate_stage"] = "initial"

    # Recover candidate_index from rewrite_id if not present.
    if "candidate_index" not in df_initial.columns:
        df_initial["candidate_index"] = df_initial["rewrite_id"].apply(
            parse_candidate_index
        )

    # ------------------------------------------------------------------
    # Coverage: items with ≥1 valid candidate before fallback retry
    # ------------------------------------------------------------------
    bon_df: pd.DataFrame | None = None
    n_total_items = df_initial["bbq_item_id"].nunique()
    n_initially_missing = 0
    initially_missing_ids: set = set()

    if args.bon_inputs.exists():
        bon_df = pd.read_parquet(args.bon_inputs)
        n_total_items = bon_df["bbq_item_id"].nunique()
        initially_missing_ids = (
            set(bon_df["bbq_item_id"].astype(str))
            - set(df_initial["bbq_item_id"].astype(str))
        )
        n_initially_missing = len(initially_missing_ids)

    if n_initially_missing > 0:
        print(
            f"[warning] {n_initially_missing}/{n_total_items} items had no valid "
            "candidate after initial pipeline."
        )

    # ------------------------------------------------------------------
    # Load and combine fallback-retry candidates (if provided)
    # ------------------------------------------------------------------
    df_retry: pd.DataFrame | None = None
    n_recovered_by_fallback = 0

    if args.retry_input is not None and args.retry_input.exists():
        df_retry = pd.read_parquet(args.retry_input)
        if not df_retry.empty:
            if "candidate_stage" not in df_retry.columns:
                df_retry["candidate_stage"] = "fallback_retry"
            if "candidate_index" not in df_retry.columns:
                df_retry["candidate_index"] = df_retry["rewrite_id"].apply(
                    parse_candidate_index
                )
            recovered_ids = (
                set(df_retry["bbq_item_id"].astype(str)) & initially_missing_ids
            )
            n_recovered_by_fallback = len(recovered_ids)

    df = df_initial if (df_retry is None or df_retry.empty) else pd.concat(
        [df_initial, df_retry], ignore_index=True
    )

    # ------------------------------------------------------------------
    # Candidate count distribution (after combining)
    # ------------------------------------------------------------------
    n_valid_candidates = len(df)
    n_items_with_candidates = df["bbq_item_id"].nunique()
    n_items_no_candidates = n_total_items - n_items_with_candidates

    candidate_counts = df.groupby("bbq_item_id").size()

    # ------------------------------------------------------------------
    # Select best candidate per item: max unused_delta_style
    # ------------------------------------------------------------------
    best_idx = df.groupby("bbq_item_id")["unused_delta_style"].idxmax()
    selected = df.loc[best_idx].reset_index(drop=True)
    selected["fallback_used"] = selected["candidate_stage"] == "fallback_retry"

    # ------------------------------------------------------------------
    # Evaluation (heldout bank — untouched during selection)
    # ------------------------------------------------------------------
    heldout_pass = (selected["heldout_style_check_status"] == "pass").sum()

    t_stat, t_pval = stats.ttest_rel(
        selected["heldout_style_sim_original"],
        selected["heldout_style_sim_natural"],
    )
    w_stat, w_pval = stats.wilcoxon(
        selected["heldout_style_sim_original"],
        selected["heldout_style_sim_natural"],
    )

    # ------------------------------------------------------------------
    # Per-cluster breakdown
    # ------------------------------------------------------------------
    cluster_stats: dict[str, dict] = {}
    for cid in sorted(selected["target_cluster_id"].unique()):
        sel_sub = selected[selected["target_cluster_id"] == cid]
        all_sub = df[df["target_cluster_id"] == cid]
        n_pass = int((sel_sub["heldout_style_check_status"] == "pass").sum())
        cluster_stats[str(int(cid))] = {
            "n_selected": int(len(sel_sub)),
            "heldout_pass": n_pass,
            "heldout_pass_rate": round(n_pass / len(sel_sub), 4),
            "mean_heldout_delta": round(float(sel_sub["heldout_delta_style"].mean()), 4),
            "mean_valid_candidates": round(
                float(all_sub.groupby("bbq_item_id").size().mean()), 2
            ),
        }

    # ------------------------------------------------------------------
    # Final missing items (after combining initial + fallback retry)
    # ------------------------------------------------------------------
    final_missing_ids: set = set()
    if bon_df is not None:
        final_missing_ids = (
            set(bon_df["bbq_item_id"].astype(str))
            - set(selected["bbq_item_id"].astype(str))
        )
    n_final_missing = len(final_missing_ids)
    final_coverage = (n_total_items - n_final_missing) / n_total_items

    if n_final_missing > 0:
        print(
            f"\n[warning] {n_final_missing}/{n_total_items} items have no valid "
            "candidate after fallback retry (final_fail)."
        )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("=" * 60)
    print(" Best-of-N candidate selection")
    print("=" * 60)
    print(f"Total items (original): {n_total_items}")
    print(f"  Initially missing:          {n_initially_missing}")
    print(f"  Recovered by fallback retry: {n_recovered_by_fallback}")
    print(f"  Final missing (final_fail):  {n_final_missing}")
    print(f"Coverage: {final_coverage:.1%}")
    print(f"Items selected: {n_items_with_candidates}")
    print(f"Total valid candidates (pool): {n_valid_candidates}")
    print()
    print("--- Candidate count per item ---")
    print(candidate_counts.describe().round(2))
    print()
    print("--- Candidate stage of selected ---")
    print(selected["candidate_stage"].value_counts())
    print()
    print("--- Heldout style check (selected candidates) ---")
    print(selected["heldout_style_check_status"].value_counts())
    print(f"pass rate: {heldout_pass}/{len(selected)} "
          f"({heldout_pass/len(selected)*100:.1f}%)")
    print(selected["heldout_delta_style"].describe().round(4))
    print(f"paired t-test: t={t_stat:.4f}, p={t_pval:.4e}")
    print(f"wilcoxon:      W={w_stat:.1f}, p={w_pval:.4e}")
    print()
    print("--- Per-cluster (selected candidates, heldout evaluation) ---")
    for cid, stats_dict in cluster_stats.items():
        print(
            f"  Cluster {cid}: "
            f"pass={stats_dict['heldout_pass']}/{stats_dict['n_selected']} "
            f"({stats_dict['heldout_pass_rate']*100:.0f}%), "
            f"δ={stats_dict['mean_heldout_delta']:+.4f}, "
            f"avg_valid_candidates={stats_dict['mean_valid_candidates']:.1f}"
        )

    # ------------------------------------------------------------------
    # Save selected
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)

    # ------------------------------------------------------------------
    # Save final_failed items (explicit, not silently dropped)
    # ------------------------------------------------------------------
    final_failed_path = args.output.parent / (args.output.stem + "_final_failed.parquet")
    if bon_df is not None and final_missing_ids:
        final_failed_df = (
            bon_df[bon_df["bbq_item_id"].astype(str).isin(final_missing_ids)]
            .drop_duplicates(subset=["bbq_item_id"])
            .reset_index(drop=True)
        )
        final_failed_df.to_parquet(final_failed_path, index=False)
        print(f"\nWrote final_failed items ({n_final_missing}): {final_failed_path}")
    elif bon_df is not None:
        # Write empty file so downstream scripts can always check for it
        bon_df.iloc[:0].to_parquet(final_failed_path, index=False)

    # ------------------------------------------------------------------
    # Build summary
    # ------------------------------------------------------------------
    stage_counts = selected["candidate_stage"].value_counts().to_dict()

    summary: dict = {
        "n_total_items": int(n_total_items),
        # --- coverage ---
        "n_items_initially_missing": int(n_initially_missing),
        "n_items_recovered_by_fallback": int(n_recovered_by_fallback),
        "n_items_final_missing": int(n_final_missing),
        "final_coverage": round(float(final_coverage), 4),
        # --- selection pool ---
        "n_valid_candidates": int(n_valid_candidates),
        "n_selected": int(len(selected)),
        "selected_candidate_stage_counts": {
            str(k): int(v) for k, v in stage_counts.items()
        },
        "candidate_counts_per_item": {
            "mean": round(float(candidate_counts.mean()), 2),
            "std": round(float(candidate_counts.std()), 2),
            "min": int(candidate_counts.min()),
            "max": int(candidate_counts.max()),
        },
        # --- heldout evaluation ---
        "heldout": {
            "pass": int(heldout_pass),
            "fail": int(len(selected) - heldout_pass),
            "pass_rate": round(heldout_pass / len(selected), 4),
            "mean_delta": round(float(selected["heldout_delta_style"].mean()), 4),
            "std_delta": round(float(selected["heldout_delta_style"].std()), 4),
            "t_stat": round(float(t_stat), 4),
            "t_pval": float(t_pval),
            "wilcoxon_W": float(w_stat),
            "wilcoxon_pval": float(w_pval),
        },
        "per_cluster": cluster_stats,
        "selection_bank": "reference_bank_unused_by_cluster (unused_delta_style)",
        "evaluation_bank": "wildchat_heldout_by_cluster (heldout_delta_style)",
    }

    # Final missing breakdown (for bias auditing)
    if bon_df is not None and final_missing_ids:
        fm = bon_df[bon_df["bbq_item_id"].astype(str).isin(final_missing_ids)]
        if "target_cluster_id" in fm.columns:
            summary["final_missing_by_cluster"] = {
                str(int(k)): int(v)
                for k, v in fm["target_cluster_id"].value_counts().sort_index().items()
            }
        if "category" in fm.columns:
            summary["final_missing_by_category"] = {
                str(k): int(v)
                for k, v in fm["category"].value_counts().items()
            }
        if "context_condition" in fm.columns:
            summary["final_missing_by_context_condition"] = {
                str(k): int(v)
                for k, v in fm["context_condition"].value_counts().items()
            }

    # Back-compat alias
    summary["n_items_with_candidates"] = int(n_items_with_candidates)
    summary["n_items_no_candidates"] = int(n_items_no_candidates)

    summary_path = args.output.parent / (args.output.stem + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nWrote: {args.output}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
