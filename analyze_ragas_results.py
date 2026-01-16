#!/usr/bin/env python3
"""
RAGAS Results Analysis Script
Aggregates all *_evaled.json files into CSV formats for analysis and plotting
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


# Configuration
INPUT_FOLDER = "5_modes_question_wise_results/5_modes_question_wise_results_priority_tickers_ALL"
OUTPUT_PREFIX = "ragas_analysis"


def load_evaluation_files(folder_path: Path) -> List[Dict[str, Any]]:
    """
    Load all *_evaled.json files from the specified folder.

    Args:
        folder_path: Path to folder containing evaluation files

    Returns:
        List of evaluation data dictionaries
    """
    eval_files = sorted(folder_path.glob("*_evaled.json"))

    if not eval_files:
        print(f"⚠ No evaluation files found in {folder_path}")
        return []

    print(f"Found {len(eval_files)} evaluation files")

    data = []
    for file_path in eval_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        except Exception as e:
            print(f"⚠ Error loading {file_path.name}: {e}")
            continue

    return data


def create_consolidated_dataframe(eval_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a consolidated DataFrame with one row per question-mode combination.

    Columns: question_id, mode, category, context_recall, context_precision,
             faithfulness, answer_correctness, question
    """
    rows = []

    for item in eval_data:
        question_id = item.get("question_id", "unknown")
        question = item.get("question", "")
        category = item.get("category", "")

        evaluations = item.get("ragas_evaluations", {})

        for mode_name, metrics in evaluations.items():
            row = {
                "question_id": question_id,
                "mode": mode_name,
                "category": category,
                "question": question
            }

            # Add metric scores
            for metric_name in ["context_recall", "context_precision", "faithfulness", "answer_correctness"]:
                if metric_name in metrics:
                    row[metric_name] = metrics[metric_name].get("score")
                    row[f"{metric_name}_reason"] = metrics[metric_name].get("reason", "")
                else:
                    row[metric_name] = None
                    row[f"{metric_name}_reason"] = ""

            rows.append(row)

    df = pd.DataFrame(rows)

    # Convert numeric columns to float
    numeric_cols = ["context_recall", "context_precision", "faithfulness", "answer_correctness"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics DataFrame grouped by mode and metric.

    Returns DataFrame with: mode, metric, count, mean, std, min, q25, median, q75, max
    """
    metric_cols = ["context_recall", "context_precision", "faithfulness", "answer_correctness"]

    summary_rows = []

    for mode in df["mode"].unique():
        mode_df = df[df["mode"] == mode]

        for metric in metric_cols:
            if metric not in mode_df.columns:
                continue

            values = mode_df[metric].dropna()

            if len(values) == 0:
                continue

            row = {
                "mode": mode,
                "metric": metric,
                "count": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "q25": values.quantile(0.25),
                "median": values.median(),
                "q75": values.quantile(0.75),
                "max": values.max()
            }

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Round numeric columns
    numeric_cols = ["mean", "std", "min", "q25", "median", "q75", "max"]
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(4)

    return summary_df


def create_mode_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create mode comparison DataFrame with metrics as rows and modes as columns.

    Returns pivot table: metrics × modes with mean scores
    """
    metric_cols = ["context_recall", "context_precision", "faithfulness", "answer_correctness"]

    comparison_rows = []

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        row = {"metric": metric}

        for mode in ["local", "global", "naive", "hybrid", "mix"]:
            mode_values = df[df["mode"] == mode][metric].dropna()
            if len(mode_values) > 0:
                row[mode] = round(mode_values.mean(), 4)
            else:
                row[mode] = None

        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def create_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create category breakdown with average scores per category and mode.

    Returns DataFrame: category, mode, metric averages
    """
    metric_cols = ["context_recall", "context_precision", "faithfulness", "answer_correctness"]

    breakdown_rows = []

    for category in df["category"].unique():
        for mode in df["mode"].unique():
            subset = df[(df["category"] == category) & (df["mode"] == mode)]

            if len(subset) == 0:
                continue

            row = {
                "category": category,
                "mode": mode,
                "count": len(subset)
            }

            for metric in metric_cols:
                if metric in subset.columns:
                    values = subset[metric].dropna()
                    if len(values) > 0:
                        row[f"{metric}_mean"] = round(values.mean(), 4)
                    else:
                        row[f"{metric}_mean"] = None

            breakdown_rows.append(row)

    return pd.DataFrame(breakdown_rows)


def create_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create correlation matrix between all metrics.
    """
    metric_cols = ["context_recall", "context_precision", "faithfulness", "answer_correctness"]

    # Get only numeric columns
    numeric_df = df[metric_cols].dropna()

    if len(numeric_df) == 0:
        print("⚠ No data available for correlation matrix")
        return pd.DataFrame()

    corr_matrix = numeric_df.corr().round(4)
    return corr_matrix


def print_quick_summary(df: pd.DataFrame):
    """Print a quick summary to console."""
    print("\n" + "="*70)
    print("QUICK SUMMARY")
    print("="*70)

    print(f"\nTotal evaluations: {len(df)}")
    print(f"Questions evaluated: {df['question_id'].nunique()}")
    print(f"Modes: {df['mode'].nunique()} ({', '.join(sorted(df['mode'].unique()))})")
    print(f"Categories: {df['category'].nunique()}")

    print("\n--- Average Scores by Mode ---")
    metric_cols = ["context_recall", "context_precision", "faithfulness", "answer_correctness"]

    for mode in sorted(df["mode"].unique()):
        mode_df = df[df["mode"] == mode]
        print(f"\n{mode.upper()}:")

        for metric in metric_cols:
            if metric in mode_df.columns:
                values = mode_df[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    print(f"  {metric:20s}: {mean_val:.4f} ± {std_val:.4f}")

    print("\n--- Best Performing Mode per Metric ---")
    for metric in metric_cols:
        if metric not in df.columns:
            continue

        mode_means = df.groupby("mode")[metric].mean()
        best_mode = mode_means.idxmax()
        best_score = mode_means.max()

        if pd.notna(best_score):
            print(f"  {metric:20s}: {best_mode} ({best_score:.4f})")

    print("="*70)


def main():
    """Main execution function."""
    print("="*70)
    print("RAGAS Results Analysis")
    print("="*70)

    # Load evaluation files
    input_path = Path(INPUT_FOLDER)
    if not input_path.exists():
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}")
        return

    print(f"\nLoading evaluation files from: {INPUT_FOLDER}")
    eval_data = load_evaluation_files(input_path)

    if not eval_data:
        print("ERROR: No evaluation data loaded")
        return

    print(f"✓ Loaded {len(eval_data)} evaluation files")

    # Create consolidated DataFrame
    print("\nCreating consolidated DataFrame...")
    df_all = create_consolidated_dataframe(eval_data)
    print(f"✓ Created DataFrame with {len(df_all)} rows")

    # Print quick summary
    print_quick_summary(df_all)

    # Save consolidated results
    print("\n--- Saving Output Files ---")

    # 1. Full results with scores only (no reasoning text)
    df_scores = df_all.drop(columns=[col for col in df_all.columns if col.endswith("_reason")])
    output_file = f"{OUTPUT_PREFIX}_all_results.csv"
    df_scores.to_csv(output_file, index=False)
    print(f"✓ {output_file} ({len(df_scores)} rows)")

    # 2. Full results with reasoning (larger file)
    output_file_full = f"{OUTPUT_PREFIX}_all_results_with_reasons.csv"
    df_all.to_csv(output_file_full, index=False)
    print(f"✓ {output_file_full} ({len(df_all)} rows)")

    # 3. Summary statistics
    print("\nCreating summary statistics...")
    df_summary = create_summary_statistics(df_all)
    output_file = f"{OUTPUT_PREFIX}_summary_stats.csv"
    df_summary.to_csv(output_file, index=False)
    print(f"✓ {output_file} ({len(df_summary)} rows)")

    # 4. Mode comparison
    print("\nCreating mode comparison...")
    df_comparison = create_mode_comparison(df_all)
    output_file = f"{OUTPUT_PREFIX}_mode_comparison.csv"
    df_comparison.to_csv(output_file, index=False)
    print(f"✓ {output_file} ({len(df_comparison)} rows)")

    # 5. Category breakdown
    print("\nCreating category breakdown...")
    df_category = create_category_breakdown(df_all)
    output_file = f"{OUTPUT_PREFIX}_category_breakdown.csv"
    df_category.to_csv(output_file, index=False)
    print(f"✓ {output_file} ({len(df_category)} rows)")

    # 6. Correlation matrix
    print("\nCreating correlation matrix...")
    df_corr = create_correlation_matrix(df_all)
    if not df_corr.empty:
        output_file = f"{OUTPUT_PREFIX}_correlation_matrix.csv"
        df_corr.to_csv(output_file)
        print(f"✓ {output_file}")

    # Generate metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_evaluations": len(df_all),
        "unique_questions": df_all["question_id"].nunique(),
        "modes": sorted(df_all["mode"].unique()),
        "categories": sorted(df_all["category"].unique()),
        "metrics": ["context_recall", "context_precision", "faithfulness", "answer_correctness"],
        "output_files": [
            f"{OUTPUT_PREFIX}_all_results.csv",
            f"{OUTPUT_PREFIX}_all_results_with_reasons.csv",
            f"{OUTPUT_PREFIX}_summary_stats.csv",
            f"{OUTPUT_PREFIX}_mode_comparison.csv",
            f"{OUTPUT_PREFIX}_category_breakdown.csv",
            f"{OUTPUT_PREFIX}_correlation_matrix.csv"
        ]
    }

    with open(f"{OUTPUT_PREFIX}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ {OUTPUT_PREFIX}_metadata.json")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nOutput files are ready for plotting and statistical analysis!")
    print("\nExample usage:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_csv('{OUTPUT_PREFIX}_all_results.csv')")
    print(f"  df.groupby('mode')['faithfulness'].mean()")
    print("="*70)


if __name__ == "__main__":
    main()
