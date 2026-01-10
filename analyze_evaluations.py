#!/usr/bin/env python3
"""
Evaluation Analysis Script
Extracts scores, creates DataFrame, and generates visualizations
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
INPUT_FILE = "test_results_priority_tickers_MIX_rerank>0.3_topk10_evaluated.json"
OUTPUT_DIR = "evaluation_analysis"

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def extract_scores(data):
    """Extract all evaluation scores into a structured format"""
    records = []

    for entry in data:
        if 'evaluation' not in entry:
            continue

        eval_data = entry['evaluation']
        question_id = entry.get('question_id', 'unknown')

        record = {
            'question_id': question_id,
            'question': entry.get('question', '')[:100],  # First 100 chars
            'category': entry.get('category', 'unknown'),
            'expected_type': entry.get('expected_type', 'unknown'),
            'reasoning': entry.get('reasoning', False),
        }

        # Answer Accuracy
        if 'answer_accuracy' in eval_data:
            acc = eval_data['answer_accuracy']
            record['answer_accuracy'] = acc.get('score', None)
            record['answer_accuracy_eval1'] = acc.get('eval1_score', None)
            record['answer_accuracy_eval2'] = acc.get('eval2_score', None)

        # Context Relevance
        if 'context_relevance' in eval_data:
            rel = eval_data['context_relevance']
            record['context_relevance'] = rel.get('score', None)
            record['context_relevance_eval1'] = rel.get('eval1_score', None)
            record['context_relevance_eval2'] = rel.get('eval2_score', None)

        # Groundedness
        if 'groundedness' in eval_data:
            ground = eval_data['groundedness']
            record['groundedness'] = ground.get('score', None)
            record['groundedness_eval1'] = ground.get('eval1_score', None)
            record['groundedness_eval2'] = ground.get('eval2_score', None)

        # Context length
        record['context_length'] = entry.get('context_length', None)

        records.append(record)

    return pd.DataFrame(records)


def create_visualizations(df, output_dir):
    """Generate all visualizations"""
    Path(output_dir).mkdir(exist_ok=True)

    # Main metric columns
    metrics = ['answer_accuracy', 'context_relevance', 'groundedness']

    print("Generating visualizations...")
    print("=" * 80)

    # =========================================================================
    # 1. DISTRIBUTION PLOTS (Histograms)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Score Distributions by Metric', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            data = df[metric].dropna()
            axes[idx].hist(data, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel('Score', fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=13)
            axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
            axes[idx].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 1_distributions.png")
    plt.close()

    # =========================================================================
    # 2. BOX PLOTS (Comparison)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = []
    labels = []
    for metric in metrics:
        if metric in df.columns:
            box_data.append(df[metric].dropna())
            labels.append(metric.replace('_', ' ').title())

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, notch=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score Comparison Across Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_boxplots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 2_boxplots.png")
    plt.close()

    # =========================================================================
    # 3. CORRELATION HEATMAP
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    corr_cols = [m for m in metrics if m in df.columns]
    if len(corr_cols) > 1:
        corr_matrix = df[corr_cols].corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                    xticklabels=[c.replace('_', ' ').title() for c in corr_cols],
                    yticklabels=[c.replace('_', ' ').title() for c in corr_cols],
                    ax=ax)

        ax.set_title('Correlation Between Metrics', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 3_correlation_heatmap.png")
        plt.close()

    # =========================================================================
    # 4. SCATTER PLOTS (Metric Relationships)
    # =========================================================================
    if all(m in df.columns for m in metrics):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Metric Relationships', fontsize=16, fontweight='bold')

        # Accuracy vs Relevance
        axes[0].scatter(df['context_relevance'], df['answer_accuracy'],
                       alpha=0.6, s=50, c='steelblue', edgecolors='black')
        axes[0].set_xlabel('Context Relevance', fontsize=11)
        axes[0].set_ylabel('Answer Accuracy', fontsize=11)
        axes[0].set_title('Accuracy vs Relevance')
        axes[0].grid(alpha=0.3)

        # Accuracy vs Groundedness
        axes[1].scatter(df['groundedness'], df['answer_accuracy'],
                       alpha=0.6, s=50, c='coral', edgecolors='black')
        axes[1].set_xlabel('Groundedness', fontsize=11)
        axes[1].set_ylabel('Answer Accuracy', fontsize=11)
        axes[1].set_title('Accuracy vs Groundedness')
        axes[1].grid(alpha=0.3)

        # Relevance vs Groundedness
        axes[2].scatter(df['context_relevance'], df['groundedness'],
                       alpha=0.6, s=50, c='mediumseagreen', edgecolors='black')
        axes[2].set_xlabel('Context Relevance', fontsize=11)
        axes[2].set_ylabel('Groundedness', fontsize=11)
        axes[2].set_title('Relevance vs Groundedness')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/4_scatter_plots.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 4_scatter_plots.png")
        plt.close()

    # =========================================================================
    # 5. SCORES BY CATEGORY
    # =========================================================================
    if 'category' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Average Scores by Question Category', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(metrics):
            if metric in df.columns:
                category_avg = df.groupby('category')[metric].mean().sort_values()

                axes[idx].barh(range(len(category_avg)), category_avg.values, color='teal', alpha=0.7)
                axes[idx].set_yticks(range(len(category_avg)))
                axes[idx].set_yticklabels(category_avg.index, fontsize=9)
                axes[idx].set_xlabel('Average Score', fontsize=11)
                axes[idx].set_title(metric.replace('_', ' ').title())
                axes[idx].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/5_scores_by_category.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 5_scores_by_category.png")
        plt.close()

    # =========================================================================
    # 6. EVALUATOR AGREEMENT (Eval1 vs Eval2)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Evaluator Agreement (Eval 1 vs Eval 2)', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        eval1_col = f'{metric}_eval1'
        eval2_col = f'{metric}_eval2'

        if eval1_col in df.columns and eval2_col in df.columns:
            eval1 = df[eval1_col].dropna()
            eval2 = df[eval2_col].dropna()

            # Align the data
            common_idx = eval1.index.intersection(eval2.index)
            eval1_aligned = eval1.loc[common_idx]
            eval2_aligned = eval2.loc[common_idx]

            axes[idx].scatter(eval1_aligned, eval2_aligned, alpha=0.6, s=50, edgecolors='black')

            # Perfect agreement line
            min_val = min(eval1_aligned.min(), eval2_aligned.min())
            max_val = max(eval1_aligned.max(), eval2_aligned.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val],
                          'r--', linewidth=2, label='Perfect Agreement')

            axes[idx].set_xlabel('Evaluator 1 Score', fontsize=11)
            axes[idx].set_ylabel('Evaluator 2 Score', fontsize=11)
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/6_evaluator_agreement.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 6_evaluator_agreement.png")
    plt.close()

    # =========================================================================
    # 7. CONTEXT LENGTH vs SCORES
    # =========================================================================
    if 'context_length' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Context Length vs Scores', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(metrics):
            if metric in df.columns:
                valid_data = df[[metric, 'context_length']].dropna()

                axes[idx].scatter(valid_data['context_length'], valid_data[metric],
                                alpha=0.6, s=50, edgecolors='black')
                axes[idx].set_xlabel('Context Length (chars)', fontsize=11)
                axes[idx].set_ylabel('Score', fontsize=11)
                axes[idx].set_title(metric.replace('_', ' ').title())
                axes[idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/7_context_length_vs_scores.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 7_context_length_vs_scores.png")
        plt.close()


def print_summary_stats(df):
    """Print summary statistics"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    metrics = ['answer_accuracy', 'context_relevance', 'groundedness']

    summary_data = []
    for metric in metrics:
        if metric in df.columns:
            data = df[metric].dropna()
            summary_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Count': len(data),
                'Mean': f"{data.mean():.3f}",
                'Std': f"{data.std():.3f}",
                'Min': f"{data.min():.3f}",
                'Q1': f"{data.quantile(0.25):.3f}",
                'Median': f"{data.median():.3f}",
                'Q3': f"{data.quantile(0.75):.3f}",
                'Max': f"{data.max():.3f}",
            })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()

    # Category breakdown
    if 'category' in df.columns:
        print("\n" + "=" * 80)
        print("SCORES BY CATEGORY")
        print("=" * 80)

        for metric in metrics:
            if metric in df.columns:
                print(f"\n{metric.replace('_', ' ').title()}:")
                category_stats = df.groupby('category')[metric].agg(['count', 'mean', 'std']).round(3)
                print(category_stats.to_string())

    # Score distribution
    print("\n" + "=" * 80)
    print("SCORE DISTRIBUTION")
    print("=" * 80)

    for metric in metrics:
        if metric in df.columns:
            data = df[metric].dropna()
            print(f"\n{metric.replace('_', ' ').title()}:")
            bins = [0, 0.25, 0.5, 0.75, 1.0]
            labels = ['0.00-0.25', '0.25-0.50', '0.50-0.75', '0.75-1.00']
            binned = pd.cut(data, bins=bins, labels=labels)
            print(binned.value_counts().sort_index())


def main():
    """Main execution function"""
    print("=" * 80)
    print("EVALUATION ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    print(f"Loading: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")
    entries_with_eval = sum(1 for e in data if 'evaluation' in e)
    print(f"Entries with evaluation: {entries_with_eval}")
    print()

    # Extract scores
    print("Extracting scores...")
    df = extract_scores(data)
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print()

    # Save DataFrame
    csv_file = f"{OUTPUT_DIR}/evaluation_scores.csv"
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved: {csv_file}")
    print()

    # Print stats
    print_summary_stats(df)

    # Generate visualizations
    print("\n" + "=" * 80)
    create_visualizations(df, OUTPUT_DIR)
    print("=" * 80)
    print()
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
