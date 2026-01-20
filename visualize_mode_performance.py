#!/usr/bin/env python3
"""
Visualize mode performance data with nice graphs.
Converts character lengths to tokens and creates comprehensive visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_analysis_data():
    """Load the analysis results from JSON file."""
    with open('mode_performance_analysis.json', 'r') as f:
        return json.load(f)

def create_processing_time_chart(summary_data):
    """Create bar chart for processing times."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']

    avg_times = []
    median_times = []

    for mode in modes:
        if mode in summary_data:
            avg_times.append(summary_data[mode].get('avg_retrieval_time', 0))
            median_times.append(summary_data[mode].get('median_retrieval_time', 0))
        else:
            avg_times.append(0)
            median_times.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(modes))
    width = 0.35

    bars1 = ax.bar(x - width/2, avg_times, width, label='Average', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, median_times, width, label='Median', color='#e74c3c', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}s',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=9)

    ax.set_xlabel('Mode', fontweight='bold')
    ax.set_ylabel('Processing Time (seconds)', fontweight='bold')
    ax.set_title('Processing Time by Mode', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_token_consumption_chart(summary_data):
    """Create stacked bar chart for token consumption."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

    input_tokens = []
    output_tokens = []

    for mode in modes:
        if mode in summary_data:
            input_tokens.append(summary_data[mode].get('avg_input_tokens', 0))
            output_tokens.append(summary_data[mode].get('avg_output_tokens', 0))
        else:
            input_tokens.append(0)
            output_tokens.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Stacked bar chart
    x = np.arange(len(modes))

    bars1 = ax1.bar(x, input_tokens, label='Input Tokens', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x, output_tokens, bottom=input_tokens, label='Output Tokens', color='#e74c3c', alpha=0.8)

    # Add total values on top
    for i, (inp, out) in enumerate(zip(input_tokens, output_tokens)):
        total = inp + out
        ax1.text(i, total + 500, f'{total:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_xlabel('Mode', fontweight='bold')
    ax1.set_ylabel('Number of Tokens', fontweight='bold')
    ax1.set_title('Token Consumption by Mode (Stacked)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Format y-axis with thousands separator
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Pie chart for total tokens
    total_tokens = [inp + out for inp, out in zip(input_tokens, output_tokens)]

    # Create pie chart with better colors and formatting
    wedges, texts, autotexts = ax2.pie(total_tokens, labels=mode_labels, colors=colors,
                                        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(total_tokens)):,})',
                                        startangle=90)

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    ax2.set_title('Token Distribution Across Modes', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

def create_efficiency_chart(summary_data):
    """Create efficiency chart comparing tokens vs time."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, mode in enumerate(modes):
        if mode in summary_data:
            time = summary_data[mode].get('avg_retrieval_time', 0)
            tokens = summary_data[mode].get('avg_total_tokens', 0)

            # Create scatter plot with different sizes based on output tokens
            output_tokens = summary_data[mode].get('avg_output_tokens', 0)
            size = 100 + (output_tokens / 10)  # Scale size based on output tokens

            ax.scatter(time, tokens, s=size, color=colors[i], alpha=0.7,
                      edgecolors='black', linewidth=2, label=mode_labels[i])

            # Add mode label next to point
            ax.annotate(mode_labels[i], (time, tokens),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')

    ax.set_xlabel('Average Processing Time (seconds)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Total Tokens', fontweight='bold', fontsize=12)
    ax.set_title('Mode Efficiency: Processing Time vs Token Consumption', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', title='Mode')

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Add a note about bubble size
    ax.text(0.02, 0.98, 'Bubble size represents output tokens',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', style='italic')

    plt.tight_layout()
    return fig

def create_detailed_comparison_chart(detailed_results):
    """Create box plots for detailed distribution analysis."""
    # Extract data for each mode from detailed results
    mode_data = {
        'local': {'times': [], 'tokens': []},
        'global': {'times': [], 'tokens': []},
        'naive': {'times': [], 'tokens': []},
        'hybrid': {'times': [], 'tokens': []},
        'mix': {'times': [], 'tokens': []}
    }

    for result in detailed_results:
        if result:
            for mode in mode_data.keys():
                if mode in result:
                    if result[mode]['retrieval_time'] is not None:
                        mode_data[mode]['times'].append(result[mode]['retrieval_time'])
                    mode_data[mode]['tokens'].append(result[mode]['total_tokens'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Box plot for processing times
    time_data = [mode_data[mode]['times'] for mode in ['local', 'global', 'naive', 'hybrid', 'mix']]
    time_data_filtered = [d for d in time_data if len(d) > 0]  # Filter out empty lists

    bp1 = ax1.boxplot(time_data_filtered, labels=['Local', 'Global', 'Naive', 'Hybrid', 'Mix'][:len(time_data_filtered)],
                      patch_artist=True, notch=True, showmeans=True)

    # Color the box plots
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Processing Time (seconds)', fontweight='bold')
    ax1.set_title('Distribution of Processing Times by Mode', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Box plot for token consumption
    token_data = [mode_data[mode]['tokens'] for mode in ['local', 'global', 'naive', 'hybrid', 'mix']]

    bp2 = ax2.boxplot(token_data, labels=['Local', 'Global', 'Naive', 'Hybrid', 'Mix'],
                      patch_artist=True, notch=True, showmeans=True)

    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('Total Tokens', fontweight='bold')
    ax2.set_title('Distribution of Token Consumption by Mode', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    plt.tight_layout()
    return fig

def create_summary_table(summary_data):
    """Create a summary table visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    columns = ['Mode', 'Avg Time (s)', 'Med Time (s)', 'Input Tokens', 'Output Tokens', 'Total Tokens']
    rows = []

    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']

    for mode, label in zip(modes, mode_labels):
        if mode in summary_data:
            data = summary_data[mode]
            row = [
                label,
                f"{data.get('avg_retrieval_time', 0):.1f}",
                f"{data.get('median_retrieval_time', 0):.1f}",
                f"{data.get('avg_input_tokens', 0):,.0f}",
                f"{data.get('avg_output_tokens', 0):,.0f}",
                f"{data.get('avg_total_tokens', 0):,.0f}"
            ]
            rows.append(row)

    # Create table
    table = ax.table(cellText=rows, colLabels=columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.12, 0.15, 0.15, 0.18, 0.18, 0.18])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    colors = ['#ecf0f1', '#ffffff']
    for i, row in enumerate(rows, 1):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(colors[i % 2])

    # Highlight best values
    # Find best (minimum) time
    times = [float(row[1]) for row in rows if float(row[1]) > 0]
    if times:
        min_time_idx = times.index(min(times)) + 1
        table[(min_time_idx, 1)].set_facecolor('#2ecc71')
        table[(min_time_idx, 1)].set_text_props(weight='bold')

    # Find best (minimum) total tokens
    tokens = [float(row[5].replace(',', '')) for row in rows]
    min_token_idx = tokens.index(min(tokens)) + 1
    table[(min_token_idx, 5)].set_facecolor('#2ecc71')
    table[(min_token_idx, 5)].set_text_props(weight='bold')

    ax.set_title('Performance Summary Table\n(Green highlights indicate best performance)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig

def main():
    """Main function to create all visualizations."""
    print("Loading analysis data...")
    data = load_analysis_data()

    summary = data['summary']
    detailed = data['detailed_results']

    print("Creating visualizations...")

    # Create individual charts
    fig1 = create_processing_time_chart(summary)
    fig1.savefig('mode_processing_times.png', dpi=300, bbox_inches='tight')
    print("  ✓ Processing time chart saved to mode_processing_times.png")

    fig2 = create_token_consumption_chart(summary)
    fig2.savefig('mode_token_consumption.png', dpi=300, bbox_inches='tight')
    print("  ✓ Token consumption chart saved to mode_token_consumption.png")

    fig3 = create_efficiency_chart(summary)
    fig3.savefig('mode_efficiency.png', dpi=300, bbox_inches='tight')
    print("  ✓ Efficiency chart saved to mode_efficiency.png")

    fig4 = create_detailed_comparison_chart(detailed)
    fig4.savefig('mode_distributions.png', dpi=300, bbox_inches='tight')
    print("  ✓ Distribution charts saved to mode_distributions.png")

    fig5 = create_summary_table(summary)
    fig5.savefig('mode_summary_table.png', dpi=300, bbox_inches='tight')
    print("  ✓ Summary table saved to mode_summary_table.png")

    # Create combined dashboard
    print("\nCreating combined dashboard...")
    fig_dashboard = plt.figure(figsize=(20, 24))

    # Add all plots to dashboard
    gs = fig_dashboard.add_gridspec(6, 2, hspace=0.3, wspace=0.25)

    # Processing time chart
    ax1 = fig_dashboard.add_subplot(gs[0, :])
    create_processing_time_subplot(ax1, summary)

    # Token consumption charts
    ax2 = fig_dashboard.add_subplot(gs[1, 0])
    ax3 = fig_dashboard.add_subplot(gs[1, 1])
    create_token_consumption_subplots(ax2, ax3, summary)

    # Efficiency chart
    ax4 = fig_dashboard.add_subplot(gs[2, :])
    create_efficiency_subplot(ax4, summary)

    # Distribution charts
    ax5 = fig_dashboard.add_subplot(gs[3:5, :])
    create_distribution_subplot(ax5, detailed)

    # Summary table
    ax6 = fig_dashboard.add_subplot(gs[5, :])
    create_table_subplot(ax6, summary)

    fig_dashboard.suptitle('LightRAG Mode Performance Analysis Dashboard',
                          fontsize=20, fontweight='bold', y=0.995)

    fig_dashboard.savefig('mode_performance_dashboard.png', dpi=300, bbox_inches='tight')
    print("  ✓ Combined dashboard saved to mode_performance_dashboard.png")

    # Show all plots
    plt.show()

    print("\n" + "="*60)
    print("All visualizations have been created successfully!")
    print("Files generated:")
    print("  1. mode_processing_times.png")
    print("  2. mode_token_consumption.png")
    print("  3. mode_efficiency.png")
    print("  4. mode_distributions.png")
    print("  5. mode_summary_table.png")
    print("  6. mode_performance_dashboard.png")

# Helper functions for dashboard subplots
def create_processing_time_subplot(ax, summary_data):
    """Create processing time chart as subplot."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']

    avg_times = [summary_data.get(mode, {}).get('avg_retrieval_time', 0) for mode in modes]
    median_times = [summary_data.get(mode, {}).get('median_retrieval_time', 0) for mode in modes]

    x = np.arange(len(modes))
    width = 0.35

    bars1 = ax.bar(x - width/2, avg_times, width, label='Average', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, median_times, width, label='Median', color='#e74c3c', alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Mode', fontweight='bold')
    ax.set_ylabel('Processing Time (seconds)', fontweight='bold')
    ax.set_title('Processing Time by Mode', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_token_consumption_subplots(ax1, ax2, summary_data):
    """Create token consumption charts as subplots."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

    input_tokens = [summary_data.get(mode, {}).get('avg_input_tokens', 0) for mode in modes]
    output_tokens = [summary_data.get(mode, {}).get('avg_output_tokens', 0) for mode in modes]

    x = np.arange(len(modes))

    bars1 = ax1.bar(x, input_tokens, label='Input Tokens', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x, output_tokens, bottom=input_tokens, label='Output Tokens', color='#e74c3c', alpha=0.8)

    for i, (inp, out) in enumerate(zip(input_tokens, output_tokens)):
        total = inp + out
        ax1.text(i, total + 500, f'{total:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_xlabel('Mode', fontweight='bold')
    ax1.set_ylabel('Number of Tokens', fontweight='bold')
    ax1.set_title('Token Consumption (Stacked)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Pie chart
    total_tokens = [inp + out for inp, out in zip(input_tokens, output_tokens)]
    wedges, texts, autotexts = ax2.pie(total_tokens, labels=mode_labels, colors=colors,
                                        autopct=lambda pct: f'{pct:.1f}%',
                                        startangle=90)

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    ax2.set_title('Token Distribution', fontsize=12, fontweight='bold')

def create_efficiency_subplot(ax, summary_data):
    """Create efficiency chart as subplot."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

    for i, mode in enumerate(modes):
        if mode in summary_data:
            time = summary_data[mode].get('avg_retrieval_time', 0)
            tokens = summary_data[mode].get('avg_total_tokens', 0)
            output_tokens = summary_data[mode].get('avg_output_tokens', 0)
            size = 100 + (output_tokens / 10)

            ax.scatter(time, tokens, s=size, color=colors[i], alpha=0.7,
                      edgecolors='black', linewidth=2, label=mode_labels[i])
            ax.annotate(mode_labels[i], (time, tokens),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')

    ax.set_xlabel('Average Processing Time (seconds)', fontweight='bold')
    ax.set_ylabel('Average Total Tokens', fontweight='bold')
    ax.set_title('Mode Efficiency: Processing Time vs Token Consumption', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

def create_distribution_subplot(ax, detailed_results):
    """Create distribution chart as subplot."""
    mode_data = {
        'local': {'tokens': []},
        'global': {'tokens': []},
        'naive': {'tokens': []},
        'hybrid': {'tokens': []},
        'mix': {'tokens': []}
    }

    for result in detailed_results:
        if result:
            for mode in mode_data.keys():
                if mode in result:
                    mode_data[mode]['tokens'].append(result[mode]['total_tokens'])

    token_data = [mode_data[mode]['tokens'] for mode in ['local', 'global', 'naive', 'hybrid', 'mix']]
    bp = ax.boxplot(token_data, labels=['Local', 'Global', 'Naive', 'Hybrid', 'Mix'],
                    patch_artist=True, notch=True, showmeans=True)

    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Total Tokens', fontweight='bold')
    ax.set_title('Distribution of Token Consumption by Mode', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

def create_table_subplot(ax, summary_data):
    """Create table as subplot."""
    ax.axis('tight')
    ax.axis('off')

    columns = ['Mode', 'Avg Time (s)', 'Input Tokens', 'Output Tokens', 'Total Tokens']
    rows = []

    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']

    for mode, label in zip(modes, mode_labels):
        if mode in summary_data:
            data = summary_data[mode]
            row = [
                label,
                f"{data.get('avg_retrieval_time', 0):.1f}",
                f"{data.get('avg_input_tokens', 0):,.0f}",
                f"{data.get('avg_output_tokens', 0):,.0f}",
                f"{data.get('avg_total_tokens', 0):,.0f}"
            ]
            rows.append(row)

    table = ax.table(cellText=rows, colLabels=columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

if __name__ == "__main__":
    main()