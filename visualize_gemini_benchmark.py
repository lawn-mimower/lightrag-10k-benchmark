#!/usr/bin/env python3
"""
Visualize Gemini API benchmark results.
Creates graphs for response times and token usage analysis.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import statistics

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_benchmark_data(file_path: str = None):
    """Load benchmark results from JSON file."""
    if file_path is None:
        # Find the most recent benchmark file
        benchmark_files = list(Path('.').glob('gemini_benchmark_*.json'))
        if not benchmark_files:
            print("Error: No benchmark result files found.")
            print("Please run benchmark_gemini_api.py (full or --quick) first.")
            sys.exit(1)

        # Sort by modification time and get the most recent
        file_path = max(benchmark_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading most recent benchmark file: {file_path}")

    with open(file_path, 'r') as f:
        return json.load(f)

def create_average_response_time_chart(data):
    """Create bar chart showing average response times per mode."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

    # Extract statistics
    avg_times = []
    error_bars = []  # For std deviation

    if 'statistics' in data:
        # Full benchmark format
        stats = data['statistics']
        for mode in modes:
            if mode in stats and 'response_time' in stats[mode]:
                avg_times.append(stats[mode]['response_time']['mean'])
                error_bars.append(stats[mode]['response_time'].get('stdev', 0))
            else:
                avg_times.append(0)
                error_bars.append(0)
    elif 'summary' in data:
        # Quick benchmark format
        stats = data['summary']
        for mode in modes:
            if mode in stats and 'avg_time' in stats[mode]:
                avg_times.append(stats[mode]['avg_time'])
                error_bars.append(0)  # No stdev in quick format
            else:
                avg_times.append(0)
                error_bars.append(0)
    else:
        print("Warning: Unexpected data format")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(modes))
    bars = ax.bar(x_pos, avg_times, yerr=error_bars if any(error_bars) else None,
                   color=colors, alpha=0.8, capsize=5)

    # Add value labels on bars
    for bar, time in zip(bars, avg_times):
        if time > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s',
                    ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Mode', fontweight='bold')
    ax.set_ylabel('Average Response Time (seconds)', fontweight='bold')
    ax.set_title('Gemini API Average Response Times by Mode', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mode_labels)
    ax.grid(True, alpha=0.3, axis='y')

    # Add horizontal line for mean across all modes
    overall_mean = np.mean([t for t in avg_times if t > 0])
    ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.5,
               label=f'Overall Average: {overall_mean:.2f}s')
    ax.legend()

    plt.tight_layout()
    return fig

def create_response_time_vs_tokens_scatter(data):
    """Create scatter plot of response time vs total tokens for all API calls."""

    # Collect all individual data points
    all_points = []

    if 'detailed_results' in data:
        # Full benchmark format
        results = data['detailed_results']
    elif 'raw_results' in data:
        # Quick benchmark format
        results = data['raw_results']
    else:
        print("Warning: No detailed/raw results found")
        return None

    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    colors = {'local': '#3498db', 'global': '#2ecc71', 'naive': '#f39c12',
              'hybrid': '#9b59b6', 'mix': '#e74c3c'}

    for mode in modes:
        if mode not in results:
            continue

        for result in results[mode]:
            if result.get('success', False):
                point = {
                    'mode': mode,
                    'response_time': result.get('response_time') or result.get('time', 0),
                    'total_tokens': result.get('total_tokens',
                                              result.get('input_tokens', 0) + result.get('output_tokens', 0)),
                    'input_tokens': result.get('input_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0)
                }
                if point['response_time'] > 0 and point['total_tokens'] > 0:
                    all_points.append(point)

    if not all_points:
        print("Warning: No valid data points found for scatter plot")
        return None

    print(f"Found {len(all_points)} data points for scatter plot")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: All points colored by mode
    for mode in modes:
        mode_points = [p for p in all_points if p['mode'] == mode]
        if mode_points:
            x = [p['total_tokens'] for p in mode_points]
            y = [p['response_time'] for p in mode_points]
            ax1.scatter(x, y, label=mode.capitalize(), color=colors[mode],
                       alpha=0.6, s=100, edgecolors='black', linewidth=1)

    ax1.set_xlabel('Total Tokens (Input + Output)', fontweight='bold')
    ax1.set_ylabel('Response Time (seconds)', fontweight='bold')
    ax1.set_title(f'API Response Time vs Token Usage ({len(all_points)} calls)',
                  fontsize=12, fontweight='bold')
    ax1.legend(title='Mode')
    ax1.grid(True, alpha=0.3)

    # Add trend line
    x_all = [p['total_tokens'] for p in all_points]
    y_all = [p['response_time'] for p in all_points]
    z = np.polyfit(x_all, y_all, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(x_all), max(x_all), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.5, label=f'Trend: {z[0]:.6f}x + {z[1]:.2f}')

    # Add correlation coefficient
    correlation = np.corrcoef(x_all, y_all)[0, 1]
    ax1.text(0.02, 0.98, f'Correlation: {correlation:.3f}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 2: Box plot by mode
    mode_data = []
    mode_labels = []

    for mode in modes:
        mode_points = [p['response_time'] for p in all_points if p['mode'] == mode]
        if mode_points:
            mode_data.append(mode_points)
            mode_labels.append(f"{mode.capitalize()}\n(n={len(mode_points)})")

    bp = ax2.boxplot(mode_data, labels=mode_labels, patch_artist=True, notch=True)

    # Color the box plots
    for patch, color in zip(bp['boxes'], [colors[m] for m in modes if any(p['mode'] == m for p in all_points)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('Response Time (seconds)', fontweight='bold')
    ax2.set_title('Response Time Distribution by Mode', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

def create_token_breakdown_chart(data):
    """Create stacked bar chart showing input vs output token breakdown."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']

    input_tokens = []
    output_tokens = []

    if 'statistics' in data:
        # Full benchmark format
        stats = data['statistics']
        for mode in modes:
            if mode in stats and 'input_tokens' in stats[mode]:
                input_tokens.append(stats[mode]['input_tokens']['mean'])
                output_tokens.append(stats[mode]['output_tokens']['mean'])
            else:
                input_tokens.append(0)
                output_tokens.append(0)
    elif 'summary' in data:
        # Quick benchmark format
        stats = data['summary']
        for mode in modes:
            if mode in stats:
                input_tokens.append(stats[mode].get('avg_input_tokens', 0))
                output_tokens.append(stats[mode].get('avg_output_tokens', 0))
            else:
                input_tokens.append(0)
                output_tokens.append(0)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Stacked bar chart
    x_pos = np.arange(len(modes))
    p1 = ax1.bar(x_pos, input_tokens, color='#3498db', alpha=0.8, label='Input Tokens')
    p2 = ax1.bar(x_pos, output_tokens, bottom=input_tokens, color='#e74c3c',
                 alpha=0.8, label='Output Tokens')

    # Add total values on top
    for i, (inp, out) in enumerate(zip(input_tokens, output_tokens)):
        total = inp + out
        if total > 0:
            ax1.text(i, total + 500, f'{total:,.0f}', ha='center',
                    va='bottom', fontweight='bold')

    ax1.set_xlabel('Mode', fontweight='bold')
    ax1.set_ylabel('Average Tokens', fontweight='bold')
    ax1.set_title('Token Usage Breakdown by Mode', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(mode_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Pie chart for total token distribution
    total_tokens = [inp + out for inp, out in zip(input_tokens, output_tokens)]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

    # Filter out zero values
    non_zero_data = [(label, tokens, color) for label, tokens, color
                     in zip(mode_labels, total_tokens, colors) if tokens > 0]

    if non_zero_data:
        labels, tokens, colors = zip(*non_zero_data)

        wedges, texts, autotexts = ax2.pie(tokens, labels=labels, colors=colors,
                                            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(tokens)):,})',
                                            startangle=90)

        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
            autotext.set_color('white')

    ax2.set_title('Token Distribution Across Modes', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig

def create_efficiency_analysis(data):
    """Create efficiency analysis chart (tokens per second, cost analysis)."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

    # Calculate tokens per second and estimated cost
    tokens_per_second = []
    estimated_costs = []

    # Gemini pricing (approximate)
    INPUT_COST_PER_1M = 0.075  # $0.075 per 1M input tokens
    OUTPUT_COST_PER_1M = 0.30  # $0.30 per 1M output tokens

    if 'statistics' in data:
        stats = data['statistics']
        for mode in modes:
            if mode in stats and 'response_time' in stats[mode]:
                avg_time = stats[mode]['response_time']['mean']
                avg_total_tokens = stats[mode]['total_tokens']['mean']
                avg_input = stats[mode]['input_tokens']['mean']
                avg_output = stats[mode]['output_tokens']['mean']

                if avg_time > 0:
                    tokens_per_second.append(avg_total_tokens / avg_time)
                else:
                    tokens_per_second.append(0)

                # Calculate cost per query
                cost = (avg_input / 1_000_000 * INPUT_COST_PER_1M +
                       avg_output / 1_000_000 * OUTPUT_COST_PER_1M)
                estimated_costs.append(cost * 1000)  # Convert to cost per 1000 queries
            else:
                tokens_per_second.append(0)
                estimated_costs.append(0)
    elif 'summary' in data:
        stats = data['summary']
        for mode in modes:
            if mode in stats:
                avg_time = stats[mode].get('avg_time', 1)
                avg_total = stats[mode].get('avg_total_tokens', 0)
                avg_input = stats[mode].get('avg_input_tokens', 0)
                avg_output = stats[mode].get('avg_output_tokens', 0)

                if avg_time > 0:
                    tokens_per_second.append(avg_total / avg_time)
                else:
                    tokens_per_second.append(0)

                cost = (avg_input / 1_000_000 * INPUT_COST_PER_1M +
                       avg_output / 1_000_000 * OUTPUT_COST_PER_1M)
                estimated_costs.append(cost * 1000)
            else:
                tokens_per_second.append(0)
                estimated_costs.append(0)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Tokens per second
    x_pos = np.arange(len(modes))
    bars1 = ax1.bar(x_pos, tokens_per_second, color=colors, alpha=0.8)

    for bar, tps in zip(bars1, tokens_per_second):
        if tps > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{tps:,.0f}',
                    ha='center', va='bottom', fontweight='bold')

    ax1.set_xlabel('Mode', fontweight='bold')
    ax1.set_ylabel('Tokens per Second', fontweight='bold')
    ax1.set_title('Processing Efficiency (Tokens/Second)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(mode_labels)
    ax1.grid(True, alpha=0.3, axis='y')

    # Cost analysis
    bars2 = ax2.bar(x_pos, estimated_costs, color=colors, alpha=0.8)

    for bar, cost in zip(bars2, estimated_costs):
        if cost > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${cost:.2f}',
                    ha='center', va='bottom', fontweight='bold')

    ax2.set_xlabel('Mode', fontweight='bold')
    ax2.set_ylabel('Estimated Cost (USD)', fontweight='bold')
    ax2.set_title('Estimated Cost per 1000 Queries', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(mode_labels)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add note about pricing
    fig.text(0.5, 0.02, 'Note: Cost estimates based on Gemini-3-Flash pricing ($0.075/1M input, $0.30/1M output)',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    return fig

def create_summary_dashboard(data):
    """Create a comprehensive dashboard with all visualizations."""
    fig = plt.figure(figsize=(20, 16))

    # Create grid
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)

    # Average response times
    ax1 = fig.add_subplot(gs[0, :])
    create_average_response_time_subplot(ax1, data)

    # Scatter plot
    ax2 = fig.add_subplot(gs[1, :])
    create_scatter_subplot(ax2, data)

    # Token breakdown
    ax3 = fig.add_subplot(gs[2, 0])
    create_token_breakdown_subplot(ax3, data)

    # Efficiency
    ax4 = fig.add_subplot(gs[2, 1])
    create_efficiency_subplot(ax4, data)

    # Summary statistics table
    ax5 = fig.add_subplot(gs[3, :])
    create_summary_table(ax5, data)

    # Add title and metadata
    config = data.get('configuration', data.get('config', {}))
    timestamp = config.get('timestamp', 'Unknown')
    sample_size = config.get('sample_size', config.get('samples', 'Unknown'))

    fig.suptitle(f'Gemini-3-Flash-Preview Benchmark Results\n'
                 f'Samples: {sample_size} | Date: {timestamp[:19] if len(timestamp) > 19 else timestamp}',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig

# Helper functions for dashboard subplots
def create_average_response_time_subplot(ax, data):
    """Create response time chart as subplot."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

    avg_times = []

    if 'statistics' in data:
        stats = data['statistics']
        for mode in modes:
            if mode in stats and 'response_time' in stats[mode]:
                avg_times.append(stats[mode]['response_time']['mean'])
            else:
                avg_times.append(0)
    elif 'summary' in data:
        stats = data['summary']
        for mode in modes:
            if mode in stats:
                avg_times.append(stats[mode].get('avg_time', 0))
            else:
                avg_times.append(0)

    x_pos = np.arange(len(modes))
    bars = ax.bar(x_pos, avg_times, color=colors, alpha=0.8)

    for bar, time in zip(bars, avg_times):
        if time > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Mode', fontweight='bold')
    ax.set_ylabel('Average Response Time (seconds)', fontweight='bold')
    ax.set_title('Average API Response Times', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mode_labels)
    ax.grid(True, alpha=0.3, axis='y')

def create_scatter_subplot(ax, data):
    """Create scatter plot as subplot."""
    all_points = []

    if 'detailed_results' in data:
        results = data['detailed_results']
    elif 'raw_results' in data:
        results = data['raw_results']
    else:
        return

    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    colors = {'local': '#3498db', 'global': '#2ecc71', 'naive': '#f39c12',
              'hybrid': '#9b59b6', 'mix': '#e74c3c'}

    for mode in modes:
        if mode not in results:
            continue

        for result in results[mode]:
            if result.get('success', False):
                point = {
                    'mode': mode,
                    'response_time': result.get('response_time') or result.get('time', 0),
                    'total_tokens': result.get('total_tokens',
                                              result.get('input_tokens', 0) + result.get('output_tokens', 0))
                }
                if point['response_time'] > 0 and point['total_tokens'] > 0:
                    all_points.append(point)

    for mode in modes:
        mode_points = [p for p in all_points if p['mode'] == mode]
        if mode_points:
            x = [p['total_tokens'] for p in mode_points]
            y = [p['response_time'] for p in mode_points]
            ax.scatter(x, y, label=mode.capitalize(), color=colors[mode],
                      alpha=0.6, s=80, edgecolors='black', linewidth=1)

    ax.set_xlabel('Total Tokens', fontweight='bold')
    ax.set_ylabel('Response Time (seconds)', fontweight='bold')
    ax.set_title(f'Response Time vs Token Usage ({len(all_points)} data points)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

def create_token_breakdown_subplot(ax, data):
    """Create token breakdown as subplot."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']

    input_tokens = []
    output_tokens = []

    if 'statistics' in data:
        stats = data['statistics']
        for mode in modes:
            if mode in stats:
                input_tokens.append(stats[mode].get('input_tokens', {}).get('mean', 0))
                output_tokens.append(stats[mode].get('output_tokens', {}).get('mean', 0))
            else:
                input_tokens.append(0)
                output_tokens.append(0)
    elif 'summary' in data:
        stats = data['summary']
        for mode in modes:
            if mode in stats:
                input_tokens.append(stats[mode].get('avg_input_tokens', 0))
                output_tokens.append(stats[mode].get('avg_output_tokens', 0))
            else:
                input_tokens.append(0)
                output_tokens.append(0)

    x_pos = np.arange(len(modes))
    p1 = ax.bar(x_pos, input_tokens, color='#3498db', alpha=0.8, label='Input')
    p2 = ax.bar(x_pos, output_tokens, bottom=input_tokens, color='#e74c3c',
                alpha=0.8, label='Output')

    ax.set_xlabel('Mode', fontweight='bold')
    ax.set_ylabel('Average Tokens', fontweight='bold')
    ax.set_title('Token Usage Breakdown', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m[:3] for m in mode_labels])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

def create_efficiency_subplot(ax, data):
    """Create efficiency metrics as subplot."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    mode_labels = ['Local', 'Global', 'Naive', 'Hybrid', 'Mix']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']

    tokens_per_second = []

    if 'statistics' in data:
        stats = data['statistics']
        for mode in modes:
            if mode in stats:
                time = stats[mode].get('response_time', {}).get('mean', 1)
                tokens = stats[mode].get('total_tokens', {}).get('mean', 0)
                tokens_per_second.append(tokens / time if time > 0 else 0)
            else:
                tokens_per_second.append(0)
    elif 'summary' in data:
        stats = data['summary']
        for mode in modes:
            if mode in stats:
                time = stats[mode].get('avg_time', 1)
                tokens = stats[mode].get('avg_total_tokens', 0)
                tokens_per_second.append(tokens / time if time > 0 else 0)
            else:
                tokens_per_second.append(0)

    x_pos = np.arange(len(modes))
    bars = ax.bar(x_pos, tokens_per_second, color=colors, alpha=0.8)

    ax.set_xlabel('Mode', fontweight='bold')
    ax.set_ylabel('Tokens/Second', fontweight='bold')
    ax.set_title('Processing Efficiency', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m[:3] for m in mode_labels])
    ax.grid(True, alpha=0.3, axis='y')

def create_summary_table(ax, data):
    """Create summary table as subplot."""
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    columns = ['Mode', 'Avg Time (s)', 'Input Tokens', 'Output Tokens', 'Total Tokens', 'Success Rate']
    rows = []

    modes = ['local', 'global', 'naive', 'hybrid', 'mix']

    if 'statistics' in data:
        stats = data['statistics']
        for mode in modes:
            if mode in stats:
                row = [
                    mode.capitalize(),
                    f"{stats[mode].get('response_time', {}).get('mean', 0):.2f}",
                    f"{stats[mode].get('input_tokens', {}).get('mean', 0):,.0f}",
                    f"{stats[mode].get('output_tokens', {}).get('mean', 0):,.0f}",
                    f"{stats[mode].get('total_tokens', {}).get('mean', 0):,.0f}",
                    f"{stats[mode].get('success_rate', 100):.0f}%"
                ]
                rows.append(row)
    elif 'summary' in data:
        stats = data['summary']
        for mode in modes:
            if mode in stats:
                row = [
                    mode.capitalize(),
                    f"{stats[mode].get('avg_time', 0):.2f}",
                    f"{stats[mode].get('avg_input_tokens', 0):,.0f}",
                    f"{stats[mode].get('avg_output_tokens', 0):,.0f}",
                    f"{stats[mode].get('avg_total_tokens', 0):,.0f}",
                    f"{stats[mode].get('success_rate', 100):.0f}%"
                ]
                rows.append(row)

    if rows:
        table = ax.table(cellText=rows, colLabels=columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

def main():
    """Main function to create all visualizations."""
    print("="*60)
    print("Gemini Benchmark Visualization")
    print("="*60)

    # Load data
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Loading specified file: {file_path}")
    else:
        file_path = None

    data = load_benchmark_data(file_path)

    # Create timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("\nCreating visualizations...")

    # 1. Average Response Times
    fig1 = create_average_response_time_chart(data)
    if fig1:
        output_file = f'gemini_benchmark_avg_times_{timestamp}.png'
        fig1.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved average response times to {output_file}")

    # 2. Response Time vs Tokens Scatter Plot
    fig2 = create_response_time_vs_tokens_scatter(data)
    if fig2:
        output_file = f'gemini_benchmark_scatter_{timestamp}.png'
        fig2.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved scatter plot to {output_file}")

    # 3. Token Breakdown
    fig3 = create_token_breakdown_chart(data)
    if fig3:
        output_file = f'gemini_benchmark_tokens_{timestamp}.png'
        fig3.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved token breakdown to {output_file}")

    # 4. Efficiency Analysis
    fig4 = create_efficiency_analysis(data)
    if fig4:
        output_file = f'gemini_benchmark_efficiency_{timestamp}.png'
        fig4.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved efficiency analysis to {output_file}")

    # 5. Summary Dashboard
    fig5 = create_summary_dashboard(data)
    if fig5:
        output_file = f'gemini_benchmark_dashboard_{timestamp}.png'
        fig5.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved dashboard to {output_file}")

    # Show all figures
    print("\nDisplaying visualizations...")
    plt.show()

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

if __name__ == "__main__":
    main()