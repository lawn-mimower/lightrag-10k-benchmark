#!/usr/bin/env python3
"""
Analyze retrieval times and token consumption across different modes
from 5_modes_question_wise_results_with_answers directory.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import statistics
from collections import defaultdict
from typing import Dict, List, Any

def estimate_tokens(text_length: int) -> int:
    """
    Estimate number of tokens from character length.
    Rough approximation: 1 token ≈ 4 characters
    """
    return text_length // 4

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO format timestamp string to datetime object."""
    try:
        # Handle both formats with and without microseconds
        if '.' in timestamp_str:
            return datetime.fromisoformat(timestamp_str)
        else:
            return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
    except:
        return None

def calculate_retrieval_time(mode_data: Dict[str, Any], prev_timestamp: datetime = None) -> float:
    """
    Calculate retrieval time for a mode.
    If prev_timestamp is provided, calculate difference from previous mode.
    Returns time in seconds.
    """
    if 'timestamp' not in mode_data:
        return None

    current_timestamp = parse_timestamp(mode_data['timestamp'])
    if current_timestamp is None:
        return None

    if prev_timestamp is not None:
        time_diff = (current_timestamp - prev_timestamp).total_seconds()
        return time_diff

    return None

def analyze_file(file_path: Path, next_file_start_time: datetime = None) -> Dict[str, Any]:
    """Analyze a single result file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    results = {}
    mode_order = ['local', 'global', 'naive', 'hybrid', 'mix']

    # First, collect all timestamps
    timestamps = {}
    for mode_name in mode_order:
        if mode_name in data.get('modes', {}) and data['modes'][mode_name].get('status') == 'success':
            if 'timestamp' in data['modes'][mode_name]:
                timestamps[mode_name] = parse_timestamp(data['modes'][mode_name]['timestamp'])

    # Now analyze each mode and calculate retrieval times
    for i, mode_name in enumerate(mode_order):
        if mode_name not in data.get('modes', {}):
            continue

        mode_data = data['modes'][mode_name]

        # Skip if status is not success
        if mode_data.get('status') != 'success':
            continue

        # Calculate retrieval time
        retrieval_time = None
        if mode_name in timestamps and timestamps[mode_name]:
            # For each mode, calculate time until the next mode starts
            # This represents the time taken for this mode's retrieval/processing
            if i < len(mode_order) - 1:  # Not the last mode
                next_mode = None
                for j in range(i + 1, len(mode_order)):
                    if mode_order[j] in timestamps:
                        next_mode = mode_order[j]
                        break

                if next_mode and timestamps[next_mode]:
                    retrieval_time = (timestamps[next_mode] - timestamps[mode_name]).total_seconds()

            else:  # Last mode (mix)
                # If we have the next file's start time, use that
                if next_file_start_time and timestamps[mode_name]:
                    retrieval_time = (next_file_start_time - timestamps[mode_name]).total_seconds()
                    # Sanity check - if time is too long (>5 minutes), ignore it
                    if retrieval_time > 300:
                        retrieval_time = None

        # Estimate tokens
        context_length = mode_data.get('context_length', 0)
        answer_length = mode_data.get('answer_length', 0)

        # For input tokens: context + prompt (we'll estimate prompt as ~100 chars for the question)
        prompt_length = len(data.get('question', '')) if 'question' in data else 100
        input_tokens = estimate_tokens(context_length + prompt_length)

        # For output tokens: just the answer
        output_tokens = estimate_tokens(answer_length)

        # Total tokens
        total_tokens = input_tokens + output_tokens

        results[mode_name] = {
            'retrieval_time': retrieval_time,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'context_length': context_length,
            'answer_length': answer_length,
            'timestamp': mode_data.get('timestamp'),
            'file_name': file_path.name
        }

    # Store the first timestamp for next file processing
    if 'local' in timestamps:
        results['_first_timestamp'] = timestamps['local']

    return results

def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across all files to compute averages."""
    mode_stats = defaultdict(lambda: {
        'retrieval_times': [],
        'input_tokens': [],
        'output_tokens': [],
        'total_tokens': [],
        'context_lengths': [],
        'answer_lengths': []
    })

    for file_result in all_results:
        if file_result is None:
            continue

        for mode_name, mode_data in file_result.items():
            if mode_data['retrieval_time'] is not None:
                mode_stats[mode_name]['retrieval_times'].append(mode_data['retrieval_time'])
            mode_stats[mode_name]['input_tokens'].append(mode_data['input_tokens'])
            mode_stats[mode_name]['output_tokens'].append(mode_data['output_tokens'])
            mode_stats[mode_name]['total_tokens'].append(mode_data['total_tokens'])
            mode_stats[mode_name]['context_lengths'].append(mode_data['context_length'])
            mode_stats[mode_name]['answer_lengths'].append(mode_data['answer_length'])

    # Calculate averages
    aggregated = {}
    for mode_name, stats in mode_stats.items():
        aggregated[mode_name] = {
            'avg_retrieval_time': statistics.mean(stats['retrieval_times']) if stats['retrieval_times'] else None,
            'median_retrieval_time': statistics.median(stats['retrieval_times']) if stats['retrieval_times'] else None,
            'avg_input_tokens': statistics.mean(stats['input_tokens']),
            'avg_output_tokens': statistics.mean(stats['output_tokens']),
            'avg_total_tokens': statistics.mean(stats['total_tokens']),
            'avg_context_length': statistics.mean(stats['context_lengths']),
            'avg_answer_length': statistics.mean(stats['answer_lengths']),
            'sample_count': len(stats['input_tokens'])
        }

    return aggregated

def main():
    # Directory containing result files
    results_dir = Path('5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL')

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return

    # Get all JSON files
    json_files = list(results_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files to analyze\n")

    # First pass: get all first timestamps to sort files by processing order
    file_timestamps = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'modes' in data and 'local' in data['modes'] and 'timestamp' in data['modes']['local']:
                timestamp = parse_timestamp(data['modes']['local']['timestamp'])
                if timestamp:
                    file_timestamps.append((file_path, timestamp))
        except:
            pass

    # Sort files by their first timestamp
    file_timestamps.sort(key=lambda x: x[1])

    # Analyze each file with next file's start time
    all_results = []
    for i, (file_path, _) in enumerate(file_timestamps):
        print(f"Processing {file_path.name}...")

        # Get next file's start time if available
        next_file_start = None
        if i < len(file_timestamps) - 1:
            next_file_start = file_timestamps[i + 1][1]

        file_result = analyze_file(file_path, next_file_start)
        if file_result:
            # Remove the internal timestamp field before storing
            if '_first_timestamp' in file_result:
                del file_result['_first_timestamp']
            all_results.append(file_result)

    print(f"\nSuccessfully processed {len(all_results)} files")

    # Aggregate results
    aggregated = aggregate_results(all_results)

    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY BY MODE")
    print("="*80)

    mode_order = ['local', 'global', 'naive', 'hybrid', 'mix']

    for mode_name in mode_order:
        if mode_name not in aggregated:
            continue

        stats = aggregated[mode_name]
        print(f"\n{mode_name.upper()} MODE:")
        print("-" * 40)

        print(f"  Sample Count: {stats['sample_count']}")

        if stats['avg_retrieval_time'] is not None:
            print(f"  Avg Processing Time: {stats['avg_retrieval_time']:.2f} seconds")
            print(f"  Median Processing Time: {stats['median_retrieval_time']:.2f} seconds")
        else:
            print(f"  Processing Time: N/A (last mode - no next timestamp)")

        print(f"\n  Token Consumption:")
        print(f"    Avg Input Tokens: {stats['avg_input_tokens']:,.0f}")
        print(f"    Avg Output Tokens: {stats['avg_output_tokens']:,.0f}")
        print(f"    Avg Total Tokens: {stats['avg_total_tokens']:,.0f}")

        print(f"\n  Context/Answer Lengths:")
        print(f"    Avg Context Length: {stats['avg_context_length']:,.0f} chars")
        print(f"    Avg Answer Length: {stats['avg_answer_length']:,.0f} chars")

    # Save detailed results to JSON
    output_file = 'mode_performance_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': aggregated,
            'detailed_results': all_results
        }, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Detailed results saved to {output_file}")

    # Calculate total time for entire pipeline
    print(f"\n{'='*80}")
    print("PIPELINE TIMING ANALYSIS")
    print("="*80)

    # For each file, calculate total time from first to last mode
    total_times = []
    for file_result in all_results:
        if file_result is None:
            continue

        # Get timestamps for first and last successful modes
        timestamps = []
        for mode in mode_order:
            if mode in file_result and file_result[mode].get('timestamp'):
                timestamps.append(parse_timestamp(file_result[mode]['timestamp']))

        if len(timestamps) >= 2:
            total_time = (timestamps[-1] - timestamps[0]).total_seconds()
            total_times.append(total_time)

    if total_times:
        print(f"Average total pipeline time (first to last mode): {statistics.mean(total_times):.2f} seconds")
        print(f"Median total pipeline time: {statistics.median(total_times):.2f} seconds")
        print(f"Min total pipeline time: {min(total_times):.2f} seconds")
        print(f"Max total pipeline time: {max(total_times):.2f} seconds")

if __name__ == "__main__":
    main()