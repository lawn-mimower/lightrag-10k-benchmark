#!/usr/bin/env python3
"""
Benchmark Gemini-3-Flash-Preview API performance across different LightRAG modes.
Measures actual API response time and token consumption for each mode.
"""

import json
import os
import time
import random
from pathlib import Path
from datetime import datetime
import statistics
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set")
    print("Please set it using: export GEMINI_API_KEY='your-api-key'")
    print("Or add it to your .env file")
    exit(1)

# Model configuration
MODEL_NAME = "gemini-3-flash-preview"

def load_sample_questions(data_dir: Path, sample_size: int = 10) -> List[Dict[str, Any]]:
    """Load sample questions with their contexts for all modes."""
    json_files = list(data_dir.glob('*.json'))

    # Randomly sample files
    if len(json_files) > sample_size:
        sampled_files = random.sample(json_files, sample_size)
    else:
        sampled_files = json_files[:sample_size]

    samples = []
    for file_path in sampled_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Check if all modes have successful results
            modes = data.get('modes', {})
            if all(mode in modes and modes[mode].get('status') == 'success'
                   for mode in ['local', 'global', 'naive', 'hybrid', 'mix']):

                sample = {
                    'question_id': data.get('question_id'),
                    'question': data.get('question'),
                    'file_name': file_path.name,
                    'contexts': {}
                }

                # Extract context for each mode
                for mode in ['local', 'global', 'naive', 'hybrid', 'mix']:
                    sample['contexts'][mode] = modes[mode].get('retrieved_context', '')

                samples.append(sample)

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue

    return samples

def create_prompt(question: str, context: str) -> str:
    """Create the prompt to send to Gemini API."""
    prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information, indicate that clearly.

Context:
{context}

Question: {question}

Answer:"""
    return prompt

def call_gemini_api(client: genai.Client, prompt: str) -> Dict[str, Any]:
    """Call Gemini API and measure performance."""
    start_time = time.time()

    try:
        # Configure generation matching generate_answers_ctas.py
        generation_config = types.GenerateContentConfig(
            max_output_tokens=65536,  # Gemini 3 Flash maximum capacity
            temperature=0.1,  # Lower temperature for consistent results
        )

        # Generate response using the client.models.generate_content pattern
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=generation_config
        )

        end_time = time.time()
        response_time = end_time - start_time

        # Extract answer text
        answer_text = ""
        if response and response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        answer_text += part.text

        # Extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            if hasattr(usage, 'prompt_token_count'):
                input_tokens = usage.prompt_token_count
            if hasattr(usage, 'candidates_token_count'):
                output_tokens = usage.candidates_token_count
            if hasattr(usage, 'total_token_count'):
                total_tokens = usage.total_token_count
        else:
            # Estimate tokens if not provided
            input_tokens = len(prompt) // 4  # Rough estimate
            output_tokens = len(answer_text) // 4 if answer_text else 0
            total_tokens = input_tokens + output_tokens

        return {
            'success': True,
            'response': answer_text.strip() if answer_text else "",
            'response_time': response_time,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        end_time = time.time()
        return {
            'success': False,
            'error': str(e),
            'response_time': end_time - start_time,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'timestamp': datetime.now().isoformat()
        }

def run_benchmark(client: genai.Client, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run benchmark for all modes across all samples."""
    modes = ['local', 'global', 'naive', 'hybrid', 'mix']
    results = {mode: [] for mode in modes}

    total_calls = len(samples) * len(modes)
    completed = 0

    print(f"Starting benchmark with {len(samples)} questions across {len(modes)} modes")
    print(f"Total API calls to make: {total_calls}\n")

    for i, sample in enumerate(samples):
        print(f"\nProcessing question {i+1}/{len(samples)}: {sample['file_name']}")
        print(f"Question: {sample['question'][:100]}...")

        for mode in modes:
            print(f"  Testing {mode} mode...", end=" ")

            # Create prompt with context for this mode
            context = sample['contexts'][mode]
            prompt = create_prompt(sample['question'], context)

            # Call API and measure performance
            result = call_gemini_api(client, prompt)

            # Store results
            result['question_id'] = sample['question_id']
            result['question'] = sample['question']
            result['mode'] = mode
            result['context_length'] = len(context)
            result['prompt_length'] = len(prompt)

            results[mode].append(result)

            completed += 1

            if result['success']:
                print(f"✓ ({result['response_time']:.2f}s, {result['total_tokens']} tokens)")
            else:
                print(f"✗ (Error: {result['error'][:50]}...)")

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        print(f"Progress: {completed}/{total_calls} calls completed ({completed*100/total_calls:.1f}%)")

    return results

def calculate_statistics(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Calculate statistics for each mode."""
    stats = {}

    for mode, mode_results in results.items():
        successful_results = [r for r in mode_results if r['success']]

        if not successful_results:
            stats[mode] = {'error': 'No successful results'}
            continue

        response_times = [r['response_time'] for r in successful_results]
        input_tokens = [r['input_tokens'] for r in successful_results]
        output_tokens = [r['output_tokens'] for r in successful_results]
        total_tokens = [r['total_tokens'] for r in successful_results]

        stats[mode] = {
            'sample_count': len(successful_results),
            'success_rate': len(successful_results) / len(mode_results) * 100,

            'response_time': {
                'mean': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'stdev': statistics.stdev(response_times) if len(response_times) > 1 else 0
            },

            'input_tokens': {
                'mean': statistics.mean(input_tokens),
                'median': statistics.median(input_tokens),
                'min': min(input_tokens),
                'max': max(input_tokens)
            },

            'output_tokens': {
                'mean': statistics.mean(output_tokens),
                'median': statistics.median(output_tokens),
                'min': min(output_tokens),
                'max': max(output_tokens)
            },

            'total_tokens': {
                'mean': statistics.mean(total_tokens),
                'median': statistics.median(total_tokens),
                'sum': sum(total_tokens)
            }
        }

    return stats

def print_summary(stats: Dict[str, Any]):
    """Print a formatted summary of the benchmark results."""
    print("\n" + "="*80)
    print("GEMINI-3-FLASH-PREVIEW BENCHMARK RESULTS")
    print("="*80)

    modes = ['local', 'global', 'naive', 'hybrid', 'mix']

    for mode in modes:
        if mode not in stats:
            continue

        mode_stats = stats[mode]

        if 'error' in mode_stats:
            print(f"\n{mode.upper()} MODE: {mode_stats['error']}")
            continue

        print(f"\n{mode.upper()} MODE:")
        print("-"*40)
        print(f"  Successful Calls: {mode_stats['sample_count']}")
        print(f"  Success Rate: {mode_stats['success_rate']:.1f}%")

        print(f"\n  Response Time (seconds):")
        print(f"    Mean: {mode_stats['response_time']['mean']:.3f}")
        print(f"    Median: {mode_stats['response_time']['median']:.3f}")
        print(f"    Min: {mode_stats['response_time']['min']:.3f}")
        print(f"    Max: {mode_stats['response_time']['max']:.3f}")
        print(f"    StdDev: {mode_stats['response_time']['stdev']:.3f}")

        print(f"\n  Token Usage:")
        print(f"    Avg Input Tokens: {mode_stats['input_tokens']['mean']:,.0f}")
        print(f"    Avg Output Tokens: {mode_stats['output_tokens']['mean']:,.0f}")
        print(f"    Avg Total Tokens: {mode_stats['total_tokens']['mean']:,.0f}")
        print(f"    Total Tokens Used: {mode_stats['total_tokens']['sum']:,}")

    # Compare modes
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    # Find best performers
    valid_modes = [m for m in modes if m in stats and 'error' not in stats[m]]

    if valid_modes:
        # Fastest mode
        fastest = min(valid_modes, key=lambda m: stats[m]['response_time']['mean'])
        print(f"\nFastest Mode: {fastest.upper()} ({stats[fastest]['response_time']['mean']:.3f}s avg)")

        # Most token efficient
        efficient = min(valid_modes, key=lambda m: stats[m]['total_tokens']['mean'])
        print(f"Most Token Efficient: {efficient.upper()} ({stats[efficient]['total_tokens']['mean']:,.0f} tokens avg)")

        # Token cost comparison
        print("\nToken Consumption Ranking (lowest to highest):")
        ranked_modes = sorted(valid_modes, key=lambda m: stats[m]['total_tokens']['mean'])
        for i, mode in enumerate(ranked_modes, 1):
            print(f"  {i}. {mode.upper()}: {stats[mode]['total_tokens']['mean']:,.0f} avg tokens")

def main():
    # Configuration
    data_dir = Path(os.getenv('RESULTS_DIR', '5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL'))
    sample_size = int(os.getenv('SAMPLE_SIZE', '10'))  # Number of questions to test

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return

    # Initialize Gemini Client (matching generate_answers_ctas.py)
    print(f"Initializing Gemini Client with {MODEL_NAME}...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"✓ Initialized Gemini Client\n")

    # Load sample questions
    print(f"Loading {sample_size} sample questions...")
    samples = load_sample_questions(data_dir, sample_size)

    if not samples:
        print("Error: No valid samples found")
        return

    print(f"Loaded {len(samples)} valid samples with all mode contexts")

    # Run benchmark
    start_time = time.time()
    results = run_benchmark(client, samples)
    total_time = time.time() - start_time

    # Calculate statistics
    stats = calculate_statistics(results)

    # Print summary
    print_summary(stats)

    print(f"\n" + "="*80)
    print(f"Total benchmark time: {total_time:.2f} seconds")
    print(f"Average time per API call: {total_time/(len(samples)*5):.2f} seconds")

    # Save detailed results
    output_file = f"gemini_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'configuration': {
                'model': 'gemini-3-flash-preview',
                'sample_size': sample_size,
                'timestamp': datetime.now().isoformat(),
                'total_time': total_time
            },
            'statistics': stats,
            'detailed_results': results
        }, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()