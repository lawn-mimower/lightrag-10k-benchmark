#!/usr/bin/env python3
"""
Quick benchmark script for Gemini-3-Flash-Preview API.
Simplified version for easy customization and testing.
"""

import json
import os
import time
import random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configuration - EDIT THESE VALUES AS NEEDED
SAMPLE_SIZE = 10  # Number of questions to test per mode
API_KEY = os.getenv('GEMINI_API_KEY')
DATA_DIR = '5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL'
OUTPUT_FILE = f"gemini_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
MODEL_NAME = "gemini-3-flash-preview"

def run_quick_benchmark():
    """Run a quick benchmark test."""

    # Initialize Gemini Client
    print(f"Initializing Gemini Client with {MODEL_NAME}...")
    client = genai.Client(api_key=API_KEY)
    print(f"✓ Initialized Gemini Client\n")

    # Load sample files
    data_path = Path(DATA_DIR)
    json_files = list(data_path.glob('*.json'))

    if len(json_files) == 0:
        print("No JSON files found in", DATA_DIR)
        return

    # Sample random files
    sample_files = random.sample(json_files, min(SAMPLE_SIZE, len(json_files)))

    results = {
        'local': [],
        'global': [],
        'naive': [],
        'hybrid': [],
        'mix': []
    }

    print(f"Testing {len(sample_files)} questions across 5 modes...")
    print(f"Total API calls: {len(sample_files) * 5}\n")

    for i, file_path in enumerate(sample_files, 1):
        print(f"Question {i}/{len(sample_files)}: {file_path.name}")

        # Load question data
        with open(file_path, 'r') as f:
            data = json.load(f)

        question = data.get('question', '')
        modes_data = data.get('modes', {})

        for mode in results.keys():
            if mode not in modes_data or modes_data[mode].get('status') != 'success':
                print(f"  {mode}: skipped (no data)")
                continue

            context = modes_data[mode].get('retrieved_context', '')

            # Create prompt (matching the style from generate_answers_ctas.py)
            prompt = f"""You are a financial document analyst specializing in SEC 10-K filings.

CRITICAL INSTRUCTIONS:
- Answer ONLY based on the provided context
- Be precise with numerical values
- Keep answers concise (2-4 sentences)

Question: {question}

Context from {mode.upper()}:
{context[:50000]}

Please provide a concise, factual answer based only on the information in the context above."""

            # Measure API call
            start_time = time.time()

            try:
                # Configure generation (matching generate_answers_ctas.py)
                generation_config = types.GenerateContentConfig(
                    max_output_tokens=65536,
                    temperature=0.1,
                )

                # Generate response
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=generation_config
                )

                elapsed = time.time() - start_time

                # Extract answer text
                answer_text = ""
                if response and response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                answer_text += part.text

                # Get token counts
                input_tokens = 0
                output_tokens = 0

                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    if hasattr(usage, 'prompt_token_count'):
                        input_tokens = usage.prompt_token_count
                    if hasattr(usage, 'candidates_token_count'):
                        output_tokens = usage.candidates_token_count
                else:
                    # Estimate if not provided
                    input_tokens = len(prompt) // 4
                    output_tokens = len(answer_text) // 4

                results[mode].append({
                    'time': elapsed,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'answer_length': len(answer_text),
                    'success': True
                })

                print(f"  {mode}: {elapsed:.2f}s, {input_tokens+output_tokens} tokens")

            except Exception as e:
                print(f"  {mode}: ERROR - {str(e)[:50]}")
                results[mode].append({
                    'time': time.time() - start_time,
                    'error': str(e),
                    'success': False
                })

            time.sleep(0.5)  # Rate limiting

    # Calculate averages
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    summary = {}
    for mode, mode_results in results.items():
        successful = [r for r in mode_results if r.get('success', False)]

        if successful:
            avg_time = sum(r['time'] for r in successful) / len(successful)
            avg_input = sum(r.get('input_tokens', 0) for r in successful) / len(successful)
            avg_output = sum(r.get('output_tokens', 0) for r in successful) / len(successful)

            summary[mode] = {
                'avg_time': avg_time,
                'avg_input_tokens': avg_input,
                'avg_output_tokens': avg_output,
                'avg_total_tokens': avg_input + avg_output,
                'success_rate': len(successful) / len(mode_results) * 100 if mode_results else 0
            }

            print(f"\n{mode.upper()}:")
            print(f"  Avg Time: {avg_time:.3f}s")
            print(f"  Avg Tokens: {avg_input + avg_output:.0f} (in: {avg_input:.0f}, out: {avg_output:.0f})")
            print(f"  Success Rate: {summary[mode]['success_rate']:.0f}%")

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({
            'config': {
                'model': 'gemini-3-flash-preview',
                'samples': len(sample_files),
                'timestamp': datetime.now().isoformat()
            },
            'summary': summary,
            'raw_results': results
        }, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    if not API_KEY:
        print("Error: Set GEMINI_API_KEY environment variable")
        exit(1)

    run_quick_benchmark()