#!/usr/bin/env python3
"""
Batch RAGAS Evaluation - Process Multiple Files in Parallel
===========================================================
Processes 3 files at a time, each evaluating all 5 modes in parallel.
Results are appended to a single JSON file.

Strategy:
- 3 files × 5 modes = 15 concurrent evaluations
- For 63 files: 21 batches × ~25 seconds = ~9 minutes total
"""

import os
import json
import time
import warnings
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper is deprecated.*")
warnings.filterwarnings("ignore", message=".*Unexpected type for token usage.*")

# RAGAS imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ============================================
# Configuration
# ============================================
RESULTS_DIR = "./lightrag-bench/5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL"
OUTPUT_FILE = "./lightrag-bench/batch_ragas_evaluation_results.json"

# Processing settings
BATCH_SIZE = 3  # Process 3 files at a time
MAX_WORKERS = 15  # 3 files × 5 modes

# Models
RAGAS_JUDGE_MODEL = "ministral-14b-2512"
RAGAS_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Query modes to evaluate
QUERY_MODES = ["local", "global", "naive", "hybrid", "mix"]

print("="*70)
print("📦 BATCH RAGAS EVALUATION")
print("="*70)
print(f"📁 Source Directory: {Path(RESULTS_DIR).name}")
print(f"💾 Output File: {Path(OUTPUT_FILE).name}")
print(f"⚡ Batch Size: {BATCH_SIZE} files concurrently")
print(f"🔧 Max Workers: {MAX_WORKERS}")
print()

# ============================================
# Setup Models (Once for entire batch)
# ============================================
print("Setting up evaluation models...")

# Check API key
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    print("❌ ERROR: MISTRAL_API_KEY not found in environment variables!")
    print("   Please set it in your .env file or environment")
    exit(1)

print(f"✓ Mistral API Key: {mistral_api_key[:8]}...")
print(f"✓ Model: {RAGAS_JUDGE_MODEL}")
print(f"✓ Endpoint: https://api.mistral.ai/v1")

# LLM for judging
base_llm = ChatOpenAI(
    model=RAGAS_JUDGE_MODEL,
    api_key=mistral_api_key,
    base_url="https://api.mistral.ai/v1",
    max_retries=5,
    request_timeout=180
)

try:
    ragas_llm = LangchainLLMWrapper(
        langchain_llm=base_llm,
        bypass_n=True
    )
except:
    ragas_llm = base_llm

# Local embeddings
print("Loading local embeddings...")
ragas_embeddings = HuggingFaceEmbeddings(
    model_name=RAGAS_EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✓ Models configured\n")

# ============================================
# Evaluation Functions
# ============================================

def evaluate_single_mode(
    question_id: str,
    question_text: str,
    mode: str,
    mode_data: Dict[str, Any],
    expected_answer: str,
    retry_count: int = 3
) -> Dict[str, Any]:
    """
    Evaluate a single mode for a question.
    This runs in a thread pool to allow parallelism.
    Includes retry logic for API connection errors.
    """
    if mode_data.get("status") != "success":
        return {
            "question_id": question_id,
            "mode": mode,
            "status": "skipped",
            "reason": "no_successful_generation"
        }

    last_error = None
    for attempt in range(retry_count):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)  # Exponential backoff: 2, 4, 8 seconds

            # Prepare RAGAS dataset
            eval_dataset = Dataset.from_dict({
                "question": [question_text],
                "answer": [mode_data["answer"]],
                "contexts": [[mode_data["retrieved_context"]]],
                "ground_truth": [str(expected_answer)]
            })

            # Run evaluation
            eval_results = evaluate(
                dataset=eval_dataset,
                metrics=[
                    Faithfulness(),
                    AnswerRelevancy(),
                    ContextRecall(),
                    ContextPrecision()
                ],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                show_progress=False
            )

            # Extract scores
            df = eval_results.to_pandas()
            scores_row = df.iloc[0]

            metrics = {
                "faithfulness": float(scores_row.get("faithfulness", 0)),
                "answer_relevancy": float(scores_row.get("answer_relevancy", 0)),
                "context_recall": float(scores_row.get("context_recall", 0)),
                "context_precision": float(scores_row.get("context_precision", 0))
            }

            # Calculate RAGAS score
            valid_metrics = [v for v in metrics.values() if not np.isnan(v)]
            ragas_score = np.mean(valid_metrics) if valid_metrics else 0

            return {
                "question_id": question_id,
                "mode": mode,
                "status": "success",
                "metrics": metrics,
                "ragas_score": round(ragas_score, 4),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            last_error = e
            if attempt < retry_count - 1:
                print(f"    Retry {attempt + 1}/{retry_count} for {mode}: {str(e)[:50]}")
                continue

    # All retries failed
    return {
        "question_id": question_id,
        "mode": mode,
        "status": "error",
        "error": str(last_error)[:200] if last_error else "Unknown error",
        "timestamp": datetime.now().isoformat()
    }


def process_file(file_path: Path, executor: ThreadPoolExecutor) -> List[Dict[str, Any]]:
    """
    Process one file, evaluating all modes in parallel.
    Returns list of evaluation results for all modes.
    """
    try:
        # Load the file
        with open(file_path, 'r') as f:
            data = json.load(f)

        question_id = data["question_id"]
        question_text = data["question"]
        expected_answer = str(data["expected_answer"])

        # Submit all mode evaluations to thread pool
        futures = []
        for mode in QUERY_MODES:
            mode_data = data["modes"].get(mode, {})
            future = executor.submit(
                evaluate_single_mode,
                question_id,
                question_text,
                mode,
                mode_data,
                expected_answer
            )
            futures.append((mode, future))

        # Collect results
        results = []
        for mode, future in futures:
            try:
                result = future.result(timeout=120)  # Increase to 120s timeout per mode
                results.append(result)
            except Exception as e:
                results.append({
                    "question_id": question_id,
                    "mode": mode,
                    "status": "timeout",
                    "error": str(e)[:100]
                })

        return results

    except Exception as e:
        # Return error for all modes if file can't be loaded
        return [{
            "question_id": file_path.stem.split('_')[-1],
            "file": str(file_path.name),
            "status": "file_error",
            "error": str(e)[:200]
        }]


async def process_batch(
    batch_files: List[Path],
    batch_num: int,
    total_batches: int,
    all_results: List[Dict[str, Any]]
) -> None:
    """
    Process a batch of files concurrently.
    """
    print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch_files)} files...")
    batch_start = time.time()

    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Process all files in batch
        batch_results = []
        for file_path in batch_files:
            file_results = process_file(file_path, executor)
            batch_results.extend(file_results)
            print(f"  ✓ {file_path.name}: {len(file_results)} modes evaluated")

    # Add to overall results
    all_results.extend(batch_results)

    batch_time = time.time() - batch_start
    print(f"[Batch {batch_num}] Completed in {batch_time:.1f}s")

    # Save intermediate results after each batch
    save_results(all_results, is_intermediate=True)


def save_results(results: List[Dict[str, Any]], is_intermediate: bool = False):
    """
    Save results to JSON file.
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_evaluations": len(results),
        "is_intermediate": is_intermediate,
        "judge_model": RAGAS_JUDGE_MODEL,
        "embedding_model": RAGAS_EMBEDDING_MODEL,
        "evaluations": results
    }

    # Save with atomic write (write to temp, then rename)
    temp_file = OUTPUT_FILE + ".tmp"
    with open(temp_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Atomic rename
    os.replace(temp_file, OUTPUT_FILE)

    if not is_intermediate:
        print(f"\n💾 Final results saved to: {OUTPUT_FILE}")


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics from evaluation results.
    """
    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        return {"error": "No successful evaluations"}

    # Group by mode
    mode_stats = {}
    for mode in QUERY_MODES:
        mode_results = [r for r in successful if r.get("mode") == mode]
        if mode_results:
            mode_stats[mode] = {
                "count": len(mode_results),
                "avg_ragas": np.mean([r["ragas_score"] for r in mode_results]),
                "avg_faithfulness": np.mean([r["metrics"]["faithfulness"] for r in mode_results]),
                "avg_relevancy": np.mean([r["metrics"]["answer_relevancy"] for r in mode_results]),
                "avg_recall": np.mean([r["metrics"]["context_recall"] for r in mode_results]),
                "avg_precision": np.mean([r["metrics"]["context_precision"] for r in mode_results])
            }

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "by_mode": mode_stats
    }


# ============================================
# Main Batch Processing
# ============================================

async def main():
    """
    Main batch processing function.
    """
    # Find all result files
    results_path = Path(RESULTS_DIR)
    all_files = sorted(results_path.glob("test_results_*_question_*.json"))

    if not all_files:
        print(f"❌ No files found in {RESULTS_DIR}")
        return

    print(f"📊 Found {len(all_files)} files to process")

    # Ask for confirmation if many files
    if len(all_files) > 10:
        response = input(f"\n⚠ This will process {len(all_files)} files. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Split into batches
    batches = [all_files[i:i+BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]
    total_batches = len(batches)

    print(f"📦 Split into {total_batches} batches of up to {BATCH_SIZE} files each")
    print(f"⏱ Estimated time: {total_batches * 25:.0f} seconds ({total_batches * 25 / 60:.1f} minutes)")
    print("="*70)

    # Process all batches
    all_results = []
    start_time = time.time()

    for batch_num, batch_files in enumerate(batches, 1):
        await process_batch(batch_files, batch_num, total_batches, all_results)

    total_time = time.time() - start_time

    # Final save
    save_results(all_results, is_intermediate=False)

    # Calculate and display statistics
    stats = calculate_statistics(all_results)

    print("\n" + "="*70)
    print("📊 EVALUATION COMPLETE")
    print("="*70)
    print(f"Total Files:        {len(all_files)}")
    print(f"Total Evaluations:  {stats['total']}")
    print(f"Successful:         {stats['successful']}")
    print(f"Failed:             {stats['failed']}")
    print(f"Total Time:         {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Avg per File:       {total_time/len(all_files):.1f}s")

    if "by_mode" in stats:
        print("\n📈 RESULTS BY MODE:")
        print(f"{'Mode':<8} | {'Count':<5} | {'RAGAS':<6} | {'Faith':<6} | {'Relev':<6} | {'Recall':<6} | {'Precis':<6}")
        print("-" * 70)

        for mode in QUERY_MODES:
            if mode in stats["by_mode"]:
                s = stats["by_mode"][mode]
                print(f"{mode.upper():<8} | {s['count']:<5} | {s['avg_ragas']:.4f} | "
                      f"{s['avg_faithfulness']:.4f} | {s['avg_relevancy']:.4f} | "
                      f"{s['avg_recall']:.4f} | {s['avg_precision']:.4f}")

    print("\n✅ Batch processing complete!")
    print(f"📁 Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())