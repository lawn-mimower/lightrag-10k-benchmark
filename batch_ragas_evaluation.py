#!/usr/bin/env python3
"""
FAST Batch RAGAS Evaluation - Optimized for Speed
==================================================
Key optimizations:
1. Process 10 files at once (was 3)
2. True async evaluation (not ThreadPoolExecutor)
3. No retries (for speed)
4. Shorter timeouts
5. Concurrent everything
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
from tqdm.asyncio import tqdm
import aiofiles

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
# OPTIMIZED Configuration
# ============================================
RESULTS_DIR = "./lightrag-bench/5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL"
OUTPUT_FILE = "./lightrag-bench/batch_ragas_evaluation_results_fast.json"

# SPEED OPTIMIZATIONS
BATCH_SIZE = 10  # Process MORE files at once (was 3)
MAX_CONCURRENT = 50  # Allow up to 50 concurrent evaluations (was 15)
NO_RETRIES = True  # Skip retries for speed
TIMEOUT_SECONDS = 30  # Shorter timeout (was 60)

# Models
RAGAS_JUDGE_MODEL = "ministral-14b-2512"
RAGAS_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Query modes
QUERY_MODES = ["local", "global", "naive", "hybrid", "mix"]

print("="*70)
print("⚡ FAST BATCH RAGAS EVALUATION")
print("="*70)
print(f"📁 Source Directory: {Path(RESULTS_DIR).name}")
print(f"💾 Output File: {Path(OUTPUT_FILE).name}")
print(f"🚀 Batch Size: {BATCH_SIZE} files concurrently")
print(f"⚡ Max Concurrent: {MAX_CONCURRENT} evaluations")
print(f"🏃 Speed Mode: No retries, 30s timeout")
print()

# ============================================
# Setup Models ONCE
# ============================================
print("Setting up models (one-time)...")

# Check API
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    print("❌ MISTRAL_API_KEY not found!")
    exit(1)

# LLM
base_llm = ChatOpenAI(
    model=RAGAS_JUDGE_MODEL,
    api_key=mistral_api_key,
    base_url="https://api.mistral.ai/v1",
    max_retries=0,  # No retries for speed
    request_timeout=30,
    temperature=0.1
    # Note: max_tokens removed as it causes issues with Mistral API
)

try:
    ragas_llm = LangchainLLMWrapper(
        langchain_llm=base_llm,
        bypass_n=True
    )
except:
    ragas_llm = base_llm

# Local embeddings (loaded once)
print("Loading embeddings (this is cached after first load)...")
ragas_embeddings = HuggingFaceEmbeddings(
    model_name=RAGAS_EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✓ Ready for FAST processing!\n")

# ============================================
# Async Evaluation Functions
# ============================================

async def evaluate_mode_async(
    question_id: str,
    question_text: str,
    mode: str,
    mode_data: Dict[str, Any],
    expected_answer: str,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    Async evaluation of a single mode.
    Uses semaphore to limit concurrent API calls.
    """
    if mode_data.get("status") != "success":
        return {
            "question_id": question_id,
            "mode": mode,
            "status": "skipped"
        }

    async with semaphore:
        try:
            # Run in executor to not block event loop
            loop = asyncio.get_event_loop()

            # Prepare dataset
            eval_dataset = Dataset.from_dict({
                "question": [question_text],
                "answer": [mode_data["answer"]],
                "contexts": [[mode_data["retrieved_context"]]],
                "ground_truth": [str(expected_answer)]
            })

            # Run evaluation in thread pool (RAGAS isn't async)
            eval_results = await loop.run_in_executor(
                None,
                lambda: evaluate(
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
                "ragas_score": round(ragas_score, 4)
            }

        except asyncio.TimeoutError:
            return {
                "question_id": question_id,
                "mode": mode,
                "status": "timeout"
            }
        except Exception as e:
            return {
                "question_id": question_id,
                "mode": mode,
                "status": "error",
                "error": str(e)[:100]
            }


async def process_file_async(
    file_path: Path,
    semaphore: asyncio.Semaphore
) -> List[Dict[str, Any]]:
    """
    Process one file with ALL modes in parallel.
    """
    try:
        # Load file asynchronously
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)

        question_id = data["question_id"]
        question_text = data["question"]
        expected_answer = str(data["expected_answer"])

        # Create tasks for ALL modes at once
        tasks = []
        for mode in QUERY_MODES:
            mode_data = data["modes"].get(mode, {})
            task = evaluate_mode_async(
                question_id,
                question_text,
                mode,
                mode_data,
                expected_answer,
                semaphore
            )
            tasks.append(task)

        # Wait for all modes to complete (with timeout)
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=TIMEOUT_SECONDS
        )

        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, dict):
                valid_results.append(r)
            else:
                valid_results.append({
                    "question_id": question_id,
                    "status": "exception",
                    "error": str(r)[:100]
                })

        return valid_results

    except Exception as e:
        return [{
            "file": str(file_path.name),
            "status": "file_error",
            "error": str(e)[:100]
        }]


async def process_batch_async(
    batch_files: List[Path],
    batch_num: int,
    total_batches: int,
    semaphore: asyncio.Semaphore
) -> List[Dict[str, Any]]:
    """
    Process a batch of files ALL IN PARALLEL.
    """
    print(f"\n[Batch {batch_num}/{total_batches}] Starting {len(batch_files)} files...")
    start_time = time.time()

    # Process ALL files in parallel
    tasks = [process_file_async(f, semaphore) for f in batch_files]

    # Progress bar
    results_lists = []
    with tqdm(total=len(tasks), desc=f"Batch {batch_num}") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results_lists.append(result)
            pbar.update(1)

    # Flatten results
    all_results = []
    for result_list in results_lists:
        all_results.extend(result_list)

    elapsed = time.time() - start_time
    print(f"[Batch {batch_num}] ✅ Completed in {elapsed:.1f}s "
          f"({len(all_results)} evaluations, "
          f"{elapsed/len(batch_files):.1f}s per file)")

    return all_results


async def main():
    """
    Main async orchestrator.
    """
    # Find all files
    results_path = Path(RESULTS_DIR)
    all_files = sorted(results_path.glob("test_results_*_question_*.json"))

    if not all_files:
        print(f"❌ No files found in {RESULTS_DIR}")
        return

    print(f"📊 Found {len(all_files)} files to process")

    # Confirmation
    if len(all_files) > 10:
        response = input(f"\n⚠ Process {len(all_files)} files in FAST mode? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Split into batches
    batches = [all_files[i:i+BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]
    total_batches = len(batches)

    print(f"📦 {total_batches} batches × {BATCH_SIZE} files")
    print(f"⚡ Estimated time: {total_batches * 15:.0f}s ({total_batches * 15 / 60:.1f} min)")
    print("="*70)

    # Semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Process all batches
    all_results = []
    start_time = time.time()

    for batch_num, batch_files in enumerate(batches, 1):
        batch_results = await process_batch_async(
            batch_files,
            batch_num,
            total_batches,
            semaphore
        )
        all_results.extend(batch_results)

        # Save intermediate
        async with aiofiles.open(OUTPUT_FILE, 'w') as f:
            await f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(all_results),
                "is_complete": batch_num == total_batches,
                "evaluations": all_results
            }, indent=2))

    total_time = time.time() - start_time

    # Calculate stats
    successful = len([r for r in all_results if r.get("status") == "success"])

    print("\n" + "="*70)
    print("⚡ FAST EVALUATION COMPLETE!")
    print("="*70)
    print(f"Total Files:        {len(all_files)}")
    print(f"Total Evaluations:  {len(all_results)}")
    print(f"Successful:         {successful}")
    print(f"Failed:             {len(all_results) - successful}")
    print(f"Total Time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg per File:       {total_time/len(all_files):.1f}s")
    print(f"Speed:              {len(all_results)/total_time:.1f} evals/sec")
    print()
    print(f"📁 Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())