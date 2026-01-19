#!/usr/bin/env python3
"""
FAST Batch RAGAS Evaluation - Fixed with Connection Handling
=============================================================
Key improvements:
1. API connection testing before batch processing
2. Reduced concurrent requests to avoid overwhelming API
3. Retry logic with exponential backoff
4. Better error handling and diagnostics
5. Connection pooling and rate limiting
"""

import os
import json
import time
import warnings
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm.asyncio import tqdm
import aiofiles
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
# OPTIMIZED Configuration with Connection Fix
# ============================================
RESULTS_DIR = "./lightrag-bench/5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL"
OUTPUT_FILE = "./lightrag-bench/batch_ragas_evaluation_results_fast.json"

# SPEED OPTIMIZATIONS - REDUCED FOR STABILITY
BATCH_SIZE = 5  # Reduced from 10 to avoid overwhelming API
MAX_CONCURRENT = 10  # Reduced from 50 to avoid connection errors
ENABLE_RETRIES = True  # Enable retries for connection errors
TIMEOUT_SECONDS = 60  # Increased timeout for stability
RATE_LIMIT_DELAY = 0.5  # Delay between API calls to avoid rate limiting

# Models
RAGAS_JUDGE_MODEL = "ministral-14b-2512"
RAGAS_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Query modes
QUERY_MODES = ["local", "global", "naive", "hybrid", "mix"]

print("="*70)
print("⚡ FAST BATCH RAGAS EVALUATION - FIXED VERSION")
print("="*70)
print(f"📁 Source Directory: {Path(RESULTS_DIR).name}")
print(f"💾 Output File: {Path(OUTPUT_FILE).name}")
print(f"🚀 Batch Size: {BATCH_SIZE} files concurrently")
print(f"⚡ Max Concurrent: {MAX_CONCURRENT} evaluations")
print(f"🔄 Retries: {'Enabled' if ENABLE_RETRIES else 'Disabled'}")
print(f"⏱️ Timeout: {TIMEOUT_SECONDS}s")
print()

# ============================================
# API Connection Test
# ============================================
async def test_api_connection():
    """Test API connection before starting batch processing."""
    print("🔍 Testing API connection...")

    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        print("❌ MISTRAL_API_KEY not found in environment!")
        print("   Please set it using: export MISTRAL_API_KEY='your-key-here'")
        print("   Or add it to your .env file")
        return False

    print(f"✓ API Key found (length: {len(mistral_api_key)} chars)")

    # Test API endpoint
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {mistral_api_key}",
                "Content-Type": "application/json"
            }

            # Test with a simple completion request
            test_payload = {
                "model": RAGAS_JUDGE_MODEL,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }

            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=test_payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    print("✅ API connection successful!")
                    return True
                elif response.status == 401:
                    print("❌ API authentication failed! Check your API key.")
                    return False
                elif response.status == 429:
                    print("⚠️ API rate limit hit. Will use rate limiting...")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ API returned status {response.status}: {error_text[:200]}")
                    return False

    except aiohttp.ClientConnectorError as e:
        print(f"❌ Connection failed: {e}")
        print("   Check your internet connection and firewall settings")
        return False
    except asyncio.TimeoutError:
        print("❌ Connection timeout! API might be slow or unreachable.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

# ============================================
# Setup Models with Better Error Handling
# ============================================
async def setup_models():
    """Setup models with connection testing."""
    print("\n📦 Setting up models...")

    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    # Create LLM with retry configuration
    base_llm = ChatOpenAI(
        model=RAGAS_JUDGE_MODEL,
        api_key=mistral_api_key,
        base_url="https://api.mistral.ai/v1",
        max_retries=3 if ENABLE_RETRIES else 0,
        request_timeout=TIMEOUT_SECONDS,
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

    # Local embeddings (cached after first load)
    print("📚 Loading embeddings (cached after first load)...")
    ragas_embeddings = HuggingFaceEmbeddings(
        model_name=RAGAS_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("✓ Models ready!\n")
    return ragas_llm, ragas_embeddings

# ============================================
# Rate Limiter
# ============================================
class RateLimiter:
    """Simple rate limiter to avoid overwhelming the API."""
    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self.last_call = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < self.delay:
                await asyncio.sleep(self.delay - time_since_last)
            self.last_call = time.time()

# Global rate limiter
rate_limiter = RateLimiter(RATE_LIMIT_DELAY)

# ============================================
# Async Evaluation Functions with Retry
# ============================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def evaluate_mode_with_retry(
    eval_dataset: Dataset,
    metrics: list,
    llm,
    embeddings
) -> dict:
    """Evaluate with retry logic for connection errors."""
    loop = asyncio.get_event_loop()

    # Add rate limiting
    await rate_limiter.acquire()

    # Run evaluation in thread pool
    eval_results = await loop.run_in_executor(
        None,
        lambda: evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False
        )
    )
    return eval_results

async def evaluate_mode_async(
    question_id: str,
    question_text: str,
    mode: str,
    mode_data: Dict[str, Any],
    expected_answer: str,
    semaphore: asyncio.Semaphore,
    ragas_llm,
    ragas_embeddings
) -> Dict[str, Any]:
    """
    Async evaluation of a single mode with improved error handling.
    """
    if mode_data.get("status") != "success":
        return {
            "question_id": question_id,
            "mode": mode,
            "status": "skipped"
        }

    async with semaphore:
        try:
            # Prepare dataset
            eval_dataset = Dataset.from_dict({
                "question": [question_text],
                "answer": [mode_data["answer"]],
                "contexts": [[mode_data["retrieved_context"]]],
                "ground_truth": [str(expected_answer)]
            })

            metrics = [
                Faithfulness(),
                AnswerRelevancy(),
                ContextRecall(),
                ContextPrecision()
            ]

            # Evaluate with retry logic
            if ENABLE_RETRIES:
                eval_results = await evaluate_mode_with_retry(
                    eval_dataset,
                    metrics,
                    ragas_llm,
                    ragas_embeddings
                )
            else:
                loop = asyncio.get_event_loop()
                await rate_limiter.acquire()
                eval_results = await loop.run_in_executor(
                    None,
                    lambda: evaluate(
                        dataset=eval_dataset,
                        metrics=metrics,
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

        except aiohttp.ClientError as e:
            return {
                "question_id": question_id,
                "mode": mode,
                "status": "connection_error",
                "error": f"API Connection Error: {str(e)[:100]}"
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
    semaphore: asyncio.Semaphore,
    ragas_llm,
    ragas_embeddings
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
                semaphore,
                ragas_llm,
                ragas_embeddings
            )
            tasks.append(task)

        # Wait for all modes to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

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
    semaphore: asyncio.Semaphore,
    ragas_llm,
    ragas_embeddings
) -> List[Dict[str, Any]]:
    """
    Process a batch of files with better error handling.
    """
    print(f"\n[Batch {batch_num}/{total_batches}] Starting {len(batch_files)} files...")
    start_time = time.time()

    # Process ALL files in parallel
    tasks = [process_file_async(f, semaphore, ragas_llm, ragas_embeddings) for f in batch_files]

    # Progress bar
    results_lists = []
    connection_errors = 0
    successes = 0

    with tqdm(total=len(tasks), desc=f"Batch {batch_num}") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results_lists.append(result)

                # Count errors
                for r in result:
                    if r.get("status") == "connection_error":
                        connection_errors += 1
                    elif r.get("status") == "success":
                        successes += 1

                pbar.update(1)
            except Exception as e:
                print(f"\n⚠️ Task failed: {e}")
                pbar.update(1)

    # Flatten results
    all_results = []
    for result_list in results_lists:
        all_results.extend(result_list)

    elapsed = time.time() - start_time
    print(f"[Batch {batch_num}] ✅ Completed in {elapsed:.1f}s "
          f"({successes} successful, {connection_errors} connection errors)")

    # If too many connection errors, pause briefly
    if connection_errors > len(batch_files):
        print(f"⚠️ High connection error rate. Pausing for 5 seconds...")
        await asyncio.sleep(5)

    return all_results

async def main():
    """
    Main async orchestrator with connection testing.
    """
    # Test API connection first
    if not await test_api_connection():
        print("\n❌ Cannot proceed without working API connection.")
        print("   Please check:")
        print("   1. Your MISTRAL_API_KEY is valid")
        print("   2. You have internet connectivity")
        print("   3. The Mistral API is accessible from your network")
        return

    # Setup models
    ragas_llm, ragas_embeddings = await setup_models()

    # Find all files
    results_path = Path(RESULTS_DIR)
    all_files = sorted(results_path.glob("test_results_*_question_*.json"))

    if not all_files:
        print(f"❌ No files found in {RESULTS_DIR}")
        return

    print(f"📊 Found {len(all_files)} files to process")

    # Confirmation
    if len(all_files) > 10:
        response = input(f"\n⚠ Process {len(all_files)} files? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Split into batches
    batches = [all_files[i:i+BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]
    total_batches = len(batches)

    print(f"📦 {total_batches} batches × {BATCH_SIZE} files")
    print(f"⏱️ Estimated time: {total_batches * 30:.0f}s ({total_batches * 30 / 60:.1f} min)")
    print("="*70)

    # Semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Process all batches
    all_results = []
    start_time = time.time()
    connection_error_count = 0

    for batch_num, batch_files in enumerate(batches, 1):
        batch_results = await process_batch_async(
            batch_files,
            batch_num,
            total_batches,
            semaphore,
            ragas_llm,
            ragas_embeddings
        )
        all_results.extend(batch_results)

        # Count connection errors
        batch_connection_errors = len([r for r in batch_results if r.get("status") == "connection_error"])
        connection_error_count += batch_connection_errors

        # If getting too many connection errors, increase delay
        if batch_connection_errors > len(batch_files) * 2:
            print(f"⚠️ Too many connection errors ({batch_connection_errors}). Adjusting rate limit...")
            rate_limiter.delay = min(rate_limiter.delay * 1.5, 5.0)

        # Save intermediate results
        async with aiofiles.open(OUTPUT_FILE, 'w') as f:
            await f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(all_results),
                "is_complete": batch_num == total_batches,
                "connection_errors": connection_error_count,
                "evaluations": all_results
            }, indent=2))

    total_time = time.time() - start_time

    # Calculate stats
    successful = len([r for r in all_results if r.get("status") == "success"])

    print("\n" + "="*70)
    print("⚡ EVALUATION COMPLETE!")
    print("="*70)
    print(f"Total Files:        {len(all_files)}")
    print(f"Total Evaluations:  {len(all_results)}")
    print(f"Successful:         {successful}")
    print(f"Connection Errors:  {connection_error_count}")
    print(f"Other Failures:     {len(all_results) - successful - connection_error_count}")
    print(f"Total Time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg per File:       {total_time/len(all_files):.1f}s")
    print(f"Success Rate:       {successful/len(all_results)*100:.1f}%")
    print()
    print(f"📁 Results saved to: {OUTPUT_FILE}")

    if connection_error_count > 0:
        print(f"\n⚠️ {connection_error_count} connection errors occurred.")
        print("   Consider:")
        print("   - Reducing MAX_CONCURRENT further")
        print("   - Increasing RATE_LIMIT_DELAY")
        print("   - Checking API quota/limits")

if __name__ == "__main__":
    asyncio.run(main())