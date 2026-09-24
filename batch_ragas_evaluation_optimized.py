#!/usr/bin/env python3
"""
OPTIMIZED BATCH RAGAS EVALUATION
=================================
Middle ground between speed and reliability.
Uses connection pooling and smart rate limiting.
"""

import os
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress warnings
warnings.filterwarnings("ignore")

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
RESULTS_DIR = os.getenv("RESULTS_DIR", "5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "batch_ragas_evaluation_results_optimized.json")
CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "batch_ragas_checkpoint_optimized.json")

# Optimized settings - balance of speed and reliability
MAX_CONCURRENT_MODES = 2  # Process 2 modes at once (not 5)
DELAY_BETWEEN_FILES = 5  # 5 second delay between files
DELAY_BETWEEN_MODES = 1  # 1 second delay between mode pairs

# Models
RAGAS_JUDGE_MODEL = "ministral-14b-2512"
RAGAS_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Query modes
QUERY_MODES = ["local", "global", "naive", "hybrid", "mix"]

print("="*70)
print("⚡ OPTIMIZED BATCH RAGAS EVALUATION")
print("="*70)
print(f"📁 Source: {Path(RESULTS_DIR).name}")
print(f"💾 Output: {Path(OUTPUT_FILE).name}")
print(f"🔄 Mode: Semi-parallel (2 modes at once)")
print(f"⏱️ Delays: Adaptive rate limiting")
print()

# ============================================
# Connection Pool Setup
# ============================================
print("🔧 Setting up connection pool...")

# Create a session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=2,  # Limit connection pool
    pool_maxsize=2,
)
session.mount("http://", adapter)
session.mount("https://", adapter)

# ============================================
# Setup Models with Shared Session
# ============================================
print("📦 Setting up models...")

mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    print("❌ MISTRAL_API_KEY not found!")
    exit(1)

# Create LLM with connection limits
base_llm = ChatOpenAI(
    model=RAGAS_JUDGE_MODEL,
    api_key=mistral_api_key,
    base_url="https://api.mistral.ai/v1",
    max_retries=3,
    request_timeout=120,
    http_client=session  # Use our custom session if supported
)

try:
    ragas_llm = LangchainLLMWrapper(
        langchain_llm=base_llm,
        bypass_n=True
    )
    print("✓ LLM ready (with connection pooling)")
except:
    ragas_llm = base_llm
    print("✓ LLM ready")

# Local embeddings (cached)
print("Loading embeddings...")
ragas_embeddings = HuggingFaceEmbeddings(
    model_name=RAGAS_EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✓ Embeddings ready")
print()

# ============================================
# Rate Limiter Class
# ============================================
class AdaptiveRateLimiter:
    """Adaptive rate limiting based on success/failure."""
    def __init__(self):
        self.delay = 0.5  # Start with small delay
        self.success_count = 0
        self.error_count = 0
        self.last_call = 0

    def wait(self):
        """Wait appropriate time before next call."""
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()

    def record_success(self):
        """Record successful call and potentially speed up."""
        self.success_count += 1
        self.error_count = 0
        if self.success_count > 5:
            self.delay = max(0.2, self.delay * 0.9)  # Speed up
            self.success_count = 0

    def record_error(self):
        """Record error and slow down."""
        self.error_count += 1
        self.delay = min(5.0, self.delay * 1.5)  # Slow down
        print(f"  ⚠️ Adjusting rate limit to {self.delay:.1f}s")

rate_limiter = AdaptiveRateLimiter()

# ============================================
# Optimized Evaluation
# ============================================
def evaluate_mode_batch(
    question_id: str,
    question_text: str,
    modes: List[str],
    modes_data: Dict[str, Any],
    expected_answer: str
) -> List[Dict[str, Any]]:
    """
    Evaluate a small batch of modes with rate limiting.
    """
    results = []

    for mode in modes:
        mode_data = modes_data.get(mode, {})

        if mode_data.get("status") != "success":
            results.append({
                "question_id": question_id,
                "mode": mode,
                "status": "skipped"
            })
            continue

        # Rate limit before each evaluation
        rate_limiter.wait()

        try:
            # Prepare dataset
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

            results.append({
                "question_id": question_id,
                "mode": mode,
                "status": "success",
                "metrics": metrics,
                "ragas_score": round(ragas_score, 4)
            })

            rate_limiter.record_success()

        except Exception as e:
            results.append({
                "question_id": question_id,
                "mode": mode,
                "status": "error",
                "error": str(e)[:200]
            })
            rate_limiter.record_error()

            # Extra delay after error
            time.sleep(2)

    return results


def main():
    """
    Main function - optimized approach
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
        response = input(f"\n⚠ Process {len(all_files)} files? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    print("\n" + "="*70)
    print("🚀 STARTING OPTIMIZED PROCESSING")
    print("="*70)

    all_results = []
    start_time = time.time()

    # Process each file
    with tqdm(total=len(all_files), desc="Files") as pbar:
        for file_idx, file_path in enumerate(all_files):
            try:
                # Load file
                with open(file_path, 'r') as f:
                    data = json.load(f)

                question_id = data["question_id"]
                question_text = data["question"]
                expected_answer = str(data["expected_answer"])

                # Process modes in small batches
                file_results = []
                for i in range(0, len(QUERY_MODES), MAX_CONCURRENT_MODES):
                    batch_modes = QUERY_MODES[i:i + MAX_CONCURRENT_MODES]
                    batch_results = evaluate_mode_batch(
                        question_id,
                        question_text,
                        batch_modes,
                        data["modes"],
                        expected_answer
                    )
                    file_results.extend(batch_results)

                    # Small delay between mode batches
                    if i + MAX_CONCURRENT_MODES < len(QUERY_MODES):
                        time.sleep(DELAY_BETWEEN_MODES)

                # Add file info to results
                for result in file_results:
                    result['file'] = file_path.name
                    all_results.append(result)

                # Save after each file
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "files_processed": file_idx + 1,
                        "total_files": len(all_files),
                        "evaluations": all_results
                    }, f, indent=2)

                # Update progress
                success_count = sum(1 for r in file_results if r.get("status") == "success")
                pbar.set_postfix({"✓": success_count, "file": file_path.name[:20]})
                pbar.update(1)

                # Delay between files
                if file_idx < len(all_files) - 1:
                    time.sleep(DELAY_BETWEEN_FILES)

            except Exception as e:
                print(f"\n❌ Error with {file_path.name}: {e}")
                pbar.update(1)

    total_time = time.time() - start_time

    # Stats
    successful = len([r for r in all_results if r.get("status") == "success"])

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"Total Files:        {len(all_files)}")
    print(f"Total Evaluations:  {len(all_results)}")
    print(f"Successful:         {successful}")
    print(f"Failed:             {len(all_results) - successful}")
    print(f"Total Time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg per File:       {total_time/len(all_files):.1f}s")
    print()
    print(f"📁 Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()