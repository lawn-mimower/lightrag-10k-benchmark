#!/usr/bin/env python3
"""
BATCH RAGAS EVALUATION - FIXED VERSION 2
=========================================
Based on the WORKING single file evaluation script.
Uses exact same configuration that we know works.
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
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# Configuration - MATCHING WORKING SCRIPT
# ============================================
RESULTS_DIR = "./lightrag-bench/5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL"
OUTPUT_FILE = "./lightrag-bench/batch_ragas_evaluation_results_fixed_v2.json"

# Processing settings - conservative to ensure stability
BATCH_SIZE = 3  # Process 3 files at once (conservative)
MAX_WORKERS = 5  # Thread pool workers

# Models - EXACT SAME AS WORKING SCRIPT
RAGAS_JUDGE_MODEL = "ministral-14b-2512"
RAGAS_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Query modes
QUERY_MODES = ["local", "global", "naive", "hybrid", "mix"]

print("="*70)
print("📦 BATCH RAGAS EVALUATION - FIXED V2")
print("="*70)
print(f"📁 Source: {Path(RESULTS_DIR).name}")
print(f"💾 Output: {Path(OUTPUT_FILE).name}")
print(f"🚀 Batch Size: {BATCH_SIZE} files")
print(f"⚡ Workers: {MAX_WORKERS}")
print()

# ============================================
# Setup Models - EXACT COPY FROM WORKING SCRIPT
# ============================================
print("Setting up RAGAS models...")

# Check API key
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    print("❌ MISTRAL_API_KEY not found!")
    exit(1)

# Create LLM - EXACT SAME PARAMETERS AS WORKING SCRIPT
base_llm = ChatOpenAI(
    model=RAGAS_JUDGE_MODEL,
    api_key=mistral_api_key,
    base_url="https://api.mistral.ai/v1",
    max_retries=5,
    request_timeout=180
    # NO max_tokens parameter!
)

# Wrap with LangchainLLMWrapper
try:
    ragas_llm = LangchainLLMWrapper(
        langchain_llm=base_llm,
        bypass_n=True
    )
    print("✓ LLM configured with bypass_n mode")
except:
    ragas_llm = base_llm
    print("✓ LLM configured (standard mode)")

# Local embeddings - EXACT SAME AS WORKING SCRIPT
print("Loading local embeddings (this may take a moment on first run)...")
ragas_embeddings = HuggingFaceEmbeddings(
    model_name=RAGAS_EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✓ Local embeddings configured (no API calls needed!)")
print()

# ============================================
# Simple Evaluation Functions
# ============================================

def evaluate_mode(
    question_id: str,
    question_text: str,
    mode: str,
    mode_data: Dict[str, Any],
    expected_answer: str
) -> Dict[str, Any]:
    """
    Evaluate a single mode - BASED ON WORKING SCRIPT
    """
    if mode_data.get("status") != "success":
        return {
            "question_id": question_id,
            "mode": mode,
            "status": "skipped"
        }

    try:
        # Prepare dataset - EXACT SAME AS WORKING SCRIPT
        eval_dataset = Dataset.from_dict({
            "question": [question_text],
            "answer": [mode_data["answer"]],
            "contexts": [[mode_data["retrieved_context"]]],  # List of list!
            "ground_truth": [str(expected_answer)]
        })

        # Run evaluation - EXACT SAME AS WORKING SCRIPT
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
            show_progress=False  # Cleaner output
        )

        # Extract scores - EXACT SAME AS WORKING SCRIPT
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

    except Exception as e:
        return {
            "question_id": question_id,
            "mode": mode,
            "status": "error",
            "error": str(e)[:200]
        }


def process_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Process one file - all modes sequentially
    """
    try:
        # Load file
        with open(file_path, 'r') as f:
            data = json.load(f)

        question_id = data["question_id"]
        question_text = data["question"]
        expected_answer = str(data["expected_answer"])

        results = []

        # Process each mode sequentially (safer)
        for mode in QUERY_MODES:
            mode_data = data["modes"].get(mode, {})
            result = evaluate_mode(
                question_id,
                question_text,
                mode,
                mode_data,
                expected_answer
            )
            results.append(result)

            # Small delay between modes to avoid overwhelming API
            time.sleep(0.5)

        return results

    except Exception as e:
        return [{
            "file": str(file_path.name),
            "status": "file_error",
            "error": str(e)[:100]
        }]


def process_batch(batch_files: List[Path], batch_num: int, total_batches: int) -> List[Dict[str, Any]]:
    """
    Process a batch of files using ThreadPoolExecutor
    """
    print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch_files)} files...")
    start_time = time.time()

    all_results = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all files
        future_to_file = {executor.submit(process_file, f): f for f in batch_files}

        # Process completed files with progress bar
        with tqdm(total=len(batch_files), desc=f"Batch {batch_num}") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    results = future.result()
                    all_results.extend(results)

                    # Count successes
                    success_count = sum(1 for r in results if r.get("status") == "success")
                    pbar.set_postfix({"ok": success_count, "file": file_path.name[:20]})

                except Exception as e:
                    all_results.append({
                        "file": str(file_path.name),
                        "status": "exception",
                        "error": str(e)[:100]
                    })

                pbar.update(1)

    elapsed = time.time() - start_time
    successful = len([r for r in all_results if r.get("status") == "success"])

    print(f"[Batch {batch_num}] ✅ Completed in {elapsed:.1f}s ({successful} successful)")

    return all_results


def main():
    """
    Main function - simple and straightforward
    """
    # Find all files
    results_path = Path(RESULTS_DIR)
    all_files = sorted(results_path.glob("test_results_*_question_*.json"))

    if not all_files:
        print(f"❌ No files found in {RESULTS_DIR}")
        return

    print(f"📊 Found {len(all_files)} files to process")

    # Confirmation for large batches
    if len(all_files) > 10:
        response = input(f"\n⚠ Process {len(all_files)} files? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Split into batches
    batches = [all_files[i:i+BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]
    total_batches = len(batches)

    print(f"📦 {total_batches} batches × {BATCH_SIZE} files")
    print(f"⏱️ Estimated time: {total_batches * 60:.0f}s ({total_batches:.0f} min)")
    print("="*70)

    # Process all batches
    all_results = []
    start_time = time.time()

    for batch_num, batch_files in enumerate(batches, 1):
        batch_results = process_batch(batch_files, batch_num, total_batches)
        all_results.extend(batch_results)

        # Save intermediate results
        with open(OUTPUT_FILE, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(all_results),
                "is_complete": batch_num == total_batches,
                "evaluations": all_results
            }, f, indent=2)

        # Pause between batches to avoid overwhelming API
        if batch_num < total_batches:
            print(f"  Pausing for 2 seconds before next batch...")
            time.sleep(2)

    total_time = time.time() - start_time

    # Calculate stats
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
    print(f"Success Rate:       {successful/len(all_results)*100:.1f}%")
    print()
    print(f"📁 Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()