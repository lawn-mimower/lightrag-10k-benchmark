#!/usr/bin/env python3
"""
SIMPLE BATCH RAGAS EVALUATION
==============================
Process one file at a time, all 5 modes in parallel.
Simple, reliable, no fancy async stuff.
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
# SIMPLE Configuration
# ============================================
RESULTS_DIR = "./lightrag-bench/5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL"
OUTPUT_FILE = "./lightrag-bench/batch_ragas_evaluation_results_simple.json"
CHECKPOINT_FILE = "./lightrag-bench/batch_ragas_checkpoint.json"

# Simple settings
TIMEOUT_SECONDS = 120  # 2 minutes timeout
MAX_WORKERS = 5  # Process 5 modes in parallel

# Models
RAGAS_JUDGE_MODEL = "ministral-14b-2512"
RAGAS_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Query modes
QUERY_MODES = ["local", "global", "naive", "hybrid", "mix"]

print("="*70)
print("📦 SIMPLE BATCH RAGAS EVALUATION")
print("="*70)
print(f"📁 Source: {Path(RESULTS_DIR).name}")
print(f"💾 Output: {Path(OUTPUT_FILE).name}")
print(f"⏱️ Timeout: {TIMEOUT_SECONDS} seconds per evaluation")
print(f"🔄 Processing: 1 file at a time, 5 modes in parallel")
print()

# ============================================
# Setup Models Once
# ============================================
print("Setting up models...")

mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    print("❌ MISTRAL_API_KEY not found!")
    exit(1)

# Create LLM - simple config that works
base_llm = ChatOpenAI(
    model=RAGAS_JUDGE_MODEL,
    api_key=mistral_api_key,
    base_url="https://api.mistral.ai/v1",
    max_retries=5,  # More retries
    request_timeout=TIMEOUT_SECONDS,
    temperature=0.1  # Add temperature for consistency
)

try:
    ragas_llm = LangchainLLMWrapper(
        langchain_llm=base_llm,
        bypass_n=True
    )
    print("✓ LLM ready")
except:
    ragas_llm = base_llm
    print("✓ LLM ready (standard mode)")

# Local embeddings
print("Loading embeddings...")
ragas_embeddings = HuggingFaceEmbeddings(
    model_name=RAGAS_EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✓ Embeddings ready")
print()

# ============================================
# Load checkpoint if exists
# ============================================
def load_checkpoint():
    """Load checkpoint to resume from where we left off."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
                print(f"📚 Resuming from checkpoint: {checkpoint['processed_files']} files already done")
                return set(checkpoint['processed_files'])
        except:
            pass
    return set()

def save_checkpoint(processed_files):
    """Save checkpoint after each file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'processed_files': list(processed_files)
        }, f)

# ============================================
# Simple Evaluation Function
# ============================================
def evaluate_single_mode(
    question_id: str,
    question_text: str,
    mode: str,
    mode_data: Dict[str, Any],
    expected_answer: str
) -> Dict[str, Any]:
    """
    Evaluate a single mode - simple and direct
    """
    if mode_data.get("status") != "success":
        return {
            "question_id": question_id,
            "mode": mode,
            "status": "skipped",
            "reason": "no_data"
        }

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

        return {
            "question_id": question_id,
            "mode": mode,
            "status": "success",
            "metrics": metrics,
            "ragas_score": round(ragas_score, 4)
        }

    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            return {
                "question_id": question_id,
                "mode": mode,
                "status": "timeout"
            }
        else:
            return {
                "question_id": question_id,
                "mode": mode,
                "status": "error",
                "error": error_msg[:200]
            }


def process_single_file(file_path: Path) -> Dict[str, Any]:
    """
    Process ONE file - evaluate all 5 modes in parallel
    """
    file_start = time.time()

    try:
        # Load file
        with open(file_path, 'r') as f:
            data = json.load(f)

        question_id = data["question_id"]
        question_text = data["question"]
        expected_answer = str(data["expected_answer"])

        # Process all 5 modes in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}

            # Submit all mode evaluations
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
                futures[future] = mode

            # Collect results
            mode_results = []
            for future in as_completed(futures):
                mode = futures[future]
                try:
                    result = future.result(timeout=TIMEOUT_SECONDS)
                    mode_results.append(result)
                except Exception as e:
                    mode_results.append({
                        "question_id": question_id,
                        "mode": mode,
                        "status": "timeout",
                        "error": str(e)[:100]
                    })

        file_time = time.time() - file_start

        # Count successes
        successes = sum(1 for r in mode_results if r.get("status") == "success")

        return {
            "file": file_path.name,
            "question_id": question_id,
            "processing_time": round(file_time, 1),
            "successful_modes": successes,
            "total_modes": len(mode_results),
            "mode_evaluations": mode_results
        }

    except Exception as e:
        return {
            "file": file_path.name,
            "status": "file_error",
            "error": str(e)[:200]
        }


def main():
    """
    Main function - process files one by one
    """
    # Find all files
    results_path = Path(RESULTS_DIR)
    all_files = sorted(results_path.glob("test_results_*_question_*.json"))

    if not all_files:
        print(f"❌ No files found in {RESULTS_DIR}")
        return

    print(f"📊 Found {len(all_files)} files to process")

    # Load checkpoint to skip already processed files
    processed_files = load_checkpoint()

    # Filter out already processed files
    remaining_files = [f for f in all_files if f.name not in processed_files]

    if len(remaining_files) < len(all_files):
        print(f"⏭️ Skipping {len(all_files) - len(remaining_files)} already processed files")
        print(f"📋 {len(remaining_files)} files left to process")

    if not remaining_files:
        print("✅ All files already processed!")
        return

    # Confirmation
    if len(remaining_files) > 10:
        response = input(f"\n⚠ Process {len(remaining_files)} files? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    print(f"\n📦 Processing {len(remaining_files)} files one by one")
    print(f"⏱️ Estimated time: {len(remaining_files) * 30:.0f}s ({len(remaining_files) * 30 / 60:.1f} min)")
    print("="*70)

    # Process each file
    all_results = []

    # Load existing results if resuming
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_data = json.load(f)
                all_results = existing_data.get("evaluations", [])
                print(f"📚 Loaded {len(all_results)} existing results")
        except:
            pass

    start_time = time.time()

    # Progress bar for all files
    with tqdm(total=len(remaining_files), desc="Files") as pbar:
        for file_num, file_path in enumerate(remaining_files, 1):
            # Process the file
            pbar.set_description(f"File {file_num}/{len(remaining_files)}: {file_path.name[:30]}")

            result = process_single_file(file_path)
            all_results.append(result)

            # Update checkpoint
            processed_files.add(file_path.name)
            save_checkpoint(processed_files)

            # Save results after each file
            with open(OUTPUT_FILE, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "total_files_processed": len(processed_files),
                    "total_files": len(all_files),
                    "is_complete": len(processed_files) == len(all_files),
                    "evaluations": all_results
                }, f, indent=2)

            # Update progress
            if "successful_modes" in result:
                pbar.set_postfix({
                    "success": f"{result['successful_modes']}/5",
                    "time": f"{result.get('processing_time', 0):.1f}s"
                })

            pbar.update(1)

            # Small pause between files to avoid overwhelming API
            if file_num < len(remaining_files):
                time.sleep(1)

    total_time = time.time() - start_time

    # Calculate final stats
    total_evaluations = sum(
        r.get("total_modes", 0) for r in all_results
        if "total_modes" in r
    )
    successful_evaluations = sum(
        r.get("successful_modes", 0) for r in all_results
        if "successful_modes" in r
    )

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"Total Files:        {len(all_files)}")
    print(f"Processed:          {len(processed_files)}")
    print(f"Total Evaluations:  {total_evaluations}")
    print(f"Successful:         {successful_evaluations}")
    print(f"Failed:             {total_evaluations - successful_evaluations}")
    print(f"Total Time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg per File:       {total_time/len(remaining_files):.1f}s")

    if successful_evaluations > 0:
        print(f"Success Rate:       {successful_evaluations/total_evaluations*100:.1f}%")

    print()
    print(f"📁 Results saved to: {OUTPUT_FILE}")

    # Clean up checkpoint file
    if len(processed_files) == len(all_files):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("🧹 Checkpoint file removed (all done)")


if __name__ == "__main__":
    main()