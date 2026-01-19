#!/usr/bin/env python3
"""
ULTRA SIMPLE BATCH RAGAS EVALUATION
====================================
Maximum simplicity, maximum reliability.
Process one file at a time, one mode at a time.
"""

import os
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

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
# ULTRA SIMPLE Configuration
# ============================================
RESULTS_DIR = "./lightrag-bench/5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL"
OUTPUT_FILE = "./lightrag-bench/batch_ragas_evaluation_results_ultra_simple.json"
CHECKPOINT_FILE = "./lightrag-bench/batch_ragas_checkpoint_ultra.json"

# Ultra conservative settings
DELAY_BETWEEN_MODES = 2  # 2 second delay between mode evaluations
DELAY_BETWEEN_FILES = 3  # 3 second delay between files
MAX_RETRIES = 3  # Retry failed evaluations

# Models
RAGAS_JUDGE_MODEL = "ministral-14b-2512"
RAGAS_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Query modes
QUERY_MODES = ["local", "global", "naive", "hybrid", "mix"]

print("="*70)
print("🐌 ULTRA SIMPLE BATCH RAGAS EVALUATION")
print("="*70)
print(f"📁 Source: {Path(RESULTS_DIR).name}")
print(f"💾 Output: {Path(OUTPUT_FILE).name}")
print(f"🐢 Mode: Sequential (1 file → 1 mode at a time)")
print(f"⏰ Delays: {DELAY_BETWEEN_MODES}s between modes, {DELAY_BETWEEN_FILES}s between files")
print()

# ============================================
# Test Connection First
# ============================================
print("🔍 Testing API connection...")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    print("❌ MISTRAL_API_KEY not found!")
    exit(1)

# Quick connection test
try:
    test_llm = ChatOpenAI(
        model=RAGAS_JUDGE_MODEL,
        api_key=mistral_api_key,
        base_url="https://api.mistral.ai/v1",
        max_retries=2,
        request_timeout=30
    )
    response = test_llm.invoke("Say 'ok' in one word")
    print(f"✅ API connection working: {response.content}")
except Exception as e:
    print(f"❌ API connection failed: {e}")
    print("\nPlease check:")
    print("1. Your MISTRAL_API_KEY is valid")
    print("2. You have internet connection")
    print("3. Mistral API is accessible")
    exit(1)

# ============================================
# Setup Models (KNOWN WORKING CONFIG)
# ============================================
print("\n📦 Setting up models...")

# Create LLM with EXACT working config
base_llm = ChatOpenAI(
    model=RAGAS_JUDGE_MODEL,
    api_key=mistral_api_key,
    base_url="https://api.mistral.ai/v1",
    max_retries=5,
    request_timeout=180  # 3 minutes
)

try:
    ragas_llm = LangchainLLMWrapper(
        langchain_llm=base_llm,
        bypass_n=True
    )
    print("✓ LLM ready (bypass_n mode)")
except:
    ragas_llm = base_llm
    print("✓ LLM ready")

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
# Checkpoint Management
# ============================================
def load_checkpoint():
    """Load checkpoint to resume."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
                return checkpoint
        except:
            pass
    return {"processed": {}, "results": []}

def save_checkpoint(checkpoint):
    """Save checkpoint."""
    checkpoint['timestamp'] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

# ============================================
# ULTRA SIMPLE Evaluation
# ============================================
def evaluate_single_mode_with_retry(
    question_id: str,
    question_text: str,
    mode: str,
    mode_data: dict[str, any],
    expected_answer: str,
    retry_count: int = 0
) -> dict[str, any]:
    """
    Evaluate with retry logic.
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
        error_str = str(e)

        # Check if it's a connection error and we can retry
        if "connection" in error_str.lower() and retry_count < MAX_RETRIES:
            print(f"    ⚠️ Connection error, retrying ({retry_count + 1}/{MAX_RETRIES})...")
            time.sleep(5 * (retry_count + 1))  # Exponential backoff
            return evaluate_single_mode_with_retry(
                question_id, question_text, mode, mode_data, expected_answer,
                retry_count + 1
            )

        # Otherwise return error
        return {
            "question_id": question_id,
            "mode": mode,
            "status": "error",
            "error": error_str[:200],
            "retries": retry_count
        }


def main():
    """
    Main function - ULTRA SIMPLE approach
    """
    # Find all files
    results_path = Path(RESULTS_DIR)
    all_files = sorted(results_path.glob("test_results_*_question_*.json"))

    if not all_files:
        print(f"❌ No files found in {RESULTS_DIR}")
        return

    print(f"📊 Found {len(all_files)} files to process")

    # Load checkpoint
    checkpoint = load_checkpoint()
    processed = checkpoint.get("processed", {})
    all_results = checkpoint.get("results", [])

    # Count remaining work
    total_evaluations = len(all_files) * len(QUERY_MODES)
    completed_evaluations = sum(len(modes) for modes in processed.values())
    remaining_evaluations = total_evaluations - completed_evaluations

    if completed_evaluations > 0:
        print(f"📚 Resuming: {completed_evaluations}/{total_evaluations} already done")

    if remaining_evaluations == 0:
        print("✅ All evaluations complete!")
        return

    # Confirmation
    if remaining_evaluations > 50:
        print(f"\n⚠ {remaining_evaluations} evaluations to process")
        print(f"⏱️ Estimated time: {remaining_evaluations * 15:.0f}s ({remaining_evaluations * 15 / 60:.1f} min)")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    print("\n" + "="*70)
    print("🚀 STARTING SEQUENTIAL PROCESSING")
    print("="*70)

    start_time = time.time()
    evaluation_count = 0

    # Process each file
    with tqdm(total=remaining_evaluations, desc="Evaluations", initial=completed_evaluations) as pbar:
        for file_path in all_files:
            file_name = file_path.name

            # Skip if already fully processed
            if file_name in processed and len(processed[file_name]) == len(QUERY_MODES):
                continue

            # Load file
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                question_id = data["question_id"]
                question_text = data["question"]
                expected_answer = str(data["expected_answer"])

                # Process each mode sequentially
                for mode in QUERY_MODES:
                    # Skip if already processed
                    if file_name in processed and mode in processed.get(file_name, []):
                        continue

                    mode_data = data["modes"].get(mode, {})

                    # Show current task
                    pbar.set_description(f"{question_id[:8]}:{mode}")

                    # Evaluate with retry
                    result = evaluate_single_mode_with_retry(
                        question_id,
                        question_text,
                        mode,
                        mode_data,
                        expected_answer
                    )

                    # Store result
                    result['file'] = file_name
                    all_results.append(result)

                    # Update checkpoint
                    if file_name not in processed:
                        processed[file_name] = []
                    processed[file_name].append(mode)

                    checkpoint = {"processed": processed, "results": all_results}
                    save_checkpoint(checkpoint)

                    # Save full results
                    with open(OUTPUT_FILE, 'w') as f:
                        json.dump({
                            "timestamp": datetime.now().isoformat(),
                            "evaluations_completed": len(all_results),
                            "evaluations_total": total_evaluations,
                            "is_complete": len(all_results) == total_evaluations,
                            "results": all_results
                        }, f, indent=2)

                    # Update progress
                    evaluation_count += 1
                    pbar.update(1)

                    if result.get("status") == "success":
                        pbar.set_postfix({"✓": evaluation_count, "mode": mode})
                    else:
                        pbar.set_postfix({"✓": evaluation_count, "⚠": result.get("status")})

                    # Delay between modes
                    if mode != QUERY_MODES[-1]:  # Not the last mode
                        time.sleep(DELAY_BETWEEN_MODES)

                # Delay between files
                time.sleep(DELAY_BETWEEN_FILES)

            except Exception as e:
                print(f"\n❌ Error processing {file_name}: {e}")
                continue

    total_time = time.time() - start_time

    # Final stats
    successful = len([r for r in all_results if r.get("status") == "success"])

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"Total Evaluations:  {len(all_results)}")
    print(f"Successful:         {successful}")
    print(f"Failed/Skipped:     {len(all_results) - successful}")
    print(f"Total Time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg per Evaluation: {total_time/len(all_results):.1f}s")

    if successful > 0:
        print(f"Success Rate:       {successful/len(all_results)*100:.1f}%")

    print()
    print(f"📁 Results saved to: {OUTPUT_FILE}")

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("🧹 Checkpoint removed (all done)")


if __name__ == "__main__":
    main()