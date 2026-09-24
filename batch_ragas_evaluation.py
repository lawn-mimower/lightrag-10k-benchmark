#!/usr/bin/env python3
"""
ULTRA SIMPLE BATCH RAGAS EVALUATION
====================================
Maximum simplicity, maximum reliability.
Process one file at a time, one mode at a time.

Strategies (--strategy):
- sequential (default): fixed delays (2s between modes, 3s between files),
  retries on connection errors, API connection test before starting.
- adaptive: modes in pairs with an adaptive per-call delay that speeds up
  after successes and backs off after errors (1s between pairs, 5s between
  files).
"""

import os
import json
import argparse
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
# Configuration
# ============================================
ALL_QUERY_MODES = ["local", "global", "naive", "hybrid", "mix"]

STRATEGIES = {
    # Ultra conservative settings
    "sequential": {
        "delay_between_modes": 2,  # 2 second delay between mode evaluations
        "delay_between_files": 3,  # 3 second delay between files
        "llm_max_retries": 5,
        "llm_timeout": 180,  # 3 minutes
        "output": "batch_ragas_evaluation_results_ultra_simple.json",
        "checkpoint": "batch_ragas_checkpoint_ultra.json",
    },
    # Balance of speed and reliability
    "adaptive": {
        "modes_per_group": 2,  # Evaluate modes in pairs
        "delay_between_modes": 1,  # 1 second delay between mode pairs
        "delay_between_files": 5,  # 5 second delay between files
        "llm_max_retries": 3,
        "llm_timeout": 120,
        "output": "batch_ragas_evaluation_results_optimized.json",
        "checkpoint": "batch_ragas_checkpoint_optimized.json",
    },
}

parser = argparse.ArgumentParser(
    description="RAGAS evaluation of per-question 5-mode LightRAG results"
)
parser.add_argument(
    "--results-dir",
    default=os.getenv("RESULTS_DIR", "5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL"),
    help="Directory with test_results_*_question_*.json files (env RESULTS_DIR)",
)
parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="sequential",
                    help="sequential (default): fixed delays and retries; "
                         "adaptive: modes in pairs with an adaptive per-call delay")
parser.add_argument("--output", default=os.getenv("OUTPUT_FILE"),
                    help="Results JSON (env OUTPUT_FILE; default batch_ragas_evaluation_results_ultra_simple.json, "
                         "or batch_ragas_evaluation_results_optimized.json for the adaptive strategy)")
parser.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_FILE"),
                    help="Checkpoint JSON used to resume (env CHECKPOINT_FILE; default batch_ragas_checkpoint_ultra.json, "
                         "or batch_ragas_checkpoint_optimized.json for the adaptive strategy)")
parser.add_argument("--modes", nargs="+", choices=ALL_QUERY_MODES, default=ALL_QUERY_MODES,
                    help="Query modes to evaluate")
parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N question files")
args = parser.parse_args()

STRATEGY = args.strategy
SETTINGS = STRATEGIES[STRATEGY]

RESULTS_DIR = args.results_dir
OUTPUT_FILE = args.output or SETTINGS["output"]
CHECKPOINT_FILE = args.checkpoint or SETTINGS["checkpoint"]

DELAY_BETWEEN_MODES = SETTINGS["delay_between_modes"]
DELAY_BETWEEN_FILES = SETTINGS["delay_between_files"]
MAX_RETRIES = 3  # Retry failed evaluations (sequential strategy)

# Models
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "ministral-14b-2512")
RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
# Any OpenAI-compatible endpoint works; defaults to the Mistral API
MISTRAL_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")

# Query modes
QUERY_MODES = args.modes

print("="*70)
if STRATEGY == "adaptive":
    print("⚡ BATCH RAGAS EVALUATION - ADAPTIVE RATE LIMITING")
else:
    print("🐌 ULTRA SIMPLE BATCH RAGAS EVALUATION")
print("="*70)
print(f"📁 Source: {Path(RESULTS_DIR).name}")
print(f"💾 Output: {Path(OUTPUT_FILE).name}")
if STRATEGY == "adaptive":
    print(f"🔄 Mode: Semi-parallel ({SETTINGS['modes_per_group']} modes at once)")
    print(f"⏱️ Delays: Adaptive rate limiting, {DELAY_BETWEEN_MODES}s between mode pairs, {DELAY_BETWEEN_FILES}s between files")
else:
    print(f"🐢 Mode: Sequential (1 file → 1 mode at a time)")
    print(f"⏰ Delays: {DELAY_BETWEEN_MODES}s between modes, {DELAY_BETWEEN_FILES}s between files")
print()


def test_connection(api_key: str):
    """Quick connection test; exits when the judge endpoint does not answer."""
    try:
        test_llm = ChatOpenAI(
            model=RAGAS_JUDGE_MODEL,
            api_key=api_key,
            base_url=MISTRAL_BASE_URL,
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
# Test Connection First
# ============================================
if STRATEGY == "sequential":
    print("🔍 Testing API connection...")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    print("❌ MISTRAL_API_KEY not found!")
    exit(1)

if STRATEGY == "sequential":
    test_connection(mistral_api_key)

# ============================================
# Setup Models (KNOWN WORKING CONFIG)
# ============================================
print("\n📦 Setting up models...")

# Create LLM with EXACT working config
base_llm = ChatOpenAI(
    model=RAGAS_JUDGE_MODEL,
    api_key=mistral_api_key,
    base_url=MISTRAL_BASE_URL,
    max_retries=SETTINGS["llm_max_retries"],
    request_timeout=SETTINGS["llm_timeout"]
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
# Rate Limiter (adaptive strategy)
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
# Evaluation
# ============================================
def run_ragas(question_text: str, mode_data: dict[str, any], expected_answer: str) -> tuple[dict, float]:
    """Score one answer with the four RAGAS metrics; returns (metrics, ragas_score)."""
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
    return metrics, ragas_score


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
        metrics, ragas_score = run_ragas(question_text, mode_data, expected_answer)

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


def evaluate_single_mode_adaptive(
    question_id: str,
    question_text: str,
    mode: str,
    mode_data: dict[str, any],
    expected_answer: str
) -> dict[str, any]:
    """
    Evaluate with the adaptive rate limiter (no retries).
    """
    if mode_data.get("status") != "success":
        return {
            "question_id": question_id,
            "mode": mode,
            "status": "skipped",
            "reason": "no_data"
        }

    # Rate limit before each evaluation
    rate_limiter.wait()

    try:
        metrics, ragas_score = run_ragas(question_text, mode_data, expected_answer)
        rate_limiter.record_success()

        return {
            "question_id": question_id,
            "mode": mode,
            "status": "success",
            "metrics": metrics,
            "ragas_score": round(ragas_score, 4)
        }

    except Exception as e:
        rate_limiter.record_error()

        # Extra delay after error
        time.sleep(2)

        return {
            "question_id": question_id,
            "mode": mode,
            "status": "error",
            "error": str(e)[:200]
        }


def pause_after_mode(mode: str):
    """Delay before the next mode: after every mode (sequential) or every pair (adaptive)."""
    position = QUERY_MODES.index(mode) + 1
    if position >= len(QUERY_MODES):  # Not after the last mode
        return
    if STRATEGY == "adaptive" and position % SETTINGS["modes_per_group"] != 0:
        return
    time.sleep(DELAY_BETWEEN_MODES)


def main():
    """
    Main function - ULTRA SIMPLE approach
    """
    # Find all files
    results_path = Path(RESULTS_DIR)
    all_files = sorted(results_path.glob("test_results_*_question_*.json"))
    if args.limit:
        all_files = all_files[:args.limit]

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
    print(f"🚀 STARTING {STRATEGY.upper()} PROCESSING")
    print("="*70)

    evaluate_mode = evaluate_single_mode_adaptive if STRATEGY == "adaptive" else evaluate_single_mode_with_retry

    start_time = time.time()
    evaluation_count = 0

    # Process each file
    with tqdm(total=remaining_evaluations, desc="Evaluations", initial=completed_evaluations) as pbar:
        for file_index, file_path in enumerate(all_files):
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

                    # Evaluate
                    result = evaluate_mode(
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
                    pause_after_mode(mode)

                # Delay between files (the adaptive strategy skips it after the last file)
                if STRATEGY == "sequential" or file_index < len(all_files) - 1:
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
