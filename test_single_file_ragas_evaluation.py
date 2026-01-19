#!/usr/bin/env python3
"""
Test RAGAS Evaluation on Single File
=====================================
This script demonstrates RAGAS evaluation on ONE test result file
to understand the process before batch processing.

What will happen:
1. Load one test result JSON file (with 5 query modes)
2. For each mode (local, global, naive, hybrid, mix):
   - Extract the question, answer, context, and ground truth
   - Run RAGAS evaluation to get 4 metrics
   - Calculate aggregate RAGAS score
3. Display results in a clear format
4. Save to a single JSON output file
"""

import os
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np

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
from langchain_huggingface import HuggingFaceEmbeddings  # Local embeddings!
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*70)
print("🧪 SINGLE FILE RAGAS EVALUATION TEST")
print("="*70)
print()

# ============================================
# Configuration
# ============================================
TEST_FILE = "./lightrag-bench/5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL/test_results_CTAS_question_f3a3d09b.json"
OUTPUT_FILE = "./lightrag-bench/single_file_ragas_test_results.json"

RAGAS_JUDGE_MODEL = "ministral-14b-2512"  # Correct Mistral API identifier
RAGAS_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Local model, 1024 dims

print("📁 Test File: test_results_CTAS_question_f3a3d09b.json")
print("🎯 Evaluation Models:")
print(f"   • Judge LLM: {RAGAS_JUDGE_MODEL} (Mistral API)")
print(f"   • Embeddings: {RAGAS_EMBEDDING_MODEL} (LOCAL, 1024 dims)")
print()

# ============================================
# Setup RAGAS Models
# ============================================
print("Setting up RAGAS models...")

# Create LLM for RAGAS
base_llm = ChatOpenAI(
    model=RAGAS_JUDGE_MODEL,
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
    max_retries=5,
    request_timeout=180
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

# Create LOCAL embeddings using HuggingFace
# Note: This will download the model (~1.3GB) on first run
print("Loading local embeddings (this may take a moment on first run)...")
ragas_embeddings = HuggingFaceEmbeddings(
    model_name=RAGAS_EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},  # Use CPU to avoid GPU memory conflicts
    encode_kwargs={'normalize_embeddings': True}
)
print("✓ Local embeddings configured (no API calls needed!)")
print()

# ============================================
# Load Test File
# ============================================
print("Loading test file...")
with open(TEST_FILE, 'r') as f:
    test_data = json.load(f)

question_id = test_data["question_id"]
question_text = test_data["question"]
expected_answer = str(test_data["expected_answer"])

print(f"✓ Question ID: {question_id}")
print(f"✓ Question: {question_text[:80]}...")
print(f"✓ Modes to evaluate: {list(test_data['modes'].keys())}")
print()

# ============================================
# Evaluate Each Mode
# ============================================
print("="*70)
print("🚀 STARTING EVALUATION")
print("="*70)
print()
print("What's happening behind the scenes:")
print("1. For each mode, RAGAS will:")
print("   - Extract claims from the answer (for Faithfulness)")
print("   - Generate questions from the answer (for Relevancy)")
print("   - Check context coverage (for Recall)")
print("   - Analyze ranking quality (for Precision)")
print("2. Each metric involves:")
print("   - 1-3 LLM calls to Ministral (for judgment)")
print("   - LOCAL embeddings using BAAI/bge-large-en-v1.5 (no API!)")
print("3. Benefits of local embeddings:")
print("   - No OpenAI API costs")
print("   - No rate limiting on embeddings")
print("   - Faster processing (no network latency)")
print("   - Complete data privacy")
print()

all_results = {}
start_time = time.time()

for mode in ["local", "global", "naive", "hybrid", "mix"]:
    mode_data = test_data["modes"].get(mode, {})

    if mode_data.get("status") != "success":
        print(f"⚠ {mode.upper():8s}: Skipped (no successful generation)")
        continue

    print(f"Evaluating {mode.upper()} mode:")
    print(f"  • Answer length: {len(mode_data['answer']):,} chars")
    print(f"  • Context length: {len(mode_data['retrieved_context']):,} chars")

    # Prepare RAGAS dataset
    eval_dataset = Dataset.from_dict({
        "question": [question_text],
        "answer": [mode_data["answer"]],
        "contexts": [[mode_data["retrieved_context"]]],  # List of list!
        "ground_truth": [expected_answer]
    })

    try:
        # Run RAGAS evaluation
        print(f"  • Running RAGAS evaluation...")
        eval_start = time.time()

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

        eval_time = time.time() - eval_start

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

        # Store results
        all_results[mode] = {
            "metrics": metrics,
            "ragas_score": round(ragas_score, 4),
            "evaluation_time_seconds": round(eval_time, 2),
            "timestamp": datetime.now().isoformat()
        }

        # Display results immediately
        print(f"  ✓ Completed in {eval_time:.1f}s")
        print(f"  📊 Scores:")
        print(f"     - Faithfulness:    {metrics['faithfulness']:.4f}")
        print(f"     - Answer Relevancy: {metrics['answer_relevancy']:.4f}")
        print(f"     - Context Recall:   {metrics['context_recall']:.4f}")
        print(f"     - Context Precision: {metrics['context_precision']:.4f}")
        print(f"     → RAGAS Score:      {ragas_score:.4f}")
        print()

    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}")
        all_results[mode] = {
            "error": str(e),
            "metrics": {},
            "ragas_score": 0,
            "timestamp": datetime.now().isoformat()
        }
        print()

total_time = time.time() - start_time

# ============================================
# Summary and Analysis
# ============================================
print("="*70)
print("📊 EVALUATION COMPLETE - SUMMARY")
print("="*70)
print()

# Find best performing mode
best_mode = max(all_results.items(),
                key=lambda x: x[1].get("ragas_score", 0))

print("🏆 Performance Ranking:")
ranked = sorted(all_results.items(),
                key=lambda x: x[1].get("ragas_score", 0),
                reverse=True)

for rank, (mode, result) in enumerate(ranked, 1):
    score = result.get("ragas_score", 0)
    print(f"{rank}. {mode.upper():8s}: {score:.4f}")

print()
print("📈 Detailed Comparison:")
print(f"{'Mode':<8} | {'Faith':<6} | {'Relev':<6} | {'Recall':<6} | {'Precis':<6} | {'RAGAS':<6}")
print("-" * 60)

for mode in ["local", "global", "naive", "hybrid", "mix"]:
    if mode in all_results:
        r = all_results[mode]
        m = r.get("metrics", {})
        print(f"{mode.upper():<8} | {m.get('faithfulness', 0):.4f} | "
              f"{m.get('answer_relevancy', 0):.4f} | "
              f"{m.get('context_recall', 0):.4f} | "
              f"{m.get('context_precision', 0):.4f} | "
              f"{r.get('ragas_score', 0):.4f}")

print()
print(f"⏱ Total evaluation time: {total_time:.1f} seconds")
print(f"  Average per mode: {total_time/len(all_results):.1f} seconds")
print()

# ============================================
# Save Results
# ============================================
output_data = {
    "test_file": TEST_FILE,
    "question_id": question_id,
    "question": question_text,
    "evaluation_timestamp": datetime.now().isoformat(),
    "total_evaluation_time_seconds": round(total_time, 2),
    "judge_model": RAGAS_JUDGE_MODEL,
    "embedding_model": RAGAS_EMBEDDING_MODEL,
    "mode_evaluations": all_results,
    "best_performing_mode": {
        "mode": best_mode[0],
        "ragas_score": best_mode[1].get("ragas_score", 0)
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"💾 Results saved to: {OUTPUT_FILE}")
print()
print("="*70)
print("✅ TEST COMPLETE!")
print("="*70)
print()
print("What this tells us:")
print("1. Each mode retrieves and generates differently")
print("2. RAGAS objectively measures quality differences")
print("3. No rate limiting needed for Ministral!")
print("4. Using LOCAL embeddings - no OpenAI API needed!")
print("5. ~20-30 seconds per mode is typical")
print()
print("Ready for batch processing? 👍")