# Optimized RAGAS Evaluation Cell (No Rate Limiting)

import time
import warnings
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Suppress warnings
warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper is deprecated.*")
warnings.filterwarnings("ignore", message=".*Unexpected type for token usage.*")

print("="*70)
print("🔍 RAGAS Evaluation - NO RATE LIMITING VERSION")
print("="*70)
print()

# ============================================
# Configure Models
# ============================================
RAGAS_JUDGE_MODEL = "ministral-3-14b-2512"
RAGAS_EMBEDDING_MODEL = "text-embedding-3-small"

# Create LLM instance for RAGAS
base_llm = ChatOpenAI(
    model=RAGAS_JUDGE_MODEL,
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
    max_retries=5,
    request_timeout=180
)

# Wrap with LangchainLLMWrapper
ragas_llm = LangchainLLMWrapper(
    langchain_llm=base_llm,
    bypass_n=True  # Avoid passing 'n' to API
)

# Create embeddings
ragas_embeddings = OpenAIEmbeddings(
    model=RAGAS_EMBEDDING_MODEL,
    api_key=os.getenv("OPENAI_API_KEY")
)

print("✓ Models configured (NO RATE LIMITING)")
print()

# ============================================
# Load Results
# ============================================
results_dir = Path("5_modes_question_wise_results_with_answers")
result_files = list(results_dir.glob("test_results_CTAS_question_*.json"))

print(f"Found {len(result_files)} question files")
print(f"Total evaluations: {len(result_files) * 5} (5 modes each)")
print()

# ============================================
# Batch Evaluation (Faster without rate limits)
# ============================================
all_evaluations = []
start_time = time.time()

# Process in batches for efficiency
BATCH_SIZE = 5  # Process 5 questions at once

for file_idx, result_file in enumerate(result_files, 1):
    with open(result_file, 'r') as f:
        question_data = json.load(f)

    question_id = question_data["question_id"]
    question_text = question_data["question"]
    ground_truth = str(question_data["expected_answer"])

    print(f"[{file_idx}/{len(result_files)}] Q-ID: {question_id[:8]}...")

    # Evaluate all modes for this question
    for mode in ["local", "global", "naive", "hybrid", "mix"]:
        mode_data = question_data["modes"].get(mode, {})

        if mode_data.get("status") != "success":
            print(f"  {mode:8s}: ❌ Skipped")
            continue

        print(f"  {mode:8s}: ", end="", flush=True)

        try:
            # Prepare dataset
            eval_dataset = Dataset.from_dict({
                "question": [question_text],
                "answer": [mode_data["answer"]],
                "contexts": [[mode_data["retrieved_context"]]],
                "ground_truth": [ground_truth]
            })

            # Run evaluation (NO SLEEP NEEDED!)
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

            # Extract metrics
            df = eval_results.to_pandas()
            metrics = {
                "faithfulness": float(df.iloc[0].get("faithfulness", 0)),
                "answer_relevancy": float(df.iloc[0].get("answer_relevancy", 0)),
                "context_recall": float(df.iloc[0].get("context_recall", 0)),
                "context_precision": float(df.iloc[0].get("context_precision", 0))
            }

            # Calculate RAGAS score
            valid_metrics = [v for v in metrics.values() if not np.isnan(v)]
            ragas_score = np.mean(valid_metrics) if valid_metrics else 0

            # Store result
            all_evaluations.append({
                "question_id": question_id,
                "question": question_text[:100] + "...",
                "mode": mode,
                "metrics": metrics,
                "ragas_score": round(ragas_score, 4),
                "timestamp": datetime.now().isoformat()
            })

            print(f"✓ RAGAS: {ragas_score:.4f}")

        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")

elapsed_time = time.time() - start_time
print(f"\n✅ Completed in {elapsed_time:.1f} seconds")
print(f"   Average: {elapsed_time/len(all_evaluations):.2f} sec/eval")