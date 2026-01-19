#!/usr/bin/env python3
"""
MISTRAL BATCH API FOR RAGAS EVALUATION
=======================================
Uses Mistral's native Batch API for efficient processing.
50% lower cost than synchronous API calls!
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
from mistralai import Mistral

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
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ============================================
# Configuration
# ============================================
RESULTS_DIR = "./lightrag-bench/5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL"
OUTPUT_FILE = "./lightrag-bench/batch_ragas_evaluation_results_batch_api.json"
BATCH_DIR = "./lightrag-bench/batch_files"

# Mistral Batch API settings
RAGAS_JUDGE_MODEL = "ministral-14b-2512"
RAGAS_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 100  # Process 100 requests per batch file

# Query modes
QUERY_MODES = ["local", "global", "naive", "hybrid", "mix"]

# RAGAS evaluation prompts (we'll construct these)
RAGAS_PROMPTS = {
    "faithfulness": "Evaluate if the answer is faithful to the context. Context: {context}\nAnswer: {answer}\nProvide a score from 0 to 1.",
    "relevancy": "Evaluate if the answer is relevant to the question. Question: {question}\nAnswer: {answer}\nProvide a score from 0 to 1.",
    "recall": "Evaluate context recall. Question: {question}\nContext: {context}\nGround Truth: {ground_truth}\nProvide a score from 0 to 1.",
    "precision": "Evaluate context precision. Question: {question}\nContext: {context}\nProvide a score from 0 to 1."
}

print("="*70)
print("🚀 MISTRAL BATCH API - RAGAS EVALUATION")
print("="*70)
print(f"📁 Source: {Path(RESULTS_DIR).name}")
print(f"💾 Output: {Path(OUTPUT_FILE).name}")
print(f"📦 Batch Size: {BATCH_SIZE} requests per file")
print(f"💰 Cost: 50% lower than synchronous API!")
print()

# ============================================
# Setup
# ============================================
# Create batch directory
os.makedirs(BATCH_DIR, exist_ok=True)

# Initialize Mistral client
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    print("❌ MISTRAL_API_KEY not found!")
    exit(1)

client = Mistral(api_key=mistral_api_key)
print("✓ Mistral client initialized")

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
# Batch Request Preparation
# ============================================
def prepare_ragas_prompt(
    question: str,
    answer: str,
    context: str,
    ground_truth: str,
    metric_type: str
) -> str:
    """
    Prepare a prompt for RAGAS evaluation.
    """
    if metric_type == "faithfulness":
        return f"""Evaluate the faithfulness of the answer based on the given context.
Context: {context}
Answer: {answer}
Score the faithfulness from 0 to 1, where 1 means the answer is completely faithful to the context.
Respond with just the score."""

    elif metric_type == "relevancy":
        return f"""Evaluate how relevant the answer is to the question.
Question: {question}
Answer: {answer}
Score the relevancy from 0 to 1, where 1 means the answer is perfectly relevant.
Respond with just the score."""

    elif metric_type == "recall":
        return f"""Evaluate how well the context covers the ground truth.
Question: {question}
Context: {context}
Ground Truth: {ground_truth}
Score the recall from 0 to 1, where 1 means the context fully covers the ground truth.
Respond with just the score."""

    elif metric_type == "precision":
        return f"""Evaluate the precision of the context for answering the question.
Question: {question}
Context: {context}
Score the precision from 0 to 1, where 1 means the context is perfectly precise.
Respond with just the score."""

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def create_batch_file(requests: List[Dict], batch_num: int) -> str:
    """
    Create a JSONL batch file for Mistral Batch API.
    """
    batch_file_path = Path(BATCH_DIR) / f"batch_{batch_num}.jsonl"

    with open(batch_file_path, 'w') as f:
        for req in requests:
            # Create Mistral API request format
            api_request = {
                "custom_id": req["custom_id"],
                "body": {
                    "model": RAGAS_JUDGE_MODEL,
                    "messages": [
                        {"role": "user", "content": req["prompt"]}
                    ],
                    "max_tokens": 10,  # Just need a score
                    "temperature": 0.1
                }
            }
            f.write(json.dumps(api_request) + '\n')

    return str(batch_file_path)


def prepare_all_batch_requests() -> List[Dict]:
    """
    Prepare all batch requests from test results.
    """
    print("📋 Preparing batch requests...")

    all_requests = []
    results_path = Path(RESULTS_DIR)
    all_files = sorted(results_path.glob("test_results_*_question_*.json"))

    if not all_files:
        print(f"❌ No files found in {RESULTS_DIR}")
        return []

    for file_path in tqdm(all_files, desc="Preparing requests"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            question_id = data["question_id"]
            question_text = data["question"]
            expected_answer = str(data["expected_answer"])

            for mode in QUERY_MODES:
                mode_data = data["modes"].get(mode, {})

                if mode_data.get("status") != "success":
                    continue

                answer = mode_data["answer"]
                context = mode_data["retrieved_context"]

                # Create requests for each metric
                for metric in ["faithfulness", "relevancy", "recall", "precision"]:
                    prompt = prepare_ragas_prompt(
                        question_text,
                        answer,
                        context,
                        expected_answer,
                        metric
                    )

                    all_requests.append({
                        "custom_id": f"{question_id}_{mode}_{metric}",
                        "prompt": prompt,
                        "metadata": {
                            "file": file_path.name,
                            "question_id": question_id,
                            "mode": mode,
                            "metric": metric
                        }
                    })

        except Exception as e:
            print(f"  ⚠️ Error with {file_path.name}: {e}")

    print(f"✓ Prepared {len(all_requests)} requests")
    return all_requests


def submit_batches(all_requests: List[Dict]) -> List[str]:
    """
    Submit batch jobs to Mistral API.
    """
    print("\n📤 Submitting batches to Mistral...")

    batch_jobs = []
    batches = [all_requests[i:i+BATCH_SIZE] for i in range(0, len(all_requests), BATCH_SIZE)]

    for batch_num, batch_requests in enumerate(batches, 1):
        print(f"\nBatch {batch_num}/{len(batches)}:")

        # Create batch file
        batch_file_path = create_batch_file(batch_requests, batch_num)
        print(f"  • Created file: {Path(batch_file_path).name}")

        # Upload file
        try:
            with open(batch_file_path, 'rb') as f:
                uploaded_file = client.files.upload(
                    file={
                        "file_name": f"batch_{batch_num}.jsonl",
                        "content": f
                    },
                    purpose="batch"
                )
            print(f"  • Uploaded: {uploaded_file.id}")

            # Create batch job
            batch_job = client.batch.jobs.create(
                input_files=[uploaded_file.id],
                model=RAGAS_JUDGE_MODEL,
                endpoint="/v1/chat/completions",
                metadata={
                    "batch_num": str(batch_num),
                    "type": "ragas_evaluation"
                }
            )
            print(f"  • Job created: {batch_job.id}")

            batch_jobs.append({
                "job_id": batch_job.id,
                "batch_num": batch_num,
                "request_count": len(batch_requests),
                "status": "QUEUED"
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Save batch job IDs
    with open(Path(BATCH_DIR) / "batch_jobs.json", 'w') as f:
        json.dump(batch_jobs, f, indent=2)

    print(f"\n✅ Submitted {len(batch_jobs)} batch jobs")
    return batch_jobs


def monitor_batches(batch_jobs: List[Dict]) -> Dict[str, Any]:
    """
    Monitor batch job progress.
    """
    print("\n⏳ Monitoring batch jobs...")

    completed_jobs = []
    failed_jobs = []

    with tqdm(total=len(batch_jobs), desc="Batch Jobs") as pbar:
        while len(completed_jobs) + len(failed_jobs) < len(batch_jobs):
            for job in batch_jobs:
                if job["job_id"] in [j["job_id"] for j in completed_jobs + failed_jobs]:
                    continue

                try:
                    status = client.batch.jobs.get(job_id=job["job_id"])

                    if status.status == "COMPLETED":
                        job["output_file"] = status.output_file
                        completed_jobs.append(job)
                        pbar.update(1)
                        print(f"\n  ✅ Job {job['batch_num']} completed")

                    elif status.status in ["FAILED", "CANCELLED", "EXPIRED"]:
                        job["error"] = status.status
                        failed_jobs.append(job)
                        pbar.update(1)
                        print(f"\n  ❌ Job {job['batch_num']} failed: {status.status}")

                except Exception as e:
                    print(f"\n  ⚠️ Error checking job {job['job_id']}: {e}")

            # Wait before checking again
            if len(completed_jobs) + len(failed_jobs) < len(batch_jobs):
                time.sleep(10)  # Check every 10 seconds

    print(f"\n📊 Results: {len(completed_jobs)} completed, {len(failed_jobs)} failed")
    return {"completed": completed_jobs, "failed": failed_jobs}


def process_results(completed_jobs: List[Dict]) -> Dict[str, Any]:
    """
    Download and process batch results.
    """
    print("\n📥 Processing results...")

    all_scores = {}

    for job in tqdm(completed_jobs, desc="Downloading results"):
        try:
            # Download result file
            output_stream = client.files.download(file_id=job["output_file"])
            result_file = Path(BATCH_DIR) / f"results_batch_{job['batch_num']}.jsonl"

            with open(result_file, 'wb') as f:
                f.write(output_stream.read())

            # Parse results
            with open(result_file, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    custom_id = result["custom_id"]

                    if result["response"]["status_code"] == 200:
                        # Extract score from response
                        content = result["response"]["body"]["choices"][0]["message"]["content"]
                        try:
                            score = float(content.strip())
                        except:
                            score = 0.0

                        all_scores[custom_id] = score
                    else:
                        all_scores[custom_id] = None

        except Exception as e:
            print(f"\n  ⚠️ Error processing job {job['job_id']}: {e}")

    # Aggregate scores by question and mode
    aggregated_results = {}

    for key, score in all_scores.items():
        parts = key.split('_')
        if len(parts) >= 3:
            question_id = parts[0]
            mode = parts[1]
            metric = parts[2]

            if question_id not in aggregated_results:
                aggregated_results[question_id] = {}
            if mode not in aggregated_results[question_id]:
                aggregated_results[question_id][mode] = {}

            aggregated_results[question_id][mode][metric] = score

    # Calculate RAGAS scores
    for question_id in aggregated_results:
        for mode in aggregated_results[question_id]:
            metrics = aggregated_results[question_id][mode]
            valid_scores = [s for s in metrics.values() if s is not None]
            if valid_scores:
                ragas_score = np.mean(valid_scores)
                aggregated_results[question_id][mode]["ragas_score"] = round(ragas_score, 4)

    return aggregated_results


def main():
    """
    Main function - orchestrates the batch processing.
    """
    print("🎯 MISTRAL BATCH API WORKFLOW")
    print("="*70)

    # Step 1: Prepare requests
    all_requests = prepare_all_batch_requests()
    if not all_requests:
        return

    # Step 2: Submit batches
    batch_jobs = submit_batches(all_requests)

    # Step 3: Monitor progress
    print("\n⏰ Note: Batch processing may take 10-30 minutes")
    print("   You can check progress at: https://console.mistral.ai/build/batches")
    job_results = monitor_batches(batch_jobs)

    # Step 4: Process results
    if job_results["completed"]:
        final_results = process_results(job_results["completed"])

        # Save final results
        with open(OUTPUT_FILE, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_requests": len(all_requests),
                "completed_jobs": len(job_results["completed"]),
                "failed_jobs": len(job_results["failed"]),
                "results": final_results
            }, f, indent=2)

        print(f"\n✅ Results saved to: {OUTPUT_FILE}")

        # Stats
        total_evaluations = sum(
            len(modes) for modes in final_results.values()
        )
        print(f"\n📊 Final Statistics:")
        print(f"  • Questions evaluated: {len(final_results)}")
        print(f"  • Total evaluations: {total_evaluations}")
        print(f"  • Cost savings: 50% compared to sync API")

    else:
        print("\n❌ No completed jobs to process")


if __name__ == "__main__":
    # Install mistralai if not present
    try:
        import mistralai
    except ImportError:
        print("📦 Installing mistralai package...")
        os.system("pip install mistralai")
        print("✓ Package installed. Please run the script again.")
        exit(0)

    main()