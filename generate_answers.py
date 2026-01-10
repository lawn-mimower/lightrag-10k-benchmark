#!/usr/bin/env python3
"""
Answer Generation Script using Gemini 2.5  Flash
Processes questions in batches of 5, rate-limited to 2 batches/minute
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configuration
INPUT_FILE = "test_results_priority_tickers_MIX_rerank>0.3_topk10.json"
OUTPUT_FILE = "test_results_priority_tickers_MIX_rerank>0.3_topk10_with_answers.json"
CHECKPOINT_FILE = "generation_checkpoint.json"
LOG_FILE = "generation_log.txt"

MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0
BATCH_SIZE = 5
BATCHES_PER_MINUTE = 2
BATCH_DELAY = 60 / BATCHES_PER_MINUTE  # 30 seconds between batches
RETRY_DELAY = 60  # 1 minute wait before retry
MAX_RETRIES = 3


def log_message(message: str):
    """Write timestamped message to log file and console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")


def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint data if exists"""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"processed_batches": [], "processed_question_ids": []}


def save_checkpoint(checkpoint_data: Dict[str, Any]):
    """Save checkpoint data"""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint_data, indent=2, fp=f)


def create_batch_prompt(questions_batch: List[Dict[str, Any]]) -> str:
    """Create prompt for a batch of 5 questions"""
    prompt_parts = [
        "You will answer multiple questions. Each question has its own separate context from different companies' SEC 10-K filings.",
        "CRITICAL: DO NOT mix information between questions. Each answer must ONLY use information from its corresponding context.",
        "",
        "Return your response as a JSON object where keys are question_ids and values are the answers.",
        "Format: {\"question_id_1\": \"answer text\", \"question_id_2\": \"answer text\", ...}",
        "",
        "---",
        ""
    ]

    for i, entry in enumerate(questions_batch, 1):
        context = entry.get('retrieved_context', '[NO CONTEXT AVAILABLE]')
        prompt_parts.extend([
            f"QUESTION {i} (ID: {entry['question_id']}):",
            f"{entry['question']}",
            "",
            f"CONTEXT {i}:",
            f"{context}",
            "",
            "---",
            ""
        ])

    prompt_parts.extend([
        "Now provide your answers in JSON format with question_ids as keys:",
        "{",
    ])

    for i, entry in enumerate(questions_batch):
        comma = "," if i < len(questions_batch) - 1 else ""
        prompt_parts.append(f'  "{entry["question_id"]}": "your answer here"{comma}')

    prompt_parts.append("}")

    return "\n".join(prompt_parts)


def generate_batch_answers(
    client: genai.Client,
    questions_batch: List[Dict[str, Any]],
    batch_num: int
) -> Dict[str, str]:
    """Generate answers for a batch of questions using Gemini"""
    log_message(f"Processing batch {batch_num} ({len(questions_batch)} questions)")

    prompt = create_batch_prompt(questions_batch)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    response_mime_type="application/json",
                ),
            )

            # Parse JSON response
            response_text = response.text.strip()
            answers_dict = json.loads(response_text)

            log_message(f"✓ Batch {batch_num} completed successfully")
            return answers_dict

        except json.JSONDecodeError as e:
            log_message(f"⚠ Batch {batch_num} attempt {attempt+1}/{MAX_RETRIES}: JSON parse error: {e}")
            log_message(f"  Raw response: {response.text[:200]}...")
            if attempt < MAX_RETRIES - 1:
                log_message(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                log_message(f"✗ Batch {batch_num} failed after {MAX_RETRIES} attempts")
                return None

        except Exception as e:
            log_message(f"⚠ Batch {batch_num} attempt {attempt+1}/{MAX_RETRIES}: API error: {e}")
            if attempt < MAX_RETRIES - 1:
                log_message(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                log_message(f"✗ Batch {batch_num} failed after {MAX_RETRIES} attempts")
                return None

    return None


def main():
    """Main execution function"""
    start_time = time.time()

    # Initialize API client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log_message("ERROR: GEMINI_API_KEY not found in environment")
        return

    client = genai.Client(api_key=api_key)
    log_message(f"Initialized Gemini client with model: {MODEL_NAME}")

    # Load input data
    log_message(f"Loading input file: {INPUT_FILE}")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    total_questions = len(data)
    log_message(f"Loaded {total_questions} questions")

    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint["processed_question_ids"])

    if processed_ids:
        log_message(f"Resuming from checkpoint: {len(processed_ids)} questions already processed")

    # Filter out entries without retrieved_context
    entries_without_context = [entry for entry in data if 'retrieved_context' not in entry]
    if entries_without_context:
        log_message(f"⚠ Skipping {len(entries_without_context)} entries without retrieved_context:")
        for entry in entries_without_context:
            log_message(f"  - {entry['question_id']}: {entry.get('status', 'unknown status')}")

    # Create batches of unprocessed questions that have context
    unprocessed = [
        entry for entry in data
        if entry["question_id"] not in processed_ids
        and 'retrieved_context' in entry
    ]
    batches = [unprocessed[i:i + BATCH_SIZE] for i in range(0, len(unprocessed), BATCH_SIZE)]

    total_batches = len(batches)
    log_message(f"Processing {len(unprocessed)} questions in {total_batches} batches")

    # Process batches
    for batch_idx, batch in enumerate(batches, 1):
        batch_start = time.time()

        # Generate answers for this batch
        answers_dict = generate_batch_answers(client, batch, batch_idx)

        if answers_dict:
            # Update entries with generated answers
            for entry in batch:
                question_id = entry["question_id"]
                if question_id in answers_dict:
                    entry["response"] = answers_dict[question_id]
                    processed_ids.add(question_id)
                    log_message(f"  ✓ Question {question_id}: answer added")
                else:
                    log_message(f"  ⚠ Question {question_id}: not found in response")

            # Update checkpoint
            checkpoint["processed_question_ids"] = list(processed_ids)
            checkpoint["processed_batches"].append({
                "batch_num": batch_idx,
                "timestamp": datetime.now().isoformat(),
                "question_ids": [e["question_id"] for e in batch]
            })
            save_checkpoint(checkpoint)

            # Save intermediate results
            with open(OUTPUT_FILE, "w") as f:
                json.dump(data, f, indent=2)
            log_message(f"Checkpoint saved: {len(processed_ids)}/{total_questions} questions completed")
        else:
            log_message(f"✗ Batch {batch_idx} failed - skipping for now")

        # Rate limiting
        if batch_idx < total_batches:
            batch_elapsed = time.time() - batch_start
            sleep_time = max(0, BATCH_DELAY - batch_elapsed)
            if sleep_time > 0:
                log_message(f"Rate limiting: waiting {sleep_time:.1f}s before next batch...")
                time.sleep(sleep_time)

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    # Summary
    elapsed = time.time() - start_time
    success_count = len(processed_ids)
    log_message("=" * 80)
    log_message("GENERATION COMPLETE")
    log_message(f"Total time: {elapsed/60:.1f} minutes")
    log_message(f"Questions processed: {success_count}/{total_questions}")
    log_message(f"Success rate: {success_count/total_questions*100:.1f}%")
    log_message(f"Output saved to: {OUTPUT_FILE}")
    log_message("=" * 80)


if __name__ == "__main__":
    main()
