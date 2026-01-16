#!/usr/bin/env python3
"""
Answer Generation Script - REASONING ALWAYS ON
(Gemini SDK with Thinking Config Always Enabled)
Preserves old responses as response_old before regenerating.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuration
RESULTS_FOLDER = "5_modes_question_wise_results/5_modes_question_wise_results_priority_tickers_ALL"
CTAS_FILE_PATTERN = "test_results_CTAS_question_*.json"
CHECKPOINT_FILE = "generation_checkpoint_reasoning.json"
LOG_FILE = "generation_log_reasoning.txt"
MODEL_NAME = "gemini-3-flash-preview"

# Parameters
FILE_PROCESSING_DELAY = 60  # 60 seconds between files (5 parallel requests per minute for 6 RPM limit)
RETRY_DELAY = 10
MAX_RETRIES = 3
MAX_PARALLEL_MODES = 5  # Process all 5 modes in parallel

def log_message(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

def load_checkpoint() -> Dict[str, Any]:
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}}

def save_checkpoint(checkpoint_data: Dict[str, Any]):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, indent=2, fp=f)

def generate_answer_logic(
    client: genai.Client,
    question: str,
    context: str,
    mode: str,
    question_id: str,
    requires_reasoning: bool = True  # Always True in this version
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate answer for a given question and context.
    Returns (answer, thought) tuple where thought contains reasoning.
    """
    prompt = f"""You are a financial document analyst specializing in SEC 10-K filings.

CRITICAL INSTRUCTIONS:
- Answer ONLY based on the provided context
- Be precise with numerical values
- Keep answers concise (2-4 sentences)
- This question requires detailed reasoning and analysis

Question: {question}

Context from {mode.upper()}:
{context}

Please provide a concise, factual answer based only on the information in the context above."""

    for attempt in range(MAX_RETRIES):
        try:
            # Configure generation with thinking always enabled
            generation_config = types.GenerateContentConfig(
                max_output_tokens=65536,  # Gemini 3 Flash maximum capacity
                temperature=0.1,
            )

            # Always add thinking config (reasoning always ON)
            generation_config.thinking_config = types.ThinkingConfig(
                include_thoughts=True
            )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=generation_config
            )

            if not response or not response.candidates:
                log_message(f"  Warning: Empty response for {question_id} [{mode}]")
                return None, None

            # Extract answer and thought from response
            answer = None
            thought = None

            # When thinking is enabled, parse multiple parts
            # Parts with thought=True are reasoning, thought=False is final answer
            thought_parts = []
            answer_parts = []

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # Check if this part is a thought (reasoning)
                        is_thought = getattr(part, 'thought', False)

                        # Extract text from this part
                        part_text = getattr(part, 'text', '')

                        if is_thought:
                            # This is reasoning/thinking
                            if part_text:
                                thought_parts.append(part_text)
                        else:
                            # This is the final answer
                            if part_text:
                                answer_parts.append(part_text)

            # Combine parts
            if thought_parts:
                thought = '\n'.join(thought_parts).strip()
            if answer_parts:
                answer = '\n'.join(answer_parts).strip()

            if answer:
                thought_info = f" (with reasoning: {len(thought)} chars)" if thought else ""
                log_message(f"  Generated answer for {question_id} [{mode}] ({len(answer)} chars{thought_info})")
                return answer, thought
            else:
                log_message(f"  Warning: No answer text in response for {question_id} [{mode}]")
                return None, None

        except Exception as e:
            log_message(f"  Warning: Attempt {attempt+1}/{MAX_RETRIES} failed for {question_id} [{mode}]: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None, None
    return None, None

def process_single_mode(
    client: genai.Client,
    file_path: Path,
    question: str,
    question_id: str,
    mode_name: str,
    mode_data: Dict[str, Any],
    requires_reasoning: bool = True  # Always True
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Process a single mode for a question.
    Returns (mode_name, answer, thought, error_message)
    """
    try:
        if not mode_data.get("retrieved_context"):
            return mode_name, None, None, "No context available"

        answer, thought = generate_answer_logic(
            client,
            question,
            mode_data["retrieved_context"],
            mode_name,
            question_id,
            requires_reasoning
        )

        return mode_name, answer, thought, None

    except Exception as e:
        return mode_name, None, None, str(e)


def process_ctas_file(
    file_path: Path,
    client: genai.Client,
    checkpoint: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single CTAS file by generating answers for all 5 modes in parallel."""
    log_message(f"\n{'='*60}")
    log_message(f"Processing file: {file_path.name}")
    log_message(f"{'='*60}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_id = data["question_id"]
    question = data["question"]

    # Always enable reasoning in this version
    requires_reasoning = True

    log_message(f"Question ID: {question_id}")
    log_message(f"Reasoning: ALWAYS ON (forced)")

    if file_path.name not in checkpoint["completed"]:
        checkpoint["completed"][file_path.name] = {
            "question_id": question_id,
            "completed_modes": []
        }

    completed_modes = []
    modes = data.get("modes", {})

    # Rename existing responses to response_old before processing
    for mode_name, mode_data in modes.items():
        if "response" in mode_data and mode_data["response"]:
            mode_data["response_old"] = mode_data["response"]
            del mode_data["response"]
            log_message(f"  -> Renamed existing response to response_old: {mode_name}")

    # Identify modes that need processing (all of them now, since we renamed responses)
    modes_to_process = []

    for mode_name, mode_data in modes.items():
        if mode_name in completed_modes:
            log_message(f"  Already completed: {mode_name}")
            continue

        modes_to_process.append((mode_name, mode_data))

    if not modes_to_process:
        log_message("  All modes already processed")
        return checkpoint

    log_message(f"\n  -> Processing {len(modes_to_process)} modes in parallel with reasoning ON...")

    # Process all modes in parallel
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_MODES) as executor:
        futures = {
            executor.submit(
                process_single_mode,
                client,
                file_path,
                question,
                question_id,
                mode_name,
                mode_data,
                requires_reasoning
            ): mode_name
            for mode_name, mode_data in modes_to_process
        }

        # Collect results as they complete
        for future in as_completed(futures):
            mode_name, answer, thought, error = future.result()
            mode_data = modes[mode_name]

            if error:
                log_message(f"  Error processing {mode_name}: {error}")
                continue

            if answer:
                mode_data["response"] = answer
                if thought:
                    mode_data["thought"] = thought
                    log_message(f"  Saved answer with reasoning for {mode_name}")
                else:
                    log_message(f"  Saved answer for {mode_name}")

                completed_modes.append(mode_name)
            else:
                log_message(f"  No answer generated for {mode_name}")

    # Save results after all parallel processing is done
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    checkpoint["completed"][file_path.name]["completed_modes"] = completed_modes
    save_checkpoint(checkpoint)

    log_message(f"\n  File processing complete: {len(completed_modes)}/{len(modes)} modes done")

    return checkpoint

def main():
    log_message("="*60)
    log_message(f"REASONING-ON Answer Generation - {MODEL_NAME}")
    log_message("Gemini SDK with Thinking Config ALWAYS ENABLED")
    log_message("="*60)

    if not GEMINI_API_KEY:
        log_message("ERROR: No API Key found.")
        log_message("Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file")
        return

    # Initialize Gemini Client
    client = genai.Client(api_key=GEMINI_API_KEY)
    log_message(f"Initialized Gemini Client with {MODEL_NAME}")

    # Get files from the results folder
    results_path = Path(RESULTS_FOLDER)
    if not results_path.exists():
        log_message(f"ERROR: Results folder not found: {RESULTS_FOLDER}")
        return

    ctas_files = sorted(results_path.glob(CTAS_FILE_PATTERN))

    if not ctas_files:
        log_message(f"ERROR: No files matching pattern '{CTAS_FILE_PATTERN}' found in {RESULTS_FOLDER}")
        return

    log_message(f"Found {len(ctas_files)} files to process")
    log_message(f"Rate limiting: Processing 1 file per {FILE_PROCESSING_DELAY} seconds (5 parallel requests per minute)")
    log_message("")

    checkpoint = load_checkpoint()

    for i, file_path in enumerate(ctas_files, 1):
        log_message(f"\n{'#'*60}")
        log_message(f"FILE {i}/{len(ctas_files)}")
        log_message(f"{'#'*60}")

        checkpoint = process_ctas_file(file_path, client, checkpoint)

        # Add delay between files to respect rate limits (except for the last file)
        if i < len(ctas_files):
            log_message(f"\nWaiting {FILE_PROCESSING_DELAY} seconds before next file (rate limiting)...")
            time.sleep(FILE_PROCESSING_DELAY)

    log_message("\n" + "="*60)
    log_message("ALL FILES PROCESSED SUCCESSFULLY")
    log_message("="*60)


if __name__ == "__main__":
    main()
