#!/usr/bin/env python3
"""
Answer Generation Script for CTAS Test Results
(Gemini SDK with Thinking Config Support)

Options:
  --always-reason   think on every question and regenerate every mode, keeping
                    the previous answer as response_old
  --vertex-express  send plain prompts to the Vertex AI express endpoint with an
                    API key instead of using the Gemini SDK
"""

import argparse
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
# Per-question result files; set RESULTS_DIR for the notebook's 5_modes_question_wise_results_with_answers/... output
RESULTS_FOLDER = os.getenv("RESULTS_DIR", "5_modes_question_wise_results/5_modes_question_wise_results_priority_tickers_ALL")
CTAS_FILE_PATTERN = "test_results_CTAS_question_*.json"
CHECKPOINT_FILE = "generation_checkpoint_ctas.json"
LOG_FILE = "generation_log_ctas.txt"
MODEL_NAME = "gemini-3-flash-preview"

# --always-reason keeps its own checkpoint and log
REASONING_CHECKPOINT_FILE = "generation_checkpoint_reasoning.json"
REASONING_LOG_FILE = "generation_log_reasoning.txt"

# --vertex-express
VERTEX_PROJECT_ID = "your-gcp-project-id"
VERTEX_LOCATION = "us-central1"               # Standard Vertex region
VERTEX_MODEL_ID = "gemini-3.0-flash-preview-001" # Vertex often uses this versioned ID

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
    requires_reasoning: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate answer for a given question and context.
    Returns (answer, thought) tuple where thought is None if not using reasoning mode.
    """
    prompt = f"""You are a financial document analyst specializing in SEC 10-K filings.

CRITICAL INSTRUCTIONS:
- Answer ONLY based on the provided context
- Be precise with numerical values
- Keep answers concise (2-4 sentences)
{f"- This question requires detailed reasoning and analysis" if requires_reasoning else ""}

Question: {question}

Context from {mode.upper()}:
{context}

Please provide a concise, factual answer based only on the information in the context above."""

    for attempt in range(MAX_RETRIES):
        try:
            # Configure generation with no output token restrictions
            generation_config = types.GenerateContentConfig(
                max_output_tokens=65536,  # Gemini 3 Flash maximum capacity
                temperature=0.1,
            )

            # Add thinking config if reasoning is required
            if requires_reasoning:
                generation_config.thinking_config = types.ThinkingConfig(
                    include_thoughts=True
                )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=generation_config
            )

            if not response or not response.candidates:
                log_message(f"  ⚠ Empty response for {question_id} [{mode}]")
                return None, None

            # Extract answer and thought based on response structure
            answer = None
            thought = None

            if requires_reasoning:
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
            else:
                # Standard response (no thinking) - simple text extraction
                if hasattr(response, 'text') and isinstance(response.text, str):
                    answer = response.text.strip()

            if answer:
                thought_info = f" (with reasoning: {len(thought)} chars)" if thought else ""
                log_message(f"  ✓ Generated answer for {question_id} [{mode}] ({len(answer)} chars{thought_info})")
                return answer, thought
            else:
                log_message(f"  ⚠ No answer text in response for {question_id} [{mode}]")
                return None, None

        except Exception as e:
            log_message(f"  ⚠ Attempt {attempt+1}/{MAX_RETRIES} failed for {question_id} [{mode}]: {e}")
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
    requires_reasoning: bool
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
    checkpoint: Dict[str, Any],
    always_reason: bool = False
) -> Tuple[Dict[str, Any], bool]:
    """Process a single CTAS file by generating answers for all 5 modes in parallel.

    With always_reason every mode is regenerated with thinking enabled and the
    previous answer is kept as response_old.
    """
    log_message(f"\n{'='*60}")
    log_message(f"Processing file: {file_path.name}")
    log_message(f"{'='*60}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_id = data["question_id"]
    question = data["question"]
    requires_reasoning = True if always_reason else data.get("reasoning", False)

    log_message(f"Question ID: {question_id}")
    if always_reason:
        log_message(f"Reasoning: ALWAYS ON (forced)")
    else:
        log_message(f"Reasoning required: {requires_reasoning}")

    if file_path.name not in checkpoint["completed"]:
        checkpoint["completed"][file_path.name] = {
            "question_id": question_id,
            "completed_modes": []
        }

    completed_modes = []
    modes = data.get("modes", {})

    if always_reason:
        # Rename existing responses to response_old before processing
        for mode_name, mode_data in modes.items():
            if "response" in mode_data and mode_data["response"]:
                mode_data["response_old"] = mode_data["response"]
                del mode_data["response"]
                log_message(f"  -> Renamed existing response to response_old: {mode_name}")

    # Identify modes that need processing
    modes_to_process = []
    
    for mode_name, mode_data in modes.items():
        if mode_name in completed_modes:
            log_message(f"  ⊙ Already completed: {mode_name}")
            continue

        if "response" in mode_data and mode_data["response"]:
            log_message(f"  ⊙ Response exists: {mode_name}")
            completed_modes.append(mode_name)
            continue

        modes_to_process.append((mode_name, mode_data))
    
    if not modes_to_process:
        log_message("  ✓ All modes already processed")
        return checkpoint, False  # No API calls made

    log_message(f"\n  → Processing {len(modes_to_process)} modes in parallel...")

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
                log_message(f"  ⊘ Error processing {mode_name}: {error}")
                continue

            if answer:
                mode_data["response"] = answer
                if thought:
                    mode_data["thought"] = thought
                    log_message(f"  ✓ Saved answer with reasoning for {mode_name}")
                else:
                    log_message(f"  ✓ Saved answer for {mode_name}")

                completed_modes.append(mode_name)
            else:
                log_message(f"  ⊘ No answer generated for {mode_name}")

    # Save results after all parallel processing is done
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    checkpoint["completed"][file_path.name]["completed_modes"] = completed_modes
    save_checkpoint(checkpoint)

    log_message(f"\n  ✓ File processing complete: {len(completed_modes)}/{len(modes)} modes done")

    return checkpoint, True  # API calls were made

def generate_vertex_express(prompt: str, question_id: str, mode: str, project_id: str, api_key: str) -> Optional[str]:
    """
    Hits the Vertex AI 'Express' endpoint which accepts API Keys.
    """
    import requests

    # Vertex AI Endpoint URL
    url = (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{project_id}/locations/{VERTEX_LOCATION}/publishers/google/models/{VERTEX_MODEL_ID}:generateContent"
    )

    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(url, params=params, headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()

        elif response.status_code == 404:
            log_message(f"  ⚠ Model not found: {VERTEX_MODEL_ID}. Trying alternative ID...")
            return None

        else:
            log_message(f"  ⚠ Vertex Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        log_message(f"  ⚠ Request Failed: {e}")
        return None


def process_file_vertex(file_path: Path, project_id: str, api_key: str):
    """Answer every mode that has context but no response yet, one request at a time."""
    log_message(f"Processing: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for mode, mode_data in data.get("modes", {}).items():
        if mode_data.get("response") or not mode_data.get("retrieved_context"):
            continue

        prompt = (f"Context: {mode_data['retrieved_context']}\n\n"
                  f"Question: {data['question']}\n\n"
                  f"Answer based strictly on context:")

        answer = generate_vertex_express(prompt, data['question_id'], mode, project_id, api_key)

        if answer:
            log_message(f"  ✓ Answered [{mode}]")
            mode_data["response"] = answer
            changed = True
            time.sleep(1)

    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def main_vertex(results_folder: str, project_id: str):
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: No API Key found.")
        return

    print("="*60)
    print(f"CTAS Gen - Vertex AI Express Mode")
    print(f"Project: {project_id}")
    print(f"Endpoint: {VERTEX_LOCATION}-aiplatform.googleapis.com")
    print("="*60)

    files = sorted(Path(results_folder).glob(CTAS_FILE_PATTERN))
    for f in files:
        process_file_vertex(f, project_id, api_key)


def main(always_reason: bool = False, results_folder: Optional[str] = None):
    global CHECKPOINT_FILE, LOG_FILE
    if always_reason:
        CHECKPOINT_FILE = REASONING_CHECKPOINT_FILE
        LOG_FILE = REASONING_LOG_FILE
    results_folder = results_folder or RESULTS_FOLDER

    log_message("="*60)
    if always_reason:
        log_message(f"REASONING-ON Answer Generation - {MODEL_NAME}")
        log_message("Gemini SDK with Thinking Config ALWAYS ENABLED")
    else:
        log_message(f"CTAS Answer Generation - {MODEL_NAME}")
        log_message("Gemini SDK with Thinking Config Support")
    log_message("="*60)

    if not GEMINI_API_KEY:
        log_message("ERROR: No API Key found.")
        log_message("Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file")
        return

    # Initialize Gemini Client
    client = genai.Client(api_key=GEMINI_API_KEY)
    log_message(f"✓ Initialized Gemini Client with {MODEL_NAME}")

    # Get files from the results folder
    results_path = Path(results_folder)
    if not results_path.exists():
        log_message(f"ERROR: Results folder not found: {results_folder}")
        return

    ctas_files = sorted(results_path.glob(CTAS_FILE_PATTERN))

    if not ctas_files:
        log_message(f"ERROR: No files matching pattern '{CTAS_FILE_PATTERN}' found in {results_folder}")
        return

    log_message(f"✓ Found {len(ctas_files)} files to process")
    log_message(f"✓ Rate limiting: Processing 1 file per {FILE_PROCESSING_DELAY} seconds (5 parallel requests per minute)")
    log_message("")

    checkpoint = load_checkpoint()

    for i, file_path in enumerate(ctas_files, 1):
        log_message(f"\n{'#'*60}")
        log_message(f"FILE {i}/{len(ctas_files)}")
        log_message(f"{'#'*60}")

        checkpoint, made_api_calls = process_ctas_file(file_path, client, checkpoint, always_reason)

        # Add delay between files to respect rate limits (only if API calls were made, and not last file;
        # --always-reason regenerates every file and always waits)
        if (made_api_calls or always_reason) and i < len(ctas_files):
            log_message(f"\n⏳ Waiting {FILE_PROCESSING_DELAY} seconds before next file (rate limiting)...")
            time.sleep(FILE_PROCESSING_DELAY)

    log_message("\n" + "="*60)
    log_message("✓ ALL FILES PROCESSED SUCCESSFULLY")
    log_message("="*60)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate answers for per-question CTAS result files")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--always-reason", action="store_true",
                         help="Enable thinking for every question and regenerate all modes; existing "
                              "answers are kept as response_old (checkpoint/log: generation_*_reasoning)")
    backend.add_argument("--vertex-express", action="store_true",
                         help="Use the Vertex AI express endpoint (API key, no SDK); "
                              "files are read from the current directory unless --results-dir is set")
    parser.add_argument("--results-dir", default=None,
                        help=f"Folder with {CTAS_FILE_PATTERN} files (default: env RESULTS_DIR or {RESULTS_FOLDER})")
    parser.add_argument("--vertex-project", default=os.getenv("VERTEX_PROJECT_ID", VERTEX_PROJECT_ID),
                        help="Google Cloud project for --vertex-express (env VERTEX_PROJECT_ID)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.vertex_express:
        main_vertex(args.results_dir or ".", args.vertex_project)
    else:
        main(always_reason=args.always_reason, results_folder=args.results_dir)