#!/usr/bin/env python3
"""
Answer Generation Script for CTAS Test Results
(Direct HTTP Mode - Bypasses SDK Auth Issues)
"""

import json
import os
import time
import requests  # Bypassing the SDK
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Configuration
CTAS_FILE_PATTERN = "test_results_CTAS_question_*.json"
CHECKPOINT_FILE = "generation_checkpoint_ctas.json"
LOG_FILE = "generation_log_ctas.txt"
MODEL_NAME = "gemini-3-flash-preview"

# Parameters
REQUEST_DELAY = 2
RETRY_DELAY = 10
MAX_RETRIES = 3

class DirectGeminiClient:
    """A minimal wrapper to replace the SDK client"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
        
    def generate_content(self, prompt: str) -> str:
        headers = {'Content-Type': 'application/json'}
        params = {'key': self.api_key}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(self.url, headers=headers, params=params, json=data)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
            
        result = response.json()
        
        # Extract text safely
        try:
            return result['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return None

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

def generate_answer_logic(client: DirectGeminiClient, question: str, context: str, mode: str, question_id: str) -> str:
    prompt = f"""You are a financial document analyst specializing in SEC 10-K filings.

CRITICAL INSTRUCTIONS:
- Answer ONLY based on the provided context
- Be precise with numerical values
- Keep answers concise (2-4 sentences)

Question: {question}

Context from {mode.upper()}:
{context}

Please provide a concise, factual answer based only on the information in the context above."""

    for attempt in range(MAX_RETRIES):
        try:
            answer = client.generate_content(prompt)
            
            if not answer:
                log_message(f"  ⚠ Empty response for {question_id} [{mode}]")
                return None

            answer = answer.strip()
            log_message(f"  ✓ Generated answer for {question_id} [{mode}] ({len(answer)} chars)")
            return answer

        except Exception as e:
            log_message(f"  ⚠ Attempt {attempt+1}/{MAX_RETRIES} failed for {question_id} [{mode}]: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None
    return None

def process_ctas_file(file_path: Path, client: DirectGeminiClient, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    log_message(f"\nProcessing file: {file_path.name}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_id = data["question_id"]
    question = data["question"]
    
    if file_path.name not in checkpoint["completed"]:
        checkpoint["completed"][file_path.name] = {
            "question_id": question_id,
            "completed_modes": []
        }
    
    completed_modes = checkpoint["completed"][file_path.name]["completed_modes"]
    modes = data.get("modes", {})

    for mode_name, mode_data in modes.items():
        if mode_name in completed_modes:
            log_message(f"  ⊙ Already completed: {mode_name}")
            continue

        if "response" in mode_data and mode_data["response"]:
            completed_modes.append(mode_name)
            continue

        if not mode_data.get("retrieved_context"):
            log_message(f"  ⊘ No context: {mode_name}")
            continue

        answer = generate_answer_logic(client, question, mode_data["retrieved_context"], mode_name, question_id)

        if answer:
            mode_data["response"] = answer
            completed_modes.append(mode_name)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            checkpoint["completed"][file_path.name]["completed_modes"] = completed_modes
            save_checkpoint(checkpoint)
            time.sleep(REQUEST_DELAY)

    return checkpoint

def main():
    log_message("="*60)
    log_message(f"CTAS Gen - {MODEL_NAME} - DIRECT HTTP MODE")
    log_message("="*60)

    if not GEMINI_API_KEY:
        log_message("ERROR: No API Key found.")
        return

    # Initialize Direct Client (No SDK, No Auth Confusion)
    client = DirectGeminiClient(api_key=GEMINI_API_KEY)
    
    ctas_files = sorted(Path(".").glob(CTAS_FILE_PATTERN))
    checkpoint = load_checkpoint()

    for file_path in ctas_files:
        checkpoint = process_ctas_file(file_path, client, checkpoint)

if __name__ == "__main__":
    main()