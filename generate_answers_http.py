#!/usr/bin/env python3
"""
Answer Generation Script - Vertex AI Express Mode
Targets the Vertex API directly to bypass AI Studio restriction errors.
"""

import json
import os
import time
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 1. Load Env
load_dotenv(override=True)
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# 2. Configuration (Hardcoded from your logs)
PROJECT_ID = "your-gcp-project-id"  # Extracted from your previous logs
LOCATION = "us-central1"               # Standard Vertex region
MODEL_ID = "gemini-3.0-flash-preview-001" # Vertex often uses this versioned ID

CTAS_FILE_PATTERN = "test_results_CTAS_question_*.json"
CHECKPOINT_FILE = "generation_checkpoint_ctas.json"
LOG_FILE = "generation_log_ctas.txt"

def log_message(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def generate_vertex_express(prompt: str, question_id: str, mode: str) -> str:
    """
    Hits the Vertex AI 'Express' endpoint which accepts API Keys.
    """
    # Vertex AI Endpoint URL
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"
    )
    
    params = {"key": API_KEY}
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
            log_message(f"  ⚠ Model not found: {MODEL_ID}. Trying alternative ID...")
            return None # Trigger fallback logic if you had it, or just fail
            
        else:
            log_message(f"  ⚠ Vertex Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        log_message(f"  ⚠ Request Failed: {e}")
        return None

def process_file(file_path: Path):
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
        
        answer = generate_vertex_express(prompt, data['question_id'], mode)
        
        if answer:
            log_message(f"  ✓ Answered [{mode}]")
            mode_data["response"] = answer
            changed = True
            time.sleep(1)
            
    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

def main():
    if not API_KEY:
        print("❌ Error: No API Key found.")
        return

    print("="*60)
    print(f"CTAS Gen - Vertex AI Express Mode")
    print(f"Project: {PROJECT_ID}")
    print(f"Endpoint: {LOCATION}-aiplatform.googleapis.com")
    print("="*60)
    
    files = sorted(Path(".").glob(CTAS_FILE_PATTERN))
    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()