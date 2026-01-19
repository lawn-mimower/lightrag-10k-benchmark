#!/usr/bin/env python3
"""
Quick test of the simple evaluation setup
==========================================
Tests with just 1 file to make sure everything works.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Check environment
print("🔍 Pre-flight checks...")
print("-" * 40)

# 1. Check API key
api_key = os.getenv("MISTRAL_API_KEY")
if api_key:
    print(f"✅ MISTRAL_API_KEY found ({len(api_key)} chars)")
else:
    print("❌ MISTRAL_API_KEY not found!")
    exit(1)

# 2. Check results directory
results_dir = Path("./lightrag-bench/5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL")
if results_dir.exists():
    files = list(results_dir.glob("test_results_*_question_*.json"))
    print(f"✅ Results directory found ({len(files)} files)")

    if files:
        # Check first file structure
        with open(files[0], 'r') as f:
            data = json.load(f)
            modes = list(data.get("modes", {}).keys())
            print(f"✅ First file has modes: {modes}")
else:
    print("❌ Results directory not found!")
    exit(1)

# 3. Test imports
print("\n📦 Testing imports...")
try:
    from datasets import Dataset
    from ragas import evaluate
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

print("\n" + "=" * 40)
print("✨ All checks passed!")
print("=" * 40)
print("\nYou can now run the evaluation:")
print("  python batch_ragas_evaluation_simple.py")
print("\nOr use the runner script:")
print("  ./run_simple_evaluation.sh")