"""
Test script to verify RAGAS setup and configuration
"""

import os
from google import genai
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    Faithfulness,
    AnswerCorrectness
)

print("Testing RAGAS setup...")

# Check API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not set!")
    exit(1)

print("✓ API key found")

# Initialize client
try:
    client = genai.Client(api_key=api_key)
    print("✓ Gemini client initialized")
except Exception as e:
    print(f"✗ Failed to initialize client: {e}")
    exit(1)

# Initialize LLM
try:
    llm = llm_factory(
        "gemini-3-flash-preview",
        provider="google",
        client=client
    )
    print("✓ LLM initialized (gemini-3-flash-preview)")
except Exception as e:
    print(f"✗ Failed to initialize LLM: {e}")
    exit(1)

# Initialize embeddings
try:
    embeddings = embedding_factory(
        "google",
        model="text-embedding-004",
        client=client
    )
    print("✓ Embeddings initialized")
except Exception as e:
    print(f"✗ Failed to initialize embeddings: {e}")
    exit(1)

# Initialize metrics
try:
    metrics = [
        ContextRecall(llm=llm),
        ContextPrecision(llm=llm),
        Faithfulness(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings)
    ]
    print(f"✓ Metrics initialized: {len(metrics)} metrics")
    for metric in metrics:
        print(f"  - {metric.__class__.__name__}: {type(metric)}")
except Exception as e:
    print(f"✗ Failed to initialize metrics: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("RAGAS is configured correctly.")
print("="*60)
