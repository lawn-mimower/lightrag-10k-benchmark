#!/usr/bin/env python3
"""
Quick test to verify Mistral API connection
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Configuration
RAGAS_JUDGE_MODEL = "ministral-14b-2512"
mistral_api_key = os.getenv("MISTRAL_API_KEY")

print("="*60)
print("🔍 Testing Mistral API Connection")
print("="*60)
print(f"Model: {RAGAS_JUDGE_MODEL}")
print(f"API Key: {mistral_api_key[:8]}..." if mistral_api_key else "❌ No API key!")
print(f"Endpoint: https://api.mistral.ai/v1")
print()

if not mistral_api_key:
    print("Please set MISTRAL_API_KEY in your .env file")
    exit(1)

# Create LLM client
llm = ChatOpenAI(
    model=RAGAS_JUDGE_MODEL,
    api_key=mistral_api_key,
    base_url="https://api.mistral.ai/v1",
    max_retries=2,
    request_timeout=30
)

# Test with simple prompt
try:
    print("Sending test prompt...")
    response = llm.invoke("Say 'API connection successful' in 5 words or less.")
    print(f"✅ Response: {response.content}")
    print("\n✨ Mistral API connection verified!")

except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nPossible issues:")
    print("1. Check your API key is valid")
    print("2. Check your internet connection")
    print("3. Check if the model name is correct")
    print(f"4. Try using 'mistral-large-latest' instead of '{RAGAS_JUDGE_MODEL}'")