#!/usr/bin/env python3
"""
Test which parameters work with Mistral API via ChatOpenAI
===========================================================
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("❌ MISTRAL_API_KEY not found!")
    exit(1)

print("Testing different parameter combinations with Mistral API...")
print("="*60)

# Test 1: WORKING configuration (from single file script)
print("\n1. Testing WORKING configuration (no max_tokens):")
try:
    llm1 = ChatOpenAI(
        model="ministral-14b-2512",
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        max_retries=5,
        request_timeout=180
    )
    response = llm1.invoke("Say 'test 1 works' in 3 words")
    print(f"   ✅ SUCCESS: {response.content}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 2: With max_tokens (might fail)
print("\n2. Testing with max_tokens parameter:")
try:
    llm2 = ChatOpenAI(
        model="ministral-14b-2512",
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        max_retries=5,
        request_timeout=180,
        max_tokens=100  # THIS MIGHT CAUSE ISSUES
    )
    response = llm2.invoke("Say 'test 2 works' in 3 words")
    print(f"   ✅ SUCCESS: {response.content}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 3: With temperature
print("\n3. Testing with temperature parameter:")
try:
    llm3 = ChatOpenAI(
        model="ministral-14b-2512",
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        max_retries=5,
        request_timeout=180,
        temperature=0.1
    )
    response = llm3.invoke("Say 'test 3 works' in 3 words")
    print(f"   ✅ SUCCESS: {response.content}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n" + "="*60)
print("RESULTS:")
print("- Use the exact configuration from test 1")
print("- DO NOT include max_tokens parameter")
print("- Temperature is OK to include")