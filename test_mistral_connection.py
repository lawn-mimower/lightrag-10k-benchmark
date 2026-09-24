#!/usr/bin/env python3
"""
Mistral API checks
==================
--check quick     (default) one ChatOpenAI request to the RAGAS judge model
--check diagnose  network, authentication, chat completion and rate-limit
                  diagnostics over plain HTTP
--check params    which ChatOpenAI parameters the Mistral API accepts
"""

import argparse
import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
RAGAS_JUDGE_MODEL = "ministral-14b-2512"


# ============================================
# --check quick: verify the Mistral API connection
# ============================================
def quick_check():
    from langchain_openai import ChatOpenAI

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


# ============================================
# --check diagnose: diagnose API connection issues
# ============================================
async def test_connection():
    """Comprehensive API connection test."""
    print("="*60)
    print("MISTRAL API CONNECTION TEST")
    print("="*60)

    # Check API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not found!")
        print("\nTo fix this:")
        print("1. Create a .env file in this directory")
        print("2. Add: MISTRAL_API_KEY='your-key-here'")
        print("3. Or export it: export MISTRAL_API_KEY='your-key-here'")
        return False

    print(f"✓ API Key found (length: {len(api_key)} chars)")
    print(f"  First 10 chars: {api_key[:10]}...")

    # Test network connectivity
    print("\n📡 Testing network connectivity...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.mistral.ai", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                print(f"✓ Can reach api.mistral.ai (status: {resp.status})")
    except Exception as e:
        print(f"❌ Cannot reach api.mistral.ai: {e}")
        print("\nPossible issues:")
        print("- No internet connection")
        print("- Firewall blocking the connection")
        print("- DNS resolution issues")
        return False

    # Test API authentication
    print("\n🔑 Testing API authentication...")
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # List models to test auth
            async with session.get(
                "https://api.mistral.ai/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Authentication successful!")
                    print(f"  Available models: {len(data.get('data', []))}")

                    # Check if our model is available
                    model_ids = [m.get('id') for m in data.get('data', [])]
                    if 'ministral-14b-2512' in model_ids:
                        print("  ✓ ministral-14b-2512 is available")
                    else:
                        print("  ⚠️ ministral-14b-2512 not found in available models")
                        print(f"  Available: {', '.join(model_ids[:5])}...")

                elif response.status == 401:
                    print("❌ Authentication failed! Invalid API key.")
                    return False
                else:
                    text = await response.text()
                    print(f"❌ API returned status {response.status}")
                    print(f"  Response: {text[:200]}")
                    return False

    except Exception as e:
        print(f"❌ Error during authentication test: {e}")
        return False

    # Test chat completion
    print("\n💬 Testing chat completion...")
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "ministral-14b-2512",
                "messages": [{"role": "user", "content": "Say 'test successful' in 3 words"}],
                "max_tokens": 10,
                "temperature": 0
            }

            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    print("✅ Chat completion successful!")
                    print(f"  Response: {content}")
                    return True
                elif response.status == 429:
                    print("⚠️ Rate limit reached!")
                    print("  The API is working but you've hit rate limits.")
                    print("  Solution: Wait a bit or reduce concurrent requests.")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Chat completion failed (status {response.status})")
                    print(f"  Response: {text[:200]}")
                    return False

    except asyncio.TimeoutError:
        print("❌ Request timeout! The API is too slow.")
        return False
    except Exception as e:
        print(f"❌ Error during chat completion: {e}")
        return False

async def test_rate_limits():
    """Test rate limits with multiple concurrent requests."""
    print("\n🚀 Testing rate limits...")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ Cannot test without API key")
        return

    async def make_request(session, i):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "ministral-14b-2512",
            "messages": [{"role": "user", "content": f"test {i}"}],
            "max_tokens": 5
        }
        try:
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                return resp.status
        except Exception as e:
            return f"error: {e}"

    async with aiohttp.ClientSession() as session:
        # Test with 5 concurrent requests
        tasks = [make_request(session, i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r == 200)
        rate_limit_count = sum(1 for r in results if r == 429)
        error_count = sum(1 for r in results if isinstance(r, str) or (isinstance(r, int) and r >= 500))

        print(f"  Results from 5 concurrent requests:")
        print(f"  ✓ Successful: {success_count}")
        print(f"  ⚠ Rate limited: {rate_limit_count}")
        print(f"  ✗ Errors: {error_count}")

        if rate_limit_count > 0:
            print("\n  📌 Recommendation: Reduce MAX_CONCURRENT in your script")
        if error_count > 0:
            print("\n  📌 Recommendation: Add retry logic for transient errors")

async def diagnose():
    """Network, authentication, chat completion and rate-limit checks."""
    global aiohttp
    import aiohttp

    success = await test_connection()

    if success:
        await test_rate_limits()
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour API connection is working correctly.")
        print("You can now run the evaluation script:")
        print("  python batch_ragas_evaluation.py")
    else:
        print("\n" + "="*60)
        print("❌ CONNECTION TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running the evaluation.")


# ============================================
# --check params: which parameters work with Mistral API via ChatOpenAI
# ============================================
def params_check():
    from langchain_openai import ChatOpenAI

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


CHECKS = {"quick": quick_check, "diagnose": lambda: asyncio.run(diagnose()), "params": params_check}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistral API checks for the RAGAS judge")
    parser.add_argument("--check", choices=list(CHECKS), default="quick",
                        help="quick (default): one chat request; diagnose: connectivity, auth, chat and "
                             "rate limits over HTTP; params: which ChatOpenAI parameters are accepted")
    CHECKS[parser.parse_args().check]()
