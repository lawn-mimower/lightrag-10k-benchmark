#!/usr/bin/env python3
"""
Test Mistral API Connection
============================
Standalone script to diagnose API connection issues.
"""

import os
import asyncio
import aiohttp
import json
from dotenv import load_dotenv

# Load environment
load_dotenv()

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

async def main():
    """Run all tests."""
    success = await test_connection()

    if success:
        await test_rate_limits()
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour API connection is working correctly.")
        print("You can now run the fixed evaluation script:")
        print("  python batch_ragas_evaluation_fast_fixed.py")
    else:
        print("\n" + "="*60)
        print("❌ CONNECTION TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running the evaluation.")

if __name__ == "__main__":
    asyncio.run(main())