#!/usr/bin/env python3
"""
Test script to verify Gemini API connection before running the full benchmark.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

def test_gemini_connection():
    """Test basic Gemini API connection."""

    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("   Please set it in your .env file or export it:")
        print("   export GEMINI_API_KEY='your-api-key'")
        return False
    else:
        print(f"✓ GEMINI_API_KEY found (length: {len(api_key)})")

    # Initialize client
    try:
        client = genai.Client(api_key=api_key)
        print("✓ Gemini client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        return False

    # Test a simple API call
    print("\nTesting API call with gemini-3-flash-preview...")

    test_prompt = "What is 2+2? Answer in one word."

    try:
        # Configure generation
        generation_config = types.GenerateContentConfig(
            max_output_tokens=100,
            temperature=0.1,
        )

        # Make API call
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=test_prompt,
            config=generation_config
        )

        # Extract answer
        answer = ""
        if response and response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        answer += part.text

        if answer:
            print(f"✓ API call successful!")
            print(f"  Response: {answer.strip()}")

            # Check for token usage
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                if hasattr(usage, 'prompt_token_count'):
                    print(f"  Input tokens: {usage.prompt_token_count}")
                if hasattr(usage, 'candidates_token_count'):
                    print(f"  Output tokens: {usage.candidates_token_count}")
                if hasattr(usage, 'total_token_count'):
                    print(f"  Total tokens: {usage.total_token_count}")
            else:
                print("  Note: Token usage metadata not available")
        else:
            print("⚠️ API call returned but no answer text found")
            return False

    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

    print("\n" + "="*60)
    print("✅ All tests passed! Ready to run the benchmark.")
    print("="*60)
    return True

def check_data_directory():
    """Check if the data directory exists and has files."""
    data_dir = "5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL"

    print(f"\nChecking data directory...")

    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False

    # Count JSON files
    from pathlib import Path
    json_files = list(Path(data_dir).glob("*.json"))

    if not json_files:
        print(f"❌ No JSON files found in {data_dir}")
        return False

    print(f"✓ Found {len(json_files)} JSON files in data directory")

    # Check a sample file structure
    import json
    sample_file = json_files[0]

    try:
        with open(sample_file, 'r') as f:
            data = json.load(f)

        required_fields = ['question', 'modes']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            print(f"⚠️ Sample file missing fields: {missing_fields}")
            return False

        # Check modes
        modes = data.get('modes', {})
        expected_modes = ['local', 'global', 'naive', 'hybrid', 'mix']
        available_modes = [m for m in expected_modes if m in modes]

        print(f"✓ Sample file structure looks good")
        print(f"  Available modes: {', '.join(available_modes)}")

        # Check for context in first available mode
        if available_modes:
            first_mode = available_modes[0]
            if 'retrieved_context' in modes[first_mode]:
                context_len = len(modes[first_mode]['retrieved_context'])
                print(f"  Sample context length ({first_mode}): {context_len:,} chars")
            else:
                print(f"⚠️ No retrieved_context in {first_mode} mode")

    except Exception as e:
        print(f"❌ Error reading sample file: {e}")
        return False

    return True

if __name__ == "__main__":
    print("="*60)
    print("Gemini Benchmark Test Suite")
    print("="*60)

    # Run tests
    api_ok = test_gemini_connection()
    data_ok = check_data_directory()

    print("\n" + "="*60)
    if api_ok and data_ok:
        print("✅ ALL CHECKS PASSED - Ready to run benchmark!")
        print("\nYou can now run:")
        print("  python3 gemini_quick_benchmark.py     # For quick test (10 questions)")
        print("  python3 benchmark_gemini_api.py       # For full benchmark")
    else:
        print("❌ Some checks failed - please fix the issues above")
    print("="*60)