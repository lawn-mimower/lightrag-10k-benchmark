"""
Test script to verify Mistral OCR API connection and basic functionality.
"""

from mistralai import Mistral
import os
import sys
from pathlib import Path

def test_api_connection():
    """Test basic API connectivity."""
    print("Testing Mistral OCR API connection...")
    print("-" * 50)

    # Check API key
    api_key = os.environ["MISTRAL_API_KEY"]
    if not api_key:
        print("❌ MISTRAL_API_KEY not found in environment variables")
        print("\nPlease set your API key using:")
        print("  export MISTRAL_API_KEY='your-api-key-here'")
        print("Or create a .env file with:")
        print("  MISTRAL_API_KEY=your-api-key-here")
        return False

    print("✅ API key found")

    try:
        # Initialize client
        client = Mistral(api_key=api_key)
        print("✅ Mistral client initialized")

        # Test with a simple text image URL (using a public example)
        print("\nTesting OCR with a sample image...")
        test_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/2201.04234"  # Small sample PDF
            },
            table_format="markdown"
        )

        print("✅ OCR API call successful")

        # Check if we got a response
        if test_response:
            print("✅ Response received from API")

            # Try to extract some content
            if hasattr(test_response, 'pages'):
                num_pages = len(test_response.pages)
                print(f"✅ Document processed: {num_pages} page(s) extracted")

                # Show first 200 characters of content
                if num_pages > 0 and hasattr(test_response.pages[0], 'content'):
                    content_preview = test_response.pages[0].content[:200]
                    print(f"\nContent preview:\n{content_preview}...")
            else:
                print("✅ Response structure validated")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nPossible issues:")
        print("1. Invalid API key")
        print("2. API key doesn't have Document AI access")
        print("3. Network connectivity issues")
        print("4. Mistral API service issues")
        return False


def test_local_file(file_path):
    """Test OCR with a local file if provided."""
    print(f"\nTesting with local file: {file_path}")
    print("-" * 50)

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("❌ API key not set")
        return False

    try:
        import base64
        from mistral_document_extraction import (
            initialize_client,
            process_document
        )

        client = initialize_client()
        result = process_document(client, Path(file_path))

        if result["status"] == "success":
            print(f"✅ Successfully processed: {result['filename']}")
            print(f"   Characters extracted: {result['markdown_length']:,}")
            print(f"   Processing time: {result['processing_time_seconds']}s")
            return True
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Main test function."""
    print("=" * 70)
    print("Mistral OCR API Test Suite")
    print("=" * 70)
    print()

    # Test API connection
    api_ok = test_api_connection()

    # If a file path is provided as argument, test with that file
    if len(sys.argv) > 1 and api_ok:
        test_file = sys.argv[1]
        test_local_file(test_file)

    print("\n" + "=" * 70)
    if api_ok:
        print("✅ API test completed successfully!")
        print("\nYou can now run the main script:")
        print("  python mistral_document_extraction.py")

        if len(sys.argv) == 1:
            print("\nTo test with a specific file:")
            print("  python test_mistral_ocr_api.py /path/to/your/file.pdf")
    else:
        print("❌ API test failed. Please check the issues above.")

    print("=" * 70)


if __name__ == "__main__":
    main()