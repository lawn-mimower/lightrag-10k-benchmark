"""
Script to parse specific documents using Mistral OCR API and export to JSON with markdown content.
Matches output format of parse_documents_with_docling.py
"""

from mistralai import Mistral
import os
import base64
import json
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, Optional
from dotenv import load_dotenv

# Base directory where documents are located
BASE_DIR = Path(os.getenv("DOCS_DIR", "documents"))
load_dotenv()

# Specific documents to process
DOCUMENTS_TO_PROCESS = [
    "sample_docs/sample_statement.xml",
    # Add more documents here as needed
]


def initialize_client():
    """Initialize Mistral client with API key."""
    print("Initializing Mistral Document AI client...")

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")

    client = Mistral(api_key=api_key)
    print("✓ Mistral client initialized successfully")

    return client


def encode_file_to_base64(file_path: Path) -> str:
    """
    Encode a file to base64 string.

    Args:
        file_path: Path to the file

    Returns:
        Base64 encoded string
    """
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')


def get_document_type_and_key(file_path: Path) -> tuple:
    """
    Determine document type and key based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (document_type, base64_key)
    """
    ext = file_path.suffix.lower()

    # Image formats
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
        return "image_base64", "image_base64"
    # Document formats (PDF, DOCX, XLSX, XML, etc.)
    else:
        return "document_base64", "document_base64"


def extract_text_from_response(response) -> str:
    """
    Extract text from Mistral OCR response.

    Args:
        response: OCR response from Mistral API

    Returns:
        Extracted text string in markdown format
    """
    try:
        # Check if response has pages attribute (typical for document OCR)
        if hasattr(response, 'pages') and response.pages:
            markdown_parts = []

            for i, page in enumerate(response.pages):
                # Add page separator for multi-page documents
                if i > 0:
                    markdown_parts.append("\n\n---\n\n")
                    markdown_parts.append(f"## Page {i + 1}\n\n")

                # Extract content from page - Mistral OCR returns markdown attribute
                if hasattr(page, 'markdown'):
                    markdown_parts.append(page.markdown)
                elif hasattr(page, 'content'):
                    markdown_parts.append(page.content)
                elif hasattr(page, 'text'):
                    markdown_parts.append(page.text)
                elif isinstance(page, dict):
                    # Try to extract from dict structure
                    content = page.get('markdown') or page.get('content') or page.get('text', '')
                    markdown_parts.append(content)

            return ''.join(markdown_parts)

        # For single page or direct response
        if hasattr(response, 'markdown'):
            return response.markdown
        if hasattr(response, 'content'):
            return response.content
        if hasattr(response, 'text'):
            return response.text

        # Handle dict response
        if isinstance(response, dict):
            # Try various possible keys
            for key in ['markdown', 'content', 'text', 'extracted_text', 'result']:
                if key in response:
                    return response[key]

            # Check for pages in dict
            if 'pages' in response:
                pages = response['pages']
                if isinstance(pages, list):
                    markdown_parts = []
                    for i, page in enumerate(pages):
                        if i > 0:
                            markdown_parts.append("\n\n---\n\n")
                            markdown_parts.append(f"## Page {i + 1}\n\n")
                        if isinstance(page, dict):
                            content = page.get('markdown') or page.get('content') or page.get('text', '')
                            markdown_parts.append(content)
                        else:
                            markdown_parts.append(str(page))
                    return ''.join(markdown_parts)

        # Fallback to string representation
        return str(response)

    except Exception as e:
        print(f"    Warning: Error extracting text from response: {e}")
        return str(response)


def process_document(client: Mistral, doc_path: Path) -> Dict:
    """
    Process a single document using Mistral OCR.

    Args:
        client: Mistral client instance
        doc_path: Path to the document

    Returns:
        Dictionary containing processing results
    """
    doc_start = time.time()
    uploaded_file = None

    try:
        # Check file size (50 MB limit)
        file_size_mb = doc_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:
            raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds 50 MB limit")

        ext = doc_path.suffix.lower()

        # For images, we can use base64 directly
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
            # Check image file size (10 MB limit for images)
            if file_size_mb > 10:
                raise ValueError(f"Image file size ({file_size_mb:.2f} MB) exceeds 10 MB limit")

            file_base64 = encode_file_to_base64(doc_path)
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.bmp': 'image/bmp',
                '.gif': 'image/gif',
                '.tiff': 'image/tiff'
            }.get(ext, 'image/jpeg')

            # Create a data URI for the image
            data_uri = f"data:{mime_type};base64,{file_base64}"

            document_data = {
                "type": "image_url",
                "image_url": data_uri
            }

        else:
            # For documents (PDF, DOCX, XML, etc.), use the signed URL approach
            print(f"    → Uploading file to Mistral...")

            # Read the file content
            with open(doc_path, 'rb') as f:
                file_content = f.read()

            # Upload the file with proper format
            uploaded_file = client.files.upload(
                file={
                    "file_name": doc_path.name,
                    "content": file_content
                },
                purpose="ocr"
            )

            print(f"    → File uploaded, getting signed URL...")

            # Get a signed URL for the uploaded file
            signed_url_response = client.files.get_signed_url(file_id=uploaded_file.id)

            document_data = {
                "type": "document_url",
                "document_url": signed_url_response.url
            }

            print(f"    → Processing document with OCR...")

        # Prepare OCR parameters
        ocr_params = {
            "model": "mistral-ocr-latest",
            "document": document_data
        }

        # Add table formatting for document types (not images)
        if document_data["type"] == "document_url":
            ocr_params["table_format"] = "markdown"  # Use markdown format for tables
            # Note: header/footer extraction only available for newer OCR models
            # ocr_params["extract_header"] = True
            # ocr_params["extract_footer"] = True
            ocr_params["include_image_base64"] = False  # Don't include base64 images in response

        # Process with Mistral OCR
        ocr_response = client.ocr.process(**ocr_params)

        # Extract text in markdown format
        markdown_content = extract_text_from_response(ocr_response)

        # Handle headers and footers if they were extracted separately
        if hasattr(ocr_response, 'header') and ocr_response.header:
            markdown_content = f"# Header\n{ocr_response.header}\n\n---\n\n" + markdown_content
        if hasattr(ocr_response, 'footer') and ocr_response.footer:
            markdown_content = markdown_content + f"\n\n---\n\n# Footer\n{ocr_response.footer}"

        processing_time = time.time() - doc_start

        return {
            "filename": doc_path.name,
            "file_path": str(doc_path),
            "status": "success",
            "markdown": markdown_content,
            "markdown_length": len(markdown_content),
            "processing_time_seconds": round(processing_time, 3),
            "timestamp": datetime.now().isoformat(),
            "processing_method": "image_url" if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'] else "signed_url"
        }

    except Exception as e:
        processing_time = time.time() - doc_start
        return {
            "filename": doc_path.name,
            "file_path": str(doc_path),
            "status": "error",
            "error": str(e),
            "markdown": None,
            "processing_time_seconds": round(processing_time, 3),
            "timestamp": datetime.now().isoformat()
        }

    finally:
        # Clean up uploaded file if it exists
        if uploaded_file:
            try:
                client.files.delete(file_id=uploaded_file.id)
                print(f"    → Cleaned up uploaded file")
            except Exception as e:
                print(f"    → Warning: Could not delete uploaded file: {e}")


def parse_documents():
    """Parse all specified documents and return JSON with markdown content."""

    start_time = datetime.now()
    client = initialize_client()

    results = {
        "metadata": {
            "start_timestamp": start_time.isoformat(),
            "source_directory": str(BASE_DIR),
            "total_documents": len(DOCUMENTS_TO_PROCESS),
            "processing_mode": "Mistral OCR API (mistral-ocr-latest)",
            "api_features": {
                "table_format": "markdown",
                "extract_headers": True,
                "extract_footers": True
            }
        },
        "documents": []
    }

    for i, doc_name in enumerate(DOCUMENTS_TO_PROCESS, 1):
        doc_path = BASE_DIR / doc_name

        if not doc_path.exists():
            print(f"[{i}/{len(DOCUMENTS_TO_PROCESS)}] SKIPPED: {doc_name} (file not found)")
            results["documents"].append({
                "filename": doc_name,
                "status": "error",
                "error": "File not found",
                "markdown": None,
                "timestamp": datetime.now().isoformat()
            })
            continue

        print(f"[{i}/{len(DOCUMENTS_TO_PROCESS)}] Processing: {doc_name}...")

        # Process the document
        doc_result = process_document(client, doc_path)
        results["documents"].append(doc_result)

        if doc_result["status"] == "success":
            print(f"    ✓ Success: {doc_result['markdown_length']:,} chars in {doc_result['processing_time_seconds']}s")
        else:
            print(f"    ✗ Error: {doc_result.get('error', 'Unknown error')}")

    results["metadata"]["end_timestamp"] = datetime.now().isoformat()
    results["metadata"]["total_time_seconds"] = round((datetime.now() - start_time).total_seconds(), 2)

    return results


def save_individual_markdowns(results: Dict, output_dir: Path = Path("mistral_extracted_texts")):
    """
    Save individual markdown files for each successfully processed document.

    Args:
        results: Processing results dictionary
        output_dir: Directory to save individual markdown files
    """
    output_dir.mkdir(exist_ok=True)

    for doc in results["documents"]:
        if doc["status"] == "success" and doc["markdown"]:
            output_file = output_dir / f"{Path(doc['filename']).stem}.md"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {doc['filename']}\n\n")
                    f.write(f"*Processed: {doc['timestamp']}*\n\n")
                    f.write("---\n\n")
                    f.write(doc["markdown"])
                print(f"    → Saved markdown: {output_file}")
            except Exception as e:
                print(f"    → Error saving markdown: {e}")


def parse_cli_args():
    import argparse
    parser = argparse.ArgumentParser(description="Parse documents; defaults to the list above")
    parser.add_argument("files", nargs="*",
                        help="Documents to parse, relative to --input-dir or absolute")
    parser.add_argument("--input-dir", default=str(BASE_DIR),
                        help="Directory containing the documents (env DOCS_DIR)")
    return parser.parse_args()


def main():
    """Main execution function."""
    global BASE_DIR, DOCUMENTS_TO_PROCESS
    args = parse_cli_args()
    BASE_DIR = Path(args.input_dir)
    if args.files:
        DOCUMENTS_TO_PROCESS = args.files
    print("=" * 70)
    print("Mistral Document AI Parser")
    print("=" * 70)
    print(f"Source directory: {BASE_DIR}")
    print(f"Documents to process: {len(DOCUMENTS_TO_PROCESS)}")
    print(f"Model: mistral-ocr-latest")
    print(f"API pricing: $0.001 per page")
    print("=" * 70)
    print()

    try:
        # Parse all documents
        results = parse_documents()

        # Save to JSON file (main output matching docling format)
        output_file = Path("mistral_parsed_documents.json")
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Also save individual markdown files
        print("\nSaving individual markdown files...")
        save_individual_markdowns(results)

        # Print summary
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        successful = sum(1 for doc in results["documents"] if doc["status"] == "success")
        failed = sum(1 for doc in results["documents"] if doc["status"] == "error")
        total_chars = sum(doc.get("markdown_length", 0) for doc in results["documents"] if doc["status"] == "success")
        avg_time = sum(doc.get("processing_time_seconds", 0) for doc in results["documents"]) / len(results["documents"]) if results["documents"] else 0

        print(f"Total documents: {len(DOCUMENTS_TO_PROCESS)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total characters extracted: {total_chars:,}")
        print(f"Average processing time: {round(avg_time, 2)}s per document")
        print(f"Total time: {results['metadata']['total_time_seconds']}s")
        print(f"\nOutput saved to: {output_file.absolute()}")
        print(f"Individual markdowns saved to: mistral_extracted_texts/")
        print("=" * 70)

        # If there were failures, print details
        if failed > 0:
            print("\nFailed documents:")
            for doc in results["documents"]:
                if doc["status"] == "error":
                    print(f"  • {doc['filename']}: {doc.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        print("\nPlease ensure:")
        print("1. MISTRAL_API_KEY is set in your environment variables")
        print("2. You have an active Mistral API subscription with Document AI access")
        print("3. The documents exist in the specified directory")
        print("4. Your files are under 50 MB and less than 1,000 pages")
        raise


if __name__ == "__main__":
    main()