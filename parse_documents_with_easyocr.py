"""
Alternative document parser using EasyOCR instead of RT-DETRv2
This avoids the transformers compatibility issue
"""

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrOptions, AcceleratorDevice, AcceleratorOptions
import json
import os
from pathlib import Path
from datetime import datetime
import time

# Base directory where documents are located
BASE_DIR = Path(os.getenv("DOCS_DIR", "documents"))

# Specific documents to process
DOCUMENTS_TO_PROCESS = [
    #"sample_docs/sample_statement.docx",
    #"sample_docs/sample_statement.docx",
    #"sample_docs/sample_statement.docx",
    "sample_docs/sample_statement.xlsx",
    #"sample_docs/sample_statement.pdf",
    #"Validation.JPG",
    #"Prescutiny.JPG",
    #"Prescrutiny CFS.JPG",
    #"Validation CFS.JPG",
    #"InstanceDocument-EXAMPLEENGINEERINGPRIVATELIMITED_Standalone.xml",
    #"InstanceDocument-EXAMPLEENGINEERINGPRIVATELIMITED_Consolidated.xml"
]


def initialize_converter():
    """Initialize Docling DocumentConverter with EasyOCR backend."""
    print("Initializing Docling DocumentConverter with EasyOCR...")

    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True  # Extract table structure

    # Configure OCR with EasyOCR (alternative to default RT-DETRv2)
    ocr_options = OcrOptions()
    ocr_options.use_gpu = False  # CPU mode
    ocr_options.lang = ["en"]  # English OCR

    pipeline_options.ocr_options = ocr_options
    pipeline_options.do_ocr = True

    # Force CPU-only processing
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CPU
    )

    # Initialize converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    return converter


def parse_documents():
    """Parse all specified documents and return JSON with markdown content."""

    start_time = datetime.now()
    converter = initialize_converter()

    results = {
        "metadata": {
            "start_timestamp": start_time.isoformat(),
            "source_directory": str(BASE_DIR),
            "total_documents": len(DOCUMENTS_TO_PROCESS),
            "processing_mode": "CPU-only with EasyOCR"
        },
        "documents": []
    }

    for i, doc_name in enumerate(DOCUMENTS_TO_PROCESS, 1):
        doc_path = BASE_DIR / doc_name
        doc_start = time.time()

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

        try:
            print(f"[{i}/{len(DOCUMENTS_TO_PROCESS)}] Processing: {doc_name}...")

            # Convert document using Docling
            result = converter.convert(str(doc_path))
            markdown_content = result.document.export_to_markdown()

            processing_time = time.time() - doc_start

            # Store result
            results["documents"].append({
                "filename": doc_name,
                "file_path": str(doc_path),
                "status": "success",
                "markdown": markdown_content,
                "markdown_length": len(markdown_content),
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": datetime.now().isoformat()
            })

            print(f"    ✓ Success: {len(markdown_content):,} chars in {round(processing_time, 2)}s")

        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            results["documents"].append({
                "filename": doc_name,
                "file_path": str(doc_path),
                "status": "error",
                "error": str(e),
                "markdown": None,
                "timestamp": datetime.now().isoformat()
            })

    results["metadata"]["end_timestamp"] = datetime.now().isoformat()
    results["metadata"]["total_time_seconds"] = (datetime.now() - start_time).total_seconds()

    return results


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
    print("Docling Document Parser (EasyOCR Backend)")
    print("=" * 70)
    print(f"Source directory: {BASE_DIR}")
    print(f"Documents to process: {len(DOCUMENTS_TO_PROCESS)}")
    print("=" * 70)
    print()

    # Parse all documents
    results = parse_documents()

    # Save to JSON file
    output_file = Path("parsed_documents_markdown_easyocr.json")
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = sum(1 for doc in results["documents"] if doc["status"] == "success")
    failed = sum(1 for doc in results["documents"] if doc["status"] == "error")

    print(f"Total documents: {len(DOCUMENTS_TO_PROCESS)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {round(results['metadata']['total_time_seconds'], 2)}s")
    print(f"\nOutput saved to: {output_file.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()