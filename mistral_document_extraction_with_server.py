"""
Script to parse documents using Mistral OCR (via Signed URL) and Pandas.

By default all results go to mistral_parsed_documents.json in the working
directory. With --markdown-dir DIR every parsed document is also written to
DIR/<name>_mistral.md and the JSON summary goes to
DIR/mistral_parsing_summary.json.
"""

import os
import time
import pandas as pd
from pathlib import Path
from mistralai import Mistral
import dotenv
import json
dotenv.load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.environ.get("MISTRAL_API_KEY")
BASE_DIR = Path(os.getenv("DOCS_DIR", "./data"))
DOCUMENTS = [
    "sample_docs/sample_statement.xml"
]

def process_signed_url(client, doc_path):
    """
    Uploads file to Mistral, generates a temp signed URL, runs OCR,
    and then deletes the file from cloud storage.
    """
    print(f"  [API] Uploading {doc_path.name}...")
    uploaded_file = None
    try:
        # STEP 1: Upload to Mistral Storage
        # We must read bytes manually to avoid Pydantic validation errors
        with open(doc_path, "rb") as f:
            file_content = f.read()

        uploaded_file = client.files.upload(
            file={
                "file_name": doc_path.name,
                "content": file_content,
            },
            purpose="ocr"
        )

        # STEP 2: Get a Signed URL (The Critical Missing Step)
        # The OCR endpoint cannot read 'file_id' directly, but it CAN read this temporary URL.
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)

        print(f"  [API] Generated Signed URL. Processing...")

        # STEP 3: Call OCR with the Signed URL
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url
            },
            include_image_base64=False
        )

        # Extract Text from all pages
        if hasattr(response, 'pages'):
             md = "\n\n".join([p.markdown for p in response.pages])
             return {"status": "success", "markdown": md, "method": "mistral-signed-url"}
        return {"status": "error", "error": "No pages returned in response"}

    except Exception as e:
        return {"status": "error", "error": str(e)}

    finally:
        # STEP 4: Cleanup (Delete file from Mistral storage)
        if uploaded_file:
            try:
                client.files.delete(file_id=uploaded_file.id)
                print(f"  [API] Cleanup: Deleted temporary file {uploaded_file.id}")
            except Exception as cleanup_err:
                print(f"  [Warning] Cleanup failed: {cleanup_err}")

def excel_to_markdown(doc_path, doc_name, per_sheet_headings=False):
    """Convert every sheet of an Excel workbook to markdown tables."""
    dfs = pd.read_excel(doc_path, sheet_name=None)
    if not per_sheet_headings:
        return f"# {doc_name}\n" + "\n".join([df.to_markdown() for df in dfs.values()])
    md = f"# {doc_name}\n\n"
    for sheet_name, df in dfs.items():
        md += f"## Sheet: {sheet_name}\n{df.to_markdown(index=False)}\n\n"
    return md

def parse_cli_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Parse documents; defaults to the list above")
    parser.add_argument("files", nargs="*",
                        help="Documents to parse, relative to --input-dir or absolute")
    parser.add_argument("--input-dir", default=str(BASE_DIR),
                        help="Directory containing the documents (env DOCS_DIR, default ./data)")
    parser.add_argument("--markdown-dir", default=None,
                        help="Also write one <name>_mistral.md per document plus "
                             "mistral_parsing_summary.json into this directory "
                             "(Excel sheets become '## Sheet:' sections)")
    return parser.parse_args(argv)


def main(argv=None):
    global BASE_DIR, DOCUMENTS
    args = parse_cli_args(argv)
    BASE_DIR = Path(args.input_dir)
    if args.files:
        DOCUMENTS = args.files
    markdown_dir = Path(args.markdown_dir) if args.markdown_dir else None
    if not API_KEY:
        print("Error: MISTRAL_API_KEY not found.")
        return

    client = Mistral(api_key=API_KEY)
    if markdown_dir:
        markdown_dir.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("Mistral Official 'Signed URL' Pipeline")
    print("="*60)
    if markdown_dir:
        print(f"Source: {BASE_DIR}")
        print(f"Output: {markdown_dir}")
        print("="*60)

    # Store results here
    all_results = {
        "metadata": {
            "source_directory": str(BASE_DIR),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "documents": []
    }

    for doc_name in DOCUMENTS:
        doc_path = BASE_DIR / doc_name
        if not doc_path.exists():
            if markdown_dir:
                print(f"Skipping {doc_name}: File not found.")
            continue

        start = time.time()

        # Route Excel to Pandas
        if doc_path.suffix.lower() in ['.xlsx', '.xls']:
            try:
                print(f"  [Native] Parsing {doc_name} via Pandas...")
                md = excel_to_markdown(doc_path, doc_name, per_sheet_headings=bool(markdown_dir))
                result = {"status": "success", "markdown": md, "method": "pandas"}
            except Exception as e:
                result = {"status": "error", "error": str(e)}
        else:
            # Route others to Mistral
            result = process_signed_url(client, doc_path)
        result["filename"] = doc_name # Ensure filename is in the result dict

        duration = time.time() - start
        if markdown_dir:
            result["processing_time_seconds"] = round(duration, 2)
        else:
            result["processing_time"] = duration

        if result['status'] == 'success':
            if markdown_dir:
                # Save the individual Markdown file
                output_md_path = markdown_dir / f"{doc_path.stem}_mistral.md"
                with open(output_md_path, "w", encoding="utf-8") as f:
                    f.write(result["markdown"])
                result["saved_to"] = str(output_md_path)
                print(f"✓ {doc_name}: {len(result['markdown']):,} chars in {duration:.2f}s via {result['method']}")
                print(f"  -> Saved to: {output_md_path.name}")
            else:
                print(f"✓ {doc_name}: {len(result['markdown']):,} chars ({duration:.2f}s) via {result['method']}")
        else:
            print(f"✗ {doc_name}: {result.get('error')}")

        # Add to list
        all_results["documents"].append(result)

    # SAVE THE FILE
    output_file = markdown_dir / "mistral_parsing_summary.json" if markdown_dir else Path("mistral_parsed_documents.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    if markdown_dir:
        print("\n" + "="*60)
        print(f"Process Complete.")
        print(f"Markdown files and JSON summary saved to:\n{markdown_dir.absolute()}")
        print("="*60)
    else:
        print(f"\n✓ Output saved to: {output_file.absolute()}")

if __name__ == "__main__":
    main()
