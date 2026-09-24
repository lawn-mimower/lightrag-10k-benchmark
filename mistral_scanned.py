"""
Script to parse documents using Mistral OCR (via Signed URL) and Pandas.
Exports to individual Markdown files and a JSON summary.
"""

import os
import time
import pandas as pd
from pathlib import Path
from mistralai import Mistral
import dotenv
import json 

# Load environment variables
dotenv.load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.environ.get("MISTRAL_API_KEY")
BASE_DIR = Path(os.getenv("DOCS_DIR", "documents"))
OUTPUT_DIR = Path("mistral_output_markdown")

DOCUMENTS = [
    "sample_docs/sample_statement.pdf",
    # Add other files here
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
        # We read bytes manually to ensure clean upload
        with open(doc_path, "rb") as f:
            file_content = f.read()
            
        uploaded_file = client.files.upload(
            file={
                "file_name": doc_path.name,
                "content": file_content, 
            },
            purpose="ocr"
        )
        
        # STEP 2: Get a Signed URL
        # The OCR endpoint needs this temporary URL to access the uploaded file
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
        
        print(f"  [API] Generated Signed URL. Processing OCR...")
        
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
             return {
                 "status": "success", 
                 "markdown": md, 
                 "method": "mistral-signed-url"
             }
        else:
            return {
                "status": "error", 
                "error": "No pages returned in response"
            }
        
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

def parse_cli_args():
    import argparse
    parser = argparse.ArgumentParser(description="Parse documents; defaults to the list above")
    parser.add_argument("files", nargs="*",
                        help="Documents to parse, relative to --input-dir or absolute")
    parser.add_argument("--input-dir", default=str(BASE_DIR),
                        help="Directory containing the documents (env DOCS_DIR)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Where to write the outputs")
    return parser.parse_args()


def main():
    global BASE_DIR, DOCUMENTS, OUTPUT_DIR
    args = parse_cli_args()
    BASE_DIR = Path(args.input_dir)
    if args.files:
        DOCUMENTS = args.files
    OUTPUT_DIR = Path(args.output_dir)
    if not API_KEY:
        print("Error: MISTRAL_API_KEY not found in environment variables.")
        return

    # Initialize Client
    client = Mistral(api_key=API_KEY)
    
    # Create Output Directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("Mistral Official 'Signed URL' Pipeline")
    print("="*60)
    print(f"Source: {BASE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60)
    
    # Store metadata results here
    all_results = {
        "metadata": {
            "source_directory": str(BASE_DIR),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "documents": []
    }
    
    for doc_name in DOCUMENTS:
        doc_path = BASE_DIR / doc_name
        
        # Skip if file doesn't exist
        if not doc_path.exists(): 
            print(f"Skipping {doc_name}: File not found.")
            continue
        
        start = time.time()
        result = {}
        
        # --- ROUTING LOGIC ---
        # 1. Excel Files -> Native Pandas
        if doc_path.suffix.lower() in ['.xlsx', '.xls']:
            try:
                print(f"  [Native] Parsing {doc_name} via Pandas...")
                dfs = pd.read_excel(doc_path, sheet_name=None)
                # Convert all sheets to markdown
                md = f"# {doc_name}\n\n" 
                for sheet_name, df in dfs.items():
                    md += f"## Sheet: {sheet_name}\n{df.to_markdown(index=False)}\n\n"
                
                result = {
                    "status": "success", 
                    "markdown": md, 
                    "method": "pandas"
                }
            except Exception as e:
                result = {"status": "error", "error": str(e)}

        # 2. PDF/Image Files -> Mistral OCR
        else:
            result = process_signed_url(client, doc_path)
            
        # Add metadata to result
        duration = time.time() - start
        result["filename"] = doc_name
        result["processing_time_seconds"] = round(duration, 2)
        
        # --- SAVE OUTPUT FILES ---
        if result['status'] == 'success':
            # Save the individual Markdown file
            output_md_name = f"{doc_path.stem}_mistral.md"
            output_md_path = OUTPUT_DIR / output_md_name
            
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(result["markdown"])
            
            result["saved_to"] = str(output_md_path)
            print(f"✓ {doc_name}: {len(result['markdown']):,} chars in {duration:.2f}s via {result['method']}")
            print(f"  -> Saved to: {output_md_path.name}")
            
        else:
            print(f"✗ {doc_name}: {result.get('error')}")

        # Add to global list
        all_results["documents"].append(result)

    # SAVE THE JSON SUMMARY
    json_output_file = OUTPUT_DIR / "mistral_parsing_summary.json"
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*60)
    print(f"Process Complete.")
    print(f"Markdown files and JSON summary saved to:\n{OUTPUT_DIR.absolute()}")
    print("="*60)

if __name__ == "__main__":
    main()