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
BASE_DIR = Path("./data")
DOCUMENTS = [
    "sample_docs/sample_statement.xml",
]

def process_signed_url(client, doc_path):
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
        
        # Extract Text
        if hasattr(response, 'pages'):
             md = "\n\n".join([p.markdown for p in response.pages])
             return {"status": "success", "markdown": md, "method": "mistral-signed-url"}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}
        
    finally:
        # STEP 4: Cleanup
        if uploaded_file:
            try:
                client.files.delete(file_id=uploaded_file.id)
            except:
                pass

def main():
    if not API_KEY:
        print("Error: MISTRAL_API_KEY not found.")
        return

    client = Mistral(api_key=API_KEY)
    print("="*60)
    print("Mistral Official 'Signed URL' Pipeline")
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
        if not doc_path.exists(): continue
        
        start = time.time()
        
        # Route Excel to Pandas
        if doc_path.suffix.lower() in ['.xlsx', '.xls']:
            try:
                print(f"  [Native] Parsing {doc_name} via Pandas...")
                dfs = pd.read_excel(doc_path, sheet_name=None)
                md = f"# {doc_name}\n" + "\n".join([df.to_markdown() for df in dfs.values()])
                result = {"filename": doc_name, "status": "success", "markdown": md, "method": "pandas"}
            except Exception as e:
                result = {"filename": doc_name, "status": "error", "error": str(e)}
        else:
            # Route others to Mistral
            result = process_signed_url(client, doc_path)
            result["filename"] = doc_name # Ensure filename is in the result dict
            
        duration = time.time() - start
        result["processing_time"] = duration
        
        # Add to list
        all_results["documents"].append(result)
        
        if result['status'] == 'success':
            print(f"✓ {doc_name}: {len(result['markdown']):,} chars ({duration:.2f}s) via {result['method']}")
        else:
            print(f"✗ {doc_name}: {result.get('error')}")

    # SAVE THE FILE
    output_file = Path("mistral_parsed_documents.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"\n✓ Output saved to: {output_file.absolute()}")

if __name__ == "__main__":
    main()