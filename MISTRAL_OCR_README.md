# Mistral Document AI OCR Scripts

This repository contains scripts for extracting text from various document formats using Mistral's Document AI OCR API.

## Prerequisites

1. **Mistral API Key**: You need an active Mistral API account with Document AI access.
2. **Python 3.8+**: Make sure Python is installed on your system.
3. **Required packages**: Install the dependencies using pip.

## Installation

1. Install the required Python packages:
```bash
pip install mistralai python-dotenv
```

2. Set your Mistral API key as an environment variable:
```bash
export MISTRAL_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project directory:
```
MISTRAL_API_KEY=your-api-key-here
```

## Available Scripts

### 1. `mistral_document_extraction_with_server.py` (Recommended)

This is the **recommended approach**: each file is uploaded to Mistral storage, OCR runs on a temporary signed URL, and the upload is deleted afterwards. Excel workbooks are converted locally with pandas.

**Usage:**
```bash
# All results in mistral_parsed_documents.json
python mistral_document_extraction_with_server.py --input-dir /path/to/docs report.pdf scan.jpg

# Also one <name>_mistral.md per document plus mistral_parsing_summary.json
python mistral_document_extraction_with_server.py --markdown-dir mistral_output_markdown report.pdf
```

**Features:**
- Handles PDF, image and Excel files
- Markdown output format
- Individual markdown files for each document (`--markdown-dir`)
- JSON summary with processing statistics

### 2. `mistral_document_extraction.py` (Direct API - Limited Support)

This script attempts to use direct API calls but has limitations due to API constraints.

**Limitations:**
- The Mistral OCR API primarily works with URLs, not direct file uploads
- Images can be processed using data URIs (base64 encoded)
- Document files (DOCX, PDF, etc.) require URLs

**Usage:**
```bash
python mistral_document_extraction.py
```

### 3. `test_mistral_ocr_api.py` (Testing Script)

Use this to test your API connection and verify everything is set up correctly.

**Usage:**
```bash
# Test API connection only
python test_mistral_ocr_api.py

# Test with a specific file
python test_mistral_ocr_api.py /path/to/your/file.pdf
```

## Output Format

The scripts generate output matching the format of `parse_documents_with_docling.py`:

### JSON Output Structure
```json
{
  "metadata": {
    "start_timestamp": "2024-01-01T10:00:00",
    "source_directory": "/path/to/documents",
    "total_documents": 11,
    "processing_mode": "Mistral OCR API",
    "end_timestamp": "2024-01-01T10:05:00",
    "total_time_seconds": 300
  },
  "documents": [
    {
      "filename": "document.pdf",
      "file_path": "/full/path/to/document.pdf",
      "status": "success",
      "markdown": "# Extracted content here...",
      "markdown_length": 5000,
      "processing_time_seconds": 2.5,
      "timestamp": "2024-01-01T10:00:05"
    }
  ]
}
```

### Individual Markdown Files
Each successfully processed document also gets saved as an individual markdown file in the `mistral_extracted_texts/` directory.

## Supported File Formats

- **Documents**: PDF, DOCX, XLSX, XML
- **Images**: JPG, JPEG, PNG, BMP, GIF, TIFF

## API Limitations

- **File size**: Maximum 50 MB per file
- **Page limit**: Maximum 1,000 pages per document
- **Pricing**: $0.001 per page processed

## Troubleshooting

### Error: "MISTRAL_API_KEY not found"
Make sure you've set your API key:
```bash
export MISTRAL_API_KEY="your-api-key-here"
```

### Error: "File not found"
Ensure the documents exist in the input directory (`--input-dir`, default `$DOCS_DIR` or `./documents`):
```bash
python mistral_document_extraction.py --input-dir /path/to/docs report.pdf scan.jpg
```

### API Validation Errors
If you encounter validation errors, use the `mistral_document_extraction_with_server.py` script, which handles files via URLs and is more compatible with the API.

## Customization

To process different files, pass them on the command line (paths are relative to `--input-dir` or absolute), or modify the `DOCUMENTS_TO_PROCESS` list in the script:

```python
DOCUMENTS_TO_PROCESS = [
    "your_document1.pdf",
    "your_document2.docx",
    "your_image.jpg"
]
```

To change the source directory, pass `--input-dir /your/document/directory` or set `DOCS_DIR`.

## Best Practices

1. **Use the server-based script** (`mistral_document_extraction_with_server.py`) for best compatibility
2. **Test with a single file first** using the test script
3. **Monitor API usage** to avoid unexpected charges
4. **Process in batches** for large document sets
5. **Check file sizes** before processing (50 MB limit)

## Example Workflow

1. Set up your environment:
```bash
export MISTRAL_API_KEY="your-key"
pip install mistralai python-dotenv
```

2. Test the connection:
```bash
python test_mistral_ocr_api.py
```

3. Run the extraction:
```bash
python mistral_document_extraction_with_server.py
```

4. Check the results:
- Main JSON output: `mistral_parsed_documents.json`
- Individual markdowns: the `--markdown-dir` directory

## Notes

- The Mistral OCR API is optimized for accuracy and handles complex layouts well
- For best results with tables, the API returns them in markdown format
- Headers and footers can be extracted separately (when supported)
- The API automatically handles OCR for scanned documents and images

## Support

For issues with:
- **Mistral API**: Check [Mistral Documentation](https://docs.mistral.ai/capabilities/document_ai)
- **Script issues**: Review error messages and check the troubleshooting section above