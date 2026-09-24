# Document parsing scripts

These scripts convert PDFs, Office files and images to Markdown with Docling or Mistral OCR. They
are separate from the 10-K pipeline, which reads iXBRL HTML through `html_parser.py`. Each script
takes the documents to parse on the command line, relative to `--input-dir` or as absolute paths. The
input directory defaults to `$DOCS_DIR`.

```bash
pip install -r requirements_document_parsing.txt   # Docling, EasyOCR, Tesseract/Poppler bindings
pip install -r requirements_mistral_ocr.txt         # mistralai (MISTRAL_API_KEY in .env)
```

## Docling (local, CPU)

```bash
python parse_documents_with_docling.py --input-dir docs_in report.pdf statement.xlsx
python parse_documents_with_docling.py --ocr easyocr --input-dir docs_in scan.pdf
```

- `parse_documents_with_docling.py` runs Docling on the CPU with table-structure extraction. `--ocr`
  selects RapidOCR (the default) or EasyOCR (English). It writes `parsed_documents_markdown.json`, or
  `parsed_documents_markdown_easyocr.json` with EasyOCR. The default input directory is `documents`.
- `docling_md_out_monitored.py [--input-dir D] [--output-dir O] files…` writes one Markdown file per
  document plus `parsing_summary_with_monitoring.json`. It also samples CPU and memory with psutil.
  When Docling's output for a PDF is too sparse, it runs Tesseract OCR directly on the page images,
  which needs the `tesseract` and Poppler binaries.
- `fix_rtdetr_model.py` pre-downloads the RT-DETR model that Docling uses, and `fix_transformers.sh`
  upgrades transformers. Both address RT-DETRv2 loading errors in Docling.

## Mistral OCR (API)

```bash
python test_mistral_ocr_api.py [file.pdf]                        # connection check, optional test file
python mistral_document_extraction_with_server.py --input-dir docs_in report.pdf statement.xlsx
python mistral_document_extraction_with_server.py --markdown-dir mistral_md report.pdf
```

- `mistral_document_extraction_with_server.py` is the recommended script. It uploads each file to
  Mistral storage, runs `mistral-ocr-latest` on a temporary signed URL, then deletes the upload. It
  converts Excel workbooks locally with pandas, which needs `openpyxl` and `tabulate`; the Docling
  requirements install both. The default input directory is `./data`. By default
  all results go to `mistral_parsed_documents.json`. With `--markdown-dir DIR`, each document is also
  written to `DIR/<name>_mistral.md`, and the summary goes to `DIR/mistral_parsing_summary.json`.
- `mistral_document_extraction.py` sends images inline as base64 data URIs and other files through
  signed URLs. It writes `mistral_parsed_documents.json` and one Markdown file per document in
  `mistral_extracted_texts/`.

## Output format

All three scripts write `{"metadata": {...}, "documents": [...]}` with one record per document. Each
record has at least `filename`, `status` and `markdown` (or `error`). The Docling script and
`mistral_document_extraction.py` also record paths, lengths and timings:

```json
{
  "metadata": {"start_timestamp": "…", "source_directory": "…", "total_documents": 2,
               "processing_mode": "…", "end_timestamp": "…", "total_time_seconds": 28.2},
  "documents": [
    {"filename": "report.pdf", "file_path": "…", "status": "success", "markdown": "# …",
     "markdown_length": 1145, "processing_time_seconds": 2.5, "timestamp": "…"}
  ]
}
```

`json_to_markdown_converter.py input.json -o report.md` renders such a file as a single Markdown
report.
