"""
Script to parse specific documents using Docling and export to Markdown files + JSON metadata.
CPU-only mode with timestamps and resource monitoring using psutil.
"""

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorDevice,
    AcceleratorOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
import json
import csv
import re
from pathlib import Path
from datetime import datetime
import time
from dotenv import load_dotenv
import psutil
import threading
from collections import defaultdict
import os
import tempfile

# Direct OCR fallback for ID documents
import pytesseract
from PIL import Image
import pdf2image

# Base directory where documents are located
BASE_DIR = Path(os.getenv("DOCS_DIR", "documents"))
# Directory to save the output Markdown files
OUTPUT_DIR = Path("output_markdown1")
# Directory to save monitoring data
MONITORING_DIR = Path("monitoring_output")

load_dotenv()

# Specific documents to process
DOCUMENTS_TO_PROCESS = [
    "sample_docs/document_1.pdf",
    "sample_docs/document_2.pdf"
    # Add other files here
    # "InstanceDocument-EXAMPLEENGINEERINGPRIVATELIMITED_Standalone.xml",
]

class ResourceMonitor:
    """Class to monitor system resources using psutil."""

    def __init__(self, interval=1.0):
        """
        Initialize resource monitor.

        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = defaultdict(list)
        self.process = psutil.Process()
        self.start_time = None

    def get_system_metrics(self):
        """Get current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        cpu_freq = psutil.cpu_freq()

        # Memory metrics
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()

        # Process-specific metrics
        process_info = {
            'cpu_percent': self.process.cpu_percent(),
            'memory_info': self.process.memory_info()._asdict(),
            'memory_percent': self.process.memory_percent(),
            'num_threads': self.process.num_threads(),
        }

        # Network I/O metrics (optional)
        net_io = psutil.net_io_counters()

        return {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': time.time() - self.start_time if self.start_time else 0,
            'cpu': {
                'percent_total': cpu_percent,
                'percent_per_core': cpu_per_core,
                'frequency_current': cpu_freq.current if cpu_freq else None,
                'frequency_min': cpu_freq.min if cpu_freq else None,
                'frequency_max': cpu_freq.max if cpu_freq else None,
            },
            'memory': {
                'total': mem.total,
                'available': mem.available,
                'percent': mem.percent,
                'used': mem.used,
                'free': mem.free,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_percent': swap.percent,
            },
            'disk_io': {
                'read_count': disk_io.read_count if disk_io else None,
                'write_count': disk_io.write_count if disk_io else None,
                'read_bytes': disk_io.read_bytes if disk_io else None,
                'write_bytes': disk_io.write_bytes if disk_io else None,
                'read_time': disk_io.read_time if disk_io else None,
                'write_time': disk_io.write_time if disk_io else None,
            },
            'network_io': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
            },
            'process': process_info,
        }

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                metrics = self.get_system_metrics()
                for key, value in self._flatten_dict(metrics).items():
                    self.metrics[key].append(value)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
            time.sleep(self.interval)

    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary for easier storage."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def start_monitoring(self):
        """Start background monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.start_time = time.time()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("Resource monitoring started...")

    def stop_monitoring(self):
        """Stop background monitoring."""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            print("Resource monitoring stopped.")

    def get_summary_stats(self):
        """Get summary statistics of collected metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values and all(isinstance(v, (int, float)) for v in values if v is not None):
                numeric_values = [v for v in values if v is not None]
                if numeric_values:
                    summary[key] = {
                        'min': min(numeric_values),
                        'max': max(numeric_values),
                        'avg': sum(numeric_values) / len(numeric_values),
                        'samples': len(numeric_values)
                    }
        return summary

    def save_metrics(self, output_dir):
        """Save collected metrics to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save raw metrics as JSON
        json_file = output_dir / "resource_metrics_raw.json"
        with open(json_file, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        print(f"  ✓ Saved raw metrics: {json_file}")

        # Save summary statistics as JSON
        summary_file = output_dir / "resource_metrics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.get_summary_stats(), f, indent=2)
        print(f"  ✓ Saved summary metrics: {summary_file}")

        # Save metrics as CSV for easier analysis
        csv_file = output_dir / "resource_metrics.csv"
        if self.metrics:
            # Get all keys (column names)
            all_keys = sorted(self.metrics.keys())

            # Determine the maximum number of samples
            max_samples = max(len(values) for values in self.metrics.values())

            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['sample_index'] + all_keys)

                # Write data rows
                for i in range(max_samples):
                    row = [i]
                    for key in all_keys:
                        values = self.metrics[key]
                        if i < len(values):
                            row.append(values[i])
                        else:
                            row.append(None)
                    writer.writerow(row)

            print(f"  ✓ Saved metrics CSV: {csv_file}")

        return {
            'json_raw': str(json_file),
            'json_summary': str(summary_file),
            'csv': str(csv_file)
        }


def initialize_converter():
    """Initialize Docling DocumentConverter with RapidOCR (primary engine for ID documents)."""
    print("Initializing Docling DocumentConverter...")

    try:
        print("  Creating PdfPipelineOptions...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.images_scale = 2.0

        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True

        print("  Configuring table structure extraction (ACCURATE mode)...")
        pipeline_options.table_structure_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True,
        )

        print("  Using RapidOcrOptions (primary engine)...")
        ocr_options = RapidOcrOptions(
            lang=["en"],
            force_full_page_ocr=True,
        )
        pipeline_options.ocr_options = ocr_options

        print("  Setting AcceleratorOptions...")
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU,
        )

        print("  Creating DocumentConverter...")
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        print("  Converter initialized successfully!")
        return converter

    except Exception as e:
        print(f"  ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        return None


def correct_orientation(image):
    """
    Detect and correct image orientation using Tesseract OSD.
    Returns the corrected image and the detected rotation angle.
    """
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotation = osd.get("rotate", 0)
        if rotation != 0:
            image = image.rotate(-rotation, expand=True)
            print(f"      Corrected rotation: {rotation}°")
        return image, rotation
    except pytesseract.TesseractError:
        # OSD failed - try all 4 rotations and pick the one with most text
        best_text = ""
        best_rotation = 0
        best_img = image
        for angle in [0, 90, 180, 270]:
            rotated = image.rotate(-angle, expand=True) if angle != 0 else image
            try:
                text = pytesseract.image_to_string(rotated, lang="eng", config="--psm 6")
                # Score by number of alphanumeric characters
                score = sum(1 for c in text if c.isalnum())
                if score > len(best_text):
                    best_text = text
                    best_rotation = angle
                    best_img = rotated
            except Exception:
                continue
        if best_rotation != 0:
            print(f"      Brute-force rotation correction: {best_rotation}°")
        return best_img, best_rotation


def direct_ocr_pdf(pdf_path, dpi=300):
    """
    Direct OCR fallback: convert PDF pages to images, correct orientation,
    and run Tesseract directly with optimal settings for ID documents.
    Returns extracted text as markdown.
    """
    print(f"    Running direct OCR fallback (DPI={dpi})...")
    pages = pdf2image.convert_from_path(str(pdf_path), dpi=dpi)
    all_text = []

    for page_num, page_img in enumerate(pages, 1):
        print(f"      Page {page_num}: {page_img.size[0]}x{page_img.size[1]}px")

        # Correct orientation
        page_img, rotation = correct_orientation(page_img)

        # Run Tesseract with PSM 6 (assume uniform text block) - avoids OSD failures
        # Also try PSM 3 (fully automatic) and pick the one with more text
        results = {}
        for psm in [6, 3, 4]:
            try:
                text = pytesseract.image_to_string(
                    page_img, lang="eng",
                    config=f"--psm {psm} --dpi {dpi}"
                )
                results[psm] = text.strip()
            except Exception:
                results[psm] = ""

        # Pick the PSM that produced the most alphanumeric characters
        best_psm = max(results, key=lambda p: sum(1 for c in results[p] if c.isalnum()))
        best_text = results[best_psm]

        if best_text:
            print(f"      Page {page_num}: {len(best_text)} chars (PSM {best_psm})")
            all_text.append(f"## Page {page_num}\n\n{best_text}")
        else:
            print(f"      Page {page_num}: No text extracted")

    return "\n\n---\n\n".join(all_text)


def extract_all_text_elements(doc_result):
    """
    Extract all text elements from the converted document, including those
    inside picture/figure regions that export_to_markdown() skips.
    """
    doc = doc_result.document
    lines = []

    for item, _level in doc.iterate_items():
        item_type = type(item).__name__

        if hasattr(item, 'text') and item.text and item.text.strip():
            text = item.text.strip()
            label = getattr(item, 'label', item_type) if hasattr(item, 'label') else item_type
            label_val = label.value if hasattr(label, 'value') else str(label)

            if label_val in ('section_header', 'title'):
                lines.append(f"## {text}")
            elif label_val == 'page_header':
                lines.append(f"**{text}**")
            elif label_val == 'caption':
                lines.append(f"*{text}*")
            else:
                lines.append(text)

        if hasattr(item, 'export_to_markdown'):
            try:
                table_md = item.export_to_markdown()
                if table_md and table_md.strip() and table_md.strip() not in [l.strip() for l in lines]:
                    lines.append(table_md)
            except Exception:
                pass

    return "\n\n".join(lines)


def parse_documents(monitor=None):
    """Parse all specified documents and return results dict with monitoring."""

    start_time = datetime.now()
    converter = initialize_converter()

    if converter is None:
        print("ERROR: Failed to initialize converter. Exiting.")
        return None

    results = {
        "metadata": {
            "start_timestamp": start_time.isoformat(),
            "source_directory": str(BASE_DIR),
            "total_documents": len(DOCUMENTS_TO_PROCESS),
            "processing_mode": "CPU-only",
            "monitoring_enabled": monitor is not None
        },
        "documents": [],
        "resource_monitoring": {}
    }

    for i, doc_name in enumerate(DOCUMENTS_TO_PROCESS, 1):
        doc_path = BASE_DIR / doc_name
        doc_start = time.time()

        # Take a snapshot of resources before processing
        if monitor:
            pre_metrics = monitor.get_system_metrics()

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

            # Pass 1: Docling with RapidOCR
            result = converter.convert(str(doc_path))
            docling_md = result.document.export_to_markdown()

            # Pass 1b: Extract all text elements (catches text in picture regions)
            enhanced_md = extract_all_text_elements(result)

            # Pick the richer Docling output
            if len(enhanced_md.strip()) > len(docling_md.strip()):
                docling_md = enhanced_md

            # Count meaningful (alphanumeric) characters
            docling_alnum = sum(1 for c in docling_md if c.isalnum())

            # Pass 2: Direct Tesseract fallback if Docling output is too sparse
            MIN_CHARS_THRESHOLD = 50
            if docling_alnum < MIN_CHARS_THRESHOLD:
                print(f"    Docling extracted only {docling_alnum} alphanumeric chars, "
                      f"running direct OCR fallback...")
                direct_md = direct_ocr_pdf(doc_path)
                direct_alnum = sum(1 for c in direct_md if c.isalnum())

                if direct_alnum > docling_alnum:
                    print(f"    Direct OCR: {direct_alnum} chars vs Docling: {docling_alnum} chars")
                    markdown_content = direct_md
                else:
                    markdown_content = docling_md
            else:
                markdown_content = docling_md

            processing_time = time.time() - doc_start

            # Take a snapshot of resources after processing
            if monitor:
                post_metrics = monitor.get_system_metrics()
                doc_metrics = {
                    'pre_processing': pre_metrics,
                    'post_processing': post_metrics,
                    'processing_time': processing_time
                }
            else:
                doc_metrics = None

            # Store result
            results["documents"].append({
                "filename": doc_name,
                "file_path": str(doc_path),
                "status": "success",
                "markdown": markdown_content,
                "markdown_length": len(markdown_content),
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": datetime.now().isoformat(),
                "resource_metrics": doc_metrics
            })

            print(f"    ✓ Success: {len(markdown_content):,} chars in {round(processing_time, 2)}s")

            if monitor:
                print(f"    CPU: {post_metrics['cpu']['percent_total']:.1f}%, "
                      f"Memory: {post_metrics['memory']['percent']:.1f}%")

        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
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
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Where to write the outputs")
    return parser.parse_args()


def main():
    """Main execution function with resource monitoring."""
    global BASE_DIR, DOCUMENTS_TO_PROCESS, OUTPUT_DIR
    args = parse_cli_args()
    BASE_DIR = Path(args.input_dir)
    if args.files:
        DOCUMENTS_TO_PROCESS = args.files
    OUTPUT_DIR = Path(args.output_dir)
    print("=" * 70)
    print("Docling Document Parser with Resource Monitoring")
    print("=" * 70)
    print(f"Source directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Monitoring directory: {MONITORING_DIR}")
    print(f"Documents to process: {len(DOCUMENTS_TO_PROCESS)}")
    print("=" * 70)
    print()

    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    MONITORING_DIR.mkdir(exist_ok=True, parents=True)

    # Create timestamped subdirectory for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_monitoring_dir = MONITORING_DIR / f"run_{run_timestamp}"
    run_monitoring_dir.mkdir(exist_ok=True, parents=True)

    # Initialize resource monitor
    monitor = ResourceMonitor(interval=0.5)  # Sample every 0.5 seconds

    # Start monitoring
    monitor.start_monitoring()

    # Parse all documents
    print("Starting document parsing with resource monitoring...")
    results = parse_documents(monitor)

    # Stop monitoring
    monitor.stop_monitoring()

    if results is None:
        print("ERROR: Document parsing failed.")
        return

    # Process results and write Markdown files
    print("\nWriting output files...")

    for doc in results["documents"]:
        if doc["status"] == "success":
            # Generate output filename (original name + .md)
            original_path = Path(doc["filename"])
            output_md_name = f"{original_path.stem}_converted.md"
            output_md_path = OUTPUT_DIR / output_md_name

            # Write the markdown content to file
            try:
                with open(output_md_path, 'w', encoding='utf-8') as f:
                    f.write(doc["markdown"])
                print(f"  ✓ Saved Markdown: {output_md_path}")

                # Update the result dict to point to the saved file
                doc["saved_to"] = str(output_md_path)
                # Remove the full markdown content from JSON to keep it cleaner
                del doc["markdown"]

            except Exception as e:
                print(f"  ✗ Failed to write markdown file for {doc['filename']}: {e}")

    # Save monitoring data
    print("\nSaving monitoring data...")
    monitoring_files = monitor.save_metrics(run_monitoring_dir)
    results["resource_monitoring"]["files"] = monitoring_files
    results["resource_monitoring"]["summary"] = monitor.get_summary_stats()

    # Save enhanced summary JSON with monitoring info
    json_output_file = OUTPUT_DIR / "parsing_summary_with_monitoring.json"
    print(f"\nSaving metadata summary to {json_output_file}...")
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also save a copy in the monitoring directory
    monitoring_json = run_monitoring_dir / "parsing_results.json"
    with open(monitoring_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate and save a monitoring report
    report_file = run_monitoring_dir / "monitoring_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RESOURCE MONITORING REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Run timestamp: {run_timestamp}\n")
        f.write(f"Total processing time: {results['metadata']['total_time_seconds']:.2f} seconds\n")
        f.write(f"Documents processed: {len(DOCUMENTS_TO_PROCESS)}\n")
        f.write("\n")

        summary = monitor.get_summary_stats()
        if summary:
            f.write("Resource Usage Summary:\n")
            f.write("-" * 40 + "\n")

            # CPU metrics
            if 'cpu_percent_total' in summary:
                stats = summary['cpu_percent_total']
                f.write(f"CPU Usage:\n")
                f.write(f"  Min: {stats['min']:.1f}%\n")
                f.write(f"  Max: {stats['max']:.1f}%\n")
                f.write(f"  Avg: {stats['avg']:.1f}%\n")
                f.write("\n")

            # Memory metrics
            if 'memory_percent' in summary:
                stats = summary['memory_percent']
                f.write(f"Memory Usage:\n")
                f.write(f"  Min: {stats['min']:.1f}%\n")
                f.write(f"  Max: {stats['max']:.1f}%\n")
                f.write(f"  Avg: {stats['avg']:.1f}%\n")
                f.write("\n")

            # Process-specific metrics
            if 'process_memory_percent' in summary:
                stats = summary['process_memory_percent']
                f.write(f"Process Memory:\n")
                f.write(f"  Min: {stats['min']:.2f}%\n")
                f.write(f"  Max: {stats['max']:.2f}%\n")
                f.write(f"  Avg: {stats['avg']:.2f}%\n")
                f.write("\n")

        f.write("=" * 70 + "\n")

    print(f"  ✓ Saved monitoring report: {report_file}")

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
    print(f"Check the '{OUTPUT_DIR}' folder for your .md files.")
    print(f"Check the '{run_monitoring_dir}' folder for monitoring data.")

    # Print resource usage summary
    summary = monitor.get_summary_stats()
    if summary and 'cpu_percent_total' in summary:
        print()
        print("Resource Usage Summary:")
        print(f"  CPU: {summary['cpu_percent_total']['avg']:.1f}% avg "
              f"({summary['cpu_percent_total']['max']:.1f}% peak)")
        if 'memory_percent' in summary:
            print(f"  Memory: {summary['memory_percent']['avg']:.1f}% avg "
                  f"({summary['memory_percent']['max']:.1f}% peak)")

    print("=" * 70)


if __name__ == "__main__":
    print("Script started...")
    try:
        # Print system information
        print(f"System: {psutil.cpu_count()} CPUs, "
              f"{psutil.virtual_memory().total / (1024**3):.1f} GB RAM")
        print()

        main()
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
    print("Script finished.")