#!/usr/bin/env python3
"""
Script to convert parsed JSON documents containing markdown to proper markdown files.
Enhanced to handle complex metadata and format markdown tables properly.
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys
import re


def format_metadata_table(metadata):
    """Format metadata as a clean markdown table."""
    md_lines = []
    md_lines.append("## Metadata\n")
    md_lines.append("| Field | Value |")
    md_lines.append("|-------|-------|")

    # Define field order for better presentation
    field_order = [
        'start_timestamp', 'end_timestamp', 'total_time_seconds',
        'source_directory', 'total_documents', 'processing_mode'
    ]

    # Add ordered fields first
    for field in field_order:
        if field in metadata:
            value = metadata[field]
            field_name = field.replace('_', ' ').title()

            # Format timestamps
            if 'timestamp' in field:
                try:
                    dt = datetime.fromisoformat(value)
                    value = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass
            # Format time duration
            elif field == 'total_time_seconds':
                value = f"{value:.2f} seconds"

            md_lines.append(f"| {field_name} | {value} |")

    # Add any remaining fields
    for field, value in metadata.items():
        if field not in field_order:
            field_name = field.replace('_', ' ').title()
            md_lines.append(f"| {field_name} | {value} |")

    md_lines.append("")
    return "\n".join(md_lines)


def clean_and_format_markdown_table(markdown_content):
    """Clean and properly format markdown table content."""
    if not markdown_content:
        return ""

    lines = markdown_content.strip().split('\n')
    formatted_lines = []
    max_widths = {}
    table_rows = []

    # First pass: collect all table rows and calculate max column widths
    for line in lines:
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|')]
            table_rows.append(cells)

            # Track max width for each column
            for i, cell in enumerate(cells):
                max_widths[i] = max(max_widths.get(i, 0), len(cell))

    # Second pass: format rows with consistent column widths
    if table_rows:
        is_separator_added = False

        for row_idx, cells in enumerate(table_rows):
            # Pad cells to max width for alignment
            padded_cells = []
            for i, cell in enumerate(cells):
                if cell and '-' in cell and all(c in '-' for c in cell.strip()):
                    # This is a separator row
                    padded_cells.append('-' * max(3, max_widths.get(i, 3)))
                else:
                    # Regular cell - pad for alignment
                    padded_cells.append(cell.ljust(max_widths.get(i, 0)))

            formatted_line = '| ' + ' | '.join(padded_cells) + ' |'
            formatted_lines.append(formatted_line)

            # Add separator after header if not present
            if row_idx == 0 and not is_separator_added:
                # Check if next row is not already a separator
                if row_idx + 1 < len(table_rows):
                    next_row = table_rows[row_idx + 1]
                    if not all('-' in str(cell).strip() for cell in next_row if cell.strip()):
                        separator_cells = ['-' * max(3, max_widths.get(i, 3)) for i in range(len(cells))]
                        separator_line = '| ' + ' | '.join(separator_cells) + ' |'
                        formatted_lines.append(separator_line)
                        is_separator_added = True

    return '\n'.join(formatted_lines)


def json_to_markdown(json_file_path, output_file_path=None):
    """
    Convert JSON containing markdown content to a proper markdown file.

    Args:
        json_file_path (str): Path to the input JSON file or '-' for stdin
        output_file_path (str): Path to the output markdown file (optional)
    """
    # Read the JSON data
    if json_file_path == '-':
        data = json.load(sys.stdin)
        default_output = 'output.md'
    else:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        input_path = Path(json_file_path)
        default_output = input_path.parent / f"{input_path.stem}_output.md"

    # Prepare the markdown content
    markdown_lines = []

    # Add title
    markdown_lines.append("# Document Processing Report")
    markdown_lines.append("")

    # Add generation timestamp
    markdown_lines.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    markdown_lines.append("")

    # Add metadata section with enhanced formatting
    if "metadata" in data:
        markdown_lines.append(format_metadata_table(data['metadata']))

    # Process each document
    if "documents" in data:
        markdown_lines.append("## Documents")
        markdown_lines.append("")

        # Summary
        total_docs = len(data["documents"])
        successful = sum(1 for doc in data["documents"] if doc.get("status") == "success")
        failed = total_docs - successful

        markdown_lines.append("### Summary")
        markdown_lines.append(f"- **Total Documents**: {total_docs}")
        markdown_lines.append(f"- **Successfully Processed**: {successful}")
        if failed > 0:
            markdown_lines.append(f"- **Failed**: {failed}")
        markdown_lines.append("")

        for idx, doc in enumerate(data["documents"], 1):
            # Document header with filename
            filename = doc.get("filename", doc.get("file", f"Document_{idx}"))
            markdown_lines.append(f"### {idx}. {filename}")
            markdown_lines.append("")

            # Document details in a clean format
            details = []
            if "file_path" in doc:
                details.append(f"**Path**: `{doc['file_path']}`")
            if "status" in doc:
                status_emoji = "✅" if doc['status'] == 'success' else "❌"
                details.append(f"**Status**: {status_emoji} {doc['status']}")
            if "error" in doc:
                details.append(f"**Error**: {doc['error']}")

            if details:
                markdown_lines.extend(details)
                markdown_lines.append("")

            # Add the markdown content with proper formatting
            if "markdown" in doc and doc["markdown"]:
                markdown_lines.append("#### Content")
                markdown_lines.append("")

                # Clean and format the markdown content
                formatted_content = clean_and_format_markdown_table(doc["markdown"])

                # If content is very long, add collapsible section
                if len(formatted_content) > 5000:
                    markdown_lines.append("<details>")
                    markdown_lines.append("<summary>Click to expand content</summary>")
                    markdown_lines.append("")
                    markdown_lines.append(formatted_content)
                    markdown_lines.append("")
                    markdown_lines.append("</details>")
                else:
                    markdown_lines.append(formatted_content)
            elif doc.get("status") == "success":
                markdown_lines.append("*No content extracted*")

            markdown_lines.append("")
            markdown_lines.append("---")
            markdown_lines.append("")

    # Join all lines
    final_markdown = "\n".join(markdown_lines)

    # Determine output file path
    if output_file_path is None:
        output_file_path = default_output

    # Write to file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(final_markdown)

    print(f"Successfully converted JSON to Markdown")
    print(f"Input: {json_file_path}")
    print(f"Output: {output_file_path}")
    return output_file_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert parsed JSON documents to markdown format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.json
  %(prog)s input.json -o report.md
  %(prog)s - < input.json  # Read from stdin
  cat input.json | %(prog)s -
  %(prog)s input.json --no-emoji --compact
        """
    )
    parser.add_argument(
        "input",
        help="Path to the input JSON file (use '-' for stdin)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output markdown file (optional)",
        default=None
    )
    parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="Don't use emoji in the output"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact formatting (no collapsible sections)"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the output to console"
    )

    args = parser.parse_args()

    # Check if input file exists (unless reading from stdin)
    if args.input != '-' and not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist")
        return 1

    try:
        output_path = json_to_markdown(args.input, args.output)

        # Print to console if requested
        if args.print:
            print("\n" + "="*60)
            print("MARKDOWN OUTPUT:")
            print("="*60 + "\n")
            with open(output_path, 'r', encoding='utf-8') as f:
                print(f.read())

        return 0
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())