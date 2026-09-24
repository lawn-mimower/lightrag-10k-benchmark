#!/usr/bin/env python3
"""
Transform retrieved_context from a single JSON string to a list of document chunk strings.
"""

import json
import re
from typing import List, Dict, Any


def extract_document_chunks(retrieved_context: str) -> List[str]:
    """
    Extract document chunk content from the retrieved_context string.

    The format is:
    Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

    ```json
    {"reference_id": "1", "content": "text content here..."}
    {"reference_id": "2", "content": "more text content..."}
    ...

    Args:
        retrieved_context: The raw retrieved context string containing JSON data

    Returns:
        List of document chunk text contents
    """
    chunks = []

    # Find the Document Chunks section
    doc_chunks_idx = retrieved_context.find("Document Chunks")
    if doc_chunks_idx == -1:
        return chunks

    # Get everything after "Document Chunks"
    doc_section = retrieved_context[doc_chunks_idx:]

    # Find the json code block
    json_start = doc_section.find("```json")
    if json_start != -1:
        # Skip past the ```json marker
        json_content = doc_section[json_start + 7:]

        # Find the end of the json block
        json_end = json_content.find("```")
        if json_end != -1:
            json_content = json_content[:json_end]

        # Split into lines and parse each JSON object
        lines = json_content.split('\n')

        for line in lines:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue

            try:
                obj = json.loads(line)
                if 'content' in obj and obj['content']:
                    chunks.append(obj['content'])
            except json.JSONDecodeError as e:
                # If we can't parse the full line, it might be truncated
                # Try to find JSON objects with regex
                json_pattern = r'\{"reference_id":\s*"[^"]+",\s*"content":\s*"((?:[^"\\]|\\.)*)"\}'
                match = re.search(json_pattern, line)
                if match:
                    try:
                        obj = json.loads(match.group(0))
                        if 'content' in obj and obj['content']:
                            chunks.append(obj['content'])
                    except:
                        pass

    return chunks


def transform_test_results(input_file: str, output_file: str):
    """
    Transform the test results file to convert retrieved_context to list format.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    print(f"Reading {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} results...")

    transformed_count = 0
    empty_count = 0

    for i, result in enumerate(data):
        if 'retrieved_context' in result:
            original_context = result['retrieved_context']

            # Extract document chunks
            chunks = extract_document_chunks(original_context)

            # Replace with list of chunks
            result['retrieved_contexts'] = chunks

            # Optionally keep the original for reference (commented out)
            # result['retrieved_context_original'] = original_context

            # Remove the old field
            del result['retrieved_context']

            if chunks:
                transformed_count += 1
            else:
                empty_count += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(data)} results...")

    print(f"\nTransformation complete:")
    print(f"  - Successfully transformed: {transformed_count}")
    print(f"  - Empty contexts: {empty_count}")
    print(f"  - Total: {len(data)}")

    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print("Done!")

    # Print a sample to verify
    if data and 'retrieved_contexts' in data[0]:
        print(f"\nSample result (first entry):")
        print(f"  Question ID: {data[0].get('question_id', 'N/A')}")
        print(f"  Number of chunks: {len(data[0]['retrieved_contexts'])}")
        if data[0]['retrieved_contexts']:
            print(f"  First chunk preview: {data[0]['retrieved_contexts'][0][:100]}...")


if __name__ == "__main__":
    import sys

    # Usage: python transform_contexts.py [input.json] [output.json]
    input_file = sys.argv[1] if len(sys.argv) > 1 else "test_results_priority_tickers.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "test_results_priority_tickers_transformed.json"

    transform_test_results(input_file, output_file)
