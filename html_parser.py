"""
HTML Parser - Parse iXBRL HTML files from SEC 10-K filings
"""

from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
import os
from pathlib import Path


def parse_ixbrl_html(file_path: str) -> Dict:
    """
    Parse single iXBRL HTML file and extract clean text

    Args:
        file_path: Path to HTML file

    Returns:
        Dict with keys:
            - ticker: Stock ticker symbol (from filename)
            - text: Extracted visible text
            - metadata: Additional metadata (filing date, etc.)
    """
    # Extract ticker from filename
    ticker = Path(file_path).stem

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            html_content = f.read()

    # Use 'html.parser' (built-in), NOT 'lxml-xml' (XML mode)
    # The XML mode extracts XBRL namespace content as text, which is wrong
    soup = BeautifulSoup(html_content, 'html.parser')

    # Try to extract metadata BEFORE removing XBRL elements
    metadata = {}
    
    # Look for document period end date (in hidden XBRL section)
    period_end = soup.find(attrs={'name': 'dei:DocumentPeriodEndDate'})
    if period_end:
        metadata['period_end_date'] = period_end.get_text(strip=True)
    
    # Look for entity name
    entity_name = soup.find(attrs={'name': 'dei:EntityRegistrantName'})
    if entity_name:
        metadata['company_name'] = entity_name.get_text(strip=True)

    # Remove ALL hidden iXBRL elements - these contain structured data, not readable text
    # This includes ix:hidden, ix:header blocks which have XBRL metadata
    for tag_name in ['ix:hidden', 'ix:header', 'ix:references', 'ix:resources']:
        for hidden in soup.find_all(tag_name):
            hidden.decompose()
    
    # Also remove elements with display:none style (hidden content)
    for hidden in soup.find_all(style=re.compile(r'display\s*:\s*none', re.IGNORECASE)):
        hidden.decompose()

    # Remove script and style elements
    for script in soup(['script', 'style', 'head', 'meta', 'link']):
        script.decompose()
    
    # Remove any remaining ix: prefixed elements (XBRL inline elements)
    # These may contain contextRef attributes with non-human-readable content
    for elem in soup.find_all(re.compile(r'^ix:')):
        # Keep the text content but remove the tag structure
        elem.unwrap() if elem.string else elem.decompose()

    # Extract visible text from the body
    body = soup.find('body')
    if body:
        text = body.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Additional cleanup: remove any XBRL-like patterns that may have leaked through
    # These are patterns like "0001326801 us-gaap:CommonClassAMember 2024-01-01"
    text = re.sub(r'\b\d{10}\s+[\w\-:]+Member\b', '', text)
    text = re.sub(r'\b\d{10}\s+[\w\-:]+\s+\d{4}-\d{2}-\d{2}\b', '', text)
    text = re.sub(r'\s+', ' ', text)  # Clean up extra spaces from removals
    text = text.strip()

    return {
        'ticker': ticker,
        'text': text,
        'metadata': metadata
    }


def batch_parse_html_files(directory: str, output_path: Optional[str] = None,
                           n_workers: int = 4, limit: Optional[int] = None) -> List[Dict]:
    """
    Parse all HTML files in directory with multiprocessing

    Args:
        directory: Path to directory containing HTML files
        output_path: Optional path to save parsed docs as JSON
        n_workers: Number of parallel workers
        limit: Optional limit on number of files to parse (for testing)

    Returns:
        List of parsed document dicts
    """
    import os
    import json
    from multiprocessing import Pool
    from tqdm import tqdm

    # Get all HTML files
    html_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.html')
    ]

    if limit:
        html_files = html_files[:limit]

    print(f"Found {len(html_files)} HTML files to parse")

    # Parse in parallel
    with Pool(n_workers) as pool:
        parsed_docs = list(tqdm(
            pool.imap(parse_ixbrl_html, html_files),
            total=len(html_files),
            desc="Parsing HTML files"
        ))

    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(parsed_docs, f, indent=2)
        print(f"Saved parsed documents to {output_path}")

    return parsed_docs


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python html_parser.py <html_file_or_directory> [output_json]")
        sys.exit(1)

    path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    if os.path.isfile(path):
        # Parse single file
        result = parse_ixbrl_html(path)
        print(f"Ticker: {result['ticker']}")
        print(f"Text length: {len(result['text'])} chars")
        print(f"Metadata: {result['metadata']}")
        print(f"\nFirst 500 chars:\n{result['text'][:500]}...")

    elif os.path.isdir(path):
        # Parse directory
        parsed_docs = batch_parse_html_files(path, output, limit=5)  # Test with 5 files
        print(f"\nParsed {len(parsed_docs)} documents")
        if parsed_docs:
            print(f"\nSample document:")
            print(f"Ticker: {parsed_docs[0]['ticker']}")
            print(f"Text length: {len(parsed_docs[0]['text'])} chars")
    else:
        print(f"Error: {path} is not a valid file or directory")
