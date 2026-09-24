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


def to_documents_by_ticker(parsed_docs: List[Dict]) -> Dict[str, Dict]:
    """
    Convert parse results into the {ticker: record} layout that the
    LightRAG notebooks load from parsed_10k_documents.json.
    """
    documents = {}
    for parsed in parsed_docs:
        ticker = parsed['ticker']
        documents[ticker] = {
            'ticker': ticker,
            'company_name': parsed['metadata'].get('company_name', ''),
            'period_end_date': parsed['metadata'].get('period_end_date', ''),
            'source_file': parsed.get('source_file', f"{ticker}.html"),
            'text': parsed['text'],
            'text_length': len(parsed['text']),
        }
    return documents


def batch_parse_html_files(directory: str, output_path: Optional[str] = None,
                           n_workers: int = 4, limit: Optional[int] = None,
                           tickers: Optional[List[str]] = None) -> List[Dict]:
    """
    Parse all HTML files in directory with multiprocessing

    Args:
        directory: Path to directory containing HTML files
        output_path: Optional path to save parsed docs as JSON ({ticker: record})
        n_workers: Number of parallel workers
        limit: Optional limit on number of files to parse (for testing)
        tickers: Optional list of tickers (file stems) to parse

    Returns:
        List of parsed document dicts
    """
    import os
    import json
    from multiprocessing import Pool
    from tqdm import tqdm

    # Get all HTML files
    html_files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.html')
    )

    if tickers:
        wanted = {t.upper() for t in tickers}
        html_files = [f for f in html_files if Path(f).stem.upper() in wanted]

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

    for parsed, html_file in zip(parsed_docs, html_files):
        parsed['source_file'] = os.path.basename(html_file)

    # Save if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(to_documents_by_ticker(parsed_docs), f, indent=2, ensure_ascii=False)
        print(f"Saved parsed documents to {output_path}")

    return parsed_docs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse SEC 10-K iXBRL HTML files into clean text"
    )
    parser.add_argument("path", help="HTML file or directory of <TICKER>.html files")
    parser.add_argument("output_json", nargs="?", default=None,
                        help="Where to write {ticker: record} JSON (directory mode)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Parse at most N files (directory mode)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Only parse these tickers, e.g. --tickers CTAS HON")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    path = args.path
    output = args.output_json

    if os.path.isfile(path):
        # Parse single file
        result = parse_ixbrl_html(path)
        print(f"Ticker: {result['ticker']}")
        print(f"Text length: {len(result['text'])} chars")
        print(f"Metadata: {result['metadata']}")
        print(f"\nFirst 500 chars:\n{result['text'][:500]}...")

    elif os.path.isdir(path):
        # Parse directory
        parsed_docs = batch_parse_html_files(path, output, n_workers=args.workers,
                                             limit=args.limit, tickers=args.tickers)
        print(f"\nParsed {len(parsed_docs)} documents")
        if parsed_docs:
            print(f"\nSample document:")
            print(f"Ticker: {parsed_docs[0]['ticker']}")
            print(f"Text length: {len(parsed_docs[0]['text'])} chars")
    else:
        print(f"Error: {path} is not a valid file or directory")
        raise SystemExit(1)
