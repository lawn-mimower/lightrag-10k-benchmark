"""
Simple script to map FinDER questions to indexed tickers
Reads from finder_train.parquet
"""
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Trying pyarrow...")
    try:
        import pyarrow.parquet as pq
        HAS_PYARROW = True
    except ImportError:
        print("Error: Neither pandas nor pyarrow available. Please install one.")
        exit(1)

# Configuration
FINDER_FILE = "finder_train.parquet"
OUTPUT_MAPPING = "ticker_question_mapping.json"
OUTPUT_FILTERED = "finder_questions_indexed_subset.jsonl"

# Define which companies you indexed (update this after Cell 6 runs)
# These should be the first N companies you indexed before interrupting
INDEXED_TICKERS = ["USB", "UHS", "ETR","CVS", "A"
    # Example - UPDATE THIS LIST after you see which companies got indexed
    # "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA", "JPM", "V", "WMT"
]

def extract_tickers_from_text(text: str) -> list:
    """Extract ticker symbols from question text.

    Looks for patterns like:
    - (TICKER)
    - TICKER at end with dash/comma
    - Standalone 1-5 letter uppercase words
    """
    tickers = []

    # Pattern 1: Ticker in parentheses like "(FICO)" or "(T)"
    paren_tickers = re.findall(r'\(([A-Z]{1,5})\)', text)
    tickers.extend(paren_tickers)

    # Pattern 2: Ticker at end with punctuation like "- T" or ", AAPL"
    end_tickers = re.findall(r'[,\-]\s*([A-Z]{1,5})\s*[,.]?\s*$', text)
    tickers.extend(end_tickers)

    # Pattern 3: Ticker with possessive like "Apple's" -> try to match known patterns
    # For now, just rely on explicit ticker mentions

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for t in tickers:
        if t not in seen and len(t) <= 5:  # Valid tickers are 1-5 chars
            seen.add(t)
            unique.append(t)

    return unique


def map_questions_to_tickers():
    """Create mapping of indexed tickers to question IDs."""

    if not INDEXED_TICKERS:
        print("⚠ WARNING: INDEXED_TICKERS list is empty!")
        print("Please update the INDEXED_TICKERS list in the script with the companies you indexed.\n")
        return

    print("="*60)
    print("MAPPING FINDER QUESTIONS TO INDEXED TICKERS")
    print("="*60)
    print(f"Indexed tickers: {', '.join(INDEXED_TICKERS)}\n")

    # Load FinDER questions from parquet
    print(f"Loading {FINDER_FILE}...")

    if HAS_PANDAS:
        df = pd.read_parquet(FINDER_FILE)
        print(f"✓ Loaded {len(df)} questions using pandas")
    else:
        table = pq.read_table(FINDER_FILE)
        df = table.to_pandas()
        print(f"✓ Loaded {len(df)} questions using pyarrow")

    print(f"Columns: {list(df.columns)}\n")

    # Convert to list of dicts for processing
    questions = df.to_dict('records')

    # Build mapping
    ticker_to_questions = defaultdict(list)
    question_id_to_tickers = {}

    matched_questions = []

    print("Extracting tickers from questions...")
    for q in questions:
        # Use 'text' attribute for question text (parquet column)
        question_text = q.get('text', '')

        # Use '_id' attribute for question ID (parquet column)
        question_id = q.get('_id', '')

        # Extract tickers from the text
        mentioned_tickers = extract_tickers_from_text(question_text)

        # Filter to only indexed tickers
        indexed_mentioned = [t for t in mentioned_tickers if t in INDEXED_TICKERS]

        if indexed_mentioned:
            # Add to mapping
            for ticker in indexed_mentioned:
                ticker_to_questions[ticker].append(question_id)

            question_id_to_tickers[question_id] = indexed_mentioned
            matched_questions.append(q)

    print(f"✓ Found {len(matched_questions)} questions mentioning indexed tickers\n")

    # Print statistics
    print("="*60)
    print("TICKER COVERAGE")
    print("="*60)

    for ticker in sorted(INDEXED_TICKERS):
        count = len(ticker_to_questions[ticker])
        if count > 0:
            print(f"{ticker:6s} {count:3d} questions")
        else:
            print(f"{ticker:6s}   0 questions (⚠ no coverage)")

    total_matched = len(matched_questions)
    coverage = (total_matched / len(questions)) * 100

    print(f"\nTotal: {total_matched} / {len(questions)} questions ({coverage:.1f}% coverage)")

    # Save mapping
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print("="*60)

    # Save ticker -> question_ids mapping
    mapping_output = {
        "indexed_tickers": INDEXED_TICKERS,
        "total_questions": len(questions),
        "matched_questions": total_matched,
        "coverage_percent": coverage,
        "ticker_to_question_ids": dict(ticker_to_questions),
        "question_id_to_tickers": question_id_to_tickers
    }

    with open(OUTPUT_MAPPING, 'w', encoding='utf-8') as f:
        json.dump(mapping_output, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved mapping to {OUTPUT_MAPPING}")

    # Save filtered questions (only those mentioning indexed tickers)
    with open(OUTPUT_FILTERED, 'w', encoding='utf-8') as f:
        for q in matched_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    print(f"✓ Saved filtered questions to {OUTPUT_FILTERED}")

    print(f"\n{'='*60}")
    print("READY FOR BENCHMARKING!")
    print("="*60)
    print(f"Use {OUTPUT_FILTERED} to evaluate your indexed subset.")
    print(f"Total answerable questions: {total_matched}")

    return mapping_output


def auto_detect_indexed_tickers():
    """Auto-detect which companies were indexed by checking workspace."""
    workspace_path = Path("./lightrag_10k_workspace/kv_store_doc_status.json")

    if not workspace_path.exists():
        print("⚠ No workspace found. Cannot auto-detect indexed tickers.")
        return []

    with open(workspace_path, 'r') as f:
        doc_status = json.load(f)

    # Doc status keys are doc IDs, not tickers
    # We'd need to correlate with our chunks to get tickers
    # For now, just return count
    print(f"Found {len(doc_status)} indexed documents in workspace")
    return []


if __name__ == "__main__":
    # Try to auto-detect
    if not INDEXED_TICKERS:
        print("Attempting to auto-detect indexed tickers...")
        detected = auto_detect_indexed_tickers()
        if detected:
            INDEXED_TICKERS = detected
        else:
            print("\n" + "="*60)
            print("⚠ PLEASE UPDATE INDEXED_TICKERS IN SCRIPT")
            print("="*60)
            print("After Cell 6 runs and indexes N companies, update the")
            print("INDEXED_TICKERS list at the top of this script with")
            print("the tickers that were actually indexed.")
            print("\nExample:")
            print('INDEXED_TICKERS = ["AAPL", "MSFT", "GOOGL", "META", ...]')
            print("="*60)
            exit(1)

    mapping_output = map_questions_to_tickers()
