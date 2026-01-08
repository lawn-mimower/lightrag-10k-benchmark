"""
Count ticker occurrences in finder_train.parquet
Helps determine which companies to index first
"""
import json
import re
from collections import Counter

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    try:
        import pyarrow.parquet as pq
        HAS_PYARROW = True
    except ImportError:
        print("Error: Neither pandas nor pyarrow available. Please install one.")
        exit(1)

FINDER_FILE = "finder_train.parquet"
OUTPUT_FILE = "ticker_frequency.json"

def extract_tickers_from_text(text: str) -> list:
    """Extract ticker symbols from text.

    Patterns:
    - (TICKER) in parentheses
    - - TICKER at end
    - , TICKER at end
    """
    tickers = []

    # Pattern 1: Ticker in parentheses
    paren_tickers = re.findall(r'\(([A-Z]{1,5})\)', text)
    tickers.extend(paren_tickers)

    # Pattern 2: Ticker at end with dash or comma
    end_tickers = re.findall(r'[,\-]\s*([A-Z]{1,5})\s*[,.]?\s*$', text)
    tickers.extend(end_tickers)

    # Remove duplicates
    return list(set(t for t in tickers if len(t) <= 5))


def count_ticker_occurrences():
    """Count how many questions mention each ticker."""

    print("="*60)
    print("COUNTING TICKER OCCURRENCES IN FINDER DATASET")
    print("="*60)

    # Load parquet file
    print(f"\nLoading {FINDER_FILE}...")

    if HAS_PANDAS:
        df = pd.read_parquet(FINDER_FILE)
        print(f"✓ Loaded {len(df)} questions using pandas")
    else:
        table = pq.read_table(FINDER_FILE)
        df = table.to_pandas()
        print(f"✓ Loaded {len(df)} questions using pyarrow")

    print(f"Columns: {list(df.columns)}\n")

    # Count ticker occurrences
    ticker_counter = Counter()
    questions_with_tickers = 0

    print("Extracting tickers from questions...")
    for idx, row in df.iterrows():
        text = row.get('text', '')
        tickers = extract_tickers_from_text(text)

        if tickers:
            questions_with_tickers += 1
            for ticker in tickers:
                ticker_counter[ticker] += 1

    print(f"✓ Processed {len(df)} questions")
    print(f"✓ Found {questions_with_tickers} questions with ticker mentions")
    print(f"✓ Found {len(ticker_counter)} unique tickers\n")

    # Display results
    print("="*60)
    print("TOP 50 TICKERS BY FREQUENCY")
    print("="*60)
    print(f"{'Rank':<6} {'Ticker':<8} {'Count':<8} {'% of Total':<10}")
    print("-"*60)

    total_questions = len(df)

    for i, (ticker, count) in enumerate(ticker_counter.most_common(50), 1):
        percentage = (count / total_questions) * 100
        print(f"{i:<6} {ticker:<8} {count:<8} {percentage:<10.2f}%")

    # Save full results
    ticker_data = {
        "total_questions": total_questions,
        "questions_with_tickers": questions_with_tickers,
        "unique_tickers": len(ticker_counter),
        "ticker_counts": dict(ticker_counter.most_common()),
        "top_10": [ticker for ticker, _ in ticker_counter.most_common(10)],
        "top_20": [ticker for ticker, _ in ticker_counter.most_common(20)]
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(ticker_data, f, indent=2)

    print(f"\n✓ Full results saved to {OUTPUT_FILE}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDED 10 COMPANIES TO INDEX FIRST")
    print("="*60)

    top_10 = [ticker for ticker, _ in ticker_counter.most_common(10)]
    top_10_count = sum(ticker_counter[t] for t in top_10)
    top_10_coverage = (top_10_count / total_questions) * 100

    print(f"Tickers: {', '.join(top_10)}")
    print(f"\nCoverage: {top_10_count} questions ({top_10_coverage:.1f}% of dataset)")

    print("\nUpdate your notebook Cell 6 with:")
    print(f"INDEXED_TICKERS = {top_10}")

    return ticker_counter


if __name__ == "__main__":
    ticker_counter = count_ticker_occurrences()
