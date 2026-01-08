"""
Estimate token count for embedding all 10-K documents
"""
import os
from html_parser import parse_ixbrl_html
from tqdm import tqdm
import json

# Try to use tiktoken for accurate counting
try:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4/embeddings tokenizer
    def count_tokens(text):
        return len(encoding.encode(text))
    print("Using tiktoken for accurate token counting")
except ImportError:
    # Fallback: rough estimate (1 token ≈ 4 chars)
    def count_tokens(text):
        return len(text) // 4
    print("Warning: tiktoken not installed. Using rough estimate (chars/4)")

HTML_DIR = "10k"

def estimate_all_docs():
    """Estimate total tokens for all 10-K documents"""

    html_files = [f for f in os.listdir(HTML_DIR) if f.endswith('.html')]
    print(f"Found {len(html_files)} HTML files\n")

    stats = {
        'total_docs': len(html_files),
        'total_tokens': 0,
        'total_chars': 0,
        'per_doc_tokens': [],
        'per_doc_chars': [],
        'tickers': []
    }

    print("Parsing and counting tokens...")
    for html_file in tqdm(html_files):
        file_path = os.path.join(HTML_DIR, html_file)

        # Parse HTML to extract clean text
        parsed = parse_ixbrl_html(file_path)
        text = parsed['text']
        ticker = parsed['ticker']

        # Count tokens
        num_tokens = count_tokens(text)
        num_chars = len(text)

        stats['total_tokens'] += num_tokens
        stats['total_chars'] += num_chars
        stats['per_doc_tokens'].append(num_tokens)
        stats['per_doc_chars'].append(num_chars)
        stats['tickers'].append(ticker)

    # Calculate statistics
    avg_tokens = stats['total_tokens'] / stats['total_docs']
    avg_chars = stats['total_chars'] / stats['total_docs']
    max_tokens = max(stats['per_doc_tokens'])
    min_tokens = min(stats['per_doc_tokens'])
    max_idx = stats['per_doc_tokens'].index(max_tokens)
    min_idx = stats['per_doc_tokens'].index(min_tokens)

    print("\n" + "="*60)
    print("TOKEN ESTIMATION RESULTS")
    print("="*60)
    print(f"Total documents:        {stats['total_docs']:,}")
    print(f"Total tokens:           {stats['total_tokens']:,}")
    print(f"Total characters:       {stats['total_chars']:,}")
    print(f"\nAverage per document:")
    print(f"  Tokens:               {avg_tokens:,.0f}")
    print(f"  Characters:           {avg_chars:,.0f}")
    print(f"\nRange:")
    print(f"  Smallest doc:         {stats['tickers'][min_idx]} ({min_tokens:,} tokens)")
    print(f"  Largest doc:          {stats['tickers'][max_idx]} ({max_tokens:,} tokens)")
    print("="*60)

    # Embedding cost estimation (example with OpenAI ada-002)
    # OpenAI ada-002: $0.0001 per 1K tokens
    cost_per_1k = 0.0001
    estimated_cost = (stats['total_tokens'] / 1000) * cost_per_1k
    print(f"\nEstimated embedding cost (OpenAI ada-002):")
    print(f"  ${estimated_cost:.2f} for one-time embedding")
    print("="*60)

    # Save detailed stats
    with open("token_stats.json", "w") as f:
        json.dump({
            'summary': {
                'total_docs': stats['total_docs'],
                'total_tokens': stats['total_tokens'],
                'avg_tokens': avg_tokens,
                'max_tokens': max_tokens,
                'min_tokens': min_tokens
            },
            'per_document': [
                {'ticker': ticker, 'tokens': tokens, 'chars': chars}
                for ticker, tokens, chars in zip(
                    stats['tickers'],
                    stats['per_doc_tokens'],
                    stats['per_doc_chars']
                )
            ]
        }, f, indent=2)
    print("\nDetailed stats saved to token_stats.json")

    return stats

if __name__ == "__main__":
    stats = estimate_all_docs()
