# Gemini-3-Flash-Preview API Benchmark Suite

## Overview
This suite benchmarks the `gemini-3-flash-preview` model across different LightRAG retrieval modes (local, global, naive, hybrid, mix) using real questions and contexts from your dataset.

## Files

All runs use **`benchmark_gemini_api.py`**:

1. `--check` - Verify the setup (API key, connection, data directory)
2. `--quick` - Quick benchmark (simplified)
3. no option - Full benchmark with detailed statistics

## Setup

### 1. Install Dependencies
```bash
pip install google-generativeai python-dotenv
```

Note: This uses the new `google.genai` client matching your existing `generate_answers_ctas.py`

### 2. Set API Key
Option A: Use .env file (recommended)
```bash
echo "GEMINI_API_KEY=your-gemini-api-key-here" > .env
```

Option B: Export environment variable
```bash
export GEMINI_API_KEY='your-gemini-api-key-here'
```

### 3. Test Your Setup
Run the setup check to verify everything is configured correctly:
```bash
python3 benchmark_gemini_api.py --check
```

This will:
- Check your API key is set
- Test the Gemini API connection
- Verify your data directory exists
- Validate the file structure

Expected output:
```
✓ GEMINI_API_KEY found
✓ Gemini client initialized successfully
✓ API call successful!
✓ Found 63 JSON files in data directory
✅ ALL CHECKS PASSED - Ready to run benchmark!
```

## Running Benchmarks

### Quick Benchmark (Recommended for Testing)
```bash
python3 benchmark_gemini_api.py --quick
```
- Tests 10 random questions across all 5 modes
- Total of 50 API calls
- Quick summary of performance
- Takes ~5-10 minutes

### Full Benchmark
```bash
python3 benchmark_gemini_api.py
```
- Detailed statistics (mean, median, min, max, stdev)
- Comprehensive error handling
- More detailed token usage analysis
- Takes ~10-15 minutes for 10 questions

## Configuration

Both benchmarks use the same configuration as your `generate_answers_ctas.py`:
- Model: `gemini-3-flash-preview`
- Temperature: `0.1` (for consistent results)
- Max output tokens: `65536`
- Uses `google.genai.Client` pattern

## What Gets Measured

For each mode (local, global, naive, hybrid, mix):
1. **Response Time** - Actual API call duration in seconds
2. **Input Tokens** - Tokens in the prompt (question + context)
3. **Output Tokens** - Tokens in the generated answer
4. **Total Tokens** - Sum of input and output tokens
5. **Success Rate** - Percentage of successful API calls

## Sample Output

```
LOCAL MODE:
  Avg Time: 2.345s
  Avg Tokens: 25,432 (in: 24,890, out: 542)
  Success Rate: 100%

NAIVE MODE:
  Avg Time: 1.234s
  Avg Tokens: 5,234 (in: 4,798, out: 436)
  Success Rate: 100%

HYBRID MODE:
  Avg Time: 3.456s
  Avg Tokens: 28,567 (in: 27,900, out: 667)
  Success Rate: 100%
```

## Expected Performance

Based on context sizes from your data:
- **Naive**: ~5,000 tokens (smallest, fastest)
- **Local/Global**: ~20,000-25,000 tokens
- **Hybrid/Mix**: ~28,000-30,000 tokens (largest, slowest)

Response times typically:
- Naive: 1-2 seconds
- Local/Global: 2-3 seconds
- Hybrid/Mix: 3-4 seconds

## Output Files

Both benchmarks generate timestamped JSON files:
- `gemini_benchmark_YYYYMMDD_HHMMSS.json`

Contains:
```json
{
  "config": {
    "model": "gemini-3-flash-preview",
    "samples": 10,
    "timestamp": "2024-..."
  },
  "summary": {
    "local": {...},
    "global": {...},
    ...
  },
  "raw_results": {...}
}
```

## Cost Estimation

Gemini-3-Flash pricing (approximate):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

For 10 questions (50 API calls):
- Est. input: ~1.2M tokens → $0.09
- Est. output: ~30K tokens → $0.01
- **Total: ~$0.10**

## Rate Limiting

The scripts include:
- 0.5 second delay between API calls
- Handles rate limit errors gracefully
- Adjustable delays if needed

## Customization

Set the number of questions with `--samples` (or `SAMPLE_SIZE`) and the
results directory with `--results-dir` (or `RESULTS_DIR`):
```bash
python3 benchmark_gemini_api.py --quick --samples 20
```

## Troubleshooting

### "No module named 'google.genai'"
```bash
pip install google-generativeai
```

### "GEMINI_API_KEY not found"
Check your .env file or:
```bash
export GEMINI_API_KEY='your-key'
```

### API errors
- Check API key is valid
- Verify you have access to gemini-3-flash-preview
- Check rate limits

### Context too long
The scripts truncate context to 50,000 chars if needed.
Adjust in the prompt creation section if needed.

## Next Steps

After running benchmarks:
1. Compare API response times across modes
2. Analyze token efficiency (cost vs quality)
3. Identify optimal mode for your use case
4. Consider adjusting context sizes

## Integration with Your Pipeline

The benchmark uses the same configuration as your existing `generate_answers_ctas.py`:
- Same client initialization pattern
- Same generation config
- Same prompt structure
- Compatible with your existing workflow