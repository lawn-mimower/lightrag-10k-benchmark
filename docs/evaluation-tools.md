# Evaluation utilities

The canonical scorer is `batch_ragas_evaluation.py`; see [configuration.md](configuration.md). The
scripts below are alternatives and earlier iterations that work on saved results. None of them is
needed to run the pipeline.

## Custom RAGAS-style metrics with Gemini

`custom_ragas_metrics.py` reimplements four RAGAS metrics with plain Gemini prompts
(`google-generativeai`) and does not use the RAGAS library. The judge model is set by
`CUSTOM_RAGAS_MODEL` (default `gemini-2.5-flash`) or by the `model_name` argument. The key comes from
`GEMINI_API_KEY`.

| Metric | Computation |
|---|---|
| Faithfulness | Extract the claims in the answer, check each against the contexts, report supported / total |
| Context recall | The same, applied to the claims in the ground truth |
| Context precision | Judge each context chunk as relevant or not, then average precision@k over the relevant positions |
| Answer correctness | 0.5 × factual F1 over TP/FP/FN facts + 0.5 × an LLM-rated semantic similarity (0–1) |

Each claim and each chunk costs one LLM call, so the cost grows with answer length and with the
number of chunks. If the judge's JSON cannot be parsed, the metric scores 0.

```python
from custom_ragas_metrics import CustomRAGASMetrics, RAGEvaluation

m = CustomRAGASMetrics()                      # or CustomRAGASMetrics(model_name="gemini-3-flash-preview")
sample = RAGEvaluation(question="…", answer="…", contexts=["…", "…"], ground_truth="…")
print(m.faithfulness(sample)["score"])
print(m.evaluate_all(sample))                 # all four metrics
```

`evaluate_with_custom_ragas.py input.json output.json` scores a JSON list of
`{question, answer, contexts, ground_truth}` records. Run without arguments, it scores one built-in
example.

## LLM-judge metrics (`evaluate_answers.py`)

This script scores answers with NVIDIA-style judge prompts. Answer accuracy uses a 0/2/4 scale;
context relevance and groundedness use 0/1/2. All three are normalised to 0–1.
`--judge gemini` (`gemini-2.5-flash`, the default) or `--judge mistral` (`ministral-14b-2512`) picks
the judge. It reads the older single-file results format, which is set in the script, and keeps a
checkpoint. `analyze_evaluations.py` summarises and plots its output.

## Answer regeneration (`generate_answers_ctas.py`)

This script adds Gemini answers (`gemini-3-flash-preview`) to per-question result files that already
hold retrieved contexts. It processes all five modes of a file in parallel and waits 60 s between
files.

- `--always-reason` turns on thinking for every question and keeps the previous answer as
  `response_old`.
- `--vertex-express` sends plain prompts to the Vertex AI express endpoint with an API key.

`--results-dir`, or `RESULTS_DIR`, chooses the folder. `generate_answers.py` is an older batch
version that sends 5 questions per request.

## Gemini latency and token benchmark (`benchmark_gemini_api.py`)

This benchmark measures answer generation only. It sends the context that LightRAG already stored
for each mode, together with the question, to `gemini-3-flash-preview` (temperature 0.1) and records
the response time and the token counts from `usage_metadata`. It samples question files in which all
five modes succeeded.

```bash
python benchmark_gemini_api.py --check              # API key, one short request, results directory
python benchmark_gemini_api.py --quick --samples 20 # short summary -> gemini_benchmark_<timestamp>.json
python benchmark_gemini_api.py                      # mean/median/min/max/stdev -> gemini_benchmark_results_<timestamp>.json
python visualize_gemini_benchmark.py [file.json]    # charts; defaults to the newest gemini_benchmark_*.json
```

`--quick` uses the prompt from `generate_answers_ctas.py` and caps the context at 50,000 characters.
Both modes pause 0.5 s between calls. `--results-dir` (or `RESULTS_DIR`) and `--samples` (or
`SAMPLE_SIZE`) set the input. `run_gemini_benchmark.sh` is a thin wrapper, and
`estimate_tokens_ctas.py` estimates token usage and cost from saved result files.

## Other RAGAS helpers

- `batch_ragas_mistral_batch_api.py` submits simplified single-prompt 0–1 scores for the same four
  metric names through Mistral's Batch API. These are not the RAGAS library's metrics.
- `test_single_file_ragas_evaluation.py` runs RAGAS on one result file.
- `test_simple_evaluation.py` checks the key, the results directory and the imports before a batch run.
- `monitor_ragas.py` follows a running batch evaluation.
- `analyze_ragas_results.py` aggregates `*_evaled.json` files into CSVs.
- `optimized_ragas_cell.py` is a notebook-cell variant without rate limiting that uses OpenAI embeddings.
- `transform_contexts.py` splits a stored context string into a list of document chunks.
