# Benchmark results

These are the outputs of the benchmark runs made with this repository's pipeline in January 2026,
curated from the original run folders. [RESULTS.md](RESULTS.md) has the analysis. This page
describes the files.

## Layout

| Path | What it is |
|---|---|
| `five_mode_run/` | **The main run**: 63 FinDER questions over 10 companies, all five LightRAG modes, answers from `gemini-3-flash-preview`, RAGAS scores for all 315 question-mode pairs |
| `five_mode_run/run_info.json` | Run conditions: models, parameters, dates, index size, question selection, RAGAS setup |
| `five_mode_run/questions/<question_id>.json` | One file per question: the question, the FinDER answer and metadata, and for each mode the answer, a summary of the retrieved context, the timing and the RAGAS scores |
| `five_mode_run/per_mode.csv` | The same run flattened to one row per question and mode (315 rows), without the text |
| `five_mode_run/ragas_scores.json` | The RAGAS results file as the scoring script wrote it (`NaN` written as `null`) |
| `five_mode_run/ragas_repeat_scoring.csv` | Pilot scorings of the same answers and contexts next to the final scores: 17 question-mode pairs scored twice, plus 3 pilot timeouts |
| `five_mode_run/mode_performance_analysis.json` | Output of `analyze_mode_performance.py` run on `questions/` |
| `five_mode_run/charts/` | `mode_*.png` from `visualize_mode_performance.py` (rendered at 100 dpi), `ragas_by_mode.png` from `summarize.py` |
| `gemini_latency_benchmark/` | Gemini-only timing: 10 of the 63 questions, each mode's saved context sent once to `gemini-3-flash-preview` (`benchmark_gemini_api.py`) |
| `mix_only_custom_judge/` | An earlier single-mode run (`mix`) with answers from `gemini-2.5-flash`, scored by the two-prompt judge in `evaluate_answers.py` |
| `local_llm_run/` | The llama.cpp run from `local-llm/benchmark_finder.py`: 10 queries, no answers produced |
| `summarize.py` | Prints every table in RESULTS.md from these files; `--chart` redraws `ragas_by_mode.png` |

```bash
python results/summarize.py            # tables
python results/summarize.py --chart    # tables + chart
RESULTS_DIR=results/five_mode_run/questions python analyze_mode_performance.py
```

## Per-question files

`five_mode_run/questions/<question_id>.json`:

```json
{
  "question_id": "38758f5a", "ticker": "MSI",
  "question": "Q4-23 avg price for share repurchase volume was reported for Motorola Solutions (MSI).",
  "expected_answer": "The company repurchased a total of 416,045 shares …",
  "category": "Shareholder return", "expected_type": "None", "reasoning": false,
  "finder_reference_count": 1, "finder_reference_chars": 1759,
  "modes": {
    "mix": {
      "status": "success", "timestamp": "2026-01-19T20:56:26.762701", "duration_ms": 19828,
      "answer": "During the fourth quarter of 2023, Motorola Solutions (MSI) repurchased …",
      "answer_length": 1445, "context_length": 121499,
      "context": {
        "entities": 35, "relations": 165, "chunks": 10, "rerank_candidates": 117,
        "chunk_tickers": ["MSI", "MSI", "…"],
        "entity_names": ["Motorola Solutions Inc.", "Fourth Quarter Of 2023", "…"],
        "chunk_excerpts": [{"ticker": "MSI", "chars": 6307, "text": "program”). The share repurchase program does not have an expiration date. …"}, "…"]
      },
      "ragas": {"faithfulness": 1.0, "answer_relevancy": 0.7383829620866207, "context_recall": 1.0,
                "context_precision": 0.9999999999, "ragas_score": 0.9346}
    },
    "local": {"…": "…"}, "global": {"…": "…"}, "naive": {"…": "…"}, "hybrid": {"…": "…"}
  }
}
```

(Text abridged with `…`.) The fields:

- `ticker`: the priority ticker found in the question with the notebook's own rule. Every question
  names exactly one.
- `expected_answer`, `category`, `expected_type`, `reasoning`: FinDER's fields, unchanged.
  `expected_type` is the string `"None"` for 60 of the 63 questions, as in the source.
- `finder_reference_count`, `finder_reference_chars`: the number and total length of FinDER's
  evidence passages for the question. The passages themselves are not copied; look them up in
  FinDER by `question_id` (FinDER's `_id`).
- `answer`, `answer_length`, `context_length`, `timestamp`, `status`: as saved by the notebook.
  `answer` is complete. `timestamp` is when the mode finished (both calls). One answer, `daff6a7b`
  in `hybrid` mode, is `null` with an `answer_omitted` note: an automated pre-publication check
  flagged a figure in it. Its length and scores are kept.
- `duration_ms`: this mode's timestamp minus the previous mode's timestamp, in milliseconds (for
  `local`, minus the previous question's `mix`). It covers both LightRAG calls, the answer call and
  the context call. It is `null` for the first mode of the first question. This is the same rule
  that `analyze_mode_performance.py` uses.
- `context`: a summary of the retrieved context. **The full context is not stored**: it averages
  19,000 to 111,000 characters per mode, about 25 MB for the run. What is kept:
  - `entities`, `relations`, `chunks`: the number of knowledge-graph entities, relations and
    document chunks in the context string that was saved and scored.
  - `rerank_candidates`: how many chunks went into the reranker, from the notebook's query log.
    The reranker keeps its top 10, and then `min_rerank_score=0.3` drops the rest of the
    low-scoring ones. `chunks` is what survived.
  - `chunk_tickers`: which filing each kept chunk came from (LightRAG's reference list).
  - `entity_names`: the first 15 entity names, in context order.
  - `chunk_excerpts`: the first 400 characters of each kept chunk, with its length and ticker.
    Runs of four or more underscores (blank form lines in the filings) are shortened to three.
    18 excerpts, marked `"text_cut_early": true`, end before a figure that the same
    pre-publication check flagged. These are exhibit numbers and table values from the filings.
- `ragas`: the four RAGAS metrics and `ragas_score` (their mean, ignoring NaN) from
  `ragas_scores.json`. Three metric values are `null` (NaN in the scoring run).

`per_mode.csv` has the numeric fields above plus `own_ticker_chunks`, the number of kept chunks
that come from the filing of the company the question asks about.

## Other files

- `ragas_repeat_scoring.csv`: `*_pilot` columns come from two pilot scorings of the same answers
  and contexts on 2026-01-19: a batch that stopped after 15 evaluations (12 scored, 3 timed out)
  and a single-question test of all five modes. `*_final` columns come from `ragas_scores.json`.
  Same judge, same embeddings.
- `gemini_latency_benchmark/summary.json`: the run's configuration and its `statistics` block,
  unchanged. `calls.csv`: one row per call, from the same file's per-call records, without the
  response text. For two calls the per-call `response_time` equals the mode's mean, while the
  statistics block only reproduces with a value of about 355 s and 362 s for those calls; the
  `note` column marks them. See RESULTS.md.
- `mix_only_custom_judge/evaluation_scores.csv`: written by `analyze_evaluations.py`, unchanged
  (questions truncated to 100 characters there). Columns `*_eval1` and `*_eval2` are the two judge
  prompts, and the unsuffixed column is their mean.
- `local_llm_run/finder_lightrag_results.csv`: the script's output with FinDER's evidence text
  replaced by its length (`finder_ground_truth_context_chars`).

## Not included

Pilot and superseded runs, raw logs, LightRAG workspaces and input data are not included. RESULTS.md
lists every run that was found and why it was left out.

## Sources and licence

- **Questions, expected answers and categories** come from FinDER (Choi et al., *FinDER: Financial
  Dataset for Question Answering and Evaluating Retrieval-Augmented Generation*, arXiv:2504.15800),
  [Linq-AI-Research/FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER) on Hugging
  Face, licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The copies here
  are unmodified, apart from the truncation noted above.
- **Retrieved-context excerpts and entity names** are text from, or extracted from, the companies'
  public SEC Form 10-K filings, as distributed in FinDER's 10-K archive.
- **Answers** were generated by `gemini-3-flash-preview` (main run), `gemini-2.5-flash` (mix-only
  run) and Llama-3.2-3B-Instruct (local run, which produced none). Scores come from the judges named
  in RESULTS.md.

The repository's MIT licence covers its code. The FinDER-derived content in `results/` (questions,
expected answers and anything built from them) remains under CC BY-NC 4.0: non-commercial use,
with attribution to FinDER.
