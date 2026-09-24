# lightrag-10k-benchmark

LightRAG can answer a question from plain vector search over text chunks (`naive`) or from a
knowledge graph it extracts while indexing: `local` retrieves through entities, `global` through
relations, `hybrid` through both, and `mix` combines the graph with vector search. This project tests
whether the graph pays off on real financial questions. It indexes SEC 10-K filings once, answers FinDER analyst
questions in all five modes against that index, and saves each mode's answer and retrieved context
next to each other. RAGAS scores and latency/context-size analyses can then compare the modes
question by question.

## How it works

```
10k/<TICKER>.html --html_parser.py--> parsed_10k_documents.json
                                              |
finder_train.parquet --> multimode notebook: index -> query in 5 modes -> answer
                                              |
            5_modes_question_wise_results_with_answers/.../one JSON per question
                     |                                        |
       batch_ragas_evaluation.py                 analyze_mode_performance.py
       (RAGAS scores per mode)                   -> visualize_mode_performance.py
```

1. **Parse.** `html_parser.py` strips the inline-XBRL layer from each filing: the hidden
   `ix:header`/`ix:hidden` blocks, `display:none` elements, scripts and styles. It keeps the
   visible text, including tagged facts, and reads the company name and period end from the `dei:` tags.
   The output is `{ticker: {company_name, period_end_date, source_file, text, text_length}}`.
2. **Index.** The notebook `lightrag_10k_priority_ticker_models_optimised_multimode.ipynb` builds one
   LightRAG workspace for ten "priority" tickers (HON, MSI, MRO, CE, HAS, VRSK, BIIB, ROP, CTAS,
   AMP). It uses 1,200-token chunks with a 100-token overlap. `ministral-14b-2512` (Mistral API, or any
   OpenAI-compatible endpoint) extracts entities and relations. `intfloat/e5-mistral-7b-instruct`
   supplies the embeddings and is loaded in 4-bit on CUDA. Ticker, company and period are stored with
   each document.
3. **Query.** The notebook keeps the FinDER questions that name one of those tickers (63 of 5,703 with
   the default list) and loads `BAAI/bge-reranker-v2-m3` as a cross-encoder. For each question and
   mode it asks LightRAG for an answer written by `gemini-3-flash-preview` (`top_k=60`,
   `chunk_top_k=10`, reranking on, `min_rerank_score=0.3`), then makes a second call for the raw
   retrieved context. Each question gets one JSON file with the expected answer, the FinDER metadata
   and the results of all five modes.
4. **Score.** `batch_ragas_evaluation.py` runs RAGAS faithfulness, answer relevancy, context recall and
   context precision for every question and mode. The judge is an OpenAI-compatible model (Ministral
   by default) and the embeddings are a local `BAAI/bge-large-en-v1.5`. The script saves a checkpoint
   after each evaluation and resumes from it.
5. **Compare.** `analyze_mode_performance.py` computes a time per mode and approximate token counts.
   The time is the gap between the completion timestamps of consecutive modes, so it covers both calls.
   Tokens are estimated as characters / 4. `visualize_mode_performance.py` plots the results.

A result file looks like this (abridged):

```json
{
  "question_id": "…", "question": "…", "expected_answer": "…",
  "expected_type": "…", "category": "…", "reasoning": false, "references": ["…"],
  "modes": {
    "naive": {"answer": "…", "retrieved_context": "…", "answer_length": 707,
              "context_length": 5499, "timestamp": "…", "status": "success"},
    "local": {"…": "…"}, "global": {"…": "…"}, "hybrid": {"…": "…"}, "mix": {"…": "…"}
  }
}
```

## Results

The pipeline above was run on 2026-01-19 over the 10 priority tickers and their 63 questions. The
run used `ministral-14b-2512` extraction, e5-mistral embeddings, `bge-reranker-v2-m3` with
`min_rerank_score=0.3` and `gemini-3-flash-preview` answers. RAGAS then scored all 315 answers with
`ministral-14b-2512` as the judge. [results/RESULTS.md](results/RESULTS.md) has the analysis:
per-mode scores, latency, breakdowns, examples of where the modes disagree, and the exact run
conditions. [results/](results/) holds the per-question answers, context summaries, timings and
scores.

| mode | RAGAS score (mean of 4) | faithfulness | answer relevancy | context recall | context precision | median time per mode (s) |
|---|---|---|---|---|---|---|
| local | 0.741 | 0.966 | 0.688 | 0.499 | 0.810 | 24.1 |
| global | 0.718 | 0.937 | 0.673 | 0.518 | 0.742 | 25.7 |
| naive | 0.513 | 0.927 | 0.419 | 0.291 | 0.429 | 11.2 |
| hybrid | 0.765 | 0.953 | 0.673 | 0.595 | 0.841 | 29.3 |
| mix | 0.760 | 0.941 | 0.674 | 0.618 | 0.810 | 30.9 |

The values are means over 63 questions. Three metric values are NaN and left out, so faithfulness
has N=62 for `naive` and `mix`, and context precision has N=62 for `global`. The time per mode covers
LightRAG's answer call and context call on one RTX 4070 SUPER, with Gemini as a hosted API.

- **Graph modes against `naive`:** the four graph modes score 0.72 to 0.77, well ahead of `naive`
  at 0.51. Part of that gap comes from the rerank threshold. `naive` sends only 10 chunks to the
  reranker, and after the 0.3 cut it kept 3.1 on average, with none at all for 13 questions.
- **Among the graph modes:** `hybrid` and `mix` are level with each other. The margins between the
  graph modes are small next to the judge's variation when the same item is scored twice.
- **Scope:** this is one run with one judge on 63 questions.

## Data (not included)

Both inputs come from the FinDER dataset on Hugging Face
([Linq-AI-Research/FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER), CC BY-NC 4.0):

```bash
# 497 10-K filings as iXBRL HTML, one per ticker (142 MB)
curl -L -o 10-k.zip "https://huggingface.co/datasets/Linq-AI-Research/FinDER/resolve/main/10-k.zip?download=true"
unzip -q 10-k.zip '10k/*'        # -> 10k/<TICKER>.html (skips the archive's __MACOSX/ folder)
# 5,703 questions with expected answers and evidence passages
curl -L -o finder_train.parquet \
  "https://huggingface.co/datasets/Linq-AI-Research/FinDER/resolve/main/data/train-00000-of-00001.parquet"
```

The notebook reads both files from the repository root. `local-llm/benchmark_finder.py` looks for
`./data/finder_train.parquet` unless you pass `--data-path`. For tests and smoke runs, the repository
includes a 100 KB excerpt of Cintas' FY2024 10-K (`tests/fixtures/10k/CTAS.html`).

## Quick start

Tested with Python 3.12, lightrag-hku 1.4.9.11, ragas 0.4.3 and transformers 4.57.3.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # fill in MISTRAL_API_KEY and GEMINI_API_KEY

python html_parser.py 10k parsed_10k_documents.json              # add --tickers CTAS HON to parse a subset
jupyter nbconvert --to notebook --execute --output-dir runs \
    lightrag_10k_priority_ticker_models_optimised_multimode.ipynb
python batch_ragas_evaluation.py --modes naive hybrid --limit 5  # no flags: every file and mode
python analyze_mode_performance.py && python visualize_mode_performance.py
```

The notebook needs Jupyter or nbconvert because it uses top-level `await`, and its `python3` kernel
must be the environment you installed into. Its last cell scores the new answers with RAGAS.
`batch_ragas_evaluation.py` does the same scoring as a standalone, resumable script. When more than
50 evaluations are left, it asks for confirmation before it starts.

**Small runs, CPU only, no hosted APIs.** Environment variables override every notebook setting.
The example below indexes the first 10,000 characters of one filing, answers one question and uses an
embedder that fits on a CPU. Extraction, answers and the RAGAS judge all go to a local
OpenAI-compatible server, such as llama.cpp's `llama-server`:

```bash
python html_parser.py 10k parsed_10k_documents.json --tickers CTAS
TICKERS=CTAS MAX_QUESTIONS=1 MAX_DOC_CHARS=10000 \
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5 EMBEDDING_DIM=1024 \
MISTRAL_BASE_URL=http://localhost:8080/v1 MISTRAL_API_KEY=local GENERATION_BACKEND=openai \
LLM_TIMEOUT=3600 EMBEDDING_TIMEOUT=900 \
jupyter nbconvert --to notebook --execute --output-dir runs \
    lightrag_10k_priority_ticker_models_optimised_multimode.ipynb
```

`MISTRAL_API_KEY` must be set even for a local server; any value works. If your server checks model
names, set `MISTRAL_MODEL` and `RAGAS_JUDGE_MODEL` to the model it serves. The two timeouts raise
LightRAG's default per-call limits, which are 180 s for the LLM and 30 s for embeddings.

### API-free variant: local llama.cpp benchmark

`local-llm/benchmark_finder.py` is self-contained. It indexes FinDER's own evidence passages rather
than the 10-K HTML, using Llama-3.2-3B-Instruct (GGUF through llama-cpp-python) and the
Qwen3-Embedding-0.6B embedder. It queries in `hybrid` mode and writes the question, expected answer
and generated answer to a CSV. It does not score the answers.

```bash
pip install -r local-llm/requirements.txt    # pinned versions; builds llama-cpp-python from source
LLM_TIMEOUT=3600 EMBEDDING_TIMEOUT=900 python local-llm/benchmark_finder.py \
    --model-path models/Llama-3.2-3B-Instruct.Q8_0.gguf \
    --embedder-path Qwen/Qwen3-Embedding-0.6B --data-path finder_train.parquet \
    --num-docs 1 --max-queries 1 --output results.csv
```

- `--device cpu` (the default) uses an 8K context and inserts passages one at a time with 2 gleaning
  passes. It indexes the first 10 passages.
- `--device gpu` puts every layer on the GPU with a 128K context and batch-inserts with 8-way
  parallelism and no gleaning. It indexes all 5,832 unique passages.
- Without `--query-only`, the script deletes and rebuilds the working directory. With it, the script
  queries the existing index in `--working-dir`.

## Configuration

`.env.example` lists every variable, and the scripts load `.env` with python-dotenv. These are the
main ones:

| Variable | Purpose |
|---|---|
| `MISTRAL_API_KEY` | Entity extraction and RAGAS judge (any value for a local endpoint) |
| `GEMINI_API_KEY` | Answer generation when `GENERATION_BACKEND=gemini`, Gemini utilities |
| `MISTRAL_BASE_URL`, `MISTRAL_MODEL` | Extraction endpoint and model (default Mistral API, `ministral-14b-2512`) |
| `GENERATION_BACKEND`, `GEMINI_MODEL` | `gemini` (default, `gemini-3-flash-preview`) or `openai` (answers from `MISTRAL_BASE_URL`) |
| `EMBEDDING_MODEL_NAME`, `EMBEDDING_DIM`, `RERANKER_MODEL_NAME` | Retrieval models |
| `TICKERS`, `TEST_TICKER`, `QUERY_MODES`, `MAX_QUESTIONS`, `MAX_DOC_CHARS` | Run scope |
| `RAGAS_JUDGE_MODEL`, `RAGAS_EMBEDDING_MODEL`, `RESULTS_DIR` | Scoring |
| `LLM_TIMEOUT`, `EMBEDDING_TIMEOUT` | LightRAG per-call timeouts (seconds) |

[docs/configuration.md](docs/configuration.md) lists the fixed notebook parameters, the RAGAS batch
strategies and every output file.

## Tests

```bash
pytest               # offline suite: 69 pass, the 4 live tests skip
pytest -m e2e        # live tests; each one runs only when its variables are set
```

The offline tests need no API keys or network access. They cover the HTML parser, the RAGAS batch CLI, mode
analysis and a check that the notebook cells compile. They also cover the answer-generation and judge
scripts, the custom metrics, the llama.cpp wrappers and the document-parsing scripts. The live tests
are:

| Test | Needs |
|---|---|
| One Gemini answer | `GEMINI_API_KEY` |
| One call to the extraction endpoint | `MISTRAL_API_KEY` (+ `MISTRAL_BASE_URL`) |
| Notebook cells 0–6 on the bundled CTAS excerpt, then `batch_ragas_evaluation.py` on one mode | `E2E_NOTEBOOK=1` + keys (or a local endpoint) |
| `local-llm/benchmark_finder.py` on CPU, 1 passage and 1 query | `LOCAL_LLM_MODEL_PATH`, `LOCAL_EMBEDDER_PATH`, `FINDER_DATA_PATH` |

The `test_*.py` scripts in the repository root are manual checks that you run by hand. pytest
collects only `tests/`.

## Status and limitations

- **What has been run:** the offline suite, and the notebook end to end on CPU against the bundled
  excerpt. That run used a small local model behind an OpenAI-compatible endpoint and bge-large
  embeddings. All five modes returned answers, and `batch_ragas_evaluation.py` scored the `naive`
  answer. `local-llm/benchmark_finder.py` also ran on CPU with 1 passage and 1 query: it gave the
  correct answer in about 9 minutes. The default hosted setup (Mistral extraction, Gemini answers,
  e5-mistral on a GPU) ran in full once, in January 2026, with the notebook as it was then. Its
  results are in [results/](results/) (see [Results](#results)). Since then, only the live tests,
  which need API keys, have exercised that setup.
- **Embedding wrapper:** the e5-mistral wrapper mean-pools the last hidden state over every position,
  padding included. It truncates input at 512 tokens, although chunks are 1,200 tokens long, and it adds no
  query instruction. The model card for e5-mistral uses last-token pooling and an instruction prefix
  for queries, so the notebook is not using the embedder the way it was trained.
- **Single-string context in RAGAS:** the whole retrieved context goes to RAGAS as one string, so
  context precision comes down to a single relevance judgement per answer.
- **Question selection:** a question is kept only if it writes the ticker in parentheses or at the end.
  Questions that name the company in another way (for example "Cboe") are skipped.
- **Result file names:** every result file is named `test_results_CTAS_question_<id>.json`, whatever
  its ticker. The downstream scripts match `test_results_*_question_*.json`, so they are unaffected.
- **Small extraction models:** LightRAG 1.4.9 drops relation lines that have the wrong number of fields
  and logs `LLM output format error; found 4/5 fields`. Llama-3.2-3B triggers this often. With a very
  small model, the graph modes can return no context at all. The notebook then stores an empty answer
  or LightRAG's `[no-context]` reply and does not fail.
- **Gemini errors:** if a Gemini call fails, the error text becomes the answer
  (`Error generating response: …`) and the mode's status is still `success`.
- **Judge defaults differ:** cell 7 of the notebook defaults to the judge `ministral-3-14b-2512`, while
  `batch_ragas_evaluation.py` defaults to `ministral-14b-2512`. Set `RAGAS_JUDGE_MODEL` to use the same
  judge in both.

## Repository layout

**Canonical pipeline**

| Path | Role |
|---|---|
| `html_parser.py` | 10-K iXBRL HTML → ticker-keyed JSON |
| `lightrag_10k_priority_ticker_models_optimised_multimode.ipynb` | Index, query in 5 modes, generate answers (cell 7: RAGAS) |
| `batch_ragas_evaluation.py`, `run_simple_evaluation.sh` | Resumable RAGAS scoring (`--strategy sequential\|adaptive`) |
| `analyze_mode_performance.py`, `visualize_mode_performance.py` | Per-mode latency and context-size comparison |
| `local-llm/benchmark_finder.py` | API-free llama.cpp variant on FinDER passages |
| `results/` | Benchmark outputs from January 2026 (per-question answers, context summaries, timings, RAGAS scores, charts), `RESULTS.md` analysis, `summarize.py` |
| `tests/` | Offline and live tests, 10-K fixture |
| `docs/` | Configuration, evaluation utilities, document parsing |

**Supporting utilities.** These are earlier iterations and side experiments, and the pipeline above
does not need them.

- *Answer generation and judging on saved results:* `generate_answers_ctas.py`, `generate_answers.py`,
  `evaluate_answers.py`, `analyze_evaluations.py`, `custom_ragas_metrics.py`,
  `evaluate_with_custom_ragas.py`, `batch_ragas_mistral_batch_api.py`, `analyze_ragas_results.py`,
  `monitor_ragas.py`, `optimized_ragas_cell.py`, `test_single_file_ragas_evaluation.py`,
  `test_simple_evaluation.py`, `transform_contexts.py`. See [docs/evaluation-tools.md](docs/evaluation-tools.md).
- *Gemini latency and token benchmark:* `benchmark_gemini_api.py`, `visualize_gemini_benchmark.py`,
  `run_gemini_benchmark.sh`, `setup_visualization.sh`, `estimate_tokens_ctas.py`.
- *Dataset exploration:* `count_ticker_occurrences.py`, `map_questions_to_tickers.py`,
  `analyze_references.py`, `estimate_tokens.py`, `finderfinder.py`, `json_schema.py`, `test.ipynb`.
- *Earlier prototypes:* `lightrag_10k_interactive.ipynb` (interactive querying with BGE embeddings,
  Ministral 8B and a ColBERT rerank server), `kgbuilder.py` + `query.py` (graph over a 10% FinDER
  sample), `local-llm/query.py`, `local-llm/extraction_formatting_debug.py`.
- *API checks:* `test_mistral_connection.py`, `test_mistral_ocr_api.py`, `test_ragas_setup.py`.
- *Document parsing (PDF, Office files and images to Markdown; separate from the 10-K HTML path):*
  `parse_documents_with_docling.py`, `docling_md_out_monitored.py`, `mistral_document_extraction*.py`,
  `json_to_markdown_converter.py`, `fix_rtdetr_model.py`, `fix_transformers.sh`, `install_md2pdf.sh`.
  See [docs/document-parsing.md](docs/document-parsing.md).

## Credits

- [LightRAG](https://github.com/HKUDS/LightRAG) (Guo et al., arXiv:2410.05779).
- FinDER: Choi et al., *FinDER: Financial Dataset for Question Answering and Evaluating
  Retrieval-Augmented Generation*, arXiv:2504.15800. The FinDER data is licensed CC BY-NC 4.0.
  `results/` includes the 63 questions and expected answers used in the benchmark, under that
  licence. The rest of the dataset is not redistributed here. The context excerpts in `results/`
  come from the companies' public SEC 10-K filings.

Licence: MIT — see [LICENSE](LICENSE). The FinDER-derived content in `results/` stays under
CC BY-NC 4.0 (see [results/README.md](results/README.md)).
