# Configuration reference

This page lists what the canonical pipeline actually uses. The values come from the code, not from
tuning notes. Environment variables are read from the shell or from `.env`, which the notebook and
scripts load with python-dotenv. `.env.example` has the complete list.

## Multimode notebook

`lightrag_10k_priority_ticker_models_optimised_multimode.ipynb`

### Settings you can override

| Variable | Default | Notes |
|---|---|---|
| `TICKERS` | `HON,MSI,MRO,CE,HAS,VRSK,BIIB,ROP,CTAS,AMP` | Tickers to index and to select questions for |
| `TEST_TICKER` | `CTAS` | Indexed first on its own (cell 4) |
| `QUERY_MODES` | `local,global,naive,hybrid,mix` | Modes run for each question, in this order |
| `MAX_QUESTIONS` | unset (all) | Keep only the first N selected questions |
| `MAX_DOC_CHARS` | unset (full text) | Index only the first N characters of each filing |
| `MISTRAL_BASE_URL` | `https://api.mistral.ai/v1` | Any OpenAI-compatible endpoint |
| `MISTRAL_MODEL` | `ministral-14b-2512` | Entity and relation extraction |
| `MISTRAL_API_KEY` | none | Required, even for a local endpoint (any value) |
| `GENERATION_BACKEND` | `gemini` | `openai` answers with `MISTRAL_BASE_URL` / `MISTRAL_MODEL` instead |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Answer generation (google-genai SDK) |
| `EMBEDDING_MODEL_NAME` | `intfloat/e5-mistral-7b-instruct` | Loaded with `transformers.AutoModel` |
| `EMBEDDING_DIM` | `4096` | Must match the model (1024 for `BAAI/bge-large-en-v1.5`) |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | `sentence_transformers.CrossEncoder` |
| `RAGAS_JUDGE_MODEL` | `ministral-3-14b-2512` | Cell 7 only. The batch script defaults to `ministral-14b-2512` |
| `RAGAS_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Local HuggingFace embeddings on CPU |
| `LLM_TIMEOUT`, `EMBEDDING_TIMEOUT` | 180, 30 | Read by LightRAG itself. Raise them for CPU runs |

### Fixed in the code

- **Paths:** the notebook reads `parsed_10k_documents.json` and `finder_train.parquet` from the
  working directory and builds its workspace in `./lightrag_10k_workspace_latest`. It writes results
  to
  `5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL/test_results_CTAS_question_<id>.json`.
  The `CTAS` in the file name is fixed, whatever the ticker.
- **LightRAG instance:** `chunk_token_size=1200` and `chunk_overlap_token_size=100`. The embedding
  function is `EmbeddingFunc(embedding_dim=EMBEDDING_DIM, max_token_size=8192)`. It has a rerank function and
  `min_rerank_score=0.3`. Everything else uses the lightrag-hku defaults. Each filing is inserted with
  `ids=<ticker>`, and a compact JSON (`ticker`, `company`, `period`, `source`) goes in `file_paths`.
- **Embedder:** on CUDA the model loads in 4-bit through bitsandbytes (NF4, double quantisation,
  fp16 compute). On CPU it loads unquantised, so a 7B model is impractical there and you should set a
  small `EMBEDDING_MODEL_NAME`. The embedder encodes in batches of 4 with `max_length=512`, mean-pools
  `last_hidden_state` over all positions (the attention mask is ignored), then L2-normalises.
- **Reranker:** it loads only before querying. It goes on the GPU in fp16 when more than 2 GB of VRAM
  is free and on the CPU otherwise, with `max_length=8192` and a prediction batch size of 8.
- **Question selection:** a question matches a ticker when the ticker appears as `(XXX)` or at the end
  of the text after a comma or dash. With the default tickers this keeps 63 of the 5,703 FinDER
  training questions.
- **Querying:** two calls per question and mode, both with
  `QueryParam(mode, top_k=60, chunk_top_k=10, enable_rerank=True)`. The first generates the answer.
  The second passes `only_need_context=True` and stores the raw context. When LightRAG finds no
  usable context, it either returns its fixed `…[no-context]` reply or returns `None`. The notebook
  stores `None` as an empty answer, and the mode's status is still `success`. With
  `GENERATION_BACKEND=gemini`, the notebook replaces `rag.llm_model_func` with the Gemini wrapper
  before querying, so every query-time LLM call, including keyword extraction, goes to Gemini.
- **Gemini wrapper:** it concatenates the system prompt, history and prompt into a single
  `generate_content` call. If the call raises, the returned answer is `Error generating response: …`.
- **Cell 7 (RAGAS):** it scores every matching result file with `Faithfulness`, `AnswerRelevancy`,
  `ContextRecall` and `ContextPrecision`. The judge is `ChatOpenAI` at `MISTRAL_BASE_URL`, wrapped in
  `LangchainLLMWrapper(bypass_n=True)`. The notebook sleeps 10 s between evaluations and writes
  `ragas_evaluation_results_<timestamp>.json`.

## RAGAS batch script

`batch_ragas_evaluation.py` scores the per-question files in `--results-dir`, which defaults to
`$RESULTS_DIR` or the notebook's output directory. Each question and mode becomes one RAGAS dataset
row: question, answer, the whole retrieved context as a single-element `contexts` list, and the FinDER
answer as `ground_truth`. `ragas_score` is the mean of the non-NaN metrics.

| `--strategy` | Pacing | LLM retries / timeout | Output (checkpoint) |
|---|---|---|---|
| `sequential` (default) | 2 s between modes, 3 s between files. Tests the connection first and retries connection errors up to 3 times with backoff | 5 / 180 s | `batch_ragas_evaluation_results_ultra_simple.json` (`batch_ragas_checkpoint_ultra.json`) |
| `adaptive` | Modes in pairs, 1 s between pairs, 5 s between files. The per-call delay starts at 0.5 s, shrinks after runs of successes (minimum 0.2 s) and grows after errors (maximum 5 s) | 3 / 120 s | `batch_ragas_evaluation_results_optimized.json` (`batch_ragas_checkpoint_optimized.json`) |

Other options: `--modes`, `--limit N` (first N files), `--output` (or `OUTPUT_FILE`) and
`--checkpoint` (or `CHECKPOINT_FILE`). The checkpoint is updated after every evaluation and removed
once the run completes, so an interrupted run resumes where it stopped. When more than 50 evaluations
remain, the script prints an estimate and waits for `y`. `run_simple_evaluation.sh` wraps the
sequential strategy.

## Mode analysis

`analyze_mode_performance.py` reads `RESULTS_DIR` and writes `mode_performance_analysis.json`:

- **Processing time per mode:** the difference between that mode's completion timestamp and the
  previous one. Files are ordered by the timestamp of their `local` result, and files without one are
  skipped. Gaps over 300 s at file boundaries are dropped.
- **Tokens:** `(context_length + question length) / 4` for input and `answer_length / 4` for output.

`visualize_mode_performance.py` reads that JSON and writes six PNGs: processing times, token
consumption, efficiency, distributions, a summary table and a combined dashboard.
