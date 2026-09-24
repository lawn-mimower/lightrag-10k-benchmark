# LightRAG 10-K Document Benchmark

A knowledge graph-powered question-answering system for SEC 10-K financial filings using [LightRAG](https://github.com/HKUDS/LightRAG). This project demonstrates hybrid search (knowledge graph + vector retrieval) with GPU-accelerated embeddings and reranking for financial document analysis.

## Overview

This project processes SEC 10-K HTML filings, builds knowledge graphs using LightRAG, and enables natural language querying with full provenance tracking. The system combines:

- **Knowledge Graph Extraction**: Entities and relationships extracted from financial documents
- **Hybrid Search**: Graph-based reasoning + semantic vector search
- **GPU Acceleration**: Fast embeddings (E5-Mistral-7B, 4-bit) and reranking (BGE-reranker-v2-m3)
- **Metadata Preservation**: Full citation tracking for answers

### Current Status

- **Tested**: HON (Honeywell) - 2 test queries successfully answered
- **Ready**: Framework for batch indexing 496 remaining company tickers
- **Pending**: Full evaluation metrics and debugging for general framework

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, ~8GB VRAM)
- 16GB+ RAM
- Mistral API key ([get one here](https://console.mistral.ai/))

### Installation

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd lightrag-bench
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # optional: local llama.cpp benchmark / document parsing scripts
   pip install -r local-llm/requirements.txt
   pip install -r requirements_document_parsing.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env   # then fill in MISTRAL_API_KEY and GEMINI_API_KEY
   ```

4. **Verify GPU** (optional but recommended):
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   ```

---

## Data Pipeline

### Step 1: Download 10-K Filings

Download the FinDER dataset (497 company 10-K HTML files, ~136MB):

```bash
wget https://huggingface.co/datasets/Linq-AI-Research/FinDER/resolve/main/10-k.zip?download=true -O 10-k.zip
```

### Step 2: Unzip Files

```bash
unzip 10-k.zip
```

This creates a `10k/` directory with 497 HTML files (one per company ticker):
```
10k/
├── AAPL.html
├── MSFT.html
├── HON.html
└── ...
```

### Step 3: Preprocess HTML Files

The HTML files contain iXBRL tags (inline XBRL) that need to be removed to extract clean text:

```bash
python html_parser.py 10k parsed_10k_documents.json
# or only some filings
python html_parser.py 10k parsed_10k_documents.json --tickers CTAS HON
```

**What this does:**
- Removes all iXBRL tags (`ix:hidden`, `ix:header`, etc.)
- Strips CSS, JavaScript, and non-visible content
- Extracts metadata (company name, filing period)
- Outputs clean JSON with structure:
  ```json
  {
    "AAPL": {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "period_end_date": "September 30, 2023",
      "text": "UNITED STATES SECURITIES AND EXCHANGE COMMISSION..."
    }
  }
  ```

**Output**: `parsed_10k_documents.json` (497 companies)

---

## Benchmark Pipeline (5 query modes)

The current pipeline indexes the priority tickers, answers every matching
FinDER question in the `local`, `global`, `naive`, `hybrid` and `mix`
modes, then scores the answers with RAGAS.

```bash
# 1. Parse filings (see above) and put finder_train.parquet in the project root
python html_parser.py 10k parsed_10k_documents.json

# 2. Index + query + answer (Mistral extraction, Gemini answers)
jupyter nbconvert --to notebook --execute --inplace \
    lightrag_10k_priority_ticker_models_optimised_multimode.ipynb
#    -> 5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL/*.json

# 3. RAGAS (faithfulness, answer relevancy, context recall/precision)
python batch_ragas_evaluation.py            # all files, all modes
python batch_ragas_evaluation.py --modes naive hybrid --limit 5
python batch_ragas_evaluation.py --strategy adaptive     # modes in pairs, adaptive delay

# 4. Latency / token analysis per mode
python analyze_mode_performance.py && python visualize_mode_performance.py
```

Quick or CPU-only runs are configured through environment variables (see
`.env.example`), for example:

```bash
TICKERS=CTAS MAX_QUESTIONS=1 MAX_DOC_CHARS=10000 \
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5 EMBEDDING_DIM=1024 \
MISTRAL_BASE_URL=http://localhost:8080/v1 GENERATION_BACKEND=openai \
LLM_TIMEOUT=3600 EMBEDDING_TIMEOUT=900 \
jupyter nbconvert --to notebook --execute --output run.ipynb \
    lightrag_10k_priority_ticker_models_optimised_multimode.ipynb
```

`MISTRAL_BASE_URL` accepts any OpenAI-compatible server (for example
llama.cpp's `llama-server`); `GENERATION_BACKEND=openai` answers with that
endpoint instead of Gemini.

### Local benchmark (llama.cpp)

```bash
python local-llm/benchmark_finder.py \
    --model-path models/Llama-3.2-3B-Instruct.Q8_0.gguf \
    --embedder-path models/qwen3-0.6b \
    --data-path finder_train.parquet --num-docs 1 --max-queries 1
```

`--device gpu` loads every layer onto the GPU with the full 128K context and
indexes all contexts in one batch; `--query-only` reuses the index in
`--working-dir` and only runs the queries.

### Tests

```bash
pytest                      # offline tests (e2e tests skip unless configured)
pytest -m e2e               # live runs; see tests/test_e2e.py for the variables
```

---

## Running the Test

### Single-ticker run (HON)

Index one company and answer a couple of its FinDER questions:

```bash
TICKERS=HON TEST_TICKER=HON MAX_QUESTIONS=2 \
jupyter nbconvert --to notebook --execute --output run_hon.ipynb \
    lightrag_10k_priority_ticker_models_optimised_multimode.ipynb
```

Or open `lightrag_10k_priority_ticker_models_optimised_multimode.ipynb` in
Jupyter and run the cells in order.

**What happens:**
1. Loads models (E5-Mistral-7B embeddings, Ministral-14B for extraction, Gemini for answers)
2. Initializes the LightRAG workspace (`lightrag_10k_workspace_latest/`)
3. Indexes the HON filing (builds the knowledge graph)
4. Answers the first 2 HON questions from FinDER in all five query modes, with the BGE reranker loaded just before querying
5. Writes one JSON file per question to `5_modes_question_wise_results_with_answers/5_modes_question_wise_results_priority_tickers_ALL/`

**Expected runtime**: 5-10 minutes on GPU (first run)

---

## Notebook Components

`lightrag_10k_priority_ticker_models_optimised_multimode.ipynb` is organized into 8 cells:

- **Cell 0: Overview** - goal, models and priority tickers
- **Cell 1: Configuration** - paths, model names, query modes and run limits; each can be overridden through an environment variable (see `.env.example`)
- **Cell 2: Model Loading** - E5-Mistral-7B embeddings (4-bit on GPU), the Ministral-14B extraction LLM and the Gemini client; the reranker is deferred to save VRAM during indexing
- **Cell 3: LightRAG Initialization** - workspace, chunking, reranker hook, document loading and indexing with ticker/company/period metadata
- **Cell 4: Indexing** - indexes `TEST_TICKER`, then every ticker in `TICKERS`
- **Cell 5: Questions** - loads `finder_train.parquet` and keeps the questions about the priority tickers (`MAX_QUESTIONS` limits the run)
- **Cell 6: Querying** - loads the reranker, answers each question in every query mode and saves one file per question
- **Cell 7: RAGAS Evaluation** - scores the saved answers (faithfulness, answer relevancy, context recall, context precision)

---

## Architecture

### LightRAG Hybrid Search

1. **Knowledge Graph Query**:
   - Extracts key entities from question (e.g., "supply chain", "Honeywell")
   - Retrieves relevant entities and 1-hop relations from graph
   - Uses cosine similarity on entity embeddings

2. **Vector Chunk Retrieval**:
   - Semantic search over text chunks
   - E5-Mistral-7B embeddings (4096-dim)

3. **Reranking**:
   - Cross-encoder (BGE-reranker-v2-m3) scores query-chunk pairs
   - Selects top 20 most relevant chunks

4. **Answer Generation**:
   - Gemini generates the answer from the reranked context
   - Includes citations from metadata

### Model Stack

| Component | Model | Device | Purpose |
|-----------|-------|--------|---------|
| Embeddings | intfloat/e5-mistral-7b-instruct (4-bit) | GPU | 4096-dim dense vectors |
| Extraction LLM | ministral-14b-2512 | API | KG extraction |
| Generation LLM | gemini-3-flash-preview | API | Answer generation |
| Reranker | BAAI/bge-reranker-v2-m3 | GPU | Relevance scoring |

### Metadata Encoding

Each document chunk includes:
```json
{
  "ticker": "HON",
  "company": "Honeywell International Inc.",
  "period": "December 31, 2023",
  "source": "HON.html"
}
```

Encoded in LightRAG's `file_path` field for full provenance tracking.

---

## Project Structure

```
lightrag-bench/
├── 10-k.zip                        # Downloaded dataset (136MB)
├── 10k/                            # Unzipped HTML files (497 companies)
│   ├── AAPL.html
│   ├── HON.html
│   └── ...
├── html_parser.py                  # HTML preprocessing script
├── parsed_10k_documents.json       # Parsed clean text (156MB)
├── lightrag_10k_priority_ticker_models_optimised_multimode.ipynb  # Pipeline notebook
├── 5_modes_question_wise_results_with_answers/  # One result file per question
├── lightrag_10k_workspace_latest/  # LightRAG storage
│   ├── kv_store_full_entities.json # Knowledge graph entities
│   ├── kv_store_full_relations.json # Knowledge graph relations
│   ├── vdb_entities.json           # Entity embeddings
│   ├── vdb_chunks.json             # Chunk embeddings
│   └── ...
├── finder_train.parquet            # FinDER benchmark questions
├── .env                            # API keys (create this)
└── README.md                       # This file
```

---

## Future Work

### Batch Indexing
Cell 4 indexes every ticker listed in `TICKERS` (the ten priority tickers by
default). To index more companies, set `TICKERS` to a longer comma-separated
list of tickers from `parsed_10k_documents.json`.

**Note**: Full indexing takes several hours on GPU (497 companies × ~2-5 min/company)

### General Framework
The current system is ready to work with any 10-K corpus:
1. Replace `10k/` with your HTML files
2. Run `html_parser.py`
3. Update `PRIORITY_TICKERS` or index all documents
4. Query with your own questions

**Status**: Framework complete, pending evaluation metrics and debugging

### Evaluation Metrics
- Accuracy vs. FinDER ground truth answers
- Context relevance (reranking effectiveness)
- Citation quality (metadata preservation)
- Retrieval recall@k (entities, relations, chunks)

---

## Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**Key packages**:
- `lightrag-hku`: Knowledge graph + hybrid RAG
- `sentence-transformers`: BGE embeddings and reranker
- `torch`: GPU acceleration (CUDA)
- `beautifulsoup4`: HTML parsing
- `pandas`, `pyarrow`: FinDER benchmark loading

**GPU Requirements**:
- Recommended: NVIDIA GPU with 8GB+ VRAM
- CPU fallback: Works but ~10x slower for embeddings

---

## Troubleshooting

### CUDA Out of Memory
Reduce batch sizes in Cell 2:
```python
# Reranker batch size
batch_size=16  # Reduce from 32
```

### API Rate Limits
Mistral API has rate limits. Add retries or use a different LLM:
```python
from lightrag.llm.openai import openai_complete_if_cache

# Replace with OpenAI GPT-4 or local model
```

### Slow Indexing
Indexing is compute-intensive. Use GPU for embeddings:
```python
assert DEVICE == "cuda", "GPU recommended for fast indexing"
```

### Missing .env File
Create `.env` with your Mistral API key:
```bash
echo "MISTRAL_API_KEY=your_key_here" > .env
```

### HTML Parser Errors
If parsing fails, check encoding:
```python
# In html_parser.py, try different encodings
with open(file_path, 'r', encoding='latin-1') as f:
    html_content = f.read()
```

---

## Expected Results

### Good Results Look Like:
1. **Context retrieval**: 80-150K characters of relevant entities, relations, chunks
2. **Metadata present**: Ticker, company name, period in context
3. **Reranking effective**: Top 20 chunks are on-topic
4. **Citations clear**: Answers reference specific document sections

### Test Output (HON):
```
Question 1: Impact on supply chain risk...
✓ Retrieved context (150,659 chars)
✓ 97 entities, 128 relations, 12 chunks
✓ Reranked: 20 chunks from 43 original

Question 2: OpMargin (Income Before Taxes/Net Sales)...
✓ Retrieved context (129,927 chars)
✓ 79 entities, 156 relations, 11 chunks
✓ Reranked: 20 chunks from 43 original
```

---

## Citation

If you use this code or dataset, please cite:

**LightRAG**:
```bibtex
@article{lightrag2024,
  title={LightRAG: Simple and Fast Retrieval-Augmented Generation},
  author={Zirui Guo et al.},
  journal={arXiv preprint arXiv:2410.05779},
  year={2024}
}
```

**FinDER Dataset**:
```bibtex
@dataset{finder2024,
  title={FinDER: Financial Document Entity Recognition Dataset},
  author={Linq AI Research},
  url={https://huggingface.co/datasets/Linq-AI-Research/FinDER},
  year={2024}
}
```

---

## License

This project is for research and educational purposes. SEC 10-K filings are public domain. Check individual package licenses (LightRAG, BGE models, Mistral API terms).

---

## Contributing

Contributions welcome! Areas for improvement:
- Add more evaluation metrics
- Support for other document types (10-Q, 8-K)
- Optimize chunking strategies
- Multi-document reasoning
- UI for interactive querying

---

## Acknowledgments

- [LightRAG](https://github.com/HKUDS/LightRAG) for the hybrid RAG framework
- [FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER) for the benchmark dataset
- [BGE Models](https://huggingface.co/BAAI) for embeddings and reranking
- [Mistral AI](https://mistral.ai/) for the LLM API

---

**Questions?** Open an issue or check the notebook comments for detailed explanations.
