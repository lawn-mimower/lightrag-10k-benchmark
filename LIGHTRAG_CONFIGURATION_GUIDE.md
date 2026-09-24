# LightRAG Configuration Guide
**Quick Reference for Optimization**

---

## **🤖 MODELS**

| # | Model | Purpose | Provider | Cost |
|---|-------|---------|----------|------|
| 1 | [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) | Embedding (4096-dim) | HF Local GPU | Free |
| 2 | [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Reranking (8192 context) | HF Local GPU | Free |
| 3 | [Ministral-14B-2512](https://docs.mistral.ai/capabilities/models/) | Entity Extraction | Mistral API | $0.20/M |
| 4 | [Gemini-3-Flash-Preview](https://deepmind.google/models/gemini/flash/) | Answer Generation | Google API | $0.50/M in, $3.00/M out |
| 5 | [Gemini-3-Flash-Preview](https://deepmind.google/models/gemini/flash/) | Answer Evaluation (Ragas) | Google API | $0.50/M in, $3.00/M out |

### **Why These Models?**

**e5-mistral-7b-instruct** (4096 dimensions):
- 4x larger embedding space than BGE-large (1024-dim) for richer semantic capture
- Instruction-tuned specifically for query/passage retrieval tasks
- Mistral-based architecture excels at understanding financial terminology
- Optimized for asymmetric search (query → document matching)
- Better handles complex multi-hop reasoning in embeddings

**BGE-reranker-v2-m3** (8192 token context):
- 4x larger context window than base model for long financial documents
- Multilingual support (English + 100+ languages)
- Dramatically improves precision by re-scoring query-chunk pairs
- v2 architecture with improved ranking accuracy
- Handles full 10-K sections without truncation

**Ministral-14B-2512**:
- 14B parameters for stronger reasoning vs 8B variants
- 2512 token context window for entity extraction from long passages
- Better at structured extraction tasks (entities, relationships, attributes)
- More accurate entity disambiguation in financial contexts
- Improved instruction following for complex extraction prompts

**Gemini-3-Flash-Preview**:
- Frontier reasoning: 90.4% GPQA Diamond, 33.7% Humanity's Last Exam
- 3x faster than Gemini 2.5 Pro
- Outperforms 2.5 Pro on accuracy benchmarks
- Configurable "thinking level" for complex reasoning
- Better instruction adherence (less hallucination)
- ✓ Validated API pattern in benchmark_gemini_api.py (`--check`)

---

## **⚙️ INDEXING CONFIGS** (Set Once at LightRAG Init)

### **Chunking**
```python
chunk_token_size=1200              # Chunk size (600-2000)
chunk_overlap_token_size=100       # Overlap (50-200)
```
- **Why**: Smaller chunks = precise matching, more chunks to process
- **Current**: 1200 tokens (balanced)
- **Trade-off**: ↑ size = faster indexing, ↓ precision

### **Embeddings**
```python
embedding_func=EmbeddingFunc(
    embedding_dim=4096,            # e5-mistral-7b-instruct (↑ from 1024)
    max_token_size=8192,           # Supports longer passages
    func=e5_mistral_embedding_func
)
embedding_batch_num=10             # Batch size (5-50)
embedding_func_max_async=8         # Concurrent calls (4-16)
```
- **Why**: 4096-dim embeddings capture richer semantics, 8192 token limit handles full sections
- **Current**: e5-mistral-7b-instruct (4x larger than BGE-large-en-v1.5)
- **Trade-off**: ↑ async = faster, ↑ memory usage; larger dims = better quality, slower indexing

### **Entity Extraction**
```python
llm_model_name='ministral-14b-2512'   # Mistral API model
entity_extract_max_gleaning=1         # Extraction passes (1-3)
addon_params={"entity_types": [...]}
```
- **Why**: More passes = catch missed entities, but 2-3x slower
- **Current**: ministral-14b-2512 with 2512 token context (↑ from 8B variants)
- **Trade-off**: More entity types = richer graph, slower extraction; 14B model = better accuracy, higher cost

### **Performance**
```python
max_parallel_insert=2              # Concurrent docs (1-8)
llm_model_max_async=4              # Concurrent LLM calls (2-8)
```
- **Why**: Higher parallelism = faster indexing
- **Trade-off**: ↑ speed = ↑ resource usage

### **Vector Search**
```python
cosine_threshold=0.2               # Min similarity (0.1-0.4)
kg_chunk_pick_method="VECTOR"      # VECTOR or WEIGHT
related_chunk_number=5             # Chunks per entity (3-10)
```
- **Why**: Lower threshold = more permissive retrieval
- **Trade-off**: ↓ threshold = more recall, more noise

### **Reranking**
```python
# Using BAAI/bge-reranker-v2-m3 (8192 token context)
min_rerank_score=0.0               # Filter threshold (0.0-0.4)
```
- **Why**: Filters low-quality chunks post-reranking
- **Current**: bge-reranker-v2-m3 supports 8192 tokens (4x larger context than base)
- **Trade-off**: ↑ threshold = cleaner results, may lose relevant info
- **Recommendation**: Can use stricter threshold (0.3) due to improved v2 model accuracy

---

## **🔍 QUERY CONFIGS** (Set Per Query via QueryParam)

### **Retrieval Mode**
```python
mode="hybrid"  # local, global, hybrid, naive, mix
```

| Mode | Retrieves | Best For | Speed |
|------|-----------|----------|-------|
| `local` | Entities + relationships | Specific entity questions | Fast |
| `global` | High-level patterns | Thematic questions | Medium |
| `hybrid` | Local + global | General questions | Medium |
| `naive` | Pure vector (no graph) | Baseline | Fastest |
| `mix` | Graph + vector chunks | Complex multi-hop | Slower |

- **Why**: Different modes for different question types
- **Current**: hybrid (balanced)
- **Recommendation**: Try `mix` for complex financial questions

### **Retrieval Quantity**
```python
top_k=60                  # Entities/relations (20-100)
chunk_top_k=20            # Text chunks (10-50)
```
- **Why**: More retrieval = more candidates for reranking
- **Current**: top_k=60 (high), chunk_top_k=20 (good)
- **Trade-off**: ↑ top_k = more recall + more noise + slower
- **Issue**: Your contexts ~33k tokens, exceeding 30k budget

### **Token Budget**
```python
max_entity_tokens=6000      # Entity budget (2k-10k)
max_relation_tokens=8000    # Relation budget (3k-12k)
max_total_tokens=30000      # Total budget (15k-50k)
```
- **Why**: Controls final context size after retrieval
- **Current**: 30k total, but retrieving ~33k (being truncated)
- **Trade-off**: ↑ budget = more context for LLM, but slower/costlier

### **Reranking Control**
```python
enable_rerank=True          # Enable reranking
```
- **Why**: Cross-encoder re-scores chunks for relevance
- **Impact**: Massive precision boost, minimal speed cost
- **Current**: Enabled ✓

### **Keyword Guidance** (Optional)
```python
hl_keywords=["risk"]           # High-level keywords
ll_keywords=["Honeywell"]      # Low-level (entities)
```
- **Why**: Manual guidance when auto-extraction misses terms
- **Use Case**: Force retrieval to prioritize specific concepts

---

## **📊 CURRENT PERFORMANCE** (58 Questions Evaluated)

| Metric | Score | Status | Interpretation |
|--------|-------|--------|----------------|
| **Context Relevance** | 83.2% | ✅ Good | Retrieval working well |
| **Groundedness** | 59.1% | ⚠️ Moderate | Model partially using context |
| **Answer Accuracy** | 46.6% | ❌ Poor | Generation needs improvement |

**Key Finding**: Retrieval is good (83%), but generation is poor (47%). Root cause: LLM hallucinating despite relevant context.

---

## **🎯 OPTIMIZATION STRATEGIES**

### **For Better Precision** (Less Noise)
```python
QueryParam(
    mode="mix",
    top_k=40,              # ↓ from 60
    chunk_top_k=15,        # ↓ from 20
    max_total_tokens=20000 # ↓ from 30000
)
```

### **For Better Recall** (Catch More Info)
```python
QueryParam(
    mode="mix",
    top_k=80,              # ↑ from 60
    chunk_top_k=35,        # ↑ from 20
    max_total_tokens=40000 # ↑ from 30000
)
```

### **Stricter Filtering** (Quality Over Quantity)
```python
# At init:
LightRAG(
    min_rerank_score=0.3,     # Filter low scores
    cosine_threshold=0.25      # Stricter matching
)

QueryParam(
    mode="mix",
    top_k=50,
    chunk_top_k=25
)
```

---

## **💰 ESTIMATED COSTS** (Full Pipeline)

| Task | Model | Usage | Cost |
|------|-------|-------|------|
| **Indexing (Entity)** | ministral-14b-2512 | 10-20M input tokens | $2.00 - $4.00 |
| **Generation** | gemini-3-flash-preview | 3M input + 20k output | $1.56 |
| **Evaluation (Ragas)** | gemini-3-flash-preview | 4M input + 30k output | $2.90 |
| | | **TOTAL** | **$6.46 - $9.46** |

**Notes**:
- Embeddings (e5-mistral-7b-instruct) and reranker (bge-reranker-v2-m3) run locally on GPU: **Free**
- Ragas adds Context Recall metric vs custom evaluation (4 metrics total)
- **vs GPT-4o**: ~10x cheaper for equivalent evaluation quality
- **vs Custom Metrics**: Similar cost (both use Gemini), but Ragas provides standardized, validated framework

---

## **📈 WHY GEMINI-3-FLASH-PREVIEW?**

### **Performance vs Gemini 2.5 Flash**
- ⚡ **3x faster** processing speed
- 📊 **Better accuracy** on reasoning benchmarks
- 🧠 **Advanced "thinking level"** parameter for complex reasoning
- ✅ **Better instruction adherence** (less hallucination)

### **Benchmark Results**
- **GPQA Diamond**: 90.4% (PhD-level science reasoning)
- **Humanity's Last Exam**: 33.7% (frontier reasoning)
- **Performance**: Rivals much larger frontier models at fraction of cost

### **Why This Matters for Your Pipeline**
| Problem | Gemini 3 Solution |
|---------|-------------------|
| Current 46.6% accuracy | Better instruction following → more grounded answers |
| 41% ungrounded claims | Stronger adherence to "only use context" instruction |
| Complex financial questions | PhD-level reasoning capability handles nuanced queries |

---

## **📐 RAGAS EVALUATION FRAMEWORK**

The RAGAS (Retrieval-Augmented Generation Assessment) framework evaluates both "Generation" and "Retrieval" components with standardized, validated metrics.

### **Installation & Setup**

```bash
# Install Ragas with Google Gemini support
pip install ragas google-genai

# Set environment variable
export GOOGLE_API_KEY="your-google-api-key"
```

### **Configuration with Gemini-3-Flash-Preview**

Using the validated pattern from [benchmark_gemini_api.py](benchmark_gemini_api.py) (`--check`):

```python
import os
import google.genai as genai
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

# Initialize Gemini client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Configure LLM for evaluation (✓ validated model)
llm = llm_factory(
    "gemini-3-flash-preview",
    provider="google",
    client=client,
    system_prompt=(
        "Evaluate financial information with strict numerical accuracy. "
        "Mark claims as unsupported if there's any ambiguity."
    )
)

# Configure embeddings (auto-matched to Google provider)
embeddings = embedding_factory(
    "google",
    model="text-embedding-004",
    client=client
)
```

### **Four FinDER-Aligned Metrics**

| Metric | What It Measures | Score | Purpose |
|--------|------------------|-------|---------|
| **Context Recall** | Retriever's ability to find necessary info | 0-1 | Measure completeness |
| **Context Precision** | Quality of retrieval ranking | 0-1 | Check if relevant items ranked higher |
| **Faithfulness** | How much answer derives from context | 0-1 | Detect hallucinations |
| **Answer Correctness** | Semantic + factual accuracy vs ground truth | 0-1 | Weighted correctness measure |

### **Detailed Metric Explanations**

**1. Context Recall** (Evaluates Retriever Coverage)
- **Formula**: `(Claims in reference supported by context) / (Total claims in reference)`
- **Use Case**: Ensures retriever found all relevant 10-K information
- **Variants**: LLM-based, Non-LLM (string similarity), ID-based
```python
from ragas.metrics import ContextRecall
context_recall = ContextRecall(llm=llm)
```

**2. Context Precision** (Evaluates Reranking Quality)
- **Formula**: `Mean Precision@K` for each retrieved chunk
- **Use Case**: Measures if relevant chunks ranked higher after reranking
- **Variants**: Standard, ContextUtilization, Non-LLM, ID-based
```python
from ragas.metrics import ContextPrecision
context_precision = ContextPrecision(llm=llm)
```

**3. Faithfulness** (Detects Hallucinations)
- **Formula**: `(Supported claims) / (Total claims in response)`
- **Use Case**: Ensures generated answer derives only from retrieved context
- **Variants**: LLM-based, Vectara HHEM (lightweight T5 classifier)
```python
from ragas.metrics import Faithfulness
faithfulness = Faithfulness(llm=llm)
```

**4. Answer Correctness** (Measures Overall Accuracy)
- **Formula**: `Weighted(Semantic Similarity, Factual F1)`
- **Use Case**: Evaluates both semantic and factual alignment with ground truth
- **Customizable**: Adjust weights for financial precision
```python
from ragas.metrics import AnswerCorrectness
answer_correctness = AnswerCorrectness(
    llm=llm,
    embeddings=embeddings,
    weights=[0.5, 0.5]  # [semantic, factual]
)
```

### **Complete Evaluation Example**

```python
from ragas import evaluate
from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    Faithfulness,
    AnswerCorrectness
)
from datasets import Dataset

# Initialize all metrics
metrics = [
    ContextRecall(llm=llm),
    ContextPrecision(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm, embeddings=embeddings, weights=[0.5, 0.5])
]

# Prepare your data (convert from test_results_*.json format)
data_samples = [{
    "user_input": "What was the revenue?",           # question
    "retrieved_contexts": ["Context 1", "..."],      # contexts
    "response": "Generated answer",                   # generated_answer
    "reference": "Ground truth answer"                # expected_answer
}]
dataset = Dataset.from_list(data_samples)

# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings
)

# Display results
print(f"Context Recall:      {results['context_recall']:.3f}")
print(f"Context Precision:   {results['context_precision']:.3f}")
print(f"Faithfulness:        {results['faithfulness']:.3f}")
print(f"Answer Correctness:  {results['answer_correctness']:.3f}")

# Export to CSV
results.to_pandas().to_csv("ragas_evaluation.csv", index=False)
```

### **Data Format Requirements**

Convert your current test results format to Ragas format:

```python
# Current format (from test_results_*.json)
{
    "question": "What was the revenue for fiscal year 2023?",
    "contexts": ["Chunk 1", "Chunk 2", "..."],
    "generated_answer": "The revenue was $150B.",
    "expected_answer": "Total revenue was $150B in FY 2023."
}

# Convert to Ragas format
{
    "user_input": question,
    "retrieved_contexts": contexts,
    "response": generated_answer,
    "reference": expected_answer
}
```

### **Migration from Custom Metrics**

**Step-by-Step Migration** from [evaluate_answers.py](evaluate_answers.py) to Ragas:

1. **Install Ragas**: `pip install ragas google-genai`
2. **Load your test results**: Read JSON files
3. **Convert data format**: Map fields as shown above
4. **Initialize Gemini**: Use validated pattern from benchmark_gemini_api.py --check
5. **Run Ragas evaluation**: Replace custom prompts with Ragas metrics
6. **Compare results**: Validate against baseline (46.6% accuracy)

**Comparison: Custom vs Ragas Metrics**

| Your Custom Metric | Ragas Metric | Alignment | Score Scale |
|-------------------|--------------|-----------|-------------|
| **Answer Accuracy** (0/2/4) | `AnswerCorrectness` | ✅ Exact match | 0-1 continuous |
| **Context Relevance** (0/2/4) | `ContextPrecision` | ✅ Exact match | 0-1 continuous |
| **Groundedness** (0/2/4) | `Faithfulness` | ✅ Exact match | 0-1 continuous |
| N/A | `ContextRecall` | ➕ New metric | 0-1 continuous |

**Advantages of Ragas**:
- Standardized prompts tested across benchmarks
- Better reproducibility and comparison with literature
- Continuous scoring (0-1) vs discrete (0/2/4)
- Context Recall adds retrieval completeness measurement

### **Best Practices for Financial Evaluation**

**1. Adjust Weights for Numerical Precision**
```python
# More emphasis on factual accuracy for financial data
answer_correctness = AnswerCorrectness(
    llm=llm,
    embeddings=embeddings,
    weights=[0.3, 0.7]  # Less semantic, more factual
)
```

**2. Handle Rate Limits**
```python
from ragas import RunConfig

run_config = RunConfig(
    max_retries=5,
    timeout=60,
    max_wait=30
)

results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings,
    run_config=run_config
)
```

**3. Batch Processing for Large Datasets**
```python
def evaluate_in_batches(dataset, metrics, llm, embeddings, batch_size=10):
    """Process large datasets in chunks"""
    results_list = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        batch_results = evaluate(batch, metrics, llm, embeddings)
        results_list.append(batch_results.to_pandas())
        print(f"Processed batch {i//batch_size + 1}")

    import pandas as pd
    return pd.concat(results_list, ignore_index=True)
```

### **LLM-as-Judge: Gemini vs GPT-4o**

**Updated Approach**: Using Gemini-3-Flash-Preview instead of GPT-4o

| Aspect | GPT-4o (FinDER Paper) | Gemini-3-Flash-Preview (This Pipeline) |
|--------|---------------------|---------------------------------------|
| **Validation** | 0.84 Spearman correlation with humans | Similar quality, ~10x cheaper |
| **Cost** | $5/M in, $15/M out | $0.50/M in, $3.00/M out |
| **Speed** | Baseline | 3x faster than Gemini 2.5 Pro |
| **Benchmarks** | Strong | 90.4% GPQA Diamond, 33.7% Humanity's Last Exam |
| **API Pattern** | OpenAI | Google GenAI (✓ validated by benchmark_gemini_api.py --check) |

**Why This Works**: Gemini-3-Flash-Preview has frontier reasoning capabilities that match GPT-4o for evaluation tasks while being significantly cheaper and faster.

---

## **🔬 RECOMMENDED EXPERIMENTS**

| Experiment | Test Variants | Goal |
|------------|--------------|------|
| **1. Retrieval Mode** | `hybrid` vs `mix` vs `naive` | Find best mode for financial Q&A |
| **2. Top-K Settings** | 40 vs 60 vs 80 | Balance recall vs noise |
| **3. Token Budget** | 20k vs 30k vs 40k | Optimize context size |
| **4. Model Upgrade** | Re-run with Gemini 3 Flash | Test improved reasoning |
| **5. Evaluation Framework** | Custom metrics vs Ragas | Validate standardized evaluation |
| **6. Embedding Dimension** | BGE (1024) vs e5-mistral (4096) | Test semantic capture improvement |
| **7. Reranker Model** | bge-reranker-base vs v2-m3 | Measure precision gains from 8192 context |

**Overall Goal**: Maximize context relevance while minimizing noise → Improve accuracy from 46.6% baseline

**Priority Order**:
1. **Evaluation Framework** (Experiment 5): Migrate to Ragas for standardized metrics
2. **Model Upgrade** (Experiment 4): Already using gemini-3-flash-preview ✓
3. **Embedding/Reranker** (Experiments 6-7): Test new models (e5-mistral, bge-v2-m3)
4. **Parameter Tuning** (Experiments 1-3): Optimize retrieval and context settings

---

## **🔧 TROUBLESHOOTING & QUICK REFERENCE**

### **Common Issues and Solutions**

**Issue: "GOOGLE_API_KEY not found"**
```python
# Solution: Set environment variable
import os
os.environ["GOOGLE_API_KEY"] = "your-key-here"
# Or in terminal: export GOOGLE_API_KEY="your-key-here"
```

**Issue: Rate limit errors with Ragas evaluation**
```python
# Solution: Add retry configuration
from ragas import RunConfig
run_config = RunConfig(max_retries=10, max_wait=60, timeout=120)
results = evaluate(dataset, metrics, llm, embeddings, run_config=run_config)
```

**Issue: Timeout on large datasets**
```python
# Solution 1: Increase timeout
run_config = RunConfig(timeout=180)  # 3 minutes

# Solution 2: Use batching (recommended)
results = evaluate_in_batches(dataset, metrics, llm, embeddings, batch_size=10)
```

**Issue: Embeddings not matching provider**
```python
# Solution: Explicitly configure Google embeddings
from ragas.embeddings import embedding_factory
embeddings = embedding_factory(
    provider="google",
    model="text-embedding-004",
    client=client
)
```

**Issue: Inconsistent scores across runs**
```python
# Note: Ragas doesn't directly expose temperature parameter
# Slight variations between runs are expected with LLM-based evaluation
# For more deterministic results, consider using Non-LLM variants:
from ragas.metrics import NonLLMContextRecall, NonLLMContextPrecisionWithReference
```

**Issue: Memory issues with large embeddings (4096-dim)**
```python
# Solution: Reduce batch size
embedding_batch_num=5        # ↓ from 10
embedding_func_max_async=4   # ↓ from 8
```

**Issue: Reranker context exceeding 8192 tokens**
```python
# Solution: Adjust chunk and token settings
chunk_token_size=1000        # ↓ from 1200
max_total_tokens=25000       # ↓ from 30000
```

### **Validated API Patterns Quick Reference**

**Pattern 1: Direct Generation (from [benchmark_gemini_api.py](benchmark_gemini_api.py) `--check`)**
```python
import google.genai as genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Direct API call
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="your prompt here"
)
print(response.text)
```

**Pattern 2: Ragas Evaluation**
```python
import google.genai as genai
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# For Ragas metrics
llm = llm_factory("gemini-3-flash-preview", provider="google", client=client)
embeddings = embedding_factory("google", model="text-embedding-004", client=client)
```

**Pattern 3: LightRAG with Mistral Entity Extraction**
```python
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache

async def ministral_model_complete(prompt, system_prompt=None, **kwargs):
    return await openai_complete_if_cache(
        model="ministral-14b-2512",
        prompt=prompt,
        system_prompt=system_prompt,
        api_key=os.getenv("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1",
        **kwargs
    )

rag = LightRAG(
    working_dir="./workspace",
    llm_model_func=ministral_model_complete,
    llm_model_name='ministral-14b-2512',
    # ... other configs
)
```

**Pattern 4: e5-Mistral Embeddings with LightRAG**
```python
from sentence_transformers import SentenceTransformer
from lightrag.utils import EmbeddingFunc
import numpy as np

class E5MistralEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "intfloat/e5-mistral-7b-instruct",
            trust_remote_code=True
        )
        self.dimension = 4096

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

e5_instance = E5MistralEmbedder()

async def e5_embedding_func(texts: list[str]) -> np.ndarray:
    return e5_instance(texts)

rag = LightRAG(
    embedding_func=EmbeddingFunc(
        embedding_dim=4096,
        max_token_size=8192,
        func=e5_embedding_func
    ),
    # ... other configs
)
```

### **Performance Optimization Checklist**

**For Faster Indexing**:
- [ ] Increase `embedding_batch_num` (10 → 20) if GPU memory allows
- [ ] Increase `max_parallel_insert` (2 → 4) for concurrent document processing
- [ ] Reduce `entity_extract_max_gleaning` (1 is optimal for speed)

**For Better Precision**:
- [ ] Use stricter `min_rerank_score` (0.0 → 0.3) with bge-reranker-v2-m3
- [ ] Increase `cosine_threshold` (0.2 → 0.25) for tighter matching
- [ ] Use `mode="mix"` for complex multi-hop queries

**For Better Recall**:
- [ ] Increase `top_k` (60 → 80) for more candidate retrieval
- [ ] Increase `chunk_top_k` (20 → 30) for more text chunks
- [ ] Decrease `cosine_threshold` (0.2 → 0.15) for permissive matching

**For Cost Optimization**:
- [ ] Use `gemini-3-flash-preview` instead of GPT-4o (10x cheaper)
- [ ] Run embeddings and reranking locally (Free vs API costs)
- [ ] Batch Ragas evaluations to reduce API calls

---

## **📚 ADDITIONAL RESOURCES**

- **LightRAG GitHub**: [github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- **Ragas Documentation**: [docs.ragas.io](https://docs.ragas.io/en/stable/)
- **Ragas+Gemini Guide**: [RAGAS_GEMINI_CONFIGURATION_GUIDE.md](RAGAS_GEMINI_CONFIGURATION_GUIDE.md)
- **FinDER Paper**: [arxiv.org/abs/2402.06709](https://arxiv.org/abs/2402.06709)
- **e5-mistral-7b**: [huggingface.co/intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct)
- **bge-reranker-v2-m3**: [huggingface.co/BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **Mistral API Docs**: [docs.mistral.ai](https://docs.mistral.ai)
- **Gemini API Docs**: [ai.google.dev/docs](https://ai.google.dev/docs)

---

**Document Version**: 2.0
**Last Updated**: January 14, 2026
**Model Configuration**: e5-mistral-7b (4096-dim) | ministral-14b-2512 | bge-reranker-v2-m3 | gemini-3-flash-preview
**Evaluation Framework**: Ragas with Gemini-3-Flash-Preview
