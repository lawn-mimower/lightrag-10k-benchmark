# Custom RAGAS Metrics Implementation

This is a custom implementation of RAGAS (Retrieval Augmented Generation Assessment) metrics using Google Gemini, without relying on the RAGAS codebase.

## Overview

RAGAS is a framework for evaluating RAG (Retrieval Augmented Generation) systems. This implementation replicates the methodology of 4 key RAGAS metrics:

1. **Context Recall** - Measures what % of ground truth claims are supported by retrieved context
2. **Context Precision** - Measures if relevant chunks are ranked higher than irrelevant ones
3. **Faithfulness** - Measures what % of answer claims are supported by retrieved context
4. **Answer Correctness** - Combines factual F1 score + semantic similarity vs ground truth

## Installation

```bash
pip install google-generativeai tqdm
```

## Setup

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
GEMINI_API_KEY=your-api-key-here
```

## Usage

### 1. Single Example Evaluation

```python
from custom_ragas_metrics import CustomRAGASMetrics, RAGEvaluation

# Create evaluation data
eval_data = RAGEvaluation(
    question="Where was Albert Einstein born?",
    answer="Albert Einstein was born in Ulm, Germany on March 14, 1879.",
    contexts=[
        "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
        "Einstein's family moved to Munich when he was an infant.",
    ],
    ground_truth="Albert Einstein was born in Ulm, Germany on March 14, 1879."
)

# Initialize metrics
metrics = CustomRAGASMetrics()

# Calculate individual metrics
faithfulness = metrics.faithfulness(eval_data)
print(f"Faithfulness: {faithfulness['score']:.3f}")

answer_correctness = metrics.answer_correctness(eval_data)
print(f"Answer Correctness: {answer_correctness['score']:.3f}")

# Or calculate all at once
all_results = metrics.evaluate_all(eval_data)
```

### 2. Batch Evaluation

Run the example script on a JSON file:

```bash
python evaluate_with_custom_ragas.py input_results.json output_evaluation.json
```

Expected input format:
```json
[
    {
        "question": "What is...",
        "answer": "The answer is...",
        "contexts": ["Context 1", "Context 2"],
        "ground_truth": "The correct answer is..."
    }
]
```

### 3. Demo Mode

Run without arguments to see a demonstration:

```bash
python evaluate_with_custom_ragas.py
```

## Metrics Explained

### 1. Context Recall

**What it measures:** How much of the ground truth information was successfully retrieved in the context.

**Formula:**
```
Context Recall = (# claims in ground truth supported by context) / (total claims in ground truth)
```

**How it works:**
1. Extracts individual claims from the ground truth answer
2. Checks if each claim can be inferred from the retrieved contexts
3. Calculates the proportion of supported claims

**When to use:**
- Evaluate if your retrieval system is finding all relevant information
- Lower scores indicate missing relevant documents

### 2. Context Precision

**What it measures:** Whether relevant chunks are ranked higher than irrelevant ones in your retrieved results.

**Formula:**
```
Context Precision = Σ(Precision@k × relevance_k) / total_relevant_items
```

**How it works:**
1. For each retrieved chunk, determines if it's relevant to answering the question
2. Calculates precision at each position k in the ranking
3. Averages the precision scores, weighted by relevance

**When to use:**
- Evaluate if your ranking/reranking is working well
- Higher scores mean relevant chunks appear first

### 3. Faithfulness

**What it measures:** How factually consistent the generated answer is with the retrieved context (i.e., no hallucinations).

**Formula:**
```
Faithfulness = (# claims in answer supported by context) / (total claims in answer)
```

**How it works:**
1. Extracts individual claims from the generated answer
2. Verifies if each claim can be inferred from the retrieved contexts
3. Calculates the proportion of supported claims

**When to use:**
- Detect hallucinations in generated answers
- Lower scores indicate the model is adding unsupported information

### 4. Answer Correctness

**What it measures:** How accurate the generated answer is compared to the ground truth.

**Formula:**
```
Answer Correctness = (weight_F1 × F1_score) + (weight_semantic × semantic_similarity)
```

Where:
- **F1 Score** = |TP| / (|TP| + 0.5 × (|FP| + |FN|))
  - TP = facts in both answer and ground truth
  - FP = facts only in answer
  - FN = facts only in ground truth
- **Semantic Similarity** = cosine similarity or LLM-based comparison

**How it works:**
1. Extracts facts from both generated answer and ground truth
2. Categorizes facts as TP, FP, or FN
3. Calculates F1 score for factual overlap
4. Calculates semantic similarity between answers
5. Combines both with equal weights (0.5 each by default)

**When to use:**
- Evaluate overall answer quality against known correct answers
- Combines factual accuracy with semantic equivalence

## Comparison with RAGAS

| Feature | This Implementation | RAGAS Framework |
|---------|---------------------|-----------------|
| Dependencies | Only `google-generativeai` | `ragas`, `langchain`, embeddings |
| Model | Gemini (configurable) | OpenAI/HuggingFace/others |
| Customization | Full control over prompts | Framework-defined |
| Cost | Pay per Gemini API call | Depends on model choice |
| Speed | Depends on Gemini API | Depends on chosen models |

## Cost Considerations

Each metric requires multiple LLM calls:
- **Faithfulness**: 1 call to extract claims + N calls to verify (where N = number of claims)
- **Context Recall**: 1 call to extract claims + N calls to verify
- **Context Precision**: K calls to verify relevance (where K = number of chunks)
- **Answer Correctness**: 2-3 calls (extract facts, compare, similarity)

For batch evaluations, consider:
- Using Gemini Flash for lower costs
- Caching contexts when possible
- Processing in batches with rate limiting

## Customization

### Change the Model

```python
metrics = CustomRAGASMetrics(model_name="gemini-3-flash-preview")
```

### Adjust Weights for Answer Correctness

Edit the weights in `answer_correctness()` method:

```python
# In custom_ragas_metrics.py
weight_factual = 0.7  # More weight on factual accuracy
weight_semantic = 0.3  # Less weight on semantic similarity
```

### Modify Prompts

All prompts are defined inline in the methods. You can edit them to:
- Change the evaluation criteria
- Add more context or examples
- Adjust the output format

## Troubleshooting

### API Key Issues

```python
# Pass API key directly
metrics = CustomRAGASMetrics(api_key="your-key-here")
```

### JSON Parsing Errors

The implementation uses JSON mode for structured output. If you get parsing errors:
1. Check your Gemini API quota
2. Try a different model (e.g., gemini-3-flash-preview)
3. Add error handling in your code

### Rate Limiting

Add delays between calls for large batches:

```python
import time
for record in records:
    results = metrics.evaluate_all(eval_data)
    time.sleep(1)  # 1 second delay
```

## References

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/vibrantlabsai/ragas)
- [Google Gemini API](https://ai.google.dev/)

## License

This is a custom implementation inspired by RAGAS methodology. Please refer to the original RAGAS project for their licensing terms.
