#!/usr/bin/env python3
"""
FinDER Benchmark with LightRAG

Benchmarks the LightRAG framework on the FinDER financial dataset using:
- LLM: Llama-3.2-3B-Instruct (llama-cpp-python, CPU or GPU inference)
- Embedder: Qwen3-0.6B (sentence-transformers)
- Dataset: FinDER train set (10-K financial filings)

Profiles (--device):
- cpu (default): strict generation parameters for small (3B) models, 8K
  context, first 10 contexts, sequential indexing, single-threaded LightRAG
- gpu: all layers on the GPU with the full 128K context, all contexts,
  batch indexing with 8-way parallelism and no gleaning retries

--query-only skips indexing and queries the index already stored in
--working-dir (32K context on CPU).
"""

import argparse
import pandas as pd
import numpy as np
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import os
import shutil

# LightRAG imports
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# Model imports
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "./models/Llama-3.2-3B-Instruct.Q8_0.gguf"
EMBEDDER_PATH = "./models/qwen3-0.6b"
DATA_PATH = "./data/finder_train.parquet"

# Flag for query detection (used by embedder to differentiate queries from documents)
QUERY_FLAG = "benchmark::query::"

DEVICE_PROFILES = {
    "cpu": {
        # llama.cpp
        "n_ctx": 8192,  # Large context for 10-K documents
        "n_threads": 8,  # Leave headroom for OS
        "n_gpu_layers": 0,  # CPU-only
        # sentence-transformers
        "embedder_device": "cpu",  # Force CPU usage (critical for CPU-only benchmark)
        # LightRAG
        "llm_model_name": "Llama-3.2-3B-Instruct",
        "llm_model_max_async": 1,  # Single-threaded to avoid CPU thrashing
        "chunk_token_size": 1024,  # Reduced from 1200 for easier processing
        "entity_extract_max_gleaning": 2,  # Allow 2 retry attempts for formatting (crucial for 3B models)
        "max_parallel_insert": 1,  # Concurrency limits for CPU
        "batch_insert": False,  # Insert contexts one at a time
        # Run defaults
        "num_docs": 10,  # Number of documents to index (limited for CPU benchmarking)
        "working_dir": "./finder_benchmark_workdir1",
        "output": "finder_lightrag_results.csv",
    },
    "gpu": {
        "n_ctx": 131072,  # Maximum context (128K) - model's native training context
        "n_threads": 4,  # Reduced threads since GPU handles compute
        "n_gpu_layers": -1,  # ALL layers on GPU
        "embedder_device": "cuda",  # GPU acceleration
        "llm_model_name": "Llama-3.2-3B-Instruct-GPU",
        "llm_model_max_async": 8,  # Increased parallelism for GPU
        "chunk_token_size": 1200,  # Balanced: not too small (more chunks) or large (harder extraction)
        "entity_extract_max_gleaning": 0,  # Disable gleaning retries (massive speedup)
        "max_parallel_insert": 8,  # Concurrency - maximized for GPU
        "batch_insert": True,  # One ainsert() call with every context
        "num_docs": None,  # No document limit - process all unique contexts
        "working_dir": "./finder_benchmark_workdir_gpu",
        "output": "finder_lightrag_results_gpu.csv",
    },
}

# Querying an existing index on CPU needs a larger window for the retrieved context
QUERY_ONLY_CPU_N_CTX = 32768

# ============================================================================
# LLM WRAPPER - Strict Parameters for a 3B Model
# ============================================================================

def format_llama3_prompt(prompt: str, system_prompt: str = None, history_messages=None) -> str:
    """Render system prompt, prior turns and the new user prompt with the Llama 3 chat template."""
    parts = []
    if system_prompt:
        parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>")
    for message in history_messages or []:
        role = message.get("role", "user")
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{message.get('content', '')}<|eot_id|>")
    parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


class LlamaCppWrapper:
    """
    LLM wrapper for llama-cpp-python.

    Optimized for 3B models with strict generation parameters to ensure
    proper formatting for LightRAG's entity extraction. The defaults are the
    CPU profile; pass n_gpu_layers=-1 to load every layer onto the GPU.
    """

    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: int = 8, n_gpu_layers: int = 0):
        print(f"Loading LLM from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        print("✓ LLM loaded successfully")
        print(f"  Context window: {n_ctx:,} tokens")
        print(f"  GPU layers: {'ALL (-1)' if n_gpu_layers == -1 else n_gpu_layers}")

    async def __call__(self, prompt: str, system_prompt: str = None, history_messages=None, **kwargs) -> str:
        """
        Async generation method compatible with LightRAG.

        Args:
            prompt: User prompt (may contain QUERY_FLAG)
            system_prompt: System instructions
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            # Strip the query flag if present
            if prompt.startswith(QUERY_FLAG):
                prompt = prompt[len(QUERY_FLAG):]

            # Format using Llama 3 chat template (entity-extraction gleaning
            # sends the previous extraction turn as history_messages)
            formatted_prompt = format_llama3_prompt(prompt, system_prompt, history_messages)

            # Run synchronous generation in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm(
                    formatted_prompt,
                    max_tokens=2048,  # Increased for dense financial text
                    temperature=0.0,  # Maximum determinism for formatting
                    top_p=1.0,  # Disable nucleus sampling
                    repeat_penalty=1.1,  # Prevent loops (common in 3B models)
                    stop=["<|eot_id|>"],
                    echo=False
                )
            )

            return response['choices'][0]['text'].strip()

        except Exception as e:
            print(f"⚠ LLM generation error: {e}")
            # Return minimal valid response to allow pipeline to continue
            return "Error: Unable to generate response."


# ============================================================================
# EMBEDDER WRAPPER - Instruction-Aware for Queries
# ============================================================================

class QwenEmbedderWrapper:
    """
    Instruction-aware embedding wrapper for Qwen3.

    Uses the QUERY_FLAG to determine when to apply instruction prefix:
    - Queries (with flag): Apply "Instruct: Given a financial query..." prefix
    - Documents/Entities (no flag): Encode as-is
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        print(f"Loading embedder from {model_path}...")
        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            device=device
        )
        print(f"✓ Embedder loaded (dimension: {self.model.get_sentence_embedding_dimension()})")
        print(f"  Device: {self.model.device}")

    async def __call__(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts with instruction-awareness.

        Args:
            texts: List of texts to embed (may contain QUERY_FLAG)

        Returns:
            Embedding matrix (n_texts x embedding_dim)
        """
        try:
            processed_texts = []

            for text in texts:
                # Check for query flag
                if text.startswith(QUERY_FLAG):
                    # Strip flag and apply instruction prefix for queries
                    query_text = text[len(QUERY_FLAG):]
                    processed_text = f"Instruct: Given a financial query, retrieve relevant 10-K passages.\nQuery: {query_text}"
                else:
                    # Documents and entities: encode as-is
                    processed_text = text

                processed_texts.append(processed_text)

            # Generate embeddings asynchronously (run sync encode in thread pool)
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    processed_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
            )

            return embeddings

        except Exception as e:
            print(f"⚠ Embedding error: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))


# ============================================================================
# INDEXING
# ============================================================================

async def index_sequentially(rag, contexts):
    """Insert contexts one at a time; failures are reported and skipped."""
    for idx, context in enumerate(contexts, 1):
        try:
            print(f"[{idx}/{len(contexts)}] Indexing context (length: {len(context)} chars)...")

            # Insert document (no flag needed - this is a document)
            await rag.ainsert(context)

            print(f"  ✓ Indexed successfully")
            print()

        except Exception as e:
            print(f"  ⚠ Error indexing document {idx}: {e}")
            print()
            continue


async def index_in_batch(rag, contexts):
    """Insert all contexts in one call (parallelised by max_parallel_insert)."""
    print(f"📦 Batch inserting {len(contexts)} documents...")
    print("   (This is much faster than sequential insertion)")
    print()

    try:
        # LightRAG's ainsert() accepts both single strings and lists
        await rag.ainsert(contexts)

        print(f"✓ Successfully indexed all {len(contexts)} documents")
        print()

    except Exception as e:
        print(f"⚠ Error during batch insertion: {e}")
        print("   Falling back to sequential insertion...")
        print()
        await index_sequentially(rag, contexts)


# ============================================================================
# MAIN BENCHMARK PIPELINE
# ============================================================================

async def main(settings):
    """Main benchmark execution pipeline."""
    profile = DEVICE_PROFILES[settings.device]
    working_dir = settings.working_dir

    title = "FinDER BENCHMARK WITH LIGHTRAG"
    if settings.device == "gpu":
        title += " - GPU ACCELERATED"
    if settings.query_only:
        title += " - QUERY ONLY"
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()

    # ------------------------------------------------------------------------
    # STEP 1: Prepare the working directory
    # ------------------------------------------------------------------------

    if settings.query_only:
        # Queries run against the index built by a previous run
        if not os.path.exists(working_dir):
            print(f"❌ ERROR: Working directory not found: {working_dir}")
            print("   Please run the indexing script first!")
            return

        print(f"✓ Found existing working directory: {working_dir}")
        print()
    else:
        # Safe-start: clear previous data
        if os.path.exists(working_dir):
            print(f"⚠ Found existing working directory: {working_dir}")
            shutil.rmtree(working_dir)
            print("✓ Cleared previous benchmark data for clean run")
            print()

        # Create fresh working directory
        Path(working_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # STEP 2: Initialize Models
    # ------------------------------------------------------------------------

    print(f"Initializing models ({settings.device.upper()})...")
    print()

    # Initialize LLM wrapper
    llm_wrapper = LlamaCppWrapper(
        settings.model_path,
        n_ctx=settings.n_ctx,
        n_threads=profile["n_threads"],
        n_gpu_layers=profile["n_gpu_layers"],
    )

    # Create LightRAG-compatible LLM function
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        return await llm_wrapper(prompt, system_prompt, history_messages=history_messages, **kwargs)

    # Initialize embedder wrapper
    embedder_wrapper = QwenEmbedderWrapper(settings.embedder_path, device=profile["embedder_device"])

    # Wrap in LightRAG's EmbeddingFunc
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=embedder_wrapper
    )

    print()

    # ------------------------------------------------------------------------
    # STEP 3: Load and Prepare FinDER Data
    # ------------------------------------------------------------------------

    print("Loading FinDER dataset...")
    df = pd.read_parquet(settings.data_path)

    print(f"✓ Loaded {len(df)} query-answer pairs")
    print(f"  Columns: {df.columns.tolist()}")
    print()

    # Extract and deduplicate contexts (10-K passages)
    # IMPORTANT: FinDER uses 'references' column (which is a numpy array) instead of 'context'
    print("Extracting unique contexts...")

    # 'references' is a numpy array, so we explode it to get individual strings
    unique_contexts = df['references'].explode().drop_duplicates().tolist()
    print(f"✓ Found {len(unique_contexts)} unique contexts")

    # The same selection is used when querying an existing index, so the
    # queries match the contexts that were indexed
    if settings.num_docs is not None:
        selected_contexts = unique_contexts[:settings.num_docs]
        print(f"✓ Selected first {settings.num_docs} contexts for benchmarking")
    else:
        selected_contexts = unique_contexts
        print(f"✓ Processing ALL {len(selected_contexts)} contexts (no limit)")

    # Filter dataframe to only queries matching selected contexts
    # This ensures 100% alignment between indexed documents and test queries
    filtered_df = df[df['references'].apply(lambda refs: any(ref in selected_contexts for ref in refs))].copy()
    if settings.max_queries is not None:
        filtered_df = filtered_df.head(settings.max_queries)
    print(f"✓ Filtered to {len(filtered_df)} queries matching selected contexts")
    print()

    if not settings.query_only:
        # Display sample
        print("=== Sample Data ===")
        print(f"Query: {filtered_df.iloc[0]['text'][:100]}...")
        print(f"Answer: {filtered_df.iloc[0]['answer'][:100]}...")
        print(f"References count: {len(filtered_df.iloc[0]['references'])}")
        print(f"First reference length: {len(filtered_df.iloc[0]['references'][0])} chars")
        print()

    # ------------------------------------------------------------------------
    # STEP 4: Initialize LightRAG
    # ------------------------------------------------------------------------

    print("Initializing LightRAG" + (" (loading existing index)..." if settings.query_only else "..."))

    rag = LightRAG(
        working_dir=working_dir,

        # LLM configuration
        llm_model_func=llm_model_func,
        llm_model_name=profile["llm_model_name"],
        llm_model_max_async=profile["llm_model_max_async"],

        # Embedding configuration
        embedding_func=embedding_func,

        # Chunking configuration (must match the indexing run when querying)
        chunk_token_size=profile["chunk_token_size"],
        chunk_overlap_token_size=100,

        # Entity extraction retry mechanism
        entity_extract_max_gleaning=profile["entity_extract_max_gleaning"],

        # Storage backends
        graph_storage="NetworkXStorage",
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",

        # Concurrency limits
        max_parallel_insert=profile["max_parallel_insert"]
    )

    # CRITICAL: Initialize storages (loads existing data when querying)
    await rag.initialize_storages()

    print("✓ LightRAG initialized successfully")
    print(f"  Working directory: {working_dir}")
    print(f"  Chunk size: {profile['chunk_token_size']} tokens")
    print(f"  Max gleaning retries: {profile['entity_extract_max_gleaning']}")
    print(f"  LLM concurrency: {profile['llm_model_max_async']}")
    print(f"  Insert parallelism: {profile['max_parallel_insert']}")
    print(f"  Context window: {settings.n_ctx:,} tokens")
    print()

    # ------------------------------------------------------------------------
    # STEP 5: Index Documents
    # ------------------------------------------------------------------------

    if not settings.query_only:
        print("=" * 70)
        print(f"INDEXING PHASE - Processing {len(selected_contexts)} documents")
        print("=" * 70)
        print()

        if profile["batch_insert"]:
            await index_in_batch(rag, selected_contexts)
        else:
            await index_sequentially(rag, selected_contexts)

        print("=" * 70)
        print("✓ INDEXING COMPLETE")
        print("=" * 70)
        print()

    # ------------------------------------------------------------------------
    # STEP 6: Query and Generate Answers
    # ------------------------------------------------------------------------

    print("=" * 70)
    print(f"QUERY PHASE - Processing {len(filtered_df)} queries")
    print("=" * 70)
    print()

    results = []

    for idx, row in filtered_df.iterrows():
        try:
            # FinDER uses 'text' for questions, not 'question'
            query = row['text']
            ground_truth_answer = row['answer']
            # 'references' is a numpy array, so we join them with separator
            ground_truth_context = "\n\n---\n\n".join(row['references'])

            print(f"[{len(results)+1}/{len(filtered_df)}] Query: {query[:80]}...")

            # IMPORTANT: Prepend query flag for proper embedding
            flagged_query = QUERY_FLAG + query

            # Execute hybrid query
            response = await rag.aquery(
                flagged_query,
                param=QueryParam(mode="hybrid")
            )

            # Handle None responses
            if response is None:
                response = "ERROR: No response generated (returned None)"
                print(f"  ⚠ Query returned None")
            else:
                print(f"  ✓ Generated answer (length: {len(response)} chars)")

            print()

            # Store results
            results.append({
                'original_query': query,
                'finder_ground_truth_answer': ground_truth_answer,
                'finder_ground_truth_context': ground_truth_context,
                'lightrag_generated_answer': response
            })

        except Exception as e:
            print(f"  ⚠ Error processing query: {e}")
            print()

            # Store partial result with error information
            results.append({
                'original_query': query if 'query' in locals() else "ERROR",
                'finder_ground_truth_answer': ground_truth_answer if 'ground_truth_answer' in locals() else "ERROR",
                'finder_ground_truth_context': ground_truth_context if 'ground_truth_context' in locals() else "ERROR",
                'lightrag_generated_answer': f"ERROR: {str(e)}"
            })
            continue

    print("=" * 70)
    print(f"✓ QUERY PHASE COMPLETE ({len(results)} results collected)")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------------
    # STEP 7: Save Results
    # ------------------------------------------------------------------------

    print("Saving results...")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = settings.output
    results_df.to_csv(output_path, index=False)

    print(f"✓ Results saved to {output_path}")
    print(f"  Total queries: {len(results_df)}")
    if len(results_df):
        errors = results_df['lightrag_generated_answer'].str.upper().str.startswith('ERROR')
        print(f"  Successful: {len(results_df[~errors])}")
        print(f"  Errors: {len(results_df[errors])}")
        print()

        # Display preview
        print("=== Results Preview ===")
        print(results_df.head(3).to_string())
    print()

    print("=" * 70)
    print("✓ BENCHMARK COMPLETE")
    print("=" * 70)
    if len(results_df):
        print()
        print(f"📊 Statistics:")
        print(f"   Contexts {'queried' if settings.query_only else 'indexed'}: {len(selected_contexts)}")
        print(f"   Total queries processed: {len(results_df)}")
        print(f"   Success rate: {len(results_df[~errors]) / len(results_df) * 100:.1f}%")


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--device", choices=sorted(DEVICE_PROFILES), default="cpu",
                        help="Inference profile: cpu (default) or gpu (all layers on the GPU)")
    parser.add_argument("--query-only", action="store_true",
                        help="Skip indexing and query the index already in --working-dir")
    parser.add_argument("--model-path", default=os.getenv("LLM_MODEL_PATH", MODEL_PATH),
                        help="Llama-3.2-3B-Instruct GGUF file (env LLM_MODEL_PATH)")
    parser.add_argument("--embedder-path", default=os.getenv("EMBEDDER_PATH", EMBEDDER_PATH),
                        help="Qwen3-Embedding-0.6B directory or HF id (env EMBEDDER_PATH)")
    parser.add_argument("--data-path", default=os.getenv("FINDER_DATA_PATH", DATA_PATH),
                        help="FinDER finder_train.parquet (env FINDER_DATA_PATH)")
    parser.add_argument("--working-dir", default=None,
                        help="LightRAG workspace, deleted and recreated unless --query-only "
                             "(default: ./finder_benchmark_workdir1 on cpu, ./finder_benchmark_workdir_gpu on gpu)")
    parser.add_argument("--num-docs", type=int, default=None,
                        help="Number of unique FinDER contexts to index (default: 10 on cpu, all on gpu)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Only run the first N matching queries")
    parser.add_argument("--n-ctx", type=int, default=None,
                        help="llama.cpp context window (default: 8192 on cpu, 32768 on cpu with "
                             "--query-only, 131072 on gpu)")
    parser.add_argument("--output", default=None,
                        help="Results CSV path (default: finder_lightrag_results[_gpu][_query_only].csv)")
    args = parser.parse_args(argv)

    profile = DEVICE_PROFILES[args.device]
    if args.working_dir is None:
        args.working_dir = profile["working_dir"]
    if args.num_docs is None:
        args.num_docs = profile["num_docs"]
    if args.n_ctx is None:
        args.n_ctx = QUERY_ONLY_CPU_N_CTX if (args.query_only and args.device == "cpu") else profile["n_ctx"]
    if args.output is None:
        stem = Path(profile["output"]).stem
        args.output = f"{stem}_query_only.csv" if args.query_only else profile["output"]
    return args


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main(parse_args()))
