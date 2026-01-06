#!/usr/bin/env python3
"""
FinDER Benchmark with LightRAG - GPU Accelerated (No Limits)

Benchmarks the LightRAG framework on the FULL FinDER financial dataset using:
- LLM: Llama-3.2-3B-Instruct (GPU inference via llama-cpp-python)
- Embedder: Qwen3-0.6B (GPU inference via sentence-transformers)
- Dataset: FinDER train set - ALL contexts (no document limit)

GPU-optimized with maximum parallelism and extended context.
"""

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
# CONFIGURATION - GPU ACCELERATED, NO LIMITS
# ============================================================================

MODEL_PATH = "./models/Llama-3.2-3B-Instruct.Q8_0.gguf"
EMBEDDER_PATH = "./models/qwen3-0.6b"
DATA_PATH = "./data/finder_train.parquet"
WORKING_DIR = "./finder_benchmark_workdir_gpu"

# Flag for query detection (used by embedder to differentiate queries from documents)
QUERY_FLAG = "benchmark::query::"

# NO DOCUMENT LIMIT - Process all unique contexts
LIMIT_DOCS = None  # Set to None for no limit, or an integer to limit

# ============================================================================
# LLM WRAPPER - GPU-Accelerated with Maximum Context
# ============================================================================

class LlamaCppWrapper:
    """
    GPU-accelerated LLM wrapper for llama-cpp-python.
    All layers loaded to GPU with maximum context window.
    """

    def __init__(self, model_path: str):
        print(f"Loading LLM from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=131072,  # Maximum context (128K) - model's native training context
            n_threads=4,  # Reduced threads since GPU handles compute
            n_gpu_layers=-1,  # ALL layers on GPU
            verbose=False
        )
        print("✓ LLM loaded successfully")
        print(f"  Context window: 131,072 tokens (128K)")
        print(f"  GPU layers: ALL (-1)")

    async def __call__(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
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

            # Format using Llama 3 chat template
            if system_prompt:
                formatted_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                formatted_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            # Run synchronous generation in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm(
                    formatted_prompt,
                    max_tokens=2048,
                    temperature=0.0,  # Deterministic for JSON extraction
                    top_p=1.0,
                    repeat_penalty=1.1,
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
# EMBEDDER WRAPPER - GPU-Accelerated with Instruction-Awareness
# ============================================================================

class QwenEmbedderWrapper:
    """
    GPU-accelerated instruction-aware embedding wrapper for Qwen3.

    Uses the QUERY_FLAG to determine when to apply instruction prefix:
    - Queries (with flag): Apply "Instruct: Given a financial query..." prefix
    - Documents/Entities (no flag): Encode as-is
    """

    def __init__(self, model_path: str):
        print(f"Loading embedder from {model_path}...")
        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            device='cuda'  # GPU acceleration
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
# MAIN BENCHMARK PIPELINE
# ============================================================================

async def main():
    """Main benchmark execution pipeline - GPU accelerated with no limits."""

    print("=" * 70)
    print("FinDER BENCHMARK WITH LIGHTRAG - GPU ACCELERATED (NO LIMITS)")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------------
    # STEP 1: Safe-Start - Clear Previous Data
    # ------------------------------------------------------------------------

    if os.path.exists(WORKING_DIR):
        print(f"⚠ Found existing working directory: {WORKING_DIR}")
        shutil.rmtree(WORKING_DIR)
        print("✓ Cleared previous benchmark data for clean run")
        print()

    # Create fresh working directory
    Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # STEP 2: Initialize Models
    # ------------------------------------------------------------------------

    print("Initializing GPU-accelerated models...")
    print()

    # Initialize LLM wrapper
    llm_wrapper = LlamaCppWrapper(MODEL_PATH)

    # Create LightRAG-compatible LLM function
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        return await llm_wrapper(prompt, system_prompt, **kwargs)

    # Initialize embedder wrapper
    embedder_wrapper = QwenEmbedderWrapper(EMBEDDER_PATH)

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
    df = pd.read_parquet(DATA_PATH)

    print(f"✓ Loaded {len(df)} query-answer pairs")
    print(f"  Columns: {df.columns.tolist()}")
    print()

    # Extract and deduplicate contexts (10-K passages)
    print("Extracting unique contexts...")

    # 'references' is a numpy array, so we explode it to get individual strings
    unique_contexts = df['references'].explode().drop_duplicates().tolist()
    print(f"✓ Found {len(unique_contexts)} unique contexts")

    # Apply limit if specified (None = no limit)
    if LIMIT_DOCS is not None:
        selected_contexts = unique_contexts[:LIMIT_DOCS]
        print(f"✓ Limited to first {LIMIT_DOCS} contexts for benchmarking")
    else:
        selected_contexts = unique_contexts
        print(f"✓ Processing ALL {len(selected_contexts)} contexts (no limit)")

    # Filter dataframe to only queries matching selected contexts
    filtered_df = df[df['references'].apply(lambda refs: any(ref in selected_contexts for ref in refs))].copy()
    print(f"✓ Filtered to {len(filtered_df)} queries matching selected contexts")
    print()

    # Display sample
    print("=== Sample Data ===")
    print(f"Query: {filtered_df.iloc[0]['text'][:100]}...")
    print(f"Answer: {filtered_df.iloc[0]['answer'][:100]}...")
    print(f"References count: {len(filtered_df.iloc[0]['references'])}")
    print(f"First reference length: {len(filtered_df.iloc[0]['references'][0])} chars")
    print()

    # ------------------------------------------------------------------------
    # STEP 4: Initialize LightRAG - GPU Optimized
    # ------------------------------------------------------------------------

    print("Initializing LightRAG (GPU-optimized)...")

    rag = LightRAG(
        working_dir=WORKING_DIR,

        # LLM configuration
        llm_model_func=llm_model_func,
        llm_model_name="Llama-3.2-3B-Instruct-GPU",
        llm_model_max_async=8,  # Increased parallelism for GPU

        # Embedding configuration
        embedding_func=embedding_func,

        # Chunking configuration - OPTIMIZED for speed
        chunk_token_size=1200,  # Balanced: not too small (more chunks) or large (harder extraction)
        chunk_overlap_token_size=100,

        # Entity extraction - REDUCED for speed
        entity_extract_max_gleaning=0,  # Disable gleaning retries (massive speedup)

        # Storage backends
        graph_storage="NetworkXStorage",
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",

        # Concurrency - MAXIMIZED for GPU
        max_parallel_insert=8  # Increased from 4 to 8
    )

    # CRITICAL: Initialize storages
    await rag.initialize_storages()

    print("✓ LightRAG initialized successfully")
    print(f"  Working directory: {WORKING_DIR}")
    print(f"  Chunk size: 1200 tokens")
    print(f"  Max gleaning retries: 0 (disabled for speed)")
    print(f"  LLM concurrency: 8")
    print(f"  Insert parallelism: 8")
    print()

    # ------------------------------------------------------------------------
    # STEP 5: Index Documents - BATCH INSERTION (OPTIMIZED)
    # ------------------------------------------------------------------------

    print("=" * 70)
    print(f"INDEXING PHASE - Processing {len(selected_contexts)} documents")
    print("=" * 70)
    print()

    print(f"📦 Batch inserting {len(selected_contexts)} documents...")
    print("   (This is much faster than sequential insertion)")
    print()

    try:
        # CRITICAL FIX: Batch insert ALL documents at once
        # LightRAG's ainsert() accepts both single strings and lists
        # This allows internal parallelization based on max_parallel_insert
        await rag.ainsert(selected_contexts)

        print(f"✓ Successfully indexed all {len(selected_contexts)} documents")
        print()

    except Exception as e:
        print(f"⚠ Error during batch insertion: {e}")
        print("   Falling back to sequential insertion...")
        print()

        # Fallback: sequential insertion if batch fails
        for idx, context in enumerate(selected_contexts, 1):
            try:
                print(f"[{idx}/{len(selected_contexts)}] Indexing context (length: {len(context)} chars)...")
                await rag.ainsert(context)
                print(f"  ✓ Indexed successfully")
                print()
            except Exception as e:
                print(f"  ⚠ Error indexing document {idx}: {e}")
                print()
                continue

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
            # FinDER uses 'text' for questions
            query = row['text']
            ground_truth_answer = row['answer']
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
    output_path = "finder_lightrag_results_gpu.csv"
    results_df.to_csv(output_path, index=False)

    print(f"✓ Results saved to {output_path}")
    print(f"  Total queries: {len(results_df)}")
    print(f"  Successful: {len(results_df[~results_df['lightrag_generated_answer'].str.startswith('ERROR')])}")
    print(f"  Errors: {len(results_df[results_df['lightrag_generated_answer'].str.startswith('ERROR')])}")
    print()

    # Display preview
    print("=== Results Preview ===")
    print(results_df.head(3).to_string())
    print()

    print("=" * 70)
    print("✓ BENCHMARK COMPLETE")
    print("=" * 70)
    print()
    print(f"📊 Statistics:")
    print(f"   Total contexts indexed: {len(selected_contexts)}")
    print(f"   Total queries processed: {len(results_df)}")
    print(f"   Success rate: {len(results_df[~results_df['lightrag_generated_answer'].str.startswith('ERROR')]) / len(results_df) * 100:.1f}%")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
