#!/usr/bin/env python3
"""
FinDER Benchmark - QUERY ONLY MODE
Loads existing LightRAG index and runs queries with increased context window.
"""

import pandas as pd
import numpy as np
import asyncio
from pathlib import Path
import os
import traceback # Added for better error visibility

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "./models/Llama-3.2-3B-Instruct.Q8_0.gguf"
EMBEDDER_PATH = "./models/qwen3-0.6b"
DATA_PATH = "./data/finder_train.parquet"
WORKING_DIR = "./finder_benchmark_workdir1" # Must match your previous run

# Flag for query detection
QUERY_FLAG = "benchmark::query::"

# ============================================================================
# LLM WRAPPER (Upgraded Context)
# ============================================================================

class LlamaCppWrapper:
    def __init__(self, model_path: str):
        print(f"Loading LLM from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=32768,       # <--- INCREASED TO 32k TO FIX OVERFLOW
            n_threads=8,       # CPU threads
            n_gpu_layers=0,    # CPU only
            verbose=False
        )
        print("✓ LLM loaded successfully with 32k context window")

    async def __call__(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        try:
            if prompt.startswith(QUERY_FLAG):
                prompt = prompt[len(QUERY_FLAG):]

            if system_prompt:
                formatted_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                formatted_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm(
                    formatted_prompt,
                    max_tokens=2048,
                    temperature=0.0,
                    top_p=1.0,
                    repeat_penalty=1.1,
                    stop=["<|eot_id|>"],
                    echo=False
                )
            )
            return response['choices'][0]['text'].strip()

        except Exception as e:
            # PRINT THE ACTUAL ERROR TRACE
            print(f"\n⚠ LLM CRASHED ON PROMPT LENGTH: {len(prompt)}")
            traceback.print_exc() 
            return f"Error: {str(e)}"

# ============================================================================
# EMBEDDER WRAPPER (Fixed for CPU)
# ============================================================================

class QwenEmbedderWrapper:
    def __init__(self, model_path: str):
        print(f"Loading embedder from {model_path}...")
        self.model = SentenceTransformer(
            model_path,
            trust_remote_code=True,
            device="cpu"  # <--- FORCED CPU
        )

    async def __call__(self, texts: list[str]) -> np.ndarray:
        try:
            processed_texts = []
            for text in texts:
                if text.startswith(QUERY_FLAG):
                    query_text = text[len(QUERY_FLAG):]
                    processed_text = f"Instruct: Given a financial query, retrieve relevant 10-K passages.\nQuery: {query_text}"
                else:
                    processed_text = text
                processed_texts.append(processed_text)

            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(processed_texts, normalize_embeddings=True, convert_to_numpy=True)
            )
            return embeddings

        except Exception as e:
            print(f"⚠ Embedding error: {e}")
            return np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))

# ============================================================================
# MAIN PIPELINE (Query Only)
# ============================================================================

async def main():
    print("=" * 70)
    print("FinDER BENCHMARK - QUERY ONLY MODE")
    print("=" * 70)

    # Check if data exists
    if not os.path.exists(WORKING_DIR) or not os.listdir(WORKING_DIR):
        print(f"❌ Error: Working directory {WORKING_DIR} is empty!")
        print("   Please run the indexing script first.")
        return

    # Initialize Models
    llm_wrapper = LlamaCppWrapper(MODEL_PATH)
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        return await llm_wrapper(prompt, system_prompt, **kwargs)

    embedder_wrapper = QwenEmbedderWrapper(EMBEDDER_PATH)
    embedding_func = EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embedder_wrapper)

    # Initialize LightRAG (Loading existing state)
    print("\nLoading existing LightRAG index...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        # We must keep storage settings same as before to load correctly
        graph_storage="NetworkXStorage",
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
    )
    # This loads the JSON/Graph files from disk (queries fail on
    # uninitialized storages)
    await rag.initialize_storages()

    # Load Data to get the queries
    print("Loading FinDER dataset to retrieve queries...")
    df = pd.read_parquet(DATA_PATH)
    
    # Re-create the filter logic to find the same 10 queries
    unique_contexts = df['references'].explode().drop_duplicates().tolist()
    # Assuming the first run used the first 10 contexts:
    selected_contexts = unique_contexts[:10] 
    filtered_df = df[df['references'].apply(lambda refs: any(ref in selected_contexts for ref in refs))].copy()
    
    print(f"✓ Found {len(filtered_df)} queries to run.")

    # Run Queries
    results = []
    print("\nStarting Queries...")
    
    for idx, row in filtered_df.iterrows():
        query = row['text']
        print(f"[{len(results)+1}/{len(filtered_df)}] Query: {query[:60]}...")
        
        flagged_query = QUERY_FLAG + query
        
        # Execute Query
        response = await rag.aquery(flagged_query, param=QueryParam(mode="hybrid"))
        
        # Check for error string we embedded in wrapper
        if "Error:" in response:
             print(f"  ❌ Failed: {response[:50]}...")
        else:
             print(f"  ✓ Success! Answer length: {len(response)}")

        results.append({
            'original_query': query,
            'ground_truth': row['answer'],
            'generated_answer': response
        })

    # Save
    pd.DataFrame(results).to_csv("finder_results_fixed.csv", index=False)
    print("\n✓ Saved to finder_results_fixed.csv")

if __name__ == "__main__":
    asyncio.run(main())