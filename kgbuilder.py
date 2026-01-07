import os
import json
import numpy as np
import asyncio
from dotenv import load_dotenv
from datasets import load_dataset
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

# 1. Load API Key
load_dotenv()
if not os.getenv("MISTRAL_API_KEY"):
    raise ValueError("MISTRAL_API_KEY not found in .env")

# --- Configuration ---
WORK_DIR = "./finder_single_kg_qwen3"
SUBSET_RATIO = 0.10
SEED = 42

# --- Mistral Connection ---
async def mistral_model_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        model="ministral-8b-latest", 
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1",
        **kwargs
    )

# --- Qwen 3 Embedding Wrapper ---
class Qwen3Embedder:
    def __init__(self):
        print("Loading Qwen3-Embedding-0.6B...")
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
        self.dimension = 1024 

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

qwen_instance = Qwen3Embedder()

async def qwen_embedding_func(texts: list[str]) -> np.ndarray:
    return qwen_instance(texts)

# --- Main Async Pipeline ---
async def main():
    # 2. Initialize LightRAG
    rag = LightRAG(
        working_dir=WORK_DIR,
        llm_model_func=mistral_model_complete,
        llm_model_name='ministral-8b-latest',
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=qwen_embedding_func
        ),
        chunk_token_size=1200, 
        chunk_overlap_token_size=100
    )

    # CRITICAL FIX: Initialize storages before use
    await rag.initialize_storages()

    # 3. Load Data
    print("Loading FinDER dataset...")
    dataset = load_dataset("Linq-AI-Research/FinDER", split="train")
    subset_size = int(len(dataset) * SUBSET_RATIO)
    subset = dataset.shuffle(seed=SEED).select(range(subset_size))
    print(f"Selected {len(subset)} items.")

    # 4. Indexing Phase
    print("\n--- Phase 1: Indexing References ---")
    
    for i, row in enumerate(subset):
        context_block = "\n\n".join(row['references'])
        # Use ainsert for async processing
        await rag.ainsert(context_block)
        
        if i % 10 == 0:
            print(f"Indexed {i}/{len(subset)}...")

    # 5. Querying Phase
    print("\n--- Phase 2: Querying ---")
    results = []
    task_instruction = "Given a web search query, retrieve relevant passages that answer the query"
    
    for row in subset:
        raw_question = row['text']
        formatted_query = f"Instruct: {task_instruction}\nQuery: {raw_question}"
        
        # Use aquery for async processing
        response = await rag.aquery(formatted_query, param=QueryParam(mode="hybrid"))
        
        results.append({
            "id": row['_id'],
            "original_question": raw_question,
            "ground_truth": row['answer'],
            "lightrag_response": response
        })

    # 6. Save Results
    with open("finder_lightrag_final.jsonl", "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print("Pipeline complete. Saved to finder_lightrag_final.jsonl")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())