import os
import json
import asyncio
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Configuration ---
# MUST match the directory used in the indexing step
WORK_DIR = "./finder_single_kg_qwen3" 
SUBSET_RATIO = 0.10
SEED = 42

# --- Model Wrappers (Must match Indexing config) ---
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

class Qwen3Embedder:
    def __init__(self):
        # We assume the model is already downloaded from the indexing phase
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    
    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

# Global instance
qwen_instance = Qwen3Embedder()

async def qwen_embedding_func(texts: list[str]) -> np.ndarray:
    return qwen_instance(texts)

async def main():
    print(f"Loading existing LightRAG from {WORK_DIR}...")
    
    # 1. Initialize RAG (Same config as Indexing)
    rag = LightRAG(
        working_dir=WORK_DIR,
        llm_model_func=mistral_model_complete,
        llm_model_name='ministral-8b-latest',
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, 
            max_token_size=8192, 
            func=qwen_embedding_func
        )
    )

    # 2. IMPORTANT: Load the stored Graph and Vectors
    # Source [6]: "LightRAG requires explicit initialization... await rag.initialize_storages()"
    await rag.initialize_storages()
    print("Graph and Vectors loaded successfully.")

    # 3. Load Dataset (Only to get the questions)
    dataset = load_dataset("Linq-AI-Research/FinDER", split="train")
    subset_size = int(len(dataset) * SUBSET_RATIO)
    subset = dataset.shuffle(seed=SEED).select(range(subset_size))
    
    results = []
    
    print(f"Starting Query Phase for {len(subset)} questions...")
    
    # 4. Query Loop
    # We use 'mix' mode to leverage the Qwen vectors + Mistral Graph
    for i, row in enumerate(subset):
        raw_question = row['text']
        
        # Qwen 3 Instruction Format (Query side only)
        # Source [7]: "Each query must come with a one-sentence instruction"
        formatted_query = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {raw_question}"
        
        # Using "mix" mode as it utilizes both the Graph and the Qwen Vectors
        response = await rag.aquery(formatted_query, param=QueryParam(mode="mix"))
        
        results.append({
            "id": row['_id'],
            "question": raw_question,
            "ground_truth": row['answer'],
            "lightrag_answer": response
        })
        
        if i % 5 == 0:
            print(f"Processed {i}/{len(subset)} queries...")

    # 5. Save
    with open("finder_query_results_mix.jsonl", "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print("Querying complete. Results saved to finder_query_results_mix.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
