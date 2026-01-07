from datasets import load_dataset
import random
import json

# Configuration
OUTPUT_FILE = "finder_subset_10pct.jsonl"

def prepare_finder_subset():
    print("Loading FinDER from Hugging Face...")
    # Load the dataset (it has a 'train' split by default usually)
    dataset = load_dataset("Linq-AI-Research/FinDER", split="train")
    
    # Calculate 10% size
    total_size = len(dataset)
    subset_size = int(total_size * 0.10)
    
    print(f"Total rows: {total_size}. Selecting random 10% ({subset_size} rows)...")
    
    # Shuffle and select top N
    # We use a fixed seed for reproducibility
    subset = dataset.shuffle(seed=42).select(range(subset_size))
    
    prepared_data = []
    
    for row in subset:
        # FinDER structure based on Source [2]:
        # _id: unique ID
        # text: the question
        # references: list of strings (the context paragraphs)
        # answer: the ground truth answer
        
        entry = {
            "id": row['_id'],
            "question": row['text'],
            "ground_truth_answer": row['answer'],
            # Join references into one text block for LightRAG indexing
            # Source [3] says these are already segmented paragraphs
            "context_to_index": "\n\n".join(row['references']) 
        }
        prepared_data.append(entry)
        
    # Save to JSONL for the pipeline to consume
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in prepared_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully saved {len(prepared_data)} items to {OUTPUT_FILE}")
    print("Ready for LightRAG indexing.")

if __name__ == "__main__":
    prepare_finder_subset()
