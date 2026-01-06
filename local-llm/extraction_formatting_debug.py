import asyncio
import pandas as pd
import re
from llama_cpp import Llama

# ============================================================================
# CONFIGURATION
# ============================================================================
# Paths based on your source [1]
MODEL_PATH = "./models/Llama-3.2-3B-Instruct.Q8_0.gguf" 
DATA_PATH = "./data/finder_train.parquet"
OUTPUT_CSV = "debug_extraction_results.csv"
NUM_SAMPLES = 3
CHUNK_SIZE = 1000  # Characters

# ============================================================================
# 1. MODEL LOADING
# ============================================================================
print(f"Loading LLM from {MODEL_PATH}...")
# CPU-optimized loading based on source [2]
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_gpu_layers=-1, # CPU only (-1 usually means all on GPU if available, or CPU if not. Source used -1)
    verbose=True
)
print("✓ LLM loaded")

# ============================================================================
# 2. LOGIC IMPLEMENTATION
# ============================================================================

async def generate_text(prompt, system_prompt):
    """Simple wrapper for Llama generation using Llama-3 format"""
    formatted_prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    # Using 0.0 temperature for maximum determinism [3]
    output = llm(
        formatted_prompt,
        max_tokens=1024,
        temperature=0.0, 
        stop=["<|eot_id|>"],
        echo=False
    )
    return output['choices'][0]['text'].strip()

def regex_repair(output_text: str) -> str:
    """
    Step 3: Regex Repair [4]
    - Ensures tuples are closed.
    - Adds default weights if missing (fixing the 'found 4/5 fields' error).
    - Ensures <|complete|> delimiter exists.
    """
    fixed_lines = []
    # Pattern looks for: ("Source", "Target" ...
    tuple_pattern = re.compile(r'^\s*\(".*?"') 
    
    for line in output_text.split('\n'):
        line = line.strip()
        if not line: continue
        
        if tuple_pattern.match(line):
            # Repair logic: If missing closing parenthesis
            if not line.endswith(')'):
                # If it ends with a comma or looks like it's missing the weight
                # (Source [4] heuristic)
                if line.endswith(',') or line.count('"') >= 4: 
                    line += " 1.0)"
                else:
                    line += ")" # Blind close
            fixed_lines.append(line)
        elif line == "<|complete|>":
            fixed_lines.append(line)
            
    # Force the delimiter LightRAG expects
    final_output = "\n".join(fixed_lines)
    if "<|complete|>" not in final_output:
        final_output += "\n<|complete|>"
        
    return final_output

async def process_chunk(text_chunk):
    """
    Executes the Two-Step Logic (Reasoning -> Formatting) + Repair
    """
    # --- Step 1: Context-Aware Extraction (Reasoning) ---
    # Improved Prompt: Explicitly handles tables and asks for "Sentencified" facts
    # to support Global Search descriptions [5].
    extraction_prompt = (
        f"Analyze the following financial text. Extract key facts involving entities (companies, metrics, people).\n"
        f"For each fact, write a complete sentence describing the relationship.\n"
        f"Pay attention to TABLES: Link the row label to the column header and value. "
        f"Be careful not to mix row values with 'Total' lines.\n\n"
        f"Text Snippet:\n{text_chunk}"
    )
    
    raw_extraction = await generate_text(
        extraction_prompt, 
        system_prompt="You are a financial analyst. Extract structured facts from text and tables."
    )

    # --- Step 2: Few-Shot Strict Formatting (Syntax & Topology) ---
    # Improved Prompt: Uses FEW-SHOT examples to fix "Sentence-as-Node" issues [5].
    # Enforces "Short Entities" for Nodes and "Full Sentences" for Descriptions.
    formatting_prompt = (
        f"Convert the extracted facts into strictly formatted tuples: "
        f"(\"Source\", \"Target\", \"Relationship Description\", \"Keywords\", weight).\n\n"
        
        f"RULES:\n"
        f"1. Source and Target must be SHORT entities (1-4 words). NO SENTENCES as nodes!\n"
        f"2. Relationship Description must be a FULL sentence summarizing the connection.\n"
        f"3. Weight must be 1.0.\n\n"
        
        f"--- EXAMPLE 1 (Standard Text) ---\n"
        f"Input: Google's revenue grew to $100M.\n"
        f"Output: (\"Google\", \"$100M\", \"Google's revenue grew to $100M\", \"revenue, growth\", 1.0)\n\n"
        
        f"--- EXAMPLE 2 (Table Data) ---\n"
        f"Input: Operating Costs in 2023 were 500.\n"
        f"Output: (\"Operating Costs\", \"500\", \"Operating Costs in 2023 were 500\", \"costs, 2023\", 1.0)\n\n"
        
        f"--- YOUR TASK ---\n"
        f"Input List:\n{raw_extraction}\n\n"
        f"Output (Tuples only, end with <|complete|>):"
    )
    
    formatted_output = await generate_text(
        formatting_prompt,
        system_prompt="You are a data formatting machine. Output valid tuples only."
    )

    # --- Step 3: Targeted Regex Repair ---
    # Catches the specific syntax errors seen in previous logs [6].
    repaired_output = regex_repair(formatted_output)

    return raw_extraction, formatted_output, repaired_output

# ============================================================================
# 3. MAIN EXECUTION LOOP
# ============================================================================
async def main():
    print("Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    
    # Grab unique context documents (Financial reports)
    contexts = df['references'].explode().drop_duplicates().tolist()
    
    results = []
    
    print(f"Processing {NUM_SAMPLES} samples...")
    
    for i in range(NUM_SAMPLES):
        # Slice a manageable chunk of text
        original_text = contexts[i][:CHUNK_SIZE]
        
        print(f"\n--- Processing Sample {i+1} ---")
        
        # Run the logic
        raw, formatted, repaired = await process_chunk(original_text)
        
        print(f"[Raw Extraction Length]: {len(raw)} chars")
        print(f"[Formatted Output Length]: {len(formatted)} chars")
        print(f"[Repaired Output Length]: {len(repaired)} chars")
        
        results.append({
            "source_text": original_text,
            "step1_raw_extraction": raw,
            "step2_llm_formatting": formatted,
            "step3_regex_repaired": repaired,
            "valid_structure": "<|complete|>" in repaired and "(" in repaired
        })

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Debugging complete. Results saved to {OUTPUT_CSV}")
    
    # Preview - FIXED: Iterate through list instead of accessing string index [7]
    print("\n=== Sample Result (Step 3 Output) ===")
    if results:
        print(results[-1]['step3_regex_repaired'])

if __name__ == "__main__":
    asyncio.run(main())