import json

def summarize_data_structure(data, indent=''):
    """
    recursively summarizes the structure of actual JSON data.
    """
    summary = ""
    data_type = type(data).__name__

    # 1. Handle Dictionaries (Objects)
    if isinstance(data, dict):
        summary += f"{indent}- Type: Object (dict)\n"
        if not data:
            summary += f"{indent}  (Empty Object)\n"
        else:
            summary += f"{indent}- Keys:\n"
            for key, value in data.items():
                summary += f"{indent}  * **{key}**:\n"
                # Recursively summarize the value
                summary += summarize_data_structure(value, indent + '    ')

    # 2. Handle Lists (Arrays)
    elif isinstance(data, list):
        summary += f"{indent}- Type: Array (list)\n"
        if not data:
            summary += f"{indent}  (Empty Array)\n"
        else:
            summary += f"{indent}  (Length: {len(data)})\n"
            # Summarize the first item as a sample structure
            summary += f"{indent}  - Sample Item Structure:\n"
            summary += summarize_data_structure(data[0], indent + '    ')

    # 3. Handle Primitives (Strings, Ints, Bools, None)
    else:
        summary += f"{indent}- Type: {data_type} | Value Example: {repr(data)[:50]}\n"

    return summary

try:
    file_path = '5_modes_question_wise_results/5_modes_question_wise_results_priority_tickers_ALL/test_results_CTAS_question_d25e0880.json'
    
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        structure_summary = summarize_data_structure(data)
        print("--- JSON Data Structure Summary ---")
        print(structure_summary)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except json.JSONDecodeError:
    print("Error: Failed to decode JSON from the file (invalid JSON format).")