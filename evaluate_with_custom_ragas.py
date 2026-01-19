"""
Example script showing how to use custom RAGAS metrics with your existing evaluation data.
This script can be adapted to work with your LightRAG evaluation results.
"""

import json
import os
from custom_ragas_metrics import CustomRAGASMetrics, RAGEvaluation
from tqdm import tqdm


def load_evaluation_results(file_path: str) -> list:
    """
    Load evaluation results from a JSON file.

    Args:
        file_path: Path to the JSON file with evaluation results

    Returns:
        List of evaluation records
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def evaluate_with_custom_ragas(input_file: str, output_file: str, api_key: str = None):
    """
    Evaluate RAG results using custom RAGAS metrics.

    Args:
        input_file: Path to input JSON file with RAG results
        output_file: Path to save evaluation results
        api_key: Gemini API key (optional, reads from env if not provided)
    """
    # Load evaluation data
    print(f"Loading data from {input_file}...")
    eval_records = load_evaluation_results(input_file)
    print(f"Loaded {len(eval_records)} records")

    # Initialize metrics calculator
    print("Initializing Custom RAGAS Metrics with Gemini...")
    metrics = CustomRAGASMetrics(api_key=api_key)

    # Process each record
    results = []
    for record in tqdm(eval_records, desc="Evaluating"):
        # Extract fields (adjust based on your data structure)
        question = record.get("question", "")
        answer = record.get("answer", "")
        contexts = record.get("contexts", [])
        ground_truth = record.get("ground_truth", "")

        # Create evaluation object
        eval_data = RAGEvaluation(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )

        # Calculate all metrics
        try:
            metric_results = metrics.evaluate_all(eval_data)

            # Add to results
            result_record = {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "metrics": {
                    "faithfulness": metric_results.get("faithfulness", {}).get("score"),
                    "answer_correctness": metric_results.get("answer_correctness", {}).get("score"),
                    "context_recall": metric_results.get("context_recall", {}).get("score"),
                    "context_precision": metric_results.get("context_precision", {}).get("score"),
                },
                "detailed_results": metric_results
            }
            results.append(result_record)

        except Exception as e:
            print(f"\nError evaluating record: {e}")
            results.append({
                "question": question,
                "error": str(e)
            })

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Calculate average scores
    scores = {
        "faithfulness": [],
        "answer_correctness": [],
        "context_recall": [],
        "context_precision": []
    }

    for result in results:
        if "metrics" in result:
            for metric_name, score in result["metrics"].items():
                if score is not None:
                    scores[metric_name].append(score)

    print("\n" + "=" * 80)
    print("AVERAGE SCORES")
    print("=" * 80)
    for metric_name, metric_scores in scores.items():
        if metric_scores:
            avg_score = sum(metric_scores) / len(metric_scores)
            print(f"{metric_name}: {avg_score:.3f} (n={len(metric_scores)})")
        else:
            print(f"{metric_name}: N/A")
    print("=" * 80)


def evaluate_single_example():
    """
    Evaluate a single example to demonstrate the metrics.
    """
    print("=" * 80)
    print("SINGLE EXAMPLE EVALUATION")
    print("=" * 80)

    # Example from Einstein (with intentional error to show faithfulness)
    eval_data = RAGEvaluation(
        question="Where was Albert Einstein born?",
        answer="Albert Einstein was born in Ulm, Germany on March 20, 1879.",  # Wrong date
        contexts=[
            "Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire.",
            "Einstein's family moved to Munich when he was an infant.",
            "The theory of relativity was developed by Einstein in the early 20th century."
        ],
        ground_truth="Albert Einstein was born in Ulm, Germany on March 14, 1879."
    )

    metrics = CustomRAGASMetrics()

    # Calculate all metrics
    print("\nCalculating metrics...")
    results = metrics.evaluate_all(eval_data)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if "faithfulness" in results:
        print(f"\nFaithfulness: {results['faithfulness']['score']:.3f}")
        print(f"  - Measures if answer claims are supported by context")
        print(f"  - Claims supported: {results['faithfulness']['supported_claims']}/{results['faithfulness']['total_claims']}")

    if "answer_correctness" in results:
        print(f"\nAnswer Correctness: {results['answer_correctness']['score']:.3f}")
        print(f"  - Combines factual F1 and semantic similarity")
        print(f"  - F1 Score: {results['answer_correctness']['f1_score']:.3f}")
        print(f"  - Semantic Similarity: {results['answer_correctness']['semantic_similarity']:.3f}")

    if "context_recall" in results:
        print(f"\nContext Recall: {results['context_recall']['score']:.3f}")
        print(f"  - Measures if ground truth is supported by retrieved context")
        print(f"  - Claims supported: {results['context_recall']['supported_claims']}/{results['context_recall']['total_claims']}")

    if "context_precision" in results:
        print(f"\nContext Precision: {results['context_precision']['score']:.3f}")
        print(f"  - Measures if relevant chunks are ranked higher")
        print(f"  - Relevant chunks: {results['context_precision']['relevant_chunks']}/{results['context_precision']['total_chunks']}")

    print("\n" + "=" * 80)

    # Save detailed results
    output_file = "single_example_ragas_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")


def main():
    """
    Main function - choose which evaluation to run.
    """
    import sys

    if len(sys.argv) > 1:
        # Batch evaluation mode
        if len(sys.argv) < 3:
            print("Usage: python evaluate_with_custom_ragas.py <input_file> <output_file>")
            sys.exit(1)

        input_file = sys.argv[1]
        output_file = sys.argv[2]

        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found")
            sys.exit(1)

        evaluate_with_custom_ragas(input_file, output_file)
    else:
        # Single example mode
        print("No input file provided. Running single example demonstration...\n")
        evaluate_single_example()
        print("\nTo evaluate a batch of results, run:")
        print("  python evaluate_with_custom_ragas.py <input_file> <output_file>")


if __name__ == "__main__":
    main()
