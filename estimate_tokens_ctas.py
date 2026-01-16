#!/usr/bin/env python3
"""
Token Estimator for CTAS Multi-Mode Question Results
Analyzes input/output token usage and estimates API costs for Gemini-3-Flash-Preview
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import sys

# ============================================
# Configuration
# ============================================

RESULTS_FOLDER = "5_modes_question_wise_results"
CTAS_FILE_PATTERN = "test_results_CTAS_question_*.json"
OUTPUT_STATS_FILE = "token_stats_ctas.json"

# Gemini-3-Flash-Preview Pricing (as of Jan 2025)
INPUT_PRICE_PER_M = 0.50  # $0.50 per million input tokens
OUTPUT_PRICE_PER_M = 3.00  # $3.00 per million output tokens

# Query modes
QUERY_MODES = ["local", "global", "naive", "hybrid", "mix"]

# ============================================
# Token Counting Setup
# ============================================

# Try to use tiktoken for accurate counting
try:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4/Gemini compatible tokenizer
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(encoding.encode(text))
    TOKEN_METHOD = "tiktoken (accurate)"
    print("✓ Using tiktoken for accurate token counting\n")
except ImportError:
    # Fallback: rough estimate (1 token ≈ 4 chars)
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(text) // 4
    TOKEN_METHOD = "character-based estimation (approximate)"
    print("⚠ Warning: tiktoken not installed. Using rough estimate (chars/4)")
    print("  Install for accurate counts: pip install tiktoken\n")


# ============================================
# Prompt Reconstruction (from generate_answers_ctas.py)
# ============================================

def reconstruct_prompt(question: str, context: str, mode: str, requires_reasoning: bool = False) -> str:
    """
    Reconstruct the exact prompt sent to Gemini API.
    This matches the prompt structure in generate_answers_ctas.py lines 64-77
    """
    prompt = f"""You are a financial document analyst specializing in SEC 10-K filings.

CRITICAL INSTRUCTIONS:
- Answer ONLY based on the provided context
- Be precise with numerical values
- Keep answers concise (2-4 sentences)
{f"- This question requires detailed reasoning and analysis" if requires_reasoning else ""}

Question: {question}

Context from {mode.upper()}:
{context}

Please provide a concise, factual answer based only on the information in the context above."""

    return prompt


# ============================================
# Token Analysis Functions
# ============================================

def analyze_mode_tokens(mode_data: Dict[str, Any], question: str, mode_name: str, requires_reasoning: bool) -> Dict[str, Any]:
    """
    Analyze token counts for a single mode.
    Returns input tokens, output tokens, and costs.
    """
    result = {
        "mode": mode_name,
        "status": mode_data.get("status", "unknown"),
        "input_tokens": 0,
        "output_tokens": 0,
        "response_tokens": 0,
        "thought_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "context_length": mode_data.get("context_length", 0),
        "has_response": False,
        "has_thought": False
    }

    # Only count if status is success
    if result["status"] != "success":
        result["error"] = mode_data.get("error", "Unknown error")
        return result

    # Count input tokens (prompt)
    context = mode_data.get("retrieved_context", "")
    if context:
        prompt = reconstruct_prompt(question, context, mode_name, requires_reasoning)
        result["input_tokens"] = count_tokens(prompt)

    # Count output tokens (response + thought)
    response = mode_data.get("response", "")
    thought = mode_data.get("thought", "")

    if response:
        result["has_response"] = True
        result["response_tokens"] = count_tokens(response)

    if thought:
        result["has_thought"] = True
        result["thought_tokens"] = count_tokens(thought)

    result["output_tokens"] = result["response_tokens"] + result["thought_tokens"]

    # Calculate costs
    result["input_cost"] = (result["input_tokens"] / 1_000_000) * INPUT_PRICE_PER_M
    result["output_cost"] = (result["output_tokens"] / 1_000_000) * OUTPUT_PRICE_PER_M
    result["total_cost"] = result["input_cost"] + result["output_cost"]

    return result


def analyze_question_file(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a single question file with all its modes.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_id = data.get("question_id")
    question = data.get("question", "")
    requires_reasoning = data.get("reasoning", False)
    modes = data.get("modes", {})

    question_result = {
        "question_id": question_id,
        "question": question[:100] + "..." if len(question) > 100 else question,
        "full_question": question,
        "requires_reasoning": requires_reasoning,
        "category": data.get("category"),
        "modes": {},
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0
    }

    # Analyze each mode
    for mode_name in QUERY_MODES:
        if mode_name in modes:
            mode_analysis = analyze_mode_tokens(
                modes[mode_name],
                question,
                mode_name,
                requires_reasoning
            )
            question_result["modes"][mode_name] = mode_analysis

            question_result["total_input_tokens"] += mode_analysis["input_tokens"]
            question_result["total_output_tokens"] += mode_analysis["output_tokens"]
            question_result["total_cost"] += mode_analysis["total_cost"]

    return question_result


# ============================================
# Statistics Aggregation
# ============================================

def aggregate_statistics(all_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate statistics across all questions and modes.
    """
    stats = {
        "total_questions": len(all_questions),
        "total_modes_analyzed": 0,
        "reasoning_questions": sum(1 for q in all_questions if q["requires_reasoning"]),
        "non_reasoning_questions": sum(1 for q in all_questions if not q["requires_reasoning"]),
        "per_mode": {},
        "overall": {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_response_tokens": 0,
            "total_thought_tokens": 0,
            "total_input_cost": 0.0,
            "total_output_cost": 0.0,
            "total_cost": 0.0
        }
    }

    # Initialize per-mode stats
    for mode in QUERY_MODES:
        stats["per_mode"][mode] = {
            "count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_response_tokens": 0,
            "total_thought_tokens": 0,
            "total_cost": 0.0,
            "input_tokens_list": [],
            "output_tokens_list": [],
            "context_lengths": []
        }

    # Aggregate from all questions
    for question in all_questions:
        for mode_name, mode_data in question["modes"].items():
            mode_stats = stats["per_mode"][mode_name]
            mode_stats["count"] += 1

            if mode_data["status"] == "success":
                mode_stats["success_count"] += 1
                mode_stats["total_input_tokens"] += mode_data["input_tokens"]
                mode_stats["total_output_tokens"] += mode_data["output_tokens"]
                mode_stats["total_response_tokens"] += mode_data["response_tokens"]
                mode_stats["total_thought_tokens"] += mode_data["thought_tokens"]
                mode_stats["total_cost"] += mode_data["total_cost"]

                mode_stats["input_tokens_list"].append(mode_data["input_tokens"])
                mode_stats["output_tokens_list"].append(mode_data["output_tokens"])
                mode_stats["context_lengths"].append(mode_data["context_length"])

                stats["total_modes_analyzed"] += 1
            else:
                mode_stats["error_count"] += 1

    # Calculate overall totals
    for mode_stats in stats["per_mode"].values():
        stats["overall"]["total_input_tokens"] += mode_stats["total_input_tokens"]
        stats["overall"]["total_output_tokens"] += mode_stats["total_output_tokens"]
        stats["overall"]["total_response_tokens"] += mode_stats["total_response_tokens"]
        stats["overall"]["total_thought_tokens"] += mode_stats["total_thought_tokens"]
        stats["overall"]["total_cost"] += mode_stats["total_cost"]

    stats["overall"]["total_input_cost"] = (stats["overall"]["total_input_tokens"] / 1_000_000) * INPUT_PRICE_PER_M
    stats["overall"]["total_output_cost"] = (stats["overall"]["total_output_tokens"] / 1_000_000) * OUTPUT_PRICE_PER_M

    # Calculate averages and ranges for each mode
    for mode_name, mode_stats in stats["per_mode"].items():
        if mode_stats["success_count"] > 0:
            mode_stats["avg_input_tokens"] = mode_stats["total_input_tokens"] / mode_stats["success_count"]
            mode_stats["avg_output_tokens"] = mode_stats["total_output_tokens"] / mode_stats["success_count"]
            mode_stats["avg_context_length"] = sum(mode_stats["context_lengths"]) / len(mode_stats["context_lengths"])
            mode_stats["min_input_tokens"] = min(mode_stats["input_tokens_list"])
            mode_stats["max_input_tokens"] = max(mode_stats["input_tokens_list"])
            mode_stats["min_output_tokens"] = min(mode_stats["output_tokens_list"]) if mode_stats["output_tokens_list"] else 0
            mode_stats["max_output_tokens"] = max(mode_stats["output_tokens_list"]) if mode_stats["output_tokens_list"] else 0
        else:
            mode_stats["avg_input_tokens"] = 0
            mode_stats["avg_output_tokens"] = 0
            mode_stats["avg_context_length"] = 0
            mode_stats["min_input_tokens"] = 0
            mode_stats["max_input_tokens"] = 0
            mode_stats["min_output_tokens"] = 0
            mode_stats["max_output_tokens"] = 0

        # Remove lists from stats (not needed in output)
        del mode_stats["input_tokens_list"]
        del mode_stats["output_tokens_list"]
        del mode_stats["context_lengths"]

    return stats


# ============================================
# Report Generation
# ============================================

def print_report(stats: Dict[str, Any], all_questions: List[Dict[str, Any]]):
    """
    Print comprehensive token usage report.
    """
    print("=" * 80)
    print("TOKEN ESTIMATION REPORT - Gemini-3-Flash-Preview")
    print("=" * 80)
    print(f"Token counting method: {TOKEN_METHOD}")
    print(f"Questions analyzed:    {stats['total_questions']}")
    print(f"Modes per question:    {len(QUERY_MODES)}")
    print(f"Total analyses:        {stats['total_modes_analyzed']}")
    print(f"Reasoning questions:   {stats['reasoning_questions']}")
    print(f"Regular questions:     {stats['non_reasoning_questions']}")
    print()

    # Per-mode statistics
    print("=" * 80)
    print("PER-MODE STATISTICS")
    print("=" * 80)
    print(f"{'Mode':<10} {'Success':<8} {'Avg Input':<12} {'Avg Output':<12} {'Total Input':<13} {'Total Output':<13} {'Est. Cost':<10}")
    print("-" * 80)

    for mode_name in QUERY_MODES:
        mode_stats = stats["per_mode"][mode_name]
        print(f"{mode_name:<10} "
              f"{mode_stats['success_count']}/{mode_stats['count']:<6} "
              f"{mode_stats['avg_input_tokens']:>11,.0f} "
              f"{mode_stats['avg_output_tokens']:>11,.0f} "
              f"{mode_stats['total_input_tokens']:>12,} "
              f"{mode_stats['total_output_tokens']:>12,} "
              f"${mode_stats['total_cost']:>9.4f}")

    print()

    # Mode comparison - context sizes
    print("=" * 80)
    print("CONTEXT SIZE STATISTICS (characters)")
    print("=" * 80)
    print(f"{'Mode':<10} {'Avg Length':<15} {'Min Length':<15} {'Max Length':<15}")
    print("-" * 80)

    for mode_name in QUERY_MODES:
        mode_stats = stats["per_mode"][mode_name]
        if mode_stats['success_count'] > 0:
            print(f"{mode_name:<10} "
                  f"{mode_stats['avg_context_length']:>14,.0f} "
                  f"{min(all_questions, key=lambda q: q['modes'].get(mode_name, {}).get('context_length', float('inf')))['modes'].get(mode_name, {}).get('context_length', 0):>14,} "
                  f"{max(all_questions, key=lambda q: q['modes'].get(mode_name, {}).get('context_length', 0))['modes'].get(mode_name, {}).get('context_length', 0):>14,}")

    print()

    # Per-question breakdown
    print("=" * 80)
    print("PER-QUESTION BREAKDOWN")
    print("=" * 80)
    print(f"{'Q-ID':<12} {'Reasoning':<10} {'Total Input':<13} {'Total Output':<13} {'Total Cost':<12} {'Question':<30}")
    print("-" * 80)

    for q in all_questions:
        reasoning_flag = "Yes" if q["requires_reasoning"] else "No"
        print(f"{q['question_id']:<12} "
              f"{reasoning_flag:<10} "
              f"{q['total_input_tokens']:>12,} "
              f"{q['total_output_tokens']:>12,} "
              f"${q['total_cost']:>11.4f} "
              f"{q['question']:<30}")

    print()

    # Overall cost summary
    print("=" * 80)
    print("COST SUMMARY - Gemini-3-Flash-Preview")
    print("=" * 80)
    print(f"Total input tokens:        {stats['overall']['total_input_tokens']:>15,}")
    print(f"Total output tokens:       {stats['overall']['total_output_tokens']:>15,}")
    print(f"  - Response tokens:       {stats['overall']['total_response_tokens']:>15,}")
    print(f"  - Thought tokens:        {stats['overall']['total_thought_tokens']:>15,}")
    print()
    print(f"Input cost (${INPUT_PRICE_PER_M}/M):   ${stats['overall']['total_input_cost']:>14.4f}")
    print(f"Output cost (${OUTPUT_PRICE_PER_M}/M):  ${stats['overall']['total_output_cost']:>14.4f}")
    print("-" * 80)
    print(f"TOTAL ESTIMATED COST:      ${stats['overall']['total_cost']:>14.4f}")
    print("=" * 80)
    print()

    # Additional insights
    if stats['overall']['total_thought_tokens'] > 0:
        thought_pct = (stats['overall']['total_thought_tokens'] / stats['overall']['total_output_tokens']) * 100
        print(f"💡 Reasoning mode usage: {thought_pct:.1f}% of output tokens are thoughts")
        print()


# ============================================
# Main Execution
# ============================================

def main():
    print("=" * 80)
    print("CTAS Multi-Mode Token Estimator")
    print("=" * 80)
    print()

    # Check if results folder exists
    results_path = Path(RESULTS_FOLDER)
    if not results_path.exists():
        print(f"❌ ERROR: Results folder not found: {RESULTS_FOLDER}")
        print(f"   Please ensure the folder exists and contains CTAS result files.")
        sys.exit(1)

    # Find all CTAS result files
    ctas_files = sorted(results_path.glob(CTAS_FILE_PATTERN))

    if not ctas_files:
        print(f"❌ ERROR: No files matching pattern '{CTAS_FILE_PATTERN}' found in {RESULTS_FOLDER}")
        sys.exit(1)

    print(f"✓ Found {len(ctas_files)} question files to analyze")
    print()

    # Analyze all questions
    print("Analyzing token usage...")
    all_questions = []

    for file_path in ctas_files:
        try:
            question_analysis = analyze_question_file(file_path)
            all_questions.append(question_analysis)
            print(f"  ✓ {file_path.name}")
        except Exception as e:
            print(f"  ❌ Error processing {file_path.name}: {e}")

    print()

    if not all_questions:
        print("❌ No questions were successfully analyzed.")
        sys.exit(1)

    # Aggregate statistics
    print("Aggregating statistics...")
    stats = aggregate_statistics(all_questions)
    print()

    # Generate report
    print_report(stats, all_questions)

    # Save detailed stats to JSON
    output_data = {
        "token_counting_method": TOKEN_METHOD,
        "pricing": {
            "input_per_million": INPUT_PRICE_PER_M,
            "output_per_million": OUTPUT_PRICE_PER_M,
            "model": "gemini-3-flash-preview"
        },
        "summary": stats,
        "questions": all_questions
    }

    with open(OUTPUT_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"📊 Detailed statistics saved to: {OUTPUT_STATS_FILE}")
    print()

    # Provide recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Find most efficient mode
    mode_efficiencies = []
    for mode_name in QUERY_MODES:
        mode_stats = stats["per_mode"][mode_name]
        if mode_stats["success_count"] > 0:
            avg_total = mode_stats["avg_input_tokens"] + mode_stats["avg_output_tokens"]
            mode_efficiencies.append((mode_name, avg_total, mode_stats["total_cost"]))

    mode_efficiencies.sort(key=lambda x: x[1])

    print(f"Most token-efficient mode:  {mode_efficiencies[0][0]} ({mode_efficiencies[0][1]:,.0f} avg tokens)")
    print(f"Most costly mode:           {mode_efficiencies[-1][0]} ({mode_efficiencies[-1][1]:,.0f} avg tokens)")
    print()

    if stats['overall']['total_thought_tokens'] == 0 and stats['reasoning_questions'] > 0:
        print("⚠  Note: Reasoning questions detected but no thoughts generated.")
        print("   Ensure thinking_config is properly enabled in generate_answers_ctas.py")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
