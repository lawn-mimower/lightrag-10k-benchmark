#!/usr/bin/env python3
"""
Answer Evaluation Script using NVIDIA Metrics Framework
Uses Ministral 3-14B via Mistral API
Implements: Answer Accuracy, Context Relevance, Response Groundedness
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
INPUT_FILE = "test_results_priority_tickers_MIX_rerank>0.3_topk10_with_answers.json"
OUTPUT_FILE = "test_results_priority_tickers_MIX_rerank>0.3_topk10_evaluated.json"
CHECKPOINT_FILE = "evaluation_checkpoint.json"
LOG_FILE = "evaluation_log.txt"

MODEL_NAME = "ministral-14b-2512"
TEMPERATURE = 0
EVALS_PER_MINUTE = 30  # Rate limit
EVAL_DELAY = 60 / EVALS_PER_MINUTE  # 2 seconds between calls
RETRY_DELAY = 60
MAX_RETRIES = 3


def log_message(message: str):
    """Write timestamped message to log file and console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")


def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint data if exists"""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"evaluated_question_ids": []}


def save_checkpoint(checkpoint_data: Dict[str, Any]):
    """Save checkpoint data"""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint_data, indent=2, fp=f)


# ==============================================================================
# METRIC 1: ANSWER ACCURACY
# Measures alignment between generated response and reference ground truth
# Scale: 0, 2, 4 → normalized to 0-1
# ==============================================================================

ANSWER_ACCURACY_PROMPT_1 = """You are evaluating the accuracy of a generated answer against a reference ground truth answer.

QUESTION:
{question}

REFERENCE ANSWER (Ground Truth):
{expected_answer}

GENERATED ANSWER:
{generated_answer}

Rate the accuracy on this scale:
- 0: The generated answer is inaccurate or doesn't address the question
- 2: The generated answer has partial alignment with the reference
- 4: The generated answer has exact or very strong alignment with the reference

Consider:
- Factual correctness
- Completeness of information
- Alignment with the reference answer's key points

Return ONLY a JSON object: {{"score": 0, "reasoning": "brief explanation"}}"""

ANSWER_ACCURACY_PROMPT_2 = """Evaluate how well the generated answer matches the expected reference answer.

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

Scoring rubric:
- 0 = Incorrect, irrelevant, or fails to answer the question
- 2 = Partially correct, missing key details or contains some errors
- 4 = Fully correct, captures all essential information accurately

Provide your evaluation as JSON: {{"score": 0, "reasoning": "explanation"}}"""


# ==============================================================================
# METRIC 2: CONTEXT RELEVANCE
# Evaluates whether retrieved contexts pertain to the user's query
# Scale: 0, 1, 2 → normalized to 0-1
# ==============================================================================

CONTEXT_RELEVANCE_PROMPT_1 = """Evaluate whether the provided context is relevant to answering the given question.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Rate the context relevance on this scale:
- 0: Context is not relevant to the question
- 1: Context is partially relevant, contains some useful information
- 2: Context is completely relevant and sufficient to answer the question

Return ONLY a JSON object: {{"score": 0, "reasoning": "brief explanation"}}"""

CONTEXT_RELEVANCE_PROMPT_2 = """Assess how relevant the retrieved context is for answering the user's question.

QUESTION:
{question}

CONTEXT:
{context}

Scoring criteria:
- 0 = Context does not contain information relevant to the question
- 1 = Context contains some relevant information but is incomplete or noisy
- 2 = Context fully pertains to the question and provides comprehensive information

Provide your evaluation as JSON: {{"score": 0, "reasoning": "explanation"}}"""


# ==============================================================================
# METRIC 3: RESPONSE GROUNDEDNESS
# Assesses how well response claims are supported by retrieved contexts
# Scale: 0, 1, 2 → normalized to 0-1
# ==============================================================================

GROUNDEDNESS_PROMPT_1 = """Evaluate whether the generated answer is grounded in (supported by) the provided context.

CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}

Rate the groundedness on this scale:
- 0: The answer is not grounded in the context (makes unsupported claims)
- 1: The answer is partially grounded (some claims supported, others not)
- 2: The answer is fully grounded (all claims are supported by the context)

Return ONLY a JSON object: {{"score": 0, "reasoning": "brief explanation"}}"""

GROUNDEDNESS_PROMPT_2 = """Assess how well the generated answer's claims are supported by the retrieved context.

CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}

Scoring rubric:
- 0 = Answer contains claims not found in or contradicted by the context
- 1 = Answer is partially supported, mix of grounded and ungrounded statements
- 2 = All statements in the answer are directly supported by the context

Provide your evaluation as JSON: {{"score": 0, "reasoning": "explanation"}}"""


def call_llm_judge(
    client: OpenAI,
    prompt: str,
    max_score: int,
    metric_name: str,
    attempt_num: int
) -> Tuple[float, str]:
    """Call LLM judge and parse response"""
    for retry in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                timeout=120
            )

            response_text = response.choices[0].message.content
            result = json.loads(response_text)

            # Handle if model returns array instead of object
            if isinstance(result, list):
                if len(result) > 0:
                    result = result[0]
                else:
                    return 0.0, "Error: Empty array returned"

            score = result.get("score", 0)
            reasoning = result.get("reasoning", "No reasoning provided")

            # Normalize to 0-1
            normalized_score = score / max_score

            return normalized_score, reasoning

        except json.JSONDecodeError as e:
            log_message(f"  ⚠ {metric_name} attempt {attempt_num} retry {retry+1}: JSON parse error")
            if retry < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                return 0.0, f"Error: Failed to parse JSON after {MAX_RETRIES} retries"

        except Exception as e:
            log_message(f"  ⚠ {metric_name} attempt {attempt_num} retry {retry+1}: {e}")
            if retry < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                return 0.0, f"Error: {str(e)}"

    return 0.0, "Error: Max retries exceeded"


def evaluate_answer_accuracy(
    client: OpenAI,
    question: str,
    expected_answer: str,
    generated_answer: str
) -> Dict[str, Any]:
    """Dual evaluation for Answer Accuracy"""
    log_message("    Evaluating: Answer Accuracy")

    # Evaluation 1
    prompt1 = ANSWER_ACCURACY_PROMPT_1.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer
    )
    score1, reasoning1 = call_llm_judge(client, prompt1, 4, "Answer Accuracy", 1)
    time.sleep(EVAL_DELAY)

    # Evaluation 2
    prompt2 = ANSWER_ACCURACY_PROMPT_2.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer
    )
    score2, reasoning2 = call_llm_judge(client, prompt2, 4, "Answer Accuracy", 2)
    time.sleep(EVAL_DELAY)

    avg_score = (score1 + score2) / 2

    return {
        "score": round(avg_score, 3),
        "eval1_score": round(score1, 3),
        "eval2_score": round(score2, 3),
        "eval1_reasoning": reasoning1,
        "eval2_reasoning": reasoning2
    }


def evaluate_context_relevance(
    client: OpenAI,
    question: str,
    context: str
) -> Dict[str, Any]:
    """Dual evaluation for Context Relevance"""
    log_message("    Evaluating: Context Relevance")

    # Truncate context if too long (50k chars = ~12.5k tokens)
    context_truncated = context[:50000]

    # Evaluation 1
    prompt1 = CONTEXT_RELEVANCE_PROMPT_1.format(
        question=question,
        context=context_truncated
    )
    score1, reasoning1 = call_llm_judge(client, prompt1, 2, "Context Relevance", 1)
    time.sleep(EVAL_DELAY)

    # Evaluation 2
    prompt2 = CONTEXT_RELEVANCE_PROMPT_2.format(
        question=question,
        context=context_truncated
    )
    score2, reasoning2 = call_llm_judge(client, prompt2, 2, "Context Relevance", 2)
    time.sleep(EVAL_DELAY)

    avg_score = (score1 + score2) / 2

    return {
        "score": round(avg_score, 3),
        "eval1_score": round(score1, 3),
        "eval2_score": round(score2, 3),
        "eval1_reasoning": reasoning1,
        "eval2_reasoning": reasoning2
    }


def evaluate_groundedness(
    client: OpenAI,
    context: str,
    generated_answer: str
) -> Dict[str, Any]:
    """Dual evaluation for Response Groundedness"""
    log_message("    Evaluating: Response Groundedness")

    # Truncate context if too long
    context_truncated = context[:50000]

    # Evaluation 1
    prompt1 = GROUNDEDNESS_PROMPT_1.format(
        context=context_truncated,
        generated_answer=generated_answer
    )
    score1, reasoning1 = call_llm_judge(client, prompt1, 2, "Groundedness", 1)
    time.sleep(EVAL_DELAY)

    # Evaluation 2
    prompt2 = GROUNDEDNESS_PROMPT_2.format(
        context=context_truncated,
        generated_answer=generated_answer
    )
    score2, reasoning2 = call_llm_judge(client, prompt2, 2, "Groundedness", 2)
    time.sleep(EVAL_DELAY)

    avg_score = (score1 + score2) / 2

    return {
        "score": round(avg_score, 3),
        "eval1_score": round(score1, 3),
        "eval2_score": round(score2, 3),
        "eval1_reasoning": reasoning1,
        "eval2_reasoning": reasoning2
    }


def evaluate_entry(client: OpenAI, entry: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
    """Evaluate a single entry on all three metrics"""
    question_id = entry["question_id"]
    log_message(f"  [{idx}/{total}] Evaluating {question_id}")

    # Skip if missing required fields
    if "response" not in entry:
        log_message(f"    ⚠ Skipping: no 'response' field")
        return None

    question = entry.get("question", "")
    expected_answer = entry.get("expected_answer", "")
    generated_answer = entry.get("response", "")
    context = entry.get("retrieved_context", "")

    evaluation = {
        "question_id": question_id,
        "timestamp": datetime.now().isoformat(),
    }

    # Metric 1: Answer Accuracy
    if expected_answer and generated_answer:
        evaluation["answer_accuracy"] = evaluate_answer_accuracy(
            client, question, expected_answer, generated_answer
        )
    else:
        log_message("    ⚠ Skipping Answer Accuracy: missing expected_answer or response")

    # Metric 2: Context Relevance
    if question and context:
        evaluation["context_relevance"] = evaluate_context_relevance(
            client, question, context
        )
    else:
        log_message("    ⚠ Skipping Context Relevance: missing question or context")

    # Metric 3: Response Groundedness
    if context and generated_answer:
        evaluation["groundedness"] = evaluate_groundedness(
            client, context, generated_answer
        )
    else:
        log_message("    ⚠ Skipping Groundedness: missing context or response")

    log_message(f"    ✓ Evaluation complete")
    return evaluation


def main():
    """Main execution function"""
    start_time = time.time()

    # Initialize Mistral client (OpenAI-compatible)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        log_message("ERROR: MISTRAL_API_KEY not found in environment")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1"
    )
    log_message(f"Initialized Mistral client with model: {MODEL_NAME}")

    # Load input data
    log_message(f"Loading input file: {INPUT_FILE}")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    total_questions = len(data)
    log_message(f"Loaded {total_questions} questions")

    # Load checkpoint
    checkpoint = load_checkpoint()
    evaluated_ids = set(checkpoint["evaluated_question_ids"])

    if evaluated_ids:
        log_message(f"Resuming from checkpoint: {len(evaluated_ids)} questions already evaluated")

    # Filter entries with responses
    entries_to_evaluate = [
        entry for entry in data
        if "response" in entry and entry["question_id"] not in evaluated_ids
    ]

    log_message(f"Evaluating {len(entries_to_evaluate)} questions")
    log_message("=" * 80)

    # Evaluate each entry
    for idx, entry in enumerate(entries_to_evaluate, 1):
        eval_result = evaluate_entry(client, entry, idx, len(entries_to_evaluate))

        if eval_result:
            # Add evaluation to entry
            entry["evaluation"] = eval_result

            # Update checkpoint
            evaluated_ids.add(entry["question_id"])
            checkpoint["evaluated_question_ids"] = list(evaluated_ids)
            save_checkpoint(checkpoint)

            # Save intermediate results
            with open(OUTPUT_FILE, "w") as f:
                json.dump(data, f, indent=2)

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    # Calculate summary statistics
    evaluations = [e.get("evaluation", {}) for e in data if "evaluation" in e]

    if evaluations:
        log_message("=" * 80)
        log_message("EVALUATION SUMMARY")

        # Answer Accuracy stats
        acc_scores = [e.get("answer_accuracy", {}).get("score", 0) for e in evaluations if "answer_accuracy" in e]
        if acc_scores:
            log_message(f"Answer Accuracy: avg={sum(acc_scores)/len(acc_scores):.3f}, min={min(acc_scores):.3f}, max={max(acc_scores):.3f}")

        # Context Relevance stats
        rel_scores = [e.get("context_relevance", {}).get("score", 0) for e in evaluations if "context_relevance" in e]
        if rel_scores:
            log_message(f"Context Relevance: avg={sum(rel_scores)/len(rel_scores):.3f}, min={min(rel_scores):.3f}, max={max(rel_scores):.3f}")

        # Groundedness stats
        ground_scores = [e.get("groundedness", {}).get("score", 0) for e in evaluations if "groundedness" in e]
        if ground_scores:
            log_message(f"Response Groundedness: avg={sum(ground_scores)/len(ground_scores):.3f}, min={min(ground_scores):.3f}, max={max(ground_scores):.3f}")

    elapsed = time.time() - start_time
    log_message(f"Total time: {elapsed/60:.1f} minutes")
    log_message(f"Questions evaluated: {len(evaluated_ids)}/{total_questions}")
    log_message(f"Output saved to: {OUTPUT_FILE}")
    log_message("=" * 80)


if __name__ == "__main__":
    main()
