"""
Custom implementation of RAGAS metrics using Google Gemini.
Replicates the methodology from RAGAS framework without using their codebase.

Metrics implemented:
1. Context Recall - measures % of ground truth claims supported by retrieved context
2. Context Precision - measures if relevant chunks are ranked higher than irrelevant ones
3. Faithfulness - measures % of answer claims supported by retrieved context
4. Answer Correctness - combines factual F1 score + semantic similarity

Based on RAGAS documentation: https://docs.ragas.io/
"""

import os
import json
import google.generativeai as genai
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class RAGEvaluation:
    """Container for RAG evaluation inputs"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str = None


class CustomRAGASMetrics:
    """
    Custom implementation of RAGAS metrics using Gemini API.
    """

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize the metrics calculator with Gemini model.

        Args:
            model_name: Name of Gemini model to use (default: CUSTOM_RAGAS_MODEL
                env var, else gemini-2.5-flash; gemini-2.0-flash-exp was retired)
            api_key: Google API key (if not provided, reads from GEMINI_API_KEY env var)
        """
        if model_name is None:
            model_name = os.getenv("CUSTOM_RAGAS_MODEL", "gemini-2.5-flash")
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _call_llm(self, prompt: str, json_output: bool = False) -> str:
        """
        Call the Gemini LLM with a prompt.

        Args:
            prompt: The prompt to send
            json_output: Whether to expect JSON output

        Returns:
            The model's response text
        """
        try:
            if json_output:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
            else:
                response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""

    def context_recall(self, eval_data: RAGEvaluation) -> Dict[str, Any]:
        """
        Calculate Context Recall metric.

        Context Recall = (# claims in ground truth supported by context) / (total claims in ground truth)

        Measures how much of the ground truth information was retrieved.

        Args:
            eval_data: RAGEvaluation object with question, contexts, and ground_truth

        Returns:
            Dict with score and details
        """
        if not eval_data.ground_truth:
            raise ValueError("Context Recall requires ground_truth")

        # Step 1: Extract claims from ground truth
        claims_prompt = f"""Given the following answer, extract all individual factual claims or statements.
Return them as a JSON array of strings.

Answer: {eval_data.ground_truth}

Return format:
{{"claims": ["claim 1", "claim 2", ...]}}
"""

        claims_response = self._call_llm(claims_prompt, json_output=True)
        try:
            claims_data = json.loads(claims_response)
            claims = claims_data.get("claims", [])
        except:
            claims = []

        if not claims:
            return {"score": 0.0, "total_claims": 0, "supported_claims": 0, "details": []}

        # Step 2: Check each claim against the context
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(eval_data.contexts)])

        supported_count = 0
        claim_details = []

        for claim in claims:
            verification_prompt = f"""Given the following claim and retrieved contexts, determine if the claim can be inferred or supported by the contexts.

Claim: {claim}

Retrieved Contexts:
{context_text}

Can this claim be inferred from or supported by the retrieved contexts?
Answer with JSON: {{"supported": true/false, "reason": "brief explanation"}}
"""

            verification_response = self._call_llm(verification_prompt, json_output=True)
            try:
                verification_data = json.loads(verification_response)
                is_supported = verification_data.get("supported", False)
                reason = verification_data.get("reason", "")
            except:
                is_supported = False
                reason = "Failed to parse LLM response"

            if is_supported:
                supported_count += 1

            claim_details.append({
                "claim": claim,
                "supported": is_supported,
                "reason": reason
            })

        score = supported_count / len(claims) if claims else 0.0

        return {
            "score": score,
            "total_claims": len(claims),
            "supported_claims": supported_count,
            "details": claim_details
        }

    def context_precision(self, eval_data: RAGEvaluation) -> Dict[str, Any]:
        """
        Calculate Context Precision metric.

        Context Precision = Mean of Precision@k for each retrieved chunk
        Measures if relevant chunks are ranked higher than irrelevant ones.

        Args:
            eval_data: RAGEvaluation object with question, contexts, and ground_truth

        Returns:
            Dict with score and details
        """
        if not eval_data.ground_truth:
            raise ValueError("Context Precision requires ground_truth")

        # Check relevance of each context chunk
        relevance_scores = []
        chunk_details = []

        for i, context in enumerate(eval_data.contexts):
            relevance_prompt = f"""Given a question and a retrieved context chunk, determine if this context chunk is relevant for answering the question.

Question: {eval_data.question}

Ground Truth Answer: {eval_data.ground_truth}

Context Chunk {i+1}: {context}

Is this context chunk relevant for answering the question?
Answer with JSON: {{"relevant": true/false, "reason": "brief explanation"}}
"""

            relevance_response = self._call_llm(relevance_prompt, json_output=True)
            try:
                relevance_data = json.loads(relevance_response)
                is_relevant = relevance_data.get("relevant", False)
                reason = relevance_data.get("reason", "")
            except:
                is_relevant = False
                reason = "Failed to parse LLM response"

            relevance_scores.append(1 if is_relevant else 0)
            chunk_details.append({
                "chunk_index": i,
                "relevant": is_relevant,
                "reason": reason
            })

        # Calculate precision@k for each position
        precision_at_k = []
        for k in range(1, len(relevance_scores) + 1):
            relevant_at_k = sum(relevance_scores[:k])
            precision_k = relevant_at_k / k
            precision_at_k.append(precision_k * relevance_scores[k-1])  # Weighted by relevance

        # Calculate final score
        total_relevant = sum(relevance_scores)
        if total_relevant == 0:
            score = 0.0
        else:
            score = sum(precision_at_k) / total_relevant

        return {
            "score": score,
            "total_chunks": len(eval_data.contexts),
            "relevant_chunks": total_relevant,
            "precision_at_k": precision_at_k,
            "details": chunk_details
        }

    def faithfulness(self, eval_data: RAGEvaluation) -> Dict[str, Any]:
        """
        Calculate Faithfulness metric.

        Faithfulness = (# claims in answer supported by context) / (total claims in answer)

        Measures how factually consistent the answer is with retrieved context.

        Args:
            eval_data: RAGEvaluation object with answer and contexts

        Returns:
            Dict with score and details
        """
        # Step 1: Extract claims from the generated answer
        claims_prompt = f"""Given the following answer, extract all individual factual claims or statements.
Return them as a JSON array of strings.

Answer: {eval_data.answer}

Return format:
{{"claims": ["claim 1", "claim 2", ...]}}
"""

        claims_response = self._call_llm(claims_prompt, json_output=True)
        try:
            claims_data = json.loads(claims_response)
            claims = claims_data.get("claims", [])
        except:
            claims = []

        if not claims:
            return {"score": 1.0, "total_claims": 0, "supported_claims": 0, "details": []}

        # Step 2: Verify each claim against the context
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(eval_data.contexts)])

        supported_count = 0
        claim_details = []

        for claim in claims:
            verification_prompt = f"""Given the following claim and retrieved contexts, determine if the claim can be inferred or supported by the contexts.

Claim: {claim}

Retrieved Contexts:
{context_text}

Can this claim be inferred from or supported by the retrieved contexts?
Answer with JSON: {{"supported": true/false, "reason": "brief explanation"}}
"""

            verification_response = self._call_llm(verification_prompt, json_output=True)
            try:
                verification_data = json.loads(verification_response)
                is_supported = verification_data.get("supported", False)
                reason = verification_data.get("reason", "")
            except:
                is_supported = False
                reason = "Failed to parse LLM response"

            if is_supported:
                supported_count += 1

            claim_details.append({
                "claim": claim,
                "supported": is_supported,
                "reason": reason
            })

        score = supported_count / len(claims) if claims else 1.0

        return {
            "score": score,
            "total_claims": len(claims),
            "supported_claims": supported_count,
            "details": claim_details
        }

    def answer_correctness(self, eval_data: RAGEvaluation) -> Dict[str, Any]:
        """
        Calculate Answer Correctness metric.

        Combines:
        1. Factual F1 score (based on TP, FP, FN of facts)
        2. Semantic similarity

        Final score = weighted average of both

        Args:
            eval_data: RAGEvaluation object with answer and ground_truth

        Returns:
            Dict with score and details
        """
        if not eval_data.ground_truth:
            raise ValueError("Answer Correctness requires ground_truth")

        # Step 1: Extract facts from both answer and ground truth
        facts_prompt = f"""Extract all factual statements from the following answer and ground truth.
Return them as JSON with two arrays.

Generated Answer: {eval_data.answer}

Ground Truth: {eval_data.ground_truth}

Return format:
{{
    "answer_facts": ["fact 1", "fact 2", ...],
    "ground_truth_facts": ["fact 1", "fact 2", ...]
}}
"""

        facts_response = self._call_llm(facts_prompt, json_output=True)
        try:
            facts_data = json.loads(facts_response)
            answer_facts = facts_data.get("answer_facts", [])
            gt_facts = facts_data.get("ground_truth_facts", [])
        except:
            answer_facts = []
            gt_facts = []

        # Step 2: Calculate TP, FP, FN
        comparison_prompt = f"""Compare the facts from the generated answer with the ground truth facts.
Categorize each fact from the generated answer as:
- TP (True Positive): Fact is present in both answer and ground truth
- FP (False Positive): Fact is only in the generated answer, not in ground truth

Also identify:
- FN (False Negative): Facts in ground truth that are missing from the answer

Generated Answer Facts:
{json.dumps(answer_facts, indent=2)}

Ground Truth Facts:
{json.dumps(gt_facts, indent=2)}

Return JSON:
{{
    "TP": ["fact 1", "fact 2", ...],
    "FP": ["fact 1", "fact 2", ...],
    "FN": ["fact 1", "fact 2", ...]
}}
"""

        comparison_response = self._call_llm(comparison_prompt, json_output=True)
        try:
            comparison_data = json.loads(comparison_response)
            tp = comparison_data.get("TP", [])
            fp = comparison_data.get("FP", [])
            fn = comparison_data.get("FN", [])
        except:
            tp, fp, fn = [], [], []

        # Calculate F1 score
        tp_count = len(tp)
        fp_count = len(fp)
        fn_count = len(fn)

        if tp_count + fp_count + fn_count == 0:
            f1_score = 1.0
        else:
            f1_score = tp_count / (tp_count + 0.5 * (fp_count + fn_count))

        # Step 3: Calculate semantic similarity
        similarity_prompt = f"""Rate the semantic similarity between the generated answer and ground truth on a scale of 0.0 to 1.0.

Generated Answer: {eval_data.answer}

Ground Truth: {eval_data.ground_truth}

Consider:
- Do they convey the same meaning?
- Are the key concepts aligned?
- Is the information equivalent even if phrased differently?

Return JSON: {{"similarity": 0.0-1.0, "explanation": "brief explanation"}}
"""

        similarity_response = self._call_llm(similarity_prompt, json_output=True)
        try:
            similarity_data = json.loads(similarity_response)
            semantic_similarity = similarity_data.get("similarity", 0.0)
            similarity_explanation = similarity_data.get("explanation", "")
        except:
            semantic_similarity = 0.0
            similarity_explanation = "Failed to parse LLM response"

        # Step 4: Combine scores (default weights: 0.5 for each)
        weight_factual = 0.5
        weight_semantic = 0.5

        final_score = (weight_factual * f1_score) + (weight_semantic * semantic_similarity)

        return {
            "score": final_score,
            "f1_score": f1_score,
            "semantic_similarity": semantic_similarity,
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "details": {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "similarity_explanation": similarity_explanation
            }
        }

    def evaluate_all(self, eval_data: RAGEvaluation) -> Dict[str, Any]:
        """
        Calculate all metrics at once.

        Args:
            eval_data: RAGEvaluation object with all required fields

        Returns:
            Dict with all metric scores
        """
        results = {}

        # Calculate each metric
        try:
            results["faithfulness"] = self.faithfulness(eval_data)
        except Exception as e:
            results["faithfulness"] = {"error": str(e)}

        try:
            if eval_data.ground_truth:
                results["answer_correctness"] = self.answer_correctness(eval_data)
        except Exception as e:
            results["answer_correctness"] = {"error": str(e)}

        try:
            if eval_data.ground_truth:
                results["context_recall"] = self.context_recall(eval_data)
        except Exception as e:
            results["context_recall"] = {"error": str(e)}

        try:
            if eval_data.ground_truth:
                results["context_precision"] = self.context_precision(eval_data)
        except Exception as e:
            results["context_precision"] = {"error": str(e)}

        return results


def main():
    """Example usage"""
    # Example evaluation
    eval_data = RAGEvaluation(
        question="Where was Albert Einstein born?",
        answer="Albert Einstein was born in Ulm, Germany on March 20, 1879.",
        contexts=[
            "Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire.",
            "Einstein's family moved to Munich when he was an infant.",
            "The theory of relativity was developed by Einstein in the early 20th century."
        ],
        ground_truth="Albert Einstein was born in Ulm, Germany on March 14, 1879."
    )

    # Initialize metrics calculator
    metrics = CustomRAGASMetrics()

    # Calculate individual metrics
    print("=" * 80)
    print("FAITHFULNESS")
    print("=" * 80)
    faithfulness_result = metrics.faithfulness(eval_data)
    print(f"Score: {faithfulness_result['score']:.3f}")
    print(f"Supported claims: {faithfulness_result['supported_claims']}/{faithfulness_result['total_claims']}")
    print()

    print("=" * 80)
    print("ANSWER CORRECTNESS")
    print("=" * 80)
    correctness_result = metrics.answer_correctness(eval_data)
    print(f"Score: {correctness_result['score']:.3f}")
    print(f"F1 Score: {correctness_result['f1_score']:.3f}")
    print(f"Semantic Similarity: {correctness_result['semantic_similarity']:.3f}")
    print()

    print("=" * 80)
    print("CONTEXT RECALL")
    print("=" * 80)
    recall_result = metrics.context_recall(eval_data)
    print(f"Score: {recall_result['score']:.3f}")
    print(f"Supported claims: {recall_result['supported_claims']}/{recall_result['total_claims']}")
    print()

    print("=" * 80)
    print("CONTEXT PRECISION")
    print("=" * 80)
    precision_result = metrics.context_precision(eval_data)
    print(f"Score: {precision_result['score']:.3f}")
    print(f"Relevant chunks: {precision_result['relevant_chunks']}/{precision_result['total_chunks']}")
    print()

    # Or evaluate all at once
    print("=" * 80)
    print("ALL METRICS")
    print("=" * 80)
    all_results = metrics.evaluate_all(eval_data)
    print(json.dumps({k: v.get("score", "error") for k, v in all_results.items()}, indent=2))


if __name__ == "__main__":
    main()
