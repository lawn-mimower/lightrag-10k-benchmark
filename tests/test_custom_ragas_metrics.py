"""Metric arithmetic of custom_ragas_metrics.py with a scripted judge (no network)."""
import json

import pytest

import custom_ragas_metrics as crm


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.delenv("CUSTOM_RAGAS_MODEL", raising=False)
    return crm.CustomRAGASMetrics(api_key="test-key")


def script_judge(metrics, monkeypatch, replies):
    """Replace the LLM call with canned JSON replies, in order."""
    replies = list(replies)
    prompts = []

    def fake_call(prompt, json_output=False):
        prompts.append(prompt)
        return json.dumps(replies.pop(0))

    monkeypatch.setattr(metrics, "_call_llm", fake_call)
    return prompts


EVAL = crm.RAGEvaluation(
    question="What was Cintas' FY2024 revenue?",
    answer="Revenue was $9.6 billion, up 9%. The CEO is Todd Schneider.",
    contexts=["Total revenue was $9.60 billion.", "Unrelated text.", "Revenue grew 8.9%."],
    ground_truth="Revenue was $9.6 billion in fiscal 2024.",
)


def test_default_model_and_override(monkeypatch):
    assert crm.CustomRAGASMetrics(api_key="k").model.model_name.endswith("gemini-2.5-flash")
    monkeypatch.setenv("CUSTOM_RAGAS_MODEL", "gemini-x")
    assert crm.CustomRAGASMetrics(api_key="k").model.model_name.endswith("gemini-x")


def test_faithfulness_counts_supported_claims(metrics, monkeypatch):
    script_judge(metrics, monkeypatch, [
        {"claims": ["revenue $9.6B", "up 9%", "CEO is Todd Schneider"]},
        {"supported": True}, {"supported": True}, {"supported": False},
    ])
    result = metrics.faithfulness(EVAL)
    assert result["score"] == pytest.approx(2 / 3)
    assert (result["supported_claims"], result["total_claims"]) == (2, 3)


def test_faithfulness_without_claims_is_perfect(metrics, monkeypatch):
    script_judge(metrics, monkeypatch, [{"claims": []}])
    assert metrics.faithfulness(EVAL)["score"] == 1.0


def test_context_recall(metrics, monkeypatch):
    script_judge(metrics, monkeypatch, [{"claims": ["a", "b"]}, {"supported": True}, {"supported": False}])
    assert metrics.context_recall(EVAL)["score"] == 0.5


def test_context_precision_rewards_relevant_chunks_ranked_first(metrics, monkeypatch):
    script_judge(metrics, monkeypatch, [{"relevant": True}, {"relevant": False}, {"relevant": True}])
    result = metrics.context_precision(EVAL)
    # precision@1 = 1, precision@3 = 2/3 -> mean over relevant chunks
    assert result["score"] == pytest.approx((1 + 2 / 3) / 2)
    assert result["relevant_chunks"] == 2


def test_answer_correctness_combines_f1_and_similarity(metrics, monkeypatch):
    script_judge(metrics, monkeypatch, [
        {"answer_facts": ["a", "b", "c"], "ground_truth_facts": ["a", "b", "d"]},
        {"TP": ["a", "b"], "FP": ["c"], "FN": ["d"]},
        {"similarity": 0.8},
    ])
    result = metrics.answer_correctness(EVAL)
    assert result["f1_score"] == pytest.approx(2 / 3)
    assert result["score"] == pytest.approx(0.5 * 2 / 3 + 0.5 * 0.8)


def test_unparseable_judge_output_scores_zero(metrics, monkeypatch):
    monkeypatch.setattr(metrics, "_call_llm", lambda prompt, json_output=False: "")
    assert metrics.context_recall(EVAL)["score"] == 0.0


def test_evaluate_all_requires_ground_truth_for_reference_metrics(metrics, monkeypatch):
    script_judge(metrics, monkeypatch, [{"claims": []}])
    no_gt = crm.RAGEvaluation(question="q", answer="a", contexts=["c"])
    assert set(metrics.evaluate_all(no_gt)) == {"faithfulness"}
