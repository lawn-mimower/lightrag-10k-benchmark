"""Answer generation and LLM-judge scripts with a fake Gemini client (no network)."""
import json
from types import SimpleNamespace

import pytest


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def fake_client(*responses):
    return SimpleNamespace(models=FakeModels(responses))


def text_response(text):
    part = SimpleNamespace(text=text, thought=False)
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


def thinking_response(thought, answer):
    parts = [SimpleNamespace(text=thought, thought=True), SimpleNamespace(text=answer, thought=False)]
    return SimpleNamespace(text=answer, candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])


@pytest.fixture
def ctas(load_script, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # log/checkpoint files are written to the working directory
    module = load_script("generate_answers_ctas.py")
    monkeypatch.setattr(module, "RETRY_DELAY", 0)
    return module


def test_generate_answer_plain(ctas):
    client = fake_client(text_response("  Revenue rose 9%.  "))
    answer, thought = ctas.generate_answer_logic(client, "Q?", "ctx", "naive", "q1")
    assert (answer, thought) == ("Revenue rose 9%.", None)
    call = client.models.calls[0]
    assert call["model"] == ctas.MODEL_NAME
    assert "Question: Q?" in call["contents"] and "Context from NAIVE:\nctx" in call["contents"]
    assert call["config"].thinking_config is None


def test_generate_answer_with_reasoning_splits_thoughts(ctas):
    client = fake_client(thinking_response("step 1", "final answer"))
    answer, thought = ctas.generate_answer_logic(client, "Q?", "ctx", "mix", "q1", requires_reasoning=True)
    assert (answer, thought) == ("final answer", "step 1")
    assert client.models.calls[0]["config"].thinking_config.include_thoughts is True


def test_generate_answer_retries_then_gives_up(ctas):
    client = fake_client(*[RuntimeError("quota")] * ctas.MAX_RETRIES)
    assert ctas.generate_answer_logic(client, "Q?", "ctx", "local", "q1") == (None, None)
    assert len(client.models.calls) == ctas.MAX_RETRIES


def test_process_file_adds_responses_for_each_mode(ctas, tmp_path):
    path = tmp_path / "test_results_CTAS_question_abc.json"
    path.write_text(json.dumps({
        "question_id": "abc",
        "question": "What drives Cintas revenue?",
        "reasoning": False,
        "modes": {
            "naive": {"retrieved_context": "uniform rental", "status": "success"},
            "local": {"retrieved_context": "", "status": "success"},
            "global": {"retrieved_context": "ctx", "response": "kept", "status": "success"},
        },
    }))
    client = fake_client(text_response("Uniform rental."))
    checkpoint, made_calls = ctas.process_ctas_file(path, client, {"completed": {}})

    data = json.loads(path.read_text())
    assert made_calls is True
    assert data["modes"]["naive"]["response"] == "Uniform rental."
    assert "response" not in data["modes"]["local"]  # no context -> skipped
    assert data["modes"]["global"]["response"] == "kept"
    assert sorted(checkpoint["completed"][path.name]["completed_modes"]) == ["global", "naive"]


def test_always_reason_regenerates_every_mode(ctas, tmp_path):
    path = tmp_path / "test_results_CTAS_question_abc.json"
    path.write_text(json.dumps({
        "question_id": "abc",
        "question": "What drives Cintas revenue?",
        "reasoning": False,
        "modes": {
            "naive": {"retrieved_context": "uniform rental", "response": "old answer", "status": "success"},
            "local": {"retrieved_context": "", "status": "success"},
        },
    }))
    client = fake_client(thinking_response("weighing segments", "Uniform rental."))
    checkpoint, made_calls = ctas.process_ctas_file(path, client, {"completed": {}}, always_reason=True)

    data = json.loads(path.read_text())
    naive = data["modes"]["naive"]
    assert made_calls is True
    assert (naive["response_old"], naive["response"], naive["thought"]) == (
        "old answer", "Uniform rental.", "weighing segments")
    assert "response" not in data["modes"]["local"]
    call = client.models.calls[0]
    assert call["config"].thinking_config.include_thoughts is True
    assert "requires detailed reasoning" in call["contents"]
    assert checkpoint["completed"][path.name]["completed_modes"] == ["naive"]


class FakeHTTPResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def test_vertex_express_answers_modes_without_response(ctas, monkeypatch, tmp_path):
    import requests

    posts = []

    def fake_post(url, params=None, headers=None, json=None, timeout=None):
        posts.append({"url": url, "params": params, "json": json})
        return FakeHTTPResponse(200, {"candidates": [{"content": {"parts": [{"text": " Uniform rental. "}]}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(ctas.time, "sleep", lambda s: None)
    path = tmp_path / "test_results_CTAS_question_abc.json"
    path.write_text(json.dumps({
        "question_id": "abc",
        "question": "What drives Cintas revenue?",
        "modes": {
            "naive": {"retrieved_context": "uniform rental"},
            "global": {"retrieved_context": "ctx", "response": "kept"},
            "local": {"retrieved_context": ""},
        },
    }))
    ctas.process_file_vertex(path, "my-project", "test-key")

    data = json.loads(path.read_text())
    assert data["modes"]["naive"]["response"] == "Uniform rental."
    assert data["modes"]["global"]["response"] == "kept"
    assert "response" not in data["modes"]["local"]
    assert len(posts) == 1
    assert "/projects/my-project/locations/us-central1/" in posts[0]["url"]
    assert posts[0]["url"].endswith("/models/gemini-3.0-flash-preview-001:generateContent")
    assert posts[0]["params"] == {"key": "test-key"}
    assert posts[0]["json"]["contents"][0]["parts"][0]["text"].endswith("Answer based strictly on context:")


def test_generation_options_are_exclusive(ctas):
    assert ctas.parse_args(["--always-reason"]).always_reason is True
    with pytest.raises(SystemExit):
        ctas.parse_args(["--always-reason", "--vertex-express"])


def test_batch_prompt_lists_every_question(load_script, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    gen = load_script("generate_answers.py")
    prompt = gen.create_batch_prompt([
        {"question_id": "q1", "question": "First?", "retrieved_context": "ctx one"},
        {"question_id": "q2", "question": "Second?"},
    ])
    assert "QUESTION 1 (ID: q1):\nFirst?" in prompt and "ctx one" in prompt
    assert "[NO CONTEXT AVAILABLE]" in prompt
    assert prompt.rstrip().endswith('"q2": "your answer here"\n}')


def test_batch_answers_parse_json(load_script, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    gen = load_script("generate_answers.py")
    client = fake_client(text_response('{"q1": "A1"}'))
    assert gen.generate_batch_answers(client, [{"question_id": "q1", "question": "Q"}], 1) == {"q1": "A1"}


def fake_openai_client(*contents):
    """OpenAI-compatible client (used for the Mistral judge) returning canned message contents."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=contents[len(calls) - 1])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions), calls=calls)


@pytest.fixture
def judge(load_script, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the judge logs to the working directory
    return load_script("evaluate_answers.py")


@pytest.mark.parametrize("judge_name", ["gemini", "mistral"])
def test_llm_judge_normalises_scores(judge, judge_name):
    judge.configure_judge(judge_name)
    reply = '{"score": 2, "reasoning": "partial"}'
    client = fake_client(text_response(reply)) if judge_name == "gemini" else fake_openai_client(reply)
    assert judge.call_llm_judge(client, "prompt", 4, "Answer Accuracy", 1) == (0.5, "partial")


def test_mistral_judge_request_and_array_reply(judge):
    judge.configure_judge("mistral")
    client = fake_openai_client('[{"score": 1, "reasoning": "ok"}]', "[]")
    assert judge.call_llm_judge(client, "prompt", 2, "Groundedness", 1) == (0.5, "ok")
    assert judge.call_llm_judge(client, "prompt", 2, "Groundedness", 1) == (0.0, "Error: Empty array returned")
    call = client.calls[0]
    assert call["model"] == "ministral-14b-2512"
    assert call["response_format"] == {"type": "json_object"} and call["timeout"] == 120
    assert judge.EVAL_DELAY == 2


def test_judge_prompts_show_scale_to_gemini_only(judge):
    fields = dict(question="Q?", expected_answer="E", generated_answer="G")
    judge.configure_judge("gemini")
    assert '"score": 0/2/4' in judge.judge_prompt(judge.ANSWER_ACCURACY_PROMPT_1, **fields)
    assert judge.EVAL_DELAY == 3
    judge.configure_judge("mistral")
    prompt = judge.judge_prompt(judge.ANSWER_ACCURACY_PROMPT_1, **fields)
    assert '"score": 0, "reasoning"' in prompt and "0/2/4" not in prompt


def test_llm_judge_reports_unparseable_output(judge, monkeypatch):
    monkeypatch.setattr(judge.time, "sleep", lambda s: None)
    client = fake_client(*[text_response("not json")] * judge.MAX_RETRIES)
    score, reason = judge.call_llm_judge(client, "prompt", 2, "Context Relevance", 1)
    assert score == 0.0 and reason.startswith("Error: Failed to parse JSON")
