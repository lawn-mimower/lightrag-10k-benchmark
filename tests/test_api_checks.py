"""Manual API check and benchmark scripts: option handling without network access."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from conftest import REPO_ROOT


def run_script(name, *args, **env_overrides):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(env_overrides)
    return subprocess.run([sys.executable, str(REPO_ROOT / name), *args],
                          capture_output=True, text=True, timeout=300, env=env)


@pytest.mark.parametrize("check, returncode, message", [
    ("quick", 1, "Please set MISTRAL_API_KEY"),
    ("params", 1, "MISTRAL_API_KEY not found"),
    ("diagnose", 0, "CONNECTION TESTS FAILED"),
])
def test_mistral_checks_stop_without_key(check, returncode, message):
    result = run_script("test_mistral_connection.py", "--check", check, MISTRAL_API_KEY="")
    assert result.returncode == returncode, result.stderr
    assert message in result.stdout


def test_mistral_check_rejects_unknown_option():
    result = run_script("test_mistral_connection.py", "--check", "ocr", MISTRAL_API_KEY="")
    assert result.returncode == 2 and "invalid choice" in result.stderr


@pytest.mark.parametrize("args, returncode, message", [
    ((), 1, "GEMINI_API_KEY environment variable not set"),
    (("--quick",), 1, "Set GEMINI_API_KEY"),
    (("--check",), 0, "Some checks failed"),
])
def test_gemini_benchmark_stops_without_key(args, returncode, message):
    result = run_script("benchmark_gemini_api.py", *args, GEMINI_API_KEY="")
    assert result.returncode == returncode, result.stderr
    assert message in result.stdout


class FakeGeminiModels:
    def __init__(self):
        self.prompts = []

    def generate_content(self, model, contents, config=None):
        self.prompts.append(contents)
        part = SimpleNamespace(text="Revenue was $9.6B.")
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=5, total_token_count=105)
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
                               usage_metadata=usage)


@pytest.fixture
def gemini_bench(load_script, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    module = load_script("benchmark_gemini_api.py")
    models = FakeGeminiModels()
    monkeypatch.setattr(module.genai, "Client", lambda api_key=None: SimpleNamespace(models=models))
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    results = tmp_path / "results"
    results.mkdir()
    modes = {m: {"status": "success", "retrieved_context": f"{m} context " + "x" * 60000}
             for m in ["local", "global", "naive", "hybrid", "mix"]}
    (results / "test_results_CTAS_question_q1.json").write_text(
        json.dumps({"question_id": "q1", "question": "Revenue?", "modes": modes}))
    return module, models, results


def test_gemini_quick_benchmark_uses_generation_prompt(gemini_bench, tmp_path):
    module, models, results = gemini_bench
    module.run_quick_benchmark(results, 1)

    assert len(models.prompts) == 5
    assert "Context from NAIVE:\nnaive context" in models.prompts[2]
    assert len(models.prompts[2]) < 51000  # context capped at 50,000 characters
    out = json.loads(next(tmp_path.glob("gemini_benchmark_*.json")).read_text())
    assert out["config"]["samples"] == 1
    assert out["summary"]["mix"]["avg_total_tokens"] == 105


def test_gemini_full_benchmark_statistics(gemini_bench, tmp_path):
    module, models, results = gemini_bench
    module.main(results, 1)

    assert len(models.prompts) == 5
    assert models.prompts[0].startswith("Based on the following context")
    out = json.loads(next(tmp_path.glob("gemini_benchmark_results_*.json")).read_text())
    assert out["configuration"]["sample_size"] == 1
    assert out["statistics"]["naive"]["total_tokens"]["sum"] == 105
