import asyncio
import sys

import numpy as np
import pytest


class FakeLlama:
    instances = []

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.prompts = []
        FakeLlama.instances.append(self)

    def __call__(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return {"choices": [{"text": "  model answer  "}]}


class FakeSentenceTransformer:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.device = kwargs.get("device", "cpu")
        self.encoded = []

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        self.encoded.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)


STUBS = {
    "llama_cpp": {"Llama": FakeLlama},
    "sentence_transformers": {"SentenceTransformer": FakeSentenceTransformer},
}


@pytest.fixture(params=["local-llm/benchmark_finder.py", "local-llm/benchmark_finder_gpu.py"])
def bench(request, load_script):
    FakeLlama.instances.clear()
    return load_script(request.param, stubs=STUBS)


def test_format_llama3_prompt_includes_history(bench):
    prompt = bench.format_llama3_prompt(
        "find more entities",
        system_prompt="You extract entities.",
        history_messages=[
            {"role": "user", "content": "extract from chunk"},
            {"role": "assistant", "content": "(entity<|#|>Cintas)"},
        ],
    )
    assert prompt == (
        "<|start_header_id|>system<|end_header_id|>\n\nYou extract entities.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nextract from chunk<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n(entity<|#|>Cintas)<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nfind more entities<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def test_format_llama3_prompt_without_system_or_history(bench):
    assert bench.format_llama3_prompt("hi") == (
        "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def test_llm_wrapper_strips_query_flag_and_passes_history(bench):
    wrapper = bench.LlamaCppWrapper("model.gguf")
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}]
    out = asyncio.run(wrapper(bench.QUERY_FLAG + "question?", "sys", history_messages=history))

    assert out == "model answer"
    prompt, kwargs = FakeLlama.instances[-1].prompts[-1]
    assert bench.QUERY_FLAG not in prompt
    assert "question?" in prompt and "reply<|eot_id|>" in prompt
    assert kwargs["temperature"] == 0.0 and kwargs["stop"] == ["<|eot_id|>"]


def test_llm_wrapper_returns_error_text_on_failure(bench):
    wrapper = bench.LlamaCppWrapper("model.gguf")

    def boom(*args, **kwargs):
        raise ValueError("context overflow")

    wrapper.llm = boom
    out = asyncio.run(wrapper("q"))
    assert out.upper().startswith("ERROR")


def test_embedder_adds_instruction_only_for_flagged_queries(bench):
    embedder = bench.QwenEmbedderWrapper("qwen3")
    vectors = asyncio.run(embedder([bench.QUERY_FLAG + "What is revenue?", "Revenue was $9.6B."]))

    assert vectors.shape == (2, 4)
    sent = embedder.model.encoded[-1]
    assert sent[0].startswith("Instruct: Given a financial query") and sent[0].endswith("Query: What is revenue?")
    assert sent[1] == "Revenue was $9.6B."


def test_parse_args_defaults_and_overrides(bench, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.delenv("LLM_MODEL_PATH", raising=False)
    args = bench.parse_args()
    assert args.model_path == "./models/Llama-3.2-3B-Instruct.Q8_0.gguf"
    assert args.max_queries is None

    monkeypatch.setenv("LLM_MODEL_PATH", "/models/x.gguf")
    monkeypatch.setattr(sys, "argv", ["prog", "--num-docs", "1", "--max-queries", "2", "--output", "o.csv"])
    args = bench.parse_args()
    assert (args.model_path, args.num_docs, args.max_queries, args.output) == ("/models/x.gguf", 1, 2, "o.csv")
