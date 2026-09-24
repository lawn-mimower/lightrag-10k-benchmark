import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
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

SCRIPT = "local-llm/benchmark_finder.py"


@pytest.fixture
def bench(load_script):
    FakeLlama.instances.clear()
    return load_script(SCRIPT, stubs=STUBS)


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


@pytest.mark.parametrize("argv, expected", [
    ([], dict(device="cpu", n_ctx=8192, num_docs=10,
              working_dir="./finder_benchmark_workdir1", output="finder_lightrag_results.csv")),
    (["--device", "gpu"], dict(device="gpu", n_ctx=131072, num_docs=None,
                               working_dir="./finder_benchmark_workdir_gpu", output="finder_lightrag_results_gpu.csv")),
    (["--query-only"], dict(device="cpu", n_ctx=32768, num_docs=10,
                            working_dir="./finder_benchmark_workdir1", output="finder_lightrag_results_query_only.csv")),
    (["--device", "gpu", "--query-only", "--n-ctx", "65536"],
     dict(device="gpu", n_ctx=65536, num_docs=None,
          working_dir="./finder_benchmark_workdir_gpu", output="finder_lightrag_results_gpu_query_only.csv")),
], ids=["cpu", "gpu", "cpu-query-only", "gpu-query-only"])
def test_profile_defaults(bench, argv, expected):
    args = bench.parse_args(argv)
    assert {key: getattr(args, key) for key in expected} == expected


@pytest.mark.parametrize("device, n_gpu_layers, n_threads, embedder_device", [
    ("cpu", 0, 8, "cpu"),
    ("gpu", -1, 4, "cuda"),
])
def test_models_load_with_profile_settings(bench, device, n_gpu_layers, n_threads, embedder_device):
    profile = bench.DEVICE_PROFILES[device]
    bench.LlamaCppWrapper("model.gguf", n_ctx=profile["n_ctx"], n_threads=profile["n_threads"],
                          n_gpu_layers=profile["n_gpu_layers"])
    kwargs = FakeLlama.instances[-1].kwargs
    assert (kwargs["n_gpu_layers"], kwargs["n_threads"], kwargs["n_ctx"]) == (n_gpu_layers, n_threads, profile["n_ctx"])

    embedder = bench.QwenEmbedderWrapper("qwen3", device=profile["embedder_device"])
    assert embedder.model.kwargs["device"] == embedder_device


class FakeRAG:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inserted = []
        FakeRAG.instances.append(self)

    async def initialize_storages(self):
        pass

    async def ainsert(self, docs):
        self.inserted.append(docs)

    async def aquery(self, query, param=None):
        return "answer for " + query


@pytest.mark.parametrize("argv, inserts", [
    ([], ["ctx a", "ctx b"]),                   # cpu: one ainsert per context
    (["--device", "gpu"], [["ctx a", "ctx b", "ctx c"]]),  # gpu: all contexts in one call
    (["--query-only"], []),                     # existing index: no inserts
], ids=["cpu", "gpu", "query-only"])
def test_main_indexing_follows_profile(bench, monkeypatch, tmp_path, argv, inserts):
    FakeRAG.instances.clear()
    monkeypatch.setattr(bench, "LightRAG", FakeRAG)
    df = pd.DataFrame({
        "text": ["Q1?", "Q2?", "Q3?"],
        "answer": ["A1", "A2", "A3"],
        "references": [["ctx a"], ["ctx b"], ["ctx c"]],
    })
    monkeypatch.setattr(bench.pd, "read_parquet", lambda path: df)
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "graph.json").write_text("{}")
    out = tmp_path / "out.csv"
    args = bench.parse_args(argv + ["--num-docs", "3" if "gpu" in argv else "2",
                                    "--working-dir", str(workdir), "--output", str(out)])

    asyncio.run(bench.main(args))

    rag = FakeRAG.instances[-1]
    assert rag.inserted == inserts
    profile = bench.DEVICE_PROFILES[args.device]
    assert rag.kwargs["chunk_token_size"] == profile["chunk_token_size"]
    assert rag.kwargs["entity_extract_max_gleaning"] == profile["entity_extract_max_gleaning"]
    # the index is kept for --query-only and recreated otherwise
    assert (workdir / "graph.json").exists() == args.query_only
    results = pd.read_csv(out)
    assert results["lightrag_generated_answer"].str.startswith("answer for ").all()


def test_query_only_requires_existing_index(bench, tmp_path, capsys):
    missing = tmp_path / "missing"
    args = bench.parse_args(["--query-only", "--working-dir", str(missing), "--output", str(tmp_path / "o.csv")])
    asyncio.run(bench.main(args))
    assert "Working directory not found" in capsys.readouterr().out
    assert not missing.exists() and not (tmp_path / "o.csv").exists()
