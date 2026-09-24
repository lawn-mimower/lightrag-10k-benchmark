"""Live end-to-end runs. Each test skips unless its keys or local models are configured.

    GEMINI_API_KEY                       -> Gemini answer generation
    MISTRAL_API_KEY (+ MISTRAL_BASE_URL) -> extraction endpoint (Mistral or any OpenAI-compatible server)
    E2E_NOTEBOOK=1                       -> full notebook pipeline on the bundled 10-K snippet
                                            (index, 5-mode query, answers, RAGAS judge)
    LOCAL_LLM_MODEL_PATH, LOCAL_EMBEDDER_PATH, FINDER_DATA_PATH
                                         -> local-llm/benchmark_finder.py on CPU (1 context, 1 query)
"""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

pytestmark = pytest.mark.e2e

NOTEBOOK = REPO_ROOT / "lightrag_10k_priority_ticker_models_optimised_multimode.ipynb"


def require_env(*names):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        pytest.skip("set " + ", ".join(missing))


def test_gemini_answer_generation_live(load_script, monkeypatch, tmp_path):
    require_env("GEMINI_API_KEY")
    from google import genai

    monkeypatch.chdir(tmp_path)
    gen = load_script("generate_answers_ctas.py")
    model = os.getenv("GEMINI_MODEL")
    if model:
        monkeypatch.setattr(gen, "MODEL_NAME", model)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    answer, _ = gen.generate_answer_logic(
        client,
        "What was Cintas' total revenue in fiscal 2024?",
        "Cintas Corporation reported total revenue of $9.60 billion for fiscal 2024.",
        "naive",
        "e2e-gemini",
    )
    assert answer and "9.6" in answer


def test_extraction_endpoint_live():
    require_env("MISTRAL_API_KEY")
    from lightrag.llm.openai import openai_complete_if_cache

    reply = asyncio.run(openai_complete_if_cache(
        os.getenv("MISTRAL_MODEL", "ministral-14b-2512"),
        "Reply with the single word OK.",
        api_key=os.environ["MISTRAL_API_KEY"],
        base_url=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
    ))
    assert reply.strip()


def _write_kernel_spec(root: Path) -> str:
    name = "lightrag10k-e2e"
    spec_dir = root / "kernels" / name
    spec_dir.mkdir(parents=True)
    (spec_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": name,
        "language": "python",
    }))
    return name


def test_pipeline_notebook_end_to_end(tmp_path, fixtures_dir, monkeypatch):
    if os.getenv("E2E_NOTEBOOK") != "1":
        pytest.skip("set E2E_NOTEBOOK=1 (slow: indexes a 10-K snippet and answers one question)")
    require_env("MISTRAL_API_KEY")
    if os.getenv("GENERATION_BACKEND", "gemini") == "gemini":
        require_env("GEMINI_API_KEY")
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pd = pytest.importorskip("pandas")

    # 1. Parse the bundled 10-K snippet exactly like the real corpus
    subprocess.run([sys.executable, str(REPO_ROOT / "html_parser.py"), str(fixtures_dir / "10k"),
                    str(tmp_path / "parsed_10k_documents.json"), "--workers", "1"], check=True)

    # 2. One FinDER-style question about that filing
    pd.DataFrame([{
        "_id": "e2e00001",
        "text": "Share of Cintas (CTAS) FY2024 revenue from Uniform Rental and Facility Services?",
        "reasoning": False,
        "category": "Financials",
        "references": ["Uniform Rental and Facility Services 77.8% of total revenue in fiscal 2024."],
        "answer": "Uniform Rental and Facility Services made up 77.8% of total revenue in fiscal 2024.",
        "type": "Compositional",
    }]).to_parquet(tmp_path / "finder_train.parquet")

    # 3. Run the notebook's indexing/query/answer cells (0-6) in tmp_path
    for key, value in {"TICKERS": "CTAS", "TEST_TICKER": "CTAS", "MAX_QUESTIONS": "1"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path / "jupyter"))
    kernel = _write_kernel_spec(tmp_path / "jupyter")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nb.cells = nb.cells[:7]
    nbclient.NotebookClient(nb, timeout=None, kernel_name=kernel,
                            resources={"metadata": {"path": str(tmp_path)}}).execute()

    results_dir = tmp_path / "5_modes_question_wise_results_with_answers" / "5_modes_question_wise_results_priority_tickers_ALL"
    files = sorted(results_dir.glob("test_results_*_question_*.json"))
    assert [f.name for f in files] == ["test_results_CTAS_question_e2e00001.json"]
    result = json.loads(files[0].read_text())
    modes = os.getenv("QUERY_MODES", "local,global,naive,hybrid,mix").split(",")
    assert list(result["modes"]) == modes
    for mode in modes:
        data = result["modes"][mode]
        assert data["status"] == "success", (mode, data.get("error"))
        assert data["answer_length"] == len(data["answer"])
    # Vector retrieval always finds the indexed chunks; graph modes can come back
    # empty when the extraction model found no matching entities/relations.
    eval_mode = "naive" if "naive" in modes else modes[0]
    assert result["modes"][eval_mode]["answer"].strip()
    assert result["modes"][eval_mode]["retrieved_context"].strip()

    # 4. RAGAS judge on one mode with the canonical batch script
    out = tmp_path / "ragas.json"
    subprocess.run([sys.executable, str(REPO_ROOT / "batch_ragas_evaluation.py"),
                    "--results-dir", str(results_dir), "--output", str(out),
                    "--checkpoint", str(tmp_path / "ragas_checkpoint.json"),
                    "--modes", eval_mode, "--limit", "1"], check=True, cwd=tmp_path)
    ragas = json.loads(out.read_text())
    assert ragas["evaluations_completed"] == 1
    assert ragas["results"][0]["status"] == "success", ragas["results"][0]
    assert set(ragas["results"][0]["metrics"]) == {
        "faithfulness", "answer_relevancy", "context_recall", "context_precision"}


def test_local_llm_benchmark_cpu(tmp_path):
    require_env("LOCAL_LLM_MODEL_PATH", "LOCAL_EMBEDDER_PATH", "FINDER_DATA_PATH")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("llama_cpp")

    out = tmp_path / "results.csv"
    env = dict(os.environ)
    # CPU inference is slow; LightRAG's default per-call timeouts are 180s/30s
    env.setdefault("LLM_TIMEOUT", "3600")
    env.setdefault("EMBEDDING_TIMEOUT", "900")
    subprocess.run([
        sys.executable, str(REPO_ROOT / "local-llm" / "benchmark_finder.py"),
        "--model-path", os.environ["LOCAL_LLM_MODEL_PATH"],
        "--embedder-path", os.environ["LOCAL_EMBEDDER_PATH"],
        "--data-path", os.environ["FINDER_DATA_PATH"],
        "--working-dir", str(tmp_path / "workdir"),
        "--num-docs", "1", "--max-queries", "1", "--output", str(out),
    ], check=True, env=env, cwd=tmp_path)

    df = pd.read_csv(out)
    assert len(df) == 1
    answer = str(df.loc[0, "lightrag_generated_answer"])
    assert answer.strip() and not answer.upper().startswith("ERROR")
