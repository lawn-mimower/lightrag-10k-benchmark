"""Offline checks for helper scripts, the RAGAS batch CLI and the notebooks."""
import ast
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

LIGHTRAG_CONTEXT = """Knowledge Graph Data (Entity):

```json
{"entity": "Cintas", "type": "organization", "description": "Uniform company."}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{"reference_id": "1", "content": "Total revenue was $9.60 billion."}
{"reference_id": "2", "content": "Uniform Rental grew 7.8%."}
```

Reference Document List:
[1] CTAS.html
"""


def test_extract_document_chunks_from_lightrag_context():
    import transform_contexts

    assert transform_contexts.extract_document_chunks(LIGHTRAG_CONTEXT) == [
        "Total revenue was $9.60 billion.",
        "Uniform Rental grew 7.8%.",
    ]
    assert transform_contexts.extract_document_chunks("no chunks here") == []


def test_transform_contexts_cli(tmp_path):
    src, dst = tmp_path / "in.json", tmp_path / "out.json"
    src.write_text(json.dumps([{"question_id": "q1", "retrieved_context": LIGHTRAG_CONTEXT}]))
    result = subprocess.run([sys.executable, str(REPO_ROOT / "transform_contexts.py"), str(src), str(dst)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    out = json.loads(dst.read_text())
    assert out[0]["retrieved_contexts"][0] == "Total revenue was $9.60 billion."
    assert "retrieved_context" not in out[0]


def test_analyze_mode_performance_on_result_file(load_script, tmp_path):
    amp = load_script("analyze_mode_performance.py")
    path = tmp_path / "test_results_CTAS_question_q1.json"
    # Timestamps are written when each mode *finishes*; modes run in this order.
    finished = {"local": 3, "global": 5, "naive": 20, "hybrid": 26, "mix": 50}
    modes = {mode: {"status": "success", "answer": "x" * 400, "retrieved_context": "c" * 4000,
                    "answer_length": 400, "context_length": 4000,
                    "timestamp": f"2026-01-01T00:00:{sec:02d}.000001"}
             for mode, sec in finished.items()}
    path.write_text(json.dumps({"question_id": "q1", "question": "q" * 100, "modes": modes}))

    result = amp.analyze_file(path)
    durations = {mode: result[mode]["retrieval_time"] for mode in finished}
    assert durations == {"local": None, "global": 2.0, "naive": 15.0, "hybrid": 6.0, "mix": 24.0}
    assert result["naive"]["input_tokens"] == (4000 + 100) // 4

    prev_question_end = amp.parse_timestamp("2026-01-01T00:00:01.000001")
    assert amp.analyze_file(path, prev_question_end)["local"]["retrieval_time"] == 2.0


RAGAS_SCRIPT = REPO_ROOT / "batch_ragas_evaluation.py"


def test_ragas_batch_help_lists_options():
    result = subprocess.run([sys.executable, str(RAGAS_SCRIPT), "--help"],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    for option in ("--results-dir", "--output", "--checkpoint", "--modes", "--limit", "--strategy"):
        assert option in result.stdout


def test_ragas_batch_requires_api_key(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("OUTPUT_FILE", "CHECKPOINT_FILE")}
    env["MISTRAL_API_KEY"] = ""
    result = subprocess.run([sys.executable, str(RAGAS_SCRIPT), "--results-dir", str(tmp_path)],
                            capture_output=True, text=True, timeout=300, env=env, cwd=tmp_path)
    assert result.returncode == 1
    assert "MISTRAL_API_KEY not found" in result.stdout
    assert "batch_ragas_evaluation_results_ultra_simple.json" in result.stdout


def test_ragas_batch_adaptive_strategy_defaults(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("OUTPUT_FILE", "CHECKPOINT_FILE")}
    env["MISTRAL_API_KEY"] = ""
    result = subprocess.run([sys.executable, str(RAGAS_SCRIPT), "--strategy", "adaptive", "--results-dir", str(tmp_path)],
                            capture_output=True, text=True, timeout=300, env=env, cwd=tmp_path)
    assert result.returncode == 1
    assert "ADAPTIVE RATE LIMITING" in result.stdout
    assert "batch_ragas_evaluation_results_optimized.json" in result.stdout
    assert "Testing API connection" not in result.stdout


NOTEBOOKS = sorted(REPO_ROOT.glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_code_cells_compile(path):
    nb = json.loads(path.read_text())
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))
        compile(source, f"{path.name}[{index}]", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_multimode_notebook_ragas_cell_uses_imported_embeddings():
    nb = json.loads((REPO_ROOT / "lightrag_10k_priority_ticker_models_optimised_multimode.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "OpenAIEmbeddings(" not in code
    assert "from langchain_huggingface import HuggingFaceEmbeddings" in code
    assert "HuggingFaceEmbeddings(" in code


DOCUMENT_PARSING_SCRIPTS = {
    "parse_documents_with_docling.py", "parse_documents_with_easyocr.py",
    "docling_md_out_monitored.py", "mistral_document_extraction.py",
    "mistral_document_extraction_with_server.py", "mistral_scanned.py",
}


def test_no_absolute_home_paths_in_pipeline_code():
    paths = list(REPO_ROOT.glob("*.py")) + list(REPO_ROOT.glob("local-llm/*.py")) + NOTEBOOKS
    offenders = [
        f"{path.name}: {line.strip()[:80]}"
        for path in paths if path.name not in DOCUMENT_PARSING_SCRIPTS
        for line in path.read_text(encoding="utf-8").splitlines() if "/home/" in line
    ]
    assert offenders == []
