"""Offline checks for helper scripts, the RAGAS batch CLI and the notebooks."""
import ast
import json
import os
import subprocess
import sys
from types import SimpleNamespace as NS

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
    "parse_documents_with_docling.py",
    "docling_md_out_monitored.py", "mistral_document_extraction.py",
    "mistral_document_extraction_with_server.py",
}


def test_no_absolute_home_paths_in_pipeline_code():
    paths = list(REPO_ROOT.glob("*.py")) + list(REPO_ROOT.glob("local-llm/*.py")) + NOTEBOOKS
    offenders = [
        f"{path.name}: {line.strip()[:80]}"
        for path in paths if path.name not in DOCUMENT_PARSING_SCRIPTS
        for line in path.read_text(encoding="utf-8").splitlines() if "/home/" in line
    ]
    assert offenders == []


@pytest.mark.parametrize("engine, options_class", [("rapidocr", "RapidOcrOptions"), ("easyocr", "EasyOcrOptions")])
def test_docling_parser_ocr_engine(load_script, monkeypatch, engine, options_class):
    parser = load_script("parse_documents_with_docling.py")
    built = {}

    def fake_converter(format_options):
        built["pipeline"] = next(iter(format_options.values())).pipeline_options
        return "converter"

    monkeypatch.setattr(parser, "DocumentConverter", fake_converter)
    assert parser.initialize_converter(engine) == "converter"
    pipeline = built["pipeline"]
    assert type(pipeline.ocr_options).__name__ == options_class
    assert pipeline.ocr_options.lang == ["en"]
    assert pipeline.do_ocr and pipeline.do_table_structure
    assert pipeline.accelerator_options.device == parser.AcceleratorDevice.CPU
    assert parser.OCR_ENGINES[engine]["output"].startswith("parsed_documents_markdown")


class FakeMistralClient:
    """Stands in for mistralai.Mistral: upload, signed URL, OCR and delete."""

    def __init__(self):
        self.deleted = []
        self.files = NS(
            upload=lambda file, purpose: NS(id="file-1"),
            get_signed_url=lambda file_id: NS(url="https://signed.example/" + file_id),
            delete=lambda file_id: self.deleted.append(file_id),
        )
        self.ocr = NS(process=lambda **kwargs: NS(pages=[NS(markdown="# Page 1"), NS(markdown="Total 42")]))


@pytest.fixture
def mistral_server(load_script, monkeypatch, tmp_path):
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    module = load_script("mistral_document_extraction_with_server.py")
    client = FakeMistralClient()
    monkeypatch.setattr(module, "API_KEY", "test-key")
    monkeypatch.setattr(module, "Mistral", lambda api_key: client)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "scan.pdf").write_bytes(b"%PDF-1.4 test")
    pd.DataFrame({"item": ["Revenue"], "value": [42]}).to_excel(docs / "sheet.xlsx", index=False)
    return module, client, docs


def test_mistral_server_default_output(mistral_server, tmp_path):
    module, client, docs = mistral_server
    module.main(["--input-dir", str(docs), "scan.pdf", "sheet.xlsx", "missing.pdf"])

    out = json.loads((tmp_path / "mistral_parsed_documents.json").read_text())
    by_name = {d["filename"]: d for d in out["documents"]}
    assert set(by_name) == {"scan.pdf", "sheet.xlsx"}
    assert by_name["scan.pdf"]["markdown"] == "# Page 1\n\nTotal 42"
    assert by_name["scan.pdf"]["method"] == "mistral-signed-url" and "processing_time" in by_name["scan.pdf"]
    assert by_name["sheet.xlsx"]["markdown"].startswith("# sheet.xlsx\n|")
    assert client.deleted == ["file-1"]


def test_mistral_server_markdown_dir(mistral_server, tmp_path):
    module, client, docs = mistral_server
    md_dir = tmp_path / "md"
    module.main(["--input-dir", str(docs), "--markdown-dir", str(md_dir), "scan.pdf", "sheet.xlsx"])

    assert (md_dir / "scan_mistral.md").read_text() == "# Page 1\n\nTotal 42"
    assert "## Sheet: Sheet1\n| item" in (md_dir / "sheet_mistral.md").read_text()
    summary = json.loads((md_dir / "mistral_parsing_summary.json").read_text())
    assert [d["saved_to"] for d in summary["documents"]] == [str(md_dir / "scan_mistral.md"), str(md_dir / "sheet_mistral.md")]
    assert all("processing_time_seconds" in d for d in summary["documents"])
    assert not (tmp_path / "mistral_parsed_documents.json").exists()
