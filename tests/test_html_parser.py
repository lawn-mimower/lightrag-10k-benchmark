import json
import subprocess
import sys

import html_parser
from conftest import REPO_ROOT


def test_parse_extracts_metadata_and_visible_text(fixtures_dir):
    parsed = html_parser.parse_ixbrl_html(str(fixtures_dir / "10k" / "CTAS.html"))

    assert parsed["ticker"] == "CTAS"
    assert parsed["metadata"] == {
        "period_end_date": "May 31, 2024",
        "company_name": "Cintas Corporation",
    }
    text = parsed["text"]
    assert text.startswith("UNITED STATES SECURITIES AND EXCHANGE COMMISSION")
    assert "helps more than one million businesses" in text
    assert "Total revenue" in text


def test_parse_keeps_nested_inline_xbrl_facts(fixtures_dir):
    # The cover-page fiscal year end is an ix:nonNumeric that contains another
    # ix:nonNumeric; it used to be deleted instead of unwrapped.
    text = html_parser.parse_ixbrl_html(str(fixtures_dir / "10k" / "CTAS.html"))["text"]
    assert "For the fiscal year ended May 31 , 2024" in text


def test_parse_drops_hidden_xbrl_and_markup(fixtures_dir):
    text = html_parser.parse_ixbrl_html(str(fixtures_dir / "10k" / "CTAS.html"))["text"]
    # Hidden ix:header facts (CIK, document fiscal period focus) must not leak into the text
    assert "0000723254" not in text
    assert "<" not in text and "ix:" not in text
    assert "  " not in text


def test_to_documents_by_ticker_layout():
    docs = html_parser.to_documents_by_ticker([
        {"ticker": "ABC", "text": "hello world", "metadata": {"company_name": "ABC Inc."},
         "source_file": "ABC.html"},
    ])
    assert docs == {
        "ABC": {
            "ticker": "ABC",
            "company_name": "ABC Inc.",
            "period_end_date": "",
            "source_file": "ABC.html",
            "text": "hello world",
            "text_length": 11,
        }
    }


def test_cli_directory_mode_writes_ticker_keyed_json(tmp_path, fixtures_dir):
    out = tmp_path / "parsed_10k_documents.json"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "html_parser.py"), str(fixtures_dir / "10k"), str(out),
         "--tickers", "ctas", "--workers", "1"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    docs = json.loads(out.read_text())
    assert list(docs) == ["CTAS"]
    record = docs["CTAS"]
    assert record["company_name"] == "Cintas Corporation"
    assert record["period_end_date"] == "May 31, 2024"
    assert record["source_file"] == "CTAS.html"
    assert record["text_length"] == len(record["text"]) > 1000


def test_cli_ticker_filter_can_select_nothing(tmp_path, fixtures_dir):
    out = tmp_path / "parsed.json"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "html_parser.py"), str(fixtures_dir / "10k"), str(out),
         "--tickers", "NOPE", "--workers", "1"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text()) == {}


def test_cli_rejects_missing_path(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "html_parser.py"), str(tmp_path / "missing")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1
    assert "not a valid file or directory" in result.stdout
