"""Manual API check scripts: option handling without network access."""
import os
import subprocess
import sys

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
