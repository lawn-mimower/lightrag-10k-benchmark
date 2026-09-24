import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def fixtures_dir():
    return FIXTURES


@pytest.fixture
def load_script(monkeypatch):
    """Import a repository script by path, optionally stubbing heavy modules."""

    def _load(relative_path, stubs=None, name=None):
        for module_name, attrs in (stubs or {}).items():
            monkeypatch.setitem(sys.modules, module_name, types.SimpleNamespace(**attrs))
        path = REPO_ROOT / relative_path
        module_name = name or "script_" + path.stem.replace("-", "_")
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load
