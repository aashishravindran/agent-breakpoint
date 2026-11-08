from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_breakpoint.proxy.server import BackgroundServer
from agent_breakpoint.taps import reset_controller


@pytest.fixture(scope="session")
def fake_llm():
    from examples.toy_bank.fake_llm import app

    server = BackgroundServer(app).start()
    os.environ["TOY_LLM_URL"] = server.url
    os.environ["TOY_PYTHON"] = sys.executable
    yield server.url
    server.stop()


def _copy_example(name: str, tmp_path: Path) -> Path:
    """Copy an example's config next to a fresh run dir so tests don't share caches."""
    src = ROOT / "examples" / name
    dest = tmp_path / name
    dest.mkdir()
    for f in src.glob("*.yaml"):
        text = f.read_text().replace('python_path: ["../.."]', f'python_path: ["{ROOT}"]')
        (dest / f.name).write_text(text)
    for f in src.glob("*.json"):
        shutil.copy(f, dest / f.name)
    return dest / "agent-breakpoint.yaml"


@pytest.fixture
def bank_config(tmp_path, fake_llm):
    from agent_breakpoint.config import load_config

    return load_config(_copy_example("toy_bank", tmp_path))


@pytest.fixture
def graph_config(tmp_path):
    from agent_breakpoint.config import load_config

    return load_config(_copy_example("toy_graph", tmp_path))


@pytest.fixture(autouse=True)
def _clean_controller(monkeypatch):
    for k in list(os.environ):
        if k.startswith("AGENT_BREAKPOINT_"):
            monkeypatch.delenv(k)
    reset_controller()
    yield
    reset_controller()
