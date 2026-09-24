"""Static checks (AST and MCP schema) for the formal/SPEC.md properties that hold by construction.

X1  stdout is the MCP transport: every print() in the package goes to stderr.
X2  R7: GenSpec/EmbedSpec are only built in routing.py, so every request carries settings.num_ctx
    and settings.keep_alive (Ollama reloads a model whose num_ctx changes between calls).
X3  S1: every call that talks to Ollama sits inside `async with app.semaphore` -- the assumption the
    TLA+ model (formal/tla/OllamaAgent.tla) makes about the code.
X4  the tool contract Claude Code sees: six tools, dedented descriptions, the "NOT for" lines
    CLAUDE.md asks to keep, read-only annotations except index_codebase, the ~32K-token note.
X5  the version string is the same in pyproject, the package and the MCP handshake.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

from ollama_agent import __version__
from ollama_agent.config import Settings
from ollama_agent.server import build_server
from tests.conftest import FakeBackend

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ollama_agent"


def _modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def test_x1_every_print_goes_to_stderr():
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and _call_name(node) == "print":
                target = {k.arg: k.value for k in node.keywords}.get("file")
                assert isinstance(target, ast.Attribute) and target.attr == "stderr", (
                    f"{path.relative_to(ROOT)}:{node.lineno} print() without file=sys.stderr"
                )


def test_x2_specs_are_only_built_in_routing():
    for path in _modules():
        if path.name == "routing.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and _call_name(node) in {"GenSpec", "EmbedSpec"}:
                pytest.fail(f"{path.relative_to(ROOT)}:{node.lineno} builds a spec outside routing.py")


GUARDED = {"run_generation", "embed_documents", "embed_query", "stream_chat", "load_model", "embed"}


def test_x3_every_backend_call_holds_the_semaphore():
    for path in [*sorted((SRC / "tools").glob("*.py")), SRC / "warmup.py"]:
        tree = ast.parse(path.read_text())
        spans = [
            (n.lineno, n.end_lineno)
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncWith)
            and any(isinstance(it.context_expr, ast.Attribute) and it.context_expr.attr == "semaphore" for it in n.items)
        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in GUARDED:
                assert any(lo <= node.lineno <= (hi or lo) for lo, hi in spans), (
                    f"{path.relative_to(ROOT)}:{node.lineno} {_call_name(node)}() outside `async with app.semaphore`"
                )


async def test_x4_mcp_tool_contract(tmp_path: Path):
    server = build_server(Settings(data_dir=tmp_path), backend=FakeBackend())
    tools = {t.name: t for t in await server.list_tools()}
    assert set(tools) == {"delegate_task", "review_diff", "summarize", "index_codebase", "search_code",
                          "local_models_status"}
    for name, t in tools.items():
        assert t.description and t.description == t.description.strip(), name
        assert not t.description.startswith(" "), f"{name}: description not dedented"
        assert t.annotations is not None and t.annotations.open_world_hint is False, name
        assert t.annotations.read_only_hint is (name != "index_codebase"), name
    for name in ("delegate_task", "summarize", "search_code"):
        assert "NOT for" in tools[name].description, f"{name}: CLAUDE.md asks to keep the NOT-for line"
    assert "~32K tokens" in (server.instructions or "") and Settings().num_ctx == 32 * 1024


def test_x5_versions_agree():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] == __version__
    assert f'version="{__version__}"' in (SRC / "server.py").read_text()
