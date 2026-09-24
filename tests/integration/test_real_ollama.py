"""Hits the real Ollama daemon. Run with:  OLLAMA_AGENT_INTEGRATION=1 uv run pytest tests/integration -s"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ollama_agent.config import Settings
from ollama_agent.routing import embed_spec
from ollama_agent.server import build_app
from ollama_agent.tools.delegate import delegate_task
from ollama_agent.tools.review import review_diff
from ollama_agent.tools.search import index_codebase, search_code
from ollama_agent.tools.status import local_models_status
from ollama_agent.tools.summarize import SINGLE_PASS_CHARS, summarize

pytestmark = pytest.mark.skipif(
    not os.environ.get("OLLAMA_AGENT_INTEGRATION"), reason="set OLLAMA_AGENT_INTEGRATION=1"
)


@pytest.fixture
def real_app(tmp_path: Path):
    return build_app(Settings.from_env(cwd=tmp_path))


async def test_status(real_app):
    out = await local_models_status(real_app)
    print(out)
    assert "ollama version:" in out and "error:" not in out


async def test_delegate_fast(real_app):
    out = await delegate_task(
        real_app, task="Reply with exactly the word PONG and nothing else.", context_files=None,
        model_tier="fast", think=False, max_tokens=16, timeout_s=120, ctx=None,
    )
    print(out)
    assert "PONG" in out.upper()


async def test_embed(real_app):
    vecs = await real_app.backend.embed(embed_spec(real_app.settings), ["alpha", "beta", "gamma"])
    assert len(vecs) == 3 and len(vecs[0]) == 1024


async def test_review_small_diff(real_app):
    diff = """--- a/div.py
+++ b/div.py
@@ -1,3 +1,3 @@
 def mean(xs):
-    return sum(xs) / len(xs) if xs else 0
+    return sum(xs) / len(xs)
"""
    res = await review_diff(real_app, diff=diff, git_range="", cwd="", focus="", think=True, timeout_s=240, ctx=None)
    print(res.model_dump_json(indent=2))
    assert res.verdict in ("approve", "request_changes", "comment")
    assert res.model.startswith("qwen")


def _log(lines: int, planted_at: int) -> str:
    out = [f"2026-09-24 10:{i % 60:02d}:{i % 57:02d} INFO worker-{i % 4} processed batch {i} ok" for i in range(lines)]
    out[planted_at] = "2026-09-24 10:31:07 ERROR payment-svc: connection refused to db-7:5432, giving up"
    return "\n".join(out)


async def test_summarize_single_pass(real_app):
    out = await summarize(real_app, paths=None, text=_log(40, 23), question="Which errors appear?",
                          timeout_s=240, ctx=None)
    print(out)
    assert "[summarize] model=qwen" in out and "(map)" not in out
    assert "db-7" in out


async def test_summarize_map_reduce(real_app, tmp_path: Path):
    log = tmp_path / "big.log"
    log.write_text(_log(900, 610))
    assert log.stat().st_size > SINGLE_PASS_CHARS
    out = await summarize(real_app, paths=[str(log)], text="", question="Which errors appear?",
                          timeout_s=300, ctx=None)
    print(out)
    assert "(map)" in out and "(reduce)" in out
    assert "db-7" in out  # the one ERROR line survives map and reduce


async def test_index_and_search(real_app, tmp_path: Path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    files = {
        "upload.py": "def retry_upload(file):\n    for attempt in range(3):\n        try:\n"
                     "            return put_object(file)\n        except IOError:\n            sleep(2 ** attempt)\n",
        "config.py": "def load_settings(path):\n    with open(path) as f:\n        return json.load(f)\n",
        "stats.py": "def mean_and_stddev(xs):\n    m = sum(xs) / len(xs)\n"
                    "    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5\n",
        "auth.py": "def verify_password(user, password):\n"
                   "    return bcrypt.checkpw(password.encode(), user.password_hash)\n",
    }
    for name, body in files.items():
        (root / "src" / name).write_text(body)

    out = await index_codebase(real_app, root=str(root), include_globs=None, force=False, ctx=None)
    print(out)
    assert "reindexed=4" in out and "embed model qwen3-embedding" in out

    for query, want in [("retry a failed upload with backoff", "upload.py"),
                        ("check whether the login password is correct", "auth.py")]:
        out = await search_code(real_app, query=query, root=str(root), top_k=4, refresh=False, ctx=None)
        print(out)
        first = next(line for line in out.splitlines() if line.startswith("1. "))
        assert want in first

    out = await index_codebase(real_app, root=str(root), include_globs=None, force=False, ctx=None)
    assert "unchanged=4" in out and "reindexed=0" in out
