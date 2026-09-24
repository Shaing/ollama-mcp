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
from ollama_agent.tools.status import local_models_status

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
