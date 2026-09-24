from pathlib import Path

from ollama_agent.tools.summarize import SINGLE_PASS_CHARS, summarize
from tests.conftest import FakeBackend


async def test_small_input_single_strong_call(app, fake: FakeBackend):
    fake.replies = ["- it adds numbers"]
    out = await summarize(app, paths=None, text="def add(a,b): return a+b", question="what does it do",
                          timeout_s=60, ctx=None)
    assert len(fake.calls) == 1
    spec, messages, _ = fake.calls[0]
    assert spec.model == "qwen3.5:latest"
    assert "what does it do" in messages[1]["content"]
    assert "it adds numbers" in out and "[summarize] model=qwen3.5:latest" in out


async def test_large_input_map_reduce(app, fake: FakeBackend, tmp_path: Path):
    log = tmp_path / "big.log"
    log.write_text("\n".join(f"2026-09-23 12:{i % 60:02d} INFO worker step {i}" for i in range(2000)))
    assert log.stat().st_size > SINGLE_PASS_CHARS
    fake.replies = []  # default "ok" for every call
    out = await summarize(app, paths=[str(log)], text="", question="", timeout_s=120, ctx=None)
    models = [c[0].model for c in fake.calls]
    assert models[-1] == "qwen3.5:latest"  # reduce on strong tier
    assert set(models[:-1]) == {"qwen3.5:4b"}  # map on fast tier
    assert len(models) > 3
    assert "lines 1-" in fake.calls[0][1][1]["content"]
    assert "(map)" in out and "(reduce)" in out


async def test_nothing_to_summarize(app, fake: FakeBackend):
    out = await summarize(app, paths=["/no/such/file"], text="", question="", timeout_s=10, ctx=None)
    assert out.startswith("error:") and "not a file" in out and fake.calls == []


async def test_call_through_mcp_server(server, fake: FakeBackend):
    fake.replies = ["- db-7 refused connections"]
    result = await server.call_tool("summarize", {"text": "ERROR db-7 connection refused", "question": "errors?"})
    assert not result.is_error, result.content[0].text
    assert "db-7 refused connections" in result.content[0].text
