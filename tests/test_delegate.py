from pathlib import Path

from ollama_agent.generate import Progress, run_generation
from ollama_agent.routing import GenSpec, gen_spec
from ollama_agent.tools.delegate import delegate_task
from tests.conftest import FakeBackend


async def test_delegate_uses_tier_and_saves_output(app, fake: FakeBackend, tmp_path: Path):
    f = tmp_path / "mod.py"
    f.write_text("def add(a, b):\n    return a + b\n")
    fake.replies = ["def test_add():\n    assert add(1, 2) == 3"]
    out = await delegate_task(
        app, task="write tests", context_files=[str(f)], model_tier="fast", think=False,
        max_tokens=512, timeout_s=30, ctx=None,
    )
    spec, messages, fmt = fake.calls[0]
    assert spec.model == "qwen3.5:4b" and spec.num_predict == 512 and fmt is None
    assert "def add" in messages[1]["content"] and str(f) in messages[1]["content"]
    assert "assert add(1, 2) == 3" in out
    assert "[delegate_task] model=qwen3.5:4b" in out
    saved = list((app.settings.data_dir / "outputs").glob("*-delegate_task-*.md"))
    assert len(saved) == 1 and "assert add" in saved[0].read_text()


async def test_delegate_reports_skipped_files(app, fake: FakeBackend):
    out = await delegate_task(
        app, task="x", context_files=["/nonexistent/file.py"], model_tier="strong", think=False,
        max_tokens=100, timeout_s=30, ctx=None,
    )
    assert "warning: skipped /nonexistent/file.py: not a file" in out
    assert fake.calls[0][0].model == "qwen3.5:latest"


async def test_delegate_rejects_oversized_input(app, fake: FakeBackend, tmp_path: Path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 88_000)  # under the 90k char budget, but ~25k tokens + 8k output > 32k ctx
    out = await delegate_task(
        app, task="summarize", context_files=[str(big)], model_tier="strong", think=False,
        max_tokens=8192, timeout_s=30, ctx=None,
    )
    assert out.startswith("error:") and "context" in out
    assert fake.calls == []


async def test_run_generation_timeout_returns_partial(fake: FakeBackend):
    fake.delay = 0.05
    fake.replies = [" ".join(["tok"] * 200)]
    spec = GenSpec(model="m", num_ctx=1024, keep_alive="1m", think=False, temperature=0, num_predict=200)
    res = await run_generation(fake, spec, [{"role": "user", "content": "go"}], timeout_s=0.3,
                               progress=Progress(None, "t"))
    assert res.truncated and res.reason == "timeout"
    assert 0 < len(res.text.split()) < 200
    assert any("timeout" in w for w in res.warnings)


async def test_call_through_mcp_server(server, fake: FakeBackend):
    fake.replies = ["hello from local"]
    result = await server.call_tool("delegate_task", {"task": "say hi", "model_tier": "fast"})
    assert not result.is_error
    assert "hello from local" in result.content[0].text
    assert fake.calls[0][0].model == "qwen3.5:4b"


def test_gen_spec_caps(app):
    spec = gen_spec(app.settings, "strong", think=False, temperature=0.2, num_predict=8192)
    assert spec.num_ctx == 32768
