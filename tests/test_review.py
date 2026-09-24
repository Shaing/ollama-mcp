import json
import subprocess
from pathlib import Path

from ollama_agent.tools.review import ReviewCore, review_diff
from tests.conftest import FakeBackend

GOOD = json.dumps(
    {
        "verdict": "request_changes",
        "summary": "One real bug.",
        "findings": [
            {"file": "a.py", "line": 3, "severity": "major", "summary": "off by one", "suggestion": "use <="}
        ],
    }
)


async def test_review_parses_structured_output(app, fake: FakeBackend):
    fake.replies = [GOOD]
    res = await review_diff(app, diff="--- a.py\n+++ a.py\n@@\n+for i in range(n-1):", git_range="", cwd="",
                            focus="loops", think=True, timeout_s=60, ctx=None)
    assert res.verdict == "request_changes" and res.findings[0].line == 3
    spec, messages, fmt = fake.calls[0]
    assert fmt == ReviewCore.model_json_schema()  # constrained decoding requested
    assert spec.model == "qwen3.5:latest" and spec.think is True
    assert messages[1]["content"].startswith("Review focus: loops")
    assert Path(res.output_path).is_file()


async def test_review_retries_then_succeeds(app, fake: FakeBackend):
    fake.replies = ["not json at all", GOOD]
    res = await review_diff(app, diff="+x", git_range="", cwd="", focus="", think=True, timeout_s=60, ctx=None)
    assert res.verdict == "request_changes"
    assert len(fake.calls) == 2
    assert fake.calls[1][0].think is False  # retry is plain, deterministic


async def test_review_verdict_matches_findings(app, fake: FakeBackend):
    fake.replies = [GOOD.replace('"request_changes"', '"approve"')]
    res = await review_diff(app, diff="+x", git_range="", cwd="", focus="", think=False, timeout_s=60, ctx=None)
    assert res.verdict == "request_changes" and any("verdict changed" in n for n in res.notes)


async def test_review_falls_back_to_raw(app, fake: FakeBackend):
    fake.replies = ["garbage", "still garbage"]
    res = await review_diff(app, diff="+x", git_range="", cwd="", focus="", think=False, timeout_s=60, ctx=None)
    assert res.verdict == "comment" and "still garbage" in res.findings[0].summary


async def test_review_runs_git_diff(app, fake: FakeBackend, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init"],
                   cwd=repo, check=True)
    (repo / "a.py").write_text("print('hi')\n")
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    fake.replies = [GOOD]
    res = await review_diff(app, diff="", git_range="--staged", cwd=str(repo), focus="", think=False,
                            timeout_s=60, ctx=None)
    assert "print('hi')" in fake.calls[0][1][1]["content"]
    assert res.diff_chars > 0


async def test_review_empty_diff_short_circuits(app, fake: FakeBackend, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init"],
                   cwd=repo, check=True)
    res = await review_diff(app, diff="", git_range="", cwd=str(repo), focus="", think=False, timeout_s=60, ctx=None)
    assert res.verdict == "approve" and "empty" in res.summary and fake.calls == []


async def test_review_reports_git_errors(app, fake: FakeBackend, tmp_path: Path):
    res = await review_diff(app, diff="", git_range="", cwd=str(tmp_path), focus="", think=False, timeout_s=60,
                            ctx=None)
    assert res.verdict == "comment" and "git diff failed" in res.summary
