"""review_diff: structured second-opinion code review on a diff."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..app import App
from ..generate import Progress, estimate_tokens, run_generation
from ..routing import gen_spec

log = logging.getLogger(__name__)

Severity = Literal["critical", "major", "minor", "nit"]
Verdict = Literal["approve", "request_changes", "comment"]


class Finding(BaseModel):
    file: str = Field(description="Path as it appears in the diff")
    line: int | None = Field(default=None, description="Line number in the NEW file, if known")
    severity: Severity
    summary: str = Field(description="One sentence: what is wrong and why it matters")
    suggestion: str = Field(default="", description="Concrete fix, or empty")


class ReviewCore(BaseModel):
    """The shape the local model is constrained to emit."""

    verdict: Verdict
    summary: str = Field(description="Two or three sentences on overall quality and risk")
    findings: list[Finding]


class ReviewResult(ReviewCore):
    """ReviewCore plus metadata about how it was produced."""

    model: str = ""
    seconds: float = 0.0
    diff_chars: int = 0
    notes: list[str] = Field(default_factory=list)
    output_path: str = ""


SYSTEM_PROMPT = (
    "You are a meticulous senior code reviewer giving a second opinion on a diff. Report only real "
    "problems you can point to in the diff: bugs, incorrect logic, unhandled errors, security issues, "
    "data loss, races, broken contracts, misleading names, missing tests for risky changes. Do not "
    "comment on style unless it hides a bug. Prefer few, high-confidence findings over many weak ones. "
    "If the diff looks correct, say so with an empty findings list and verdict 'approve'. "
    "Respond with JSON only, matching the provided schema."
)

GIT_TIMEOUT_S = 30
MAX_PREDICT = 4096


async def _git_diff(cwd: Path, git_range: str) -> tuple[str, str | None]:
    """Run git diff in cwd. Returns (diff, error)."""
    args = ["git", "-C", str(cwd), "diff", "--no-color", "--no-ext-diff"]
    rng = git_range.strip()
    if rng in ("--staged", "--cached", "staged"):
        args.append("--staged")
    elif rng:
        args.extend(rng.split())
    else:
        args.append("HEAD")
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=GIT_TIMEOUT_S)
    except FileNotFoundError:
        return "", "git is not installed or not on PATH"
    except asyncio.TimeoutError:
        return "", f"git diff timed out after {GIT_TIMEOUT_S}s"
    if proc.returncode != 0:
        return "", f"git diff failed: {err.decode(errors='replace').strip()}"
    return out.decode(errors="replace"), None


def _fallback(raw: str, reason: str) -> ReviewCore:
    return ReviewCore(
        verdict="comment",
        summary=f"The local model did not return valid JSON ({reason}); its raw output is attached as a finding.",
        findings=[
            Finding(file="", line=None, severity="minor", summary=raw.strip()[:2000] or "(empty output)")
        ],
    )


def _normalize_verdict(core: ReviewCore, notes: list[str]) -> ReviewCore:
    """Small models sometimes say 'approve' while listing a critical bug; make the verdict match."""
    worst = {f.severity for f in core.findings}
    if core.verdict == "approve" and worst & {"critical", "major"}:
        notes.append("verdict changed from 'approve' to 'request_changes' because of critical/major findings")
        return core.model_copy(update={"verdict": "request_changes"})
    return core


async def review_diff(
    app: App,
    *,
    diff: str,
    git_range: str,
    cwd: str,
    focus: str,
    think: bool,
    timeout_s: int,
    ctx: Any | None,
) -> ReviewResult:
    settings = app.settings
    notes: list[str] = []
    base = Path(cwd).expanduser() if cwd else Path.cwd()

    diff_text = diff or ""
    if not diff_text.strip():
        diff_text, err = await _git_diff(base, git_range)
        if err:
            return ReviewResult(verdict="comment", summary=f"Could not obtain a diff: {err}", findings=[], notes=[err])
        if not diff_text.strip():
            what = git_range or "HEAD (all uncommitted changes)"
            return ReviewResult(
                verdict="approve", summary=f"git diff {what} is empty; nothing to review.", findings=[]
            )

    if len(diff_text) > settings.max_input_chars:
        notes.append(
            f"diff truncated from {len(diff_text)} to {settings.max_input_chars} chars; review only covers the head"
        )
        diff_text = diff_text[: settings.max_input_chars]

    spec = gen_spec(settings, "strong", think=think, temperature=0.1, num_predict=MAX_PREDICT)
    est = estimate_tokens(diff_text) + 400
    if est + MAX_PREDICT > spec.num_ctx:
        keep = int((spec.num_ctx - MAX_PREDICT - 400) * 3.5)
        notes.append(f"diff further cut to ~{keep} chars to fit the {spec.num_ctx}-token context")
        diff_text = diff_text[:keep]

    user = ((f"Review focus: {focus.strip()}\n\n" if focus.strip() else "") + f"```diff\n{diff_text}\n```")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    schema = ReviewCore.model_json_schema()
    timeout = app.clamp_timeout(timeout_s, 180)
    progress = Progress(ctx, "review_diff")

    total_seconds = 0.0
    core: ReviewCore | None = None
    raw = ""
    model = spec.model
    async with app.semaphore:
        for attempt in range(2):
            result = await run_generation(
                app.backend, spec, messages, timeout_s=timeout, progress=progress, format=schema
            )
            total_seconds += result.seconds
            raw = result.text
            model = result.model
            notes.extend(result.warnings)
            try:
                core = ReviewCore.model_validate_json(raw)
                break
            except ValidationError as exc:
                log.warning("review JSON invalid on attempt %d: %s", attempt + 1, exc.errors()[:2])
                if result.truncated:
                    break  # a retry would time out again
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "That was not valid JSON for the schema. Emit only the JSON object."},
                ]
                spec = gen_spec(settings, "strong", think=False, temperature=0.0, num_predict=MAX_PREDICT)
    if core is None:
        core = _fallback(raw, "schema validation failed")
    else:
        core = _normalize_verdict(core, notes)

    path = app.outputs.save(
        "review_diff",
        core.model_dump_json(indent=2),
        {"model": model, "seconds": round(total_seconds, 2), "diff_chars": len(diff_text), "focus": focus, "notes": notes},
    )
    return ReviewResult(
        **core.model_dump(),
        model=model,
        seconds=round(total_seconds, 2),
        diff_chars=len(diff_text),
        notes=notes,
        output_path=str(path),
    )
