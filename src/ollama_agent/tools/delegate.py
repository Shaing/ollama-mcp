"""delegate_task: run a self-contained subtask on a local model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..app import App
from ..files import format_blocks, read_paths
from ..generate import Progress, estimate_tokens, run_generation
from ..outputs import clip, render
from ..routing import Tier, gen_spec

SYSTEM_PROMPT = (
    "You are a focused local coding assistant working for a senior engineer who will review your "
    "output. Do exactly the task requested. Return only the deliverable (code, tests, text) with "
    "brief notes where a choice was non-obvious. Do not restate the task. If the task cannot be "
    "completed with the material given, say precisely what is missing."
)

MAX_PREDICT = 8192


async def delegate_task(
    app: App,
    *,
    task: str,
    context_files: list[str] | None,
    model_tier: Tier,
    think: bool,
    max_tokens: int,
    timeout_s: int,
    ctx: Any | None,
) -> str:
    settings = app.settings
    task = task.strip()
    if not task:
        return "error: `task` is empty."

    budget = settings.max_input_chars - len(task)
    blocks, notes = read_paths(context_files or [], budget_chars=max(0, budget), base=Path.cwd())
    user = task if not blocks else f"{task}\n\n## Context files\n\n{format_blocks(blocks)}"

    num_predict = max(64, min(int(max_tokens or 4096), MAX_PREDICT))
    spec = gen_spec(settings, model_tier, think=think, temperature=0.2, num_predict=num_predict)
    est = estimate_tokens(user) + estimate_tokens(SYSTEM_PROMPT)
    if est + num_predict > spec.num_ctx:
        return (
            f"error: input is ~{est} tokens plus {num_predict} output tokens, over the {spec.num_ctx}-token "
            "context of the local model. Pass fewer/smaller context_files, lower max_tokens, or use "
            "`summarize` first."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    timeout = app.clamp_timeout(timeout_s, 120)
    async with app.semaphore:
        result = await run_generation(
            app.backend, spec, messages, timeout_s=timeout, progress=Progress(ctx, "delegate_task")
        )

    body = result.text.strip() or "(model returned no text)"
    path = app.outputs.save(
        "delegate_task",
        body,
        {
            "task": task[:500],
            "context_files": [str(b.path) for b in blocks],
            "model": result.model,
            "tier": model_tier,
            "think": think,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "seconds": round(result.seconds, 2),
            "truncated": result.truncated,
            "reason": result.reason,
        },
    )
    head, clipped = clip(body, settings.max_return_chars)
    return render(
        head,
        tool="delegate_task",
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        seconds=result.seconds,
        output_path=path,
        truncated=clipped or result.truncated,
        warnings=result.warnings + notes,
    )
