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
DEFAULT_PREDICT = 4096


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

    est = estimate_tokens(user) + estimate_tokens(SYSTEM_PROMPT)
    if max_tokens:
        num_predict = max(64, min(int(max_tokens), MAX_PREDICT))
    else:
        # Thinking tokens count against num_predict: with think on, 4096 can all go on reasoning
        # (9B, measured 2026-09-24). Take up to the cap, but never less than the plain default and
        # never more than the context has room for, so every input accepted without think still is.
        want = MAX_PREDICT if think else DEFAULT_PREDICT
        num_predict = max(DEFAULT_PREDICT, min(want, settings.num_ctx - est))
    spec = gen_spec(settings, model_tier, think=think, temperature=0.2, num_predict=num_predict)
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
    warnings = list(result.warnings)
    if not result.text.strip() and result.thinking.strip():
        if result.reason != "length":
            warnings.append("The model returned thinking but no answer; the thinking is in the output file.")
        else:
            retry = f"a larger max_tokens (cap {MAX_PREDICT}), " if num_predict < MAX_PREDICT else ""
            warnings.append(
                f"The model spent its {num_predict}-token budget thinking and gave no answer; the thinking is in "
                f"the output file. Retry with {retry}a smaller task, or think=false."
            )
    path = app.outputs.save(
        "delegate_task",
        body,
        {
            "task": task[:500],
            "context_files": [str(b.path) for b in blocks],
            "model": result.model,
            "tier": model_tier,
            "think": think,
            "max_tokens": num_predict,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "seconds": round(result.seconds, 2),
            "truncated": result.truncated,
            "reason": result.reason,
        },
        thinking=result.thinking,
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
        warnings=warnings + notes,
    )
