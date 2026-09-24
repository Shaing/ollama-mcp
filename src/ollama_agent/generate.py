"""Shared streaming generation runner: progress reporting, timeouts, stats."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .backend import Backend
from .routing import GenSpec

log = logging.getLogger(__name__)

PROGRESS_INTERVAL_S = 2.0


class Progress:
    """Best-effort progress sink. Never raises: progress must not kill a tool."""

    def __init__(self, ctx: Any | None, label: str) -> None:
        self._ctx = ctx
        self._label = label
        self._last = 0.0

    async def tick(self, done: float, total: float | None, message: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < PROGRESS_INTERVAL_S:
            return
        self._last = now
        if self._ctx is None:
            return
        try:
            await self._ctx.report_progress(done, total, f"{self._label}: {message}")
        except Exception as exc:  # noqa: BLE001 - progress is advisory
            log.debug("progress notification failed: %s", exc)


@dataclass
class GenResult:
    text: str
    thinking: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float
    load_seconds: float | None
    truncated: bool = False
    reason: str | None = None  # "timeout" | "length" | None
    warnings: list[str] = field(default_factory=list)


async def run_generation(
    backend: Backend,
    spec: GenSpec,
    messages: Sequence[dict[str, Any]],
    *,
    timeout_s: float,
    progress: Progress,
    format: dict[str, Any] | None = None,
) -> GenResult:
    """Stream a chat completion; on timeout return what arrived so far."""
    started = time.monotonic()
    text: list[str] = []
    thinking: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    load_ns: int | None = None
    done_reason: str | None = None
    n_tokens = 0
    truncated = False
    reason: str | None = None

    await progress.tick(0, spec.num_predict, f"{spec.model} starting", force=True)

    async def consume() -> None:
        nonlocal prompt_tokens, completion_tokens, load_ns, done_reason, n_tokens
        async for chunk in backend.stream_chat(spec, messages, format=format):
            if chunk.thinking:
                thinking.append(chunk.thinking)
            if chunk.content:
                text.append(chunk.content)
            n_tokens += 1
            await progress.tick(n_tokens, spec.num_predict, f"{spec.model} {n_tokens} tok")
            if chunk.done:
                prompt_tokens = chunk.prompt_eval_count
                completion_tokens = chunk.eval_count
                load_ns = chunk.load_duration_ns
                done_reason = chunk.done_reason

    try:
        await asyncio.wait_for(consume(), timeout=timeout_s)
    except asyncio.TimeoutError:
        truncated = True
        reason = "timeout"
        log.warning("generation on %s timed out after %.0fs (%d tokens)", spec.model, timeout_s, n_tokens)

    if done_reason == "length":
        truncated = True
        reason = reason or "length"

    seconds = time.monotonic() - started
    await progress.tick(n_tokens, n_tokens or None, f"{spec.model} done in {seconds:.0f}s", force=True)

    warnings: list[str] = []
    if reason == "timeout":
        warnings.append(
            f"Stopped after {timeout_s:.0f}s (timeout); returned partial output. "
            "Raise timeout_s or reduce the task."
        )
    elif reason == "length":
        warnings.append(f"Output hit the {spec.num_predict}-token cap; raise max_tokens if it is cut off.")

    return GenResult(
        text="".join(text),
        thinking="".join(thinking),
        model=spec.model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens if completion_tokens is not None else (n_tokens or None),
        seconds=seconds,
        load_seconds=(load_ns / 1e9) if load_ns else None,
        truncated=truncated,
        reason=reason,
        warnings=warnings,
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate for budgeting (code averages ~3.5 chars/token)."""
    return int(len(text) / 3.5) + 1
