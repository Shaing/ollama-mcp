"""summarize: map-reduce summarization of large files, logs and text."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..app import App
from ..chunking import Chunk, chunk_lines
from ..files import FileBlock, is_text
from ..generate import Progress, run_generation
from ..outputs import clip, render
from ..routing import gen_spec

SINGLE_PASS_CHARS = 45_000  # below this, one strong-tier call; above, map-reduce
CHUNK_CHARS = 12_000  # ~3.5k tokens per map call
MAP_PREDICT = 600
REDUCE_PREDICT = 2048
MAX_TOTAL_CHARS = 20_000_000

MAP_SYSTEM = (
    "You condense one section of a larger document for a later merge step. Keep every concrete detail "
    "that matters: identifiers, file names, numbers, error messages, timestamps, decisions. Drop "
    "repetition and filler. Output terse bullet points, no preamble."
)
REDUCE_SYSTEM = (
    "You merge section notes into one coherent summary for a senior engineer. Preserve concrete "
    "details (identifiers, numbers, errors), order events chronologically where relevant, and "
    "resolve duplicates. Output Markdown with short headings and bullets, no preamble."
)


def _focus(question: str) -> str:
    q = question.strip()
    return f"\nAnswer/focus on this question: {q}\n" if q else ""


def _gather(paths: list[str], text: str, base: Path) -> tuple[list[FileBlock], list[str]]:
    blocks: list[FileBlock] = []
    notes: list[str] = []
    total = 0
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = base / p
        if not p.is_file():
            notes.append(f"skipped {raw}: not a file")
            continue
        if not is_text(p):
            notes.append(f"skipped {raw}: binary")
            continue
        size = p.stat().st_size
        if total + size > MAX_TOTAL_CHARS:
            notes.append(f"skipped {raw}: total input would exceed {MAX_TOTAL_CHARS} bytes")
            continue
        total += size
        blocks.append(FileBlock(path=p, text=p.read_text(encoding="utf-8", errors="replace")))
    if text.strip():
        blocks.append(FileBlock(path=Path("<inline text>"), text=text))
    return blocks, notes


async def summarize(
    app: App,
    *,
    paths: list[str] | None,
    text: str,
    question: str,
    timeout_s: int,
    ctx: Any | None,
) -> str:
    settings = app.settings
    blocks, notes = _gather(paths or [], text or "", Path.cwd())
    if not blocks:
        return "error: nothing to summarize (no readable paths and empty text)." + (
            "\n" + "\n".join(notes) if notes else ""
        )

    deadline = time.monotonic() + app.clamp_timeout(timeout_s, 300)
    progress = Progress(ctx, "summarize")
    total_chars = sum(len(b.text) for b in blocks)
    prompt_tokens = 0
    completion_tokens = 0
    seconds = 0.0
    model_used = ""
    truncated = False
    started = time.monotonic()

    def remaining() -> float:
        return max(5.0, deadline - time.monotonic())

    async with app.semaphore:
        if total_chars <= SINGLE_PASS_CHARS:
            spec = gen_spec(settings, "strong", think=False, temperature=0.1, num_predict=REDUCE_PREDICT)
            body_in = "\n\n".join(f"### {b.path}\n{b.text}" for b in blocks)
            messages = [
                {"role": "system", "content": REDUCE_SYSTEM},
                {"role": "user", "content": f"Summarize the following.{_focus(question)}\n{body_in}"},
            ]
            res = await run_generation(app.backend, spec, messages, timeout_s=remaining(), progress=progress)
            summary = res.text
            model_used = res.model
            prompt_tokens += res.prompt_tokens or 0
            completion_tokens += res.completion_tokens or 0
            truncated = res.truncated
            notes.extend(res.warnings)
        else:
            # Map: fast tier per chunk, sequential so VRAM stays predictable.
            units: list[tuple[str, Chunk]] = [
                (str(b.path), c) for b in blocks for c in chunk_lines(b.text, max_chars=CHUNK_CHARS, overlap_lines=8)
            ]
            map_spec = gen_spec(settings, "fast", think=False, temperature=0.1, num_predict=MAP_PREDICT)
            partials: list[str] = []
            for i, (name, chunk) in enumerate(units, 1):
                await progress.tick(i - 1, len(units) + 1, f"chunk {i}/{len(units)}", force=True)
                if time.monotonic() >= deadline:
                    notes.append(f"timeout: stopped after {i - 1}/{len(units)} chunks; summary covers only those")
                    truncated = True
                    break
                messages = [
                    {"role": "system", "content": MAP_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Section: {name} lines {chunk.start}-{chunk.end}.{_focus(question)}\n{chunk.text}",
                    },
                ]
                res = await run_generation(
                    app.backend, map_spec, messages, timeout_s=remaining(), progress=progress
                )
                prompt_tokens += res.prompt_tokens or 0
                completion_tokens += res.completion_tokens or 0
                partials.append(f"## {name} lines {chunk.start}-{chunk.end}\n{res.text.strip()}")
                if res.reason == "timeout":
                    truncated = True
            # Reduce: strong tier; if the notes themselves are too big, fold them once more.
            reduce_spec = gen_spec(settings, "strong", think=False, temperature=0.1, num_predict=REDUCE_PREDICT)
            joined = "\n\n".join(partials)
            while len(joined) > SINGLE_PASS_CHARS and time.monotonic() < deadline:
                folded: list[str] = []
                for c in chunk_lines(joined, max_chars=CHUNK_CHARS, overlap_lines=0):
                    res = await run_generation(
                        app.backend,
                        map_spec,
                        [
                            {"role": "system", "content": MAP_SYSTEM},
                            {"role": "user", "content": f"Merge these notes.{_focus(question)}\n{c.text}"},
                        ],
                        timeout_s=remaining(),
                        progress=progress,
                    )
                    prompt_tokens += res.prompt_tokens or 0
                    completion_tokens += res.completion_tokens or 0
                    folded.append(res.text.strip())
                joined = "\n\n".join(folded)
            await progress.tick(len(units), len(units) + 1, "merging", force=True)
            res = await run_generation(
                app.backend,
                reduce_spec,
                [
                    {"role": "system", "content": REDUCE_SYSTEM},
                    {"role": "user", "content": f"Merge these section notes into one summary.{_focus(question)}\n{joined}"},
                ],
                timeout_s=remaining(),
                progress=progress,
            )
            summary = res.text
            model_used = f"{map_spec.model} (map) + {res.model} (reduce)"
            prompt_tokens += res.prompt_tokens or 0
            completion_tokens += res.completion_tokens or 0
            truncated = truncated or res.truncated
            notes.extend(res.warnings)

    seconds = time.monotonic() - started
    body = summary.strip() or "(model returned no text)"
    path = app.outputs.save(
        "summarize",
        body,
        {
            "paths": [str(b.path) for b in blocks],
            "question": question,
            "input_chars": total_chars,
            "model": model_used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "seconds": round(seconds, 2),
            "truncated": truncated,
        },
    )
    head, clipped = clip(body, settings.max_return_chars)
    return render(
        head,
        tool="summarize",
        model=model_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        seconds=seconds,
        output_path=path,
        truncated=clipped or truncated,
        warnings=notes,
    )
