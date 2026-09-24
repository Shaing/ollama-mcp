"""Persist full tool outputs to disk and clip what goes back to Claude."""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def clip(text: str, max_chars: int) -> tuple[str, bool]:
    """Return (head, truncated). Cuts at a line boundary when possible."""
    if len(text) <= max_chars:
        return text, False
    head = text[:max_chars]
    nl = head.rfind("\n")
    if nl > max_chars * 0.6:
        head = head[:nl]
    return head, True


class OutputStore:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "outputs"

    def save(self, tool: str, text: str, meta: dict[str, Any]) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.dir / f"{stamp}-{tool}-{secrets.token_hex(4)}.md"
        front = json.dumps(meta, indent=2, default=str, ensure_ascii=False)
        path.write_text(f"<!-- ollama-agent {tool}\n{front}\n-->\n\n{text}\n", encoding="utf-8")
        return path


def render(
    body: str,
    *,
    tool: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    seconds: float,
    output_path: Path | None,
    truncated: bool,
    warnings: list[str] | None = None,
) -> str:
    """Body plus a compact metadata footer Claude can read at a glance."""
    lines = [body.rstrip(), "", "---"]
    tok = f"{prompt_tokens or '?'} in / {completion_tokens or '?'} out"
    lines.append(f"[{tool}] model={model} tokens={tok} time={seconds:.1f}s")
    if truncated:
        lines.append("[truncated: the text above is the head; full output is in the file below]")
    if output_path is not None:
        lines.append(f"full output: {output_path}")
    for w in warnings or []:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
