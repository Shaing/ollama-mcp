"""Reading and enumerating source files within a byte budget."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CODE_GLOBS: tuple[str, ...] = (
    "*.py", "*.pyi", "*.js", "*.mjs", "*.cjs", "*.ts", "*.tsx", "*.jsx", "*.go", "*.rs",
    "*.java", "*.kt", "*.c", "*.h", "*.cc", "*.cpp", "*.hpp", "*.cs", "*.rb", "*.php",
    "*.swift", "*.scala", "*.sh", "*.bash", "*.zsh", "*.sql", "*.proto", "*.md", "*.rst",
    "*.toml", "*.yaml", "*.yml", "*.json", "*.ini", "*.cfg", "*.env.example",
    "Dockerfile", "Makefile", "CMakeLists.txt",
)

DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
     ".pytest_cache", ".ruff_cache", "dist", "build", "target", ".ollama-agent", ".idea",
     ".vscode", ".next", ".cache", "coverage"}
)

MAX_FILE_BYTES = 2_000_000


@dataclass
class FileBlock:
    path: Path
    text: str


def is_text(path: Path, sample: int = 4096) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(sample)
    except OSError:
        return False
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def expand_globs(
    root: Path,
    include: tuple[str, ...] | list[str] = DEFAULT_CODE_GLOBS,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
) -> list[Path]:
    """Walk root, returning text files whose basename matches an include glob."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs and not d.startswith(".git"))
        for name in sorted(filenames):
            if not any(fnmatch.fnmatch(name, pat) for pat in include):
                continue
            p = Path(dirpath) / name
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            if is_text(p):
                out.append(p)
    return out


def read_paths(
    paths: list[str],
    budget_chars: int,
    base: Path | None = None,
) -> tuple[list[FileBlock], list[str]]:
    """Read files (relative paths resolve against base). Returns (blocks, notes).

    Stops adding files once the budget is exhausted; notes explain every skip.
    """
    blocks: list[FileBlock] = []
    notes: list[str] = []
    used = 0
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.is_absolute() and base is not None:
            p = base / p
        if not p.is_file():
            notes.append(f"skipped {raw}: not a file")
            continue
        if not is_text(p):
            notes.append(f"skipped {raw}: binary")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if used + len(text) > budget_chars:
            notes.append(
                f"skipped {raw}: {len(text)} chars would exceed the {budget_chars}-char input budget "
                f"({used} used); use summarize or pass fewer files"
            )
            continue
        used += len(text)
        blocks.append(FileBlock(path=p, text=text))
    return blocks, notes


def format_blocks(blocks: list[FileBlock]) -> str:
    parts = []
    for b in blocks:
        parts.append(f"### FILE: {b.path}\n```\n{b.text.rstrip()}\n```")
    return "\n\n".join(parts)
