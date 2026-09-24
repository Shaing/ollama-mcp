"""Line-based chunking with overlap, for map-reduce summarization and indexing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    start: int  # 1-based inclusive line
    end: int  # 1-based inclusive line
    text: str


def chunk_lines(text: str, max_chars: int = 6000, overlap_lines: int = 10) -> list[Chunk]:
    """Split text into chunks of at most max_chars, breaking on line boundaries.

    Consecutive chunks share `overlap_lines` lines so nothing is lost at a seam.
    A single line longer than max_chars becomes its own chunk (never split mid-line).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[Chunk] = []
    i = 0
    n = len(lines)
    while i < n:
        size = 0
        j = i
        while j < n and (size + len(lines[j]) + 1 <= max_chars or j == i):
            size += len(lines[j]) + 1
            j += 1
        chunks.append(Chunk(start=i + 1, end=j, text="\n".join(lines[i:j])))
        if j >= n:
            break
        i = max(j - overlap_lines, i + 1)
    return chunks
