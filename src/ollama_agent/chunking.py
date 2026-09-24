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

    Consecutive chunks share up to `overlap_lines` lines so nothing is lost at a seam, but
    never more than half a chunk, and every chunk ends past the previous one. So long lines
    cannot inflate the output beyond about twice the input, and no chunk is a mere
    sub-window of its predecessor (formal/SPEC.md C5-C8). A single line longer than
    max_chars becomes its own chunk (never split mid-line).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")
    lines = text.splitlines()
    n = len(lines)
    if not lines:
        return []

    def window_end(i: int) -> int:
        size = 0
        j = i
        while j < n and (size + len(lines[j]) + 1 <= max_chars or j == i):
            size += len(lines[j]) + 1
            j += 1
        return j

    chunks: list[Chunk] = []
    i = 0
    prev_end = 0
    while i < n:
        j = window_end(i)
        if j <= prev_end:  # the overlap window adds nothing new: continue right after the previous chunk
            i = prev_end
            j = window_end(i)
        chunks.append(Chunk(start=i + 1, end=j, text="\n".join(lines[i:j])))
        if j >= n:
            break
        prev_end = j
        i = max(j - min(overlap_lines, (j - i) // 2), i + 1)
    return chunks
