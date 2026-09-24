"""Contracts for CrossHair: symbolic execution of the REAL functions (not a model of them).

Run:  uv run crosshair check formal/crosshair/contracts.py --analysis_kind=PEP316

Each wrapper calls the production function and states pre/postconditions in PEP 316
docstring syntax.  CrossHair executes the function on symbolic inputs backed by Z3 and
reports any concrete input that violates a postcondition.  The preconditions bound the
search space (string lengths etc.) so every check finishes in seconds; within those bounds
"no counterexample" is a proof, outside them it is not.
"""

from __future__ import annotations

from ollama_agent.chunking import chunk_lines
from ollama_agent.generate import estimate_tokens
from ollama_agent.outputs import clip


def clip_contract(text: str, max_chars: int) -> tuple[str, bool]:
    """
    pre: 1 <= max_chars <= 12
    pre: len(text) <= 24
    post: len(__return__[0]) <= max_chars
    post: __return__[1] == (len(text) > max_chars)
    post: text.startswith(__return__[0])
    post: (not __return__[1]) or 5 * len(__return__[0]) >= 3 * max_chars
    """
    return clip(text, max_chars)


def estimate_tokens_contract(text: str) -> int:
    """
    pre: len(text) <= 40
    post: __return__ >= 1
    post: 3.5 * (__return__ - 1) <= len(text) < 3.5 * __return__
    """
    return estimate_tokens(text)


def chunk_lines_contract(lines: list[str], max_chars: int, overlap_lines: int) -> list[tuple[int, int]]:
    """
    pre: 1 <= max_chars <= 12
    pre: 0 <= overlap_lines <= 2
    pre: 1 <= len(lines) <= 4
    pre: all(1 <= len(ln) <= 5 and all(ch in "ab" for ch in ln) for ln in lines)
    post: __return__[0][0] == 1 and __return__[-1][1] == len(lines)
    post: all(s <= e for s, e in __return__)
    post: all(b[0] == max(a[1] - overlap_lines + 1, a[0] + 1) for a, b in zip(__return__, __return__[1:]))
    post: all(len(chr(10).join(lines[s - 1:e])) <= max_chars or s == e for s, e in __return__)
    """
    return [(c.start, c.end) for c in chunk_lines("\n".join(lines), max_chars=max_chars, overlap_lines=overlap_lines)]
