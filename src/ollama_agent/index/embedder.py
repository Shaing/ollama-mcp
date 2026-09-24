"""Batch embedding through the backend with progress callbacks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from ..backend import Backend
from ..routing import EmbedSpec

# Qwen3-Embedding is instruction-tuned: queries carry a task prefix, documents do not.
QUERY_INSTRUCTION = (
    "Instruct: Given a natural-language question about a codebase, retrieve the code or "
    "documentation chunks that answer it\nQuery: "
)

BATCH = 32


async def embed_documents(
    backend: Backend,
    spec: EmbedSpec,
    texts: Sequence[str],
    progress: Callable[[int, int], Awaitable[None]] | None = None,
    batch: int = BATCH,
) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(await backend.embed(spec, list(texts[i : i + batch])))
        if progress is not None:
            await progress(min(i + batch, len(texts)), len(texts))
    return out


async def embed_query(backend: Backend, spec: EmbedSpec, query: str) -> list[float]:
    vecs = await backend.embed(spec, [QUERY_INSTRUCTION + query.strip()])
    return vecs[0]
