"""index_codebase / search_code: local semantic code search."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from ..app import App
from ..chunking import chunk_lines
from ..files import DEFAULT_CODE_GLOBS, expand_globs
from ..generate import Progress
from ..index.embedder import embed_documents, embed_query
from ..index.store import IndexStore
from ..routing import embed_spec

log = logging.getLogger(__name__)

CHUNK_CHARS = 1500
OVERLAP_LINES = 3
SNIPPET_LINES = 25


def index_path_for(app: App, root: Path) -> Path:
    if os.access(root, os.W_OK):
        return root / ".ollama-agent" / "index.sqlite"
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:12]
    return app.settings.data_dir / f"index-{digest}.sqlite"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve_root(root: str) -> Path | str:
    p = Path(root).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if not p.is_dir():
        return f"error: root {root!r} is not a directory"
    return p


async def refresh_index(
    app: App,
    root: Path,
    include_globs: list[str] | None,
    force: bool,
    progress: Progress,
) -> tuple[IndexStore, dict[str, int]]:
    """Incrementally (re)index root. Returns the open store and counters."""
    store = IndexStore(index_path_for(app, root))
    spec = embed_spec(app.settings)
    include = tuple(include_globs) if include_globs else DEFAULT_CODE_GLOBS
    known = store.file_states()
    seen: set[str] = set()
    counters = {"scanned": 0, "unchanged": 0, "reindexed": 0, "removed": 0, "chunks": 0}

    files = expand_globs(root, include)
    counters["scanned"] = len(files)
    todo: list[tuple[Path, str, float]] = []
    for p in files:
        rel = str(p.relative_to(root))
        seen.add(rel)
        mtime = p.stat().st_mtime
        state = known.get(rel)
        if state is not None and not force and abs(state.mtime - mtime) < 1e-6:
            counters["unchanged"] += 1
            continue
        todo.append((p, rel, mtime))

    async with app.semaphore:
        for i, (p, rel, mtime) in enumerate(todo, 1):
            await progress.tick(i - 1, len(todo), f"indexing {rel}")
            sha = _sha256(p)
            state = known.get(rel)
            if state is not None and not force and state.sha256 == sha:
                store.touch_mtime(rel, mtime)
                counters["unchanged"] += 1
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_lines(text, max_chars=CHUNK_CHARS, overlap_lines=OVERLAP_LINES)
            if not chunks:
                store.replace_file(rel, mtime, sha, [], [])
                continue
            docs = [f"File: {rel} (lines {c.start}-{c.end})\n{c.text}" for c in chunks]
            vecs = await embed_documents(app.backend, spec, docs)
            store.replace_file(rel, mtime, sha, [(c.start, c.end, c.text) for c in chunks], vecs)
            counters["reindexed"] += 1
            counters["chunks"] += len(chunks)

    gone = [rel for rel in known if rel not in seen]
    store.remove_files(gone)
    counters["removed"] = len(gone)
    store.set_meta("last_refresh", str(time.time()))
    store.set_meta("embed_model", spec.model)
    return store, counters


async def index_codebase(
    app: App,
    *,
    root: str,
    include_globs: list[str] | None,
    force: bool,
    ctx: Any | None,
) -> str:
    r = _resolve_root(root)
    if isinstance(r, str):
        return r
    started = time.monotonic()
    store, c = await refresh_index(app, r, include_globs, force, Progress(ctx, "index_codebase"))
    nf, nc = store.stats()
    store.close()
    return (
        f"index: {store.path}\n"
        f"root: {r}\n"
        f"scanned={c['scanned']} unchanged={c['unchanged']} reindexed={c['reindexed']} "
        f"removed={c['removed']} new_chunks={c['chunks']}\n"
        f"total: {nf} files, {nc} chunks, embed model {embed_spec(app.settings).model}\n"
        f"time: {time.monotonic() - started:.1f}s"
    )


def _snippet(text: str, max_lines: int = SNIPPET_LINES) -> str:
    lines = text.splitlines()
    body = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        body += f"\n… ({len(lines) - max_lines} more lines)"
    return body


async def search_code(
    app: App,
    *,
    query: str,
    root: str,
    top_k: int,
    refresh: bool,
    ctx: Any | None,
) -> str:
    q = query.strip()
    if not q:
        return "error: `query` is empty."
    r = _resolve_root(root)
    if isinstance(r, str):
        return r
    started = time.monotonic()
    progress = Progress(ctx, "search_code")
    if refresh:
        store, _ = await refresh_index(app, r, None, False, progress)
    else:
        store = IndexStore(index_path_for(app, r))
    nf, nc = store.stats()
    if nc == 0:
        store.close()
        return f"index for {r} is empty — call index_codebase(root={str(r)!r}) first."
    spec = embed_spec(app.settings)
    async with app.semaphore:
        qvec = await embed_query(app.backend, spec, q)
    hits = store.search(qvec, max(1, min(int(top_k or 8), 50)))
    store.close()

    out = [f"query: {q}", f"index: {nf} files / {nc} chunks under {r}", ""]
    for i, (row, score) in enumerate(hits, 1):
        out.append(f"{i}. {row.path}:{row.start}-{row.end}  (score {score:.3f})")
        out.append("```")
        out.append(_snippet(row.text))
        out.append("```")
        out.append("")
    out.append(f"[search_code] embed model={spec.model} time={time.monotonic() - started:.1f}s")
    return "\n".join(out)
