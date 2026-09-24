"""SQLite-backed embedding index: one row per chunk, vectors as float32 blobs."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    sha256 TEXT NOT NULL,
    nchunks INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    start INTEGER NOT NULL,
    "end" INTEGER NOT NULL,
    text TEXT NOT NULL,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True)
class ChunkRow:
    id: int
    path: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class FileState:
    mtime: float
    sha256: str
    nchunks: int


class IndexStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(str(path))
        self._db.executescript(SCHEMA)
        self._matrix_cache: tuple[list[ChunkRow], np.ndarray] | None = None

    def close(self) -> None:
        self._db.close()

    # --- metadata -----------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self._db.commit()

    # --- files --------------------------------------------------------------
    def file_states(self) -> dict[str, FileState]:
        rows = self._db.execute("SELECT path, mtime, sha256, nchunks FROM files").fetchall()
        return {p: FileState(m, s, n) for p, m, s, n in rows}

    def touch_mtime(self, path: str, mtime: float) -> None:
        self._db.execute("UPDATE files SET mtime=? WHERE path=?", (mtime, path))
        self._db.commit()

    def replace_file(
        self,
        path: str,
        mtime: float,
        sha256: str,
        chunks: list[tuple[int, int, str]],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        cur = self._db.cursor()
        cur.execute("DELETE FROM chunks WHERE path=?", (path,))
        cur.executemany(
            'INSERT INTO chunks(path, start, "end", text, vec) VALUES (?, ?, ?, ?, ?)',
            [
                (path, s, e, t, np.asarray(v, dtype=np.float32).tobytes())
                for (s, e, t), v in zip(chunks, vectors, strict=True)
            ],
        )
        cur.execute(
            "INSERT OR REPLACE INTO files(path, mtime, sha256, nchunks) VALUES (?, ?, ?, ?)",
            (path, mtime, sha256, len(chunks)),
        )
        self._db.commit()
        self._matrix_cache = None

    def remove_files(self, paths: list[str]) -> None:
        if not paths:
            return
        cur = self._db.cursor()
        cur.executemany("DELETE FROM chunks WHERE path=?", [(p,) for p in paths])
        cur.executemany("DELETE FROM files WHERE path=?", [(p,) for p in paths])
        self._db.commit()
        self._matrix_cache = None

    # --- search -------------------------------------------------------------
    def stats(self) -> tuple[int, int]:
        (nf,) = self._db.execute("SELECT COUNT(*) FROM files").fetchone()
        (nc,) = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return nf, nc

    def matrix(self) -> tuple[list[ChunkRow], np.ndarray]:
        """All chunks plus an L2-normalised float32 matrix (rows align with the list)."""
        if self._matrix_cache is not None:
            return self._matrix_cache
        rows = self._db.execute('SELECT id, path, start, "end", text, vec FROM chunks ORDER BY id').fetchall()
        chunks = [ChunkRow(r[0], r[1], r[2], r[3], r[4]) for r in rows]
        if rows:
            mat = np.stack([np.frombuffer(r[5], dtype=np.float32) for r in rows])
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        else:
            mat = np.zeros((0, 0), dtype=np.float32)
        self._matrix_cache = (chunks, mat)
        return self._matrix_cache

    def search(self, query_vec: list[float], top_k: int) -> list[tuple[ChunkRow, float]]:
        chunks, mat = self.matrix()
        if not chunks:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        n = np.linalg.norm(q)
        q = q / n if n else q
        scores = mat @ q
        k = min(top_k, len(chunks))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(chunks[i], float(scores[i])) for i in idx]
