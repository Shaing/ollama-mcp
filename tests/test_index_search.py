import os
import time
from pathlib import Path

from ollama_agent.index.store import IndexStore
from ollama_agent.tools.search import index_codebase, search_code
from ollama_agent.tools.status import local_models_status
from ollama_agent.backend import LoadedModel
from tests.conftest import FakeBackend, bow_vector


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "node_modules" / "junk").mkdir(parents=True)
    (root / "src" / "upload.py").write_text(
        "def retry_upload(file):\n    for attempt in range(3):\n        try:\n            return upload(file)\n"
        "        except IOError:\n            pass  # retry failed upload\n"
    )
    (root / "src" / "config.py").write_text("def parse_config(path):\n    return json.load(open(path))\n")
    (root / "node_modules" / "junk" / "x.js").write_text("ignored()")
    (root / "img.png").write_bytes(b"\x89PNG\x00\x00binary")
    return root


async def test_index_is_incremental(app, fake: FakeBackend, tmp_path: Path):
    root = _repo(tmp_path)
    out = await index_codebase(app, root=str(root), include_globs=None, force=False, ctx=None)
    assert "scanned=2" in out and "reindexed=2" in out
    first = fake.embed_calls
    assert first == 2  # one batch per file
    assert not any("node_modules" in t for t in fake.embedded)

    out = await index_codebase(app, root=str(root), include_globs=None, force=False, ctx=None)
    assert "unchanged=2" in out and fake.embed_calls == first  # nothing re-embedded

    # touch mtime but same content -> hash check saves the embed call
    p = root / "src" / "config.py"
    os.utime(p, (time.time() + 5, time.time() + 5))
    out = await index_codebase(app, root=str(root), include_globs=None, force=False, ctx=None)
    assert "unchanged=2" in out and fake.embed_calls == first

    # real change -> exactly one file re-embedded
    p.write_text("def parse_config(path):\n    return toml.load(path)\n")
    os.utime(p, (time.time() + 10, time.time() + 10))
    out = await index_codebase(app, root=str(root), include_globs=None, force=False, ctx=None)
    assert "reindexed=1" in out and fake.embed_calls == first + 1

    # deletion -> removed
    p.unlink()
    out = await index_codebase(app, root=str(root), include_globs=None, force=False, ctx=None)
    assert "removed=1" in out and "total: 1 files" in out


async def test_index_lives_in_data_dir_not_repo(app, fake: FakeBackend, tmp_path: Path):
    root = _repo(tmp_path)
    out = await index_codebase(app, root=str(root), include_globs=None, force=False, ctx=None)
    assert not (root / ".ollama-agent").exists()
    assert f"index: {app.settings.data_dir}" in out


async def test_search_ranks_semantically_similar_chunk(app, fake: FakeBackend, tmp_path: Path):
    root = _repo(tmp_path)
    out = await search_code(app, query="retry failed upload", root=str(root), top_k=2, refresh=True, ctx=None)
    lines = out.splitlines()
    assert lines[0] == "query: retry failed upload"
    assert "1. src/upload.py:1-" in out
    assert out.index("src/upload.py") < out.index("src/config.py")
    assert fake.embedded[-1].startswith("Instruct:")  # query got the instruction prefix


async def test_search_without_index(app, fake: FakeBackend, tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    out = await search_code(app, query="x", root=str(root), top_k=3, refresh=False, ctx=None)
    assert "is empty" in out


def test_store_search_math(tmp_path: Path):
    store = IndexStore(tmp_path / "i.sqlite")
    store.replace_file("a", 1.0, "h", [(1, 2, "alpha beta")], [bow_vector("alpha beta")])
    store.replace_file("b", 1.0, "h", [(1, 2, "gamma delta")], [bow_vector("gamma delta")])
    hits = store.search(bow_vector("alpha"), top_k=5)
    assert hits[0][0].path == "a" and hits[0][1] > hits[1][1]
    assert store.stats() == (2, 2)
    store.remove_files(["a"])
    assert store.stats() == (1, 1)


async def test_status_warns_on_foreign_models(app, fake: FakeBackend):
    fake.loaded = [
        LoadedModel(name="qwen3.5:latest", size=10, size_vram=10),
        LoadedModel(name="llama3:8b", size=10, size_vram=5),
    ]
    fake.models = ["qwen3.5:latest", "qwen3-embedding:0.6b"]  # 4b not pulled
    out = await local_models_status(app)
    assert "llama3:8b" in out and "reload thrash" in out
    assert "ollama pull qwen3.5:4b" in out
    assert "50% GPU" in out
