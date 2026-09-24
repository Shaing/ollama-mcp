"""Property-based checks of formal/SPEC.md against the real code (Hypothesis).

IDs in the test names and comments refer to the properties in formal/SPEC.md.  Tests marked
`xfail(strict=True)` pin down deviations the formal analysis found: fixing the code makes
them pass, which fails the suite until the marker (and the SPEC.md row) is removed.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import AsyncIterator, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hsettings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from ollama_agent.backend import ChatChunk
from ollama_agent.chunking import chunk_lines
from ollama_agent.config import Settings, load_dotenv
from ollama_agent.files import read_paths
from ollama_agent.generate import Progress, estimate_tokens, run_generation
from ollama_agent.index.store import IndexStore
from ollama_agent.outputs import clip
from ollama_agent.routing import (
    PROFILES,
    GenSpec,
    embed_spec,
    gen_spec,
    model_for,
    profile_models,
    warmup_specs,
)
from ollama_agent.server import build_app
from ollama_agent.tools.review import (
    MAX_PREDICT,
    Finding,
    ReviewCore,
    _normalize_verdict,
    review_diff,
)
from tests.conftest import FakeBackend

# str.splitlines() also breaks on these; keeping them out of generated lines makes `lines`
# the reference the chunker must reproduce.
LINE_BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85  "
line_st = st.text(st.characters(blacklist_categories=("Cs",), blacklist_characters=LINE_BREAKS), max_size=40)
lines_st = st.lists(line_st, max_size=30)
short_lines_st = st.lists(st.text("ab", min_size=1, max_size=4), min_size=3, max_size=8)


# --- C: chunking.chunk_lines -------------------------------------------------------------
@given(lines_st, st.integers(1, 200), st.integers(0, 12))
def test_chunk_lines_covers_every_line_in_order(lines, max_chars, overlap):
    text = "\n".join(lines)
    ref = text.splitlines()
    chunks = chunk_lines(text, max_chars=max_chars, overlap_lines=overlap)
    n = len(ref)
    if n == 0:
        assert chunks == []  # C7
        return
    assert chunks[0].start == 1 and chunks[-1].end == n  # C1: from the first line to the last
    for c in chunks:
        assert 1 <= c.start <= c.end <= n
        assert c.text == "\n".join(ref[c.start - 1 : c.end])  # C2: the text is exactly its lines
        assert len(c.text) <= max_chars or c.start == c.end  # C3: size bound; a long line stands alone
    for a, b in pairwise(chunks):
        assert b.start == max(a.end - overlap + 1, a.start + 1)  # C5: the exact overlap rule
        assert b.start <= a.end + 1  # C1: no gap between chunks
        assert b.end >= a.end  # C6: ends never move backwards
    assert {ln for c in chunks for ln in range(c.start, c.end + 1)} == set(range(1, n + 1))  # C1


@pytest.mark.xfail(strict=True, reason="SPEC C8: a chunk can be a strict sub-window of its predecessor")
@given(short_lines_st, st.integers(2, 12), st.integers(1, 4))
def test_chunk_lines_every_chunk_adds_a_line(lines, max_chars, overlap):
    chunks = chunk_lines("\n".join(lines), max_chars=max_chars, overlap_lines=overlap)
    for a, b in pairwise(chunks):
        assert b.end > a.end, f"{b} is contained in {a}"


@pytest.mark.xfail(strict=True, reason="SPEC C5: a negative overlap_lines silently skips lines")
@given(short_lines_st, st.integers(-5, -1))
def test_chunk_lines_negative_overlap_is_rejected_or_harmless(lines, overlap):
    text = "\n".join(lines)
    try:
        chunks = chunk_lines(text, max_chars=1, overlap_lines=overlap)
    except ValueError:
        return  # rejecting it is fine too
    covered = {ln for c in chunks for ln in range(c.start, c.end + 1)}
    assert covered == set(range(1, len(text.splitlines()) + 1))


# --- O: outputs.clip ---------------------------------------------------------------------
@given(st.text(max_size=300), st.integers(1, 200))
def test_clip_bounds_prefix_and_line_boundary(text, max_chars):
    head, truncated = clip(text, max_chars)
    assert truncated == (len(text) > max_chars)  # O2
    assert text.startswith(head)  # O4: the head is a prefix
    assert len(head) <= max_chars  # O1
    if truncated:
        assert 5 * len(head) >= 3 * max_chars  # O3: at least 60% of the budget survives
        if len(head) < max_chars:
            assert text[len(head)] == "\n"  # O5: a shorter cut lands on a line boundary
    else:
        assert head == text


# --- F: files.read_paths -----------------------------------------------------------------
@hsettings(max_examples=40, deadline=None)
@given(st.lists(st.integers(0, 60), max_size=6), st.integers(0, 150), st.data())
def test_read_paths_budget_and_accounting(sizes, budget, data):
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for k, size in enumerate(sizes):
            p = Path(d) / f"f{k}.txt"
            p.write_text("x" * size)
            paths.append(str(p))
        paths.insert(data.draw(st.integers(0, len(paths))), str(Path(d) / "missing.txt"))
        blocks, notes = read_paths(paths, budget_chars=budget)
    taken = [str(b.path) for b in blocks]
    assert sum(len(b.text) for b in blocks) <= budget  # F1: the budget is never exceeded
    assert len(blocks) + len(notes) == len(paths)  # F2: every path is either read or explained
    assert taken == [p for p in paths if p in set(taken)]  # F3: input order is preserved
    assert not any(t.endswith("missing.txt") for t in taken)


# --- K: config.load_dotenv ---------------------------------------------------------------
key_st = st.from_regex(r"\AOLLAMA_PBT_[A-Z0-9_]{1,8}\Z")
value_st = st.text(st.characters(min_codepoint=33, max_codepoint=126, blacklist_characters="#\"'"), min_size=1, max_size=20)


@given(key_st, value_st, value_st)
def test_dotenv_parses_values_and_never_overrides(key, value, existing):
    with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ):
        os.environ.pop(key, None)
        os.environ.pop(key + "_Q", None)
        env = Path(d) / ".env"
        env.write_text(f"# comment\n\n{key}={value}   # trailing comment\n{key}_Q=\"{value} # kept\"\n")
        load_dotenv(env)
        assert os.environ[key] == value  # K1: unquoted value, trailing comment stripped
        assert os.environ[key + "_Q"] == f"{value} # kept"  # K2: quotes protect '#'
        os.environ[key] = existing
        load_dotenv(env)
        assert os.environ[key] == existing  # K3: the existing environment wins


# --- R: routing ----------------------------------------------------------------------------
model_name_st = st.text(st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=12)
maybe_model_st = st.one_of(st.none(), model_name_st)


@given(st.sampled_from(sorted(PROFILES)), maybe_model_st, maybe_model_st, maybe_model_st,
       st.integers(1024, 131072), st.text(min_size=1, max_size=4))
def test_routing_resolution(profile, strong, fast, embed, num_ctx, keep_alive):
    s = Settings(profile=profile, strong_model=strong, fast_model=fast, embed_model=embed,
                 num_ctx=num_ctx, keep_alive=keep_alive)
    p = PROFILES[profile]
    assert model_for(s, "strong") == (strong or p.strong)  # R3: an override wins over the profile
    assert model_for(s, "fast") == (fast or p.fast)
    es = embed_spec(s)
    assert (es.model, es.num_gpu, es.keep_alive) == (embed or p.embed, p.embed_num_gpu, keep_alive)
    assert profile_models(s) == {"strong": model_for(s, "strong"), "fast": model_for(s, "fast"), "embed": es.model}
    ws = warmup_specs(s)
    assert [w.model for w in ws] == list(dict.fromkeys([model_for(s, "strong"), model_for(s, "fast")]))  # R4
    for w in ws:
        assert (w.num_ctx, w.keep_alive, w.num_predict, w.think) == (num_ctx, keep_alive, 1, False)  # R6
    g = gen_spec(s, "fast", think=True, temperature=0.3, num_predict=77)
    assert (g.num_ctx, g.keep_alive) == (num_ctx, keep_alive)  # R7: one num_ctx per process
    assert g.options() == {"num_ctx": num_ctx, "temperature": 0.3, "num_predict": 77}


def test_big_profile_is_one_llm_without_overrides():
    s = Settings(profile="big")
    assert len(warmup_specs(s)) == 1 and embed_spec(s).num_gpu == 0  # R2


# --- V: review._normalize_verdict ---------------------------------------------------------
finding_st = st.builds(
    Finding,
    file=st.text(max_size=6),
    line=st.one_of(st.none(), st.integers(0, 999)),
    severity=st.sampled_from(["critical", "major", "minor", "nit"]),
    summary=st.text(max_size=12),
    suggestion=st.text(max_size=6),
)
core_st = st.builds(
    ReviewCore,
    verdict=st.sampled_from(["approve", "request_changes", "comment"]),
    summary=st.text(max_size=12),
    findings=st.lists(finding_st, max_size=4),
)


@given(core_st)
def test_normalize_verdict_never_approves_serious_findings(core):
    notes: list[str] = []
    out = _normalize_verdict(core, notes)
    serious = any(f.severity in ("critical", "major") for f in core.findings)
    assert not (out.verdict == "approve" and serious)  # V1
    assert (out.findings, out.summary) == (core.findings, core.summary)  # V2: only the verdict may change
    if out.verdict != core.verdict:
        assert core.verdict == "approve" and serious and len(notes) == 1
    else:
        assert notes == []


GOOD_JSON = json.dumps({"verdict": "approve", "summary": "fine", "findings": []})


@pytest.mark.xfail(strict=True, reason="SPEC R1: the 'cut to fit the context' is off by one token")
def test_review_cut_fits_the_context_exactly(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, num_ctx=16384, max_input_chars=90_000)
    fake = FakeBackend(replies=[GOOD_JSON])
    app = build_app(settings, backend=fake)
    res = asyncio.run(review_diff(app, diff="+x\n" * 30_000, git_range="", cwd="", focus="", think=False,
                                  timeout_s=30, ctx=None))
    assert any("further cut" in n for n in res.notes)
    sent = fake.calls[0][1][1]["content"]
    body = sent[len("```diff\n") : -len("\n```")]
    assert estimate_tokens(body) + 400 + MAX_PREDICT <= settings.num_ctx


# --- G: generate.run_generation -----------------------------------------------------------
class _ReasonBackend:
    """Streams `words`, then a final chunk carrying `done_reason`."""

    def __init__(self, words: list[str], done_reason: str | None) -> None:
        self.words, self.done_reason = words, done_reason

    async def stream_chat(self, spec: GenSpec, messages: Sequence[dict[str, Any]],
                          format: dict[str, Any] | None = None) -> AsyncIterator[ChatChunk]:
        for w in self.words:
            yield ChatChunk(content=w)
        yield ChatChunk(done=True, done_reason=self.done_reason, prompt_eval_count=1, eval_count=len(self.words))


@given(st.lists(st.text("ab", min_size=1, max_size=3), max_size=5), st.sampled_from(["stop", "length", None]))
def test_run_generation_reports_the_length_cap(words, done_reason):
    spec = GenSpec(model="m", num_ctx=1024, keep_alive="1m", think=False, temperature=0.0, num_predict=8)
    res = asyncio.run(run_generation(_ReasonBackend(words, done_reason), spec, [{"role": "user", "content": "x"}],
                                     timeout_s=5, progress=Progress(None, "t")))
    assert res.text == "".join(words)  # G1: everything streamed is returned
    assert res.truncated == (done_reason == "length")  # G2
    assert res.reason == ("length" if done_reason == "length" else None)
    assert len(res.warnings) == (1 if res.truncated else 0)  # G3: a warning iff truncated
    assert res.completion_tokens == len(words)


@given(st.text(max_size=500))
def test_estimate_tokens_bounds(text):
    e = estimate_tokens(text)
    assert e >= 1 and 3.5 * (e - 1) <= len(text) < 3.5 * e  # E1


@given(st.text(max_size=200), st.text(max_size=200))
def test_estimate_tokens_is_monotone(a, b):
    assert estimate_tokens(a) <= estimate_tokens(a + b)  # E2


# --- I: index.store.IndexStore (model-based, stateful) ------------------------------------
DIM = 4
vec_st = st.lists(st.floats(-1, 1, allow_nan=False, allow_infinity=False, width=32), min_size=DIM, max_size=DIM)
path_st = st.sampled_from(["a.py", "b.py", "c/d.py"])
mtime_st = st.floats(0, 2e9, allow_nan=False, allow_infinity=False)


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n else a


class IndexStoreMachine(RuleBasedStateMachine):
    """The SQLite store must behave like a dict path -> (mtime, sha, chunks, vectors)."""

    def __init__(self) -> None:
        super().__init__()
        self._tmp = tempfile.TemporaryDirectory()
        self.store = IndexStore(Path(self._tmp.name) / "index.sqlite")
        self.model: dict[str, tuple[float, str, list[tuple[int, int, str]], list[list[float]]]] = {}

    @rule(path=path_st, mtime=mtime_st, sha=st.text("0123456789abcdef", min_size=1, max_size=8),
          vecs=st.lists(vec_st, max_size=3))
    def replace(self, path, mtime, sha, vecs):
        chunks = [(10 * k + 1, 10 * k + 10, f"{path}#{k}") for k in range(len(vecs))]
        self.store.replace_file(path, mtime, sha, chunks, vecs)
        self.model[path] = (mtime, sha, chunks, vecs)  # I1: replace is total for that path

    @rule(paths=st.lists(path_st, max_size=3))
    def remove(self, paths):
        self.store.remove_files(paths)
        for p in paths:
            self.model.pop(p, None)  # I2: chunks and file row go together

    @rule(path=path_st, mtime=mtime_st)
    def touch(self, path, mtime):
        self.store.touch_mtime(path, mtime)
        if path in self.model:
            _, sha, chunks, vecs = self.model[path]
            self.model[path] = (mtime, sha, chunks, vecs)

    @rule(q=vec_st, k=st.integers(1, 6))
    def search(self, q, k):
        hits = self.store.search(q, k)
        rows = [(p, c, v) for p, (_, _, cs, vs) in self.model.items() for c, v in zip(cs, vs)]
        assert len(hits) == min(k, len(rows))  # I5: exactly min(top_k, n) results
        qn = _unit(q)
        expected = sorted((float(_unit(v) @ qn) for _, _, v in rows), reverse=True)[:k]
        got = [s for _, s in hits]
        assert got == sorted(got, reverse=True)  # I5: best first
        assert got == pytest.approx(expected, abs=1e-5)  # I5: these are the true top-k cosines
        for row, score in hits:
            match = [v for p, (s, e, t), v in rows if (p, s, e, t) == (row.path, row.start, row.end, row.text)]
            assert len(match) == 1 and float(_unit(match[0]) @ qn) == pytest.approx(score, abs=1e-5)

    @invariant()
    def store_matches_model(self):
        nf, nc = self.store.stats()
        assert nf == len(self.model)  # I3
        assert nc == sum(len(cs) for _, _, cs, _ in self.model.values())  # I3: files.nchunks == real chunks
        states = self.store.file_states()
        assert set(states) == set(self.model)
        for p, fs in states.items():
            mtime, sha, chunks, _ = self.model[p]
            assert (fs.mtime, fs.sha256, fs.nchunks) == (mtime, sha, len(chunks))
        rows, mat = self.store.matrix()
        assert len(rows) == nc and all(r.path in self.model for r in rows)  # I4: no orphan chunks, cache is fresh
        assert nc == 0 or mat.shape == (nc, DIM)

    def teardown(self) -> None:
        self.store.close()
        self._tmp.cleanup()


TestIndexStore = IndexStoreMachine.TestCase
TestIndexStore.settings = hsettings(max_examples=50, stateful_step_count=12, deadline=None)
