from pathlib import Path

from ollama_agent.chunking import chunk_lines
from ollama_agent.outputs import OutputStore, clip, render


def test_chunks_cover_everything_with_overlap():
    text = "\n".join(f"line {i:03d}" for i in range(100))
    chunks = chunk_lines(text, max_chars=200, overlap_lines=3)
    assert chunks[0].start == 1
    assert chunks[-1].end == 100
    for a, b in zip(chunks, chunks[1:]):
        assert b.start == a.end - 3 + 1  # 3 shared lines
        assert len(a.text) <= 200
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end + 1))
    assert covered == set(range(1, 101))


def test_long_line_is_its_own_chunk():
    text = "short\n" + "x" * 500 + "\nshort"
    chunks = chunk_lines(text, max_chars=100, overlap_lines=0)
    assert [c.text for c in chunks] == ["short", "x" * 500, "short"]


def test_empty_text():
    assert chunk_lines("", 100) == []


def test_clip_prefers_line_boundary():
    text = "\n".join(["a" * 40] * 10)
    head, truncated = clip(text, 100)
    assert truncated and head.endswith("a" * 40) and len(head) <= 100
    assert clip("short", 100) == ("short", False)


def test_store_and_render(tmp_path: Path):
    store = OutputStore(tmp_path)
    p = store.save("delegate_task", "hello\nworld", {"model": "m", "n": 1})
    assert p.parent == tmp_path / "outputs"
    body = p.read_text()
    assert body.startswith("<!-- ollama-agent delegate_task") and body.rstrip().endswith("hello\nworld")
    out = render("body", tool="t", model="m", prompt_tokens=10, completion_tokens=5, seconds=1.234,
                 output_path=p, truncated=True, warnings=["w1"])
    assert "[t] model=m tokens=10 in / 5 out time=1.2s" in out
    assert "[truncated" in out and str(p) in out and "warning: w1" in out
