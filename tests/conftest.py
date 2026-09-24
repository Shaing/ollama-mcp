"""Shared fixtures: a scripted fake Ollama backend and an isolated App."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from ollama_agent.backend import ChatChunk, LoadedModel
from ollama_agent.config import Settings
from ollama_agent.routing import EmbedSpec, GenSpec
from ollama_agent.server import build_app, build_server

Reply = str | Callable[[GenSpec, Sequence[dict[str, Any]]], str]


def bow_vector(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-words embedding so shared words raise cosine similarity."""
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z_]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    return vec


class FakeBackend:
    def __init__(self, replies: list[Reply] | None = None, *, delay: float = 0.0) -> None:
        self.replies: list[Reply] = list(replies or [])
        self.delay = delay
        self.calls: list[tuple[GenSpec, list[dict[str, Any]], dict[str, Any] | None]] = []
        self.embed_calls = 0
        self.embedded: list[str] = []
        self.loaded: list[LoadedModel] = []
        self.models = ["qwen3.5:latest", "qwen3.5:4b", "qwen3-embedding:0.6b", "qwen3.6:35b-a3b"]

    async def stream_chat(
        self, spec: GenSpec, messages: Sequence[dict[str, Any]], format: dict[str, Any] | None = None
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append((spec, list(messages), format))
        reply: Reply = self.replies.pop(0) if self.replies else "ok"
        text = reply(spec, messages) if callable(reply) else reply
        words = text.split(" ")
        for i, w in enumerate(words):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield ChatChunk(content=w + (" " if i < len(words) - 1 else ""))
        yield ChatChunk(
            done=True,
            done_reason="stop",
            prompt_eval_count=sum(len(m["content"]) for m in messages) // 4,
            eval_count=len(words),
            total_duration_ns=1_000_000,
            load_duration_ns=0,
        )

    async def embed(self, spec: EmbedSpec, texts: Sequence[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.embedded.extend(texts)
        return [bow_vector(t) for t in texts]

    async def load_model(self, spec: GenSpec) -> None:
        self.loaded.append(LoadedModel(name=spec.model, size=1, size_vram=1))

    async def ps(self) -> list[LoadedModel]:
        return self.loaded

    async def list_models(self) -> list[str]:
        return self.models

    async def version(self) -> str:
        return "fake"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", max_timeout_s=600)


@pytest.fixture
def fake() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def app(settings: Settings, fake: FakeBackend):
    return build_app(settings, backend=fake)


@pytest.fixture
def server(settings: Settings, fake: FakeBackend):
    return build_server(settings, backend=fake)
