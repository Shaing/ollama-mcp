"""Thin async wrapper over the Ollama client, behind a Protocol so tests can
substitute a fake."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ollama import AsyncClient

from .routing import EmbedSpec, GenSpec


@dataclass
class ChatChunk:
    content: str = ""
    thinking: str = ""
    done: bool = False
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None


@dataclass
class LoadedModel:
    name: str
    size: int
    size_vram: int
    context_length: int | None = None

    @property
    def gpu_fraction(self) -> float:
        return self.size_vram / self.size if self.size else 0.0


class Backend(Protocol):
    def stream_chat(
        self,
        spec: GenSpec,
        messages: Sequence[dict[str, Any]],
        format: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatChunk]: ...

    async def embed(self, spec: EmbedSpec, texts: Sequence[str]) -> list[list[float]]: ...

    async def load_model(self, spec: GenSpec) -> None: ...

    async def ps(self) -> list[LoadedModel]: ...

    async def list_models(self) -> list[str]: ...

    async def version(self) -> str: ...


@dataclass
class OllamaBackend:
    host: str
    _client: AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = AsyncClient(host=self.host)

    async def stream_chat(
        self,
        spec: GenSpec,
        messages: Sequence[dict[str, Any]],
        format: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        stream = await self._client.chat(
            model=spec.model,
            messages=list(messages),
            stream=True,
            think=spec.think,
            format=format,
            options=spec.options(),
            keep_alive=spec.keep_alive,
        )
        async for part in stream:
            msg = part.message
            yield ChatChunk(
                content=msg.content or "",
                thinking=msg.thinking or "",
                done=bool(part.done),
                done_reason=part.done_reason,
                prompt_eval_count=part.prompt_eval_count,
                eval_count=part.eval_count,
                total_duration_ns=part.total_duration,
                load_duration_ns=part.load_duration,
            )

    async def embed(self, spec: EmbedSpec, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embed(
            model=spec.model,
            input=list(texts),
            options=spec.options(),
            keep_alive=spec.keep_alive,
        )
        return [list(vec) for vec in resp.embeddings]

    async def load_model(self, spec: GenSpec) -> None:
        """Load a model without generating: Ollama treats an empty message list as a load request."""
        await self._client.chat(
            model=spec.model, messages=[], options=spec.options(), keep_alive=spec.keep_alive
        )

    async def ps(self) -> list[LoadedModel]:
        resp = await self._client.ps()
        return [
            LoadedModel(
                name=m.name or m.model or "?",
                size=m.size or 0,
                size_vram=m.size_vram or 0,
                context_length=m.context_length,
            )
            for m in resp.models
        ]

    async def list_models(self) -> list[str]:
        resp = await self._client.list()
        return [m.model for m in resp.models if m.model]

    async def version(self) -> str:
        # The client has no typed helper for /api/version; go through its HTTP layer.
        r = await self._client._request_raw("GET", "/api/version")
        return r.json().get("version", "?")
