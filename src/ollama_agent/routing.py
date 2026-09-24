"""Model routing: which Ollama model serves which tier under the active profile.

The profile is fixed for the lifetime of the server process so that one Claude
Code session never alternates between the 35B model and the 9B/4B pair (each
swap costs 15-20 s of reload on a 16 GB GPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .config import Settings

Tier = Literal["strong", "fast"]


@dataclass(frozen=True)
class ProfileSpec:
    strong: str
    fast: str
    embed: str
    embed_num_gpu: int | None  # 0 forces the embedder onto the CPU
    note: str


PROFILES: dict[str, ProfileSpec] = {
    "trio": ProfileSpec(
        strong="qwen3.5:latest",
        fast="qwen3.5:4b",
        embed="qwen3-embedding:0.6b",
        embed_num_gpu=None,
        note="9B strong + 4B fast + embeddings all resident on the GPU; zero reloads.",
    ),
    "big": ProfileSpec(
        strong="qwen3.6:35b-a3b",
        fast="qwen3.6:35b-a3b",  # never load a second LLM next to the 35B
        embed="qwen3-embedding:0.6b",
        embed_num_gpu=0,  # keep the 35B's GPU slice intact
        note="35B-A3B for both tiers (12 GiB on GPU, rest in RAM); embeddings on CPU.",
    ),
}


@dataclass(frozen=True)
class GenSpec:
    model: str
    num_ctx: int
    keep_alive: str
    think: bool
    temperature: float
    num_predict: int
    num_gpu: int | None = None

    def options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
        }
        if self.num_gpu is not None:
            opts["num_gpu"] = self.num_gpu
        return opts


EMBED_NUM_CTX = 2048  # chunks are ~1500 chars; a small window keeps the embedder's VRAM footprint low


@dataclass(frozen=True)
class EmbedSpec:
    model: str
    keep_alive: str
    num_gpu: int | None = None
    num_ctx: int = EMBED_NUM_CTX

    def options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {"num_ctx": self.num_ctx}
        if self.num_gpu is not None:
            opts["num_gpu"] = self.num_gpu
        return opts


def profile_spec(settings: Settings) -> ProfileSpec:
    return PROFILES[settings.profile]


def model_for(settings: Settings, tier: Tier) -> str:
    spec = profile_spec(settings)
    if tier == "strong":
        return settings.strong_model or spec.strong
    if tier == "fast":
        return settings.fast_model or spec.fast
    raise ValueError(f"unknown tier {tier!r}")


def embed_model_for(settings: Settings) -> str:
    return settings.embed_model or profile_spec(settings).embed


def profile_models(settings: Settings) -> dict[str, str]:
    """tier -> model, as resolved for the active profile plus overrides."""
    return {
        "strong": model_for(settings, "strong"),
        "fast": model_for(settings, "fast"),
        "embed": embed_model_for(settings),
    }


def gen_spec(
    settings: Settings,
    tier: Tier,
    *,
    think: bool,
    temperature: float,
    num_predict: int,
) -> GenSpec:
    return GenSpec(
        model=model_for(settings, tier),
        num_ctx=settings.num_ctx,
        keep_alive=settings.keep_alive,
        think=think,
        temperature=temperature,
        num_predict=num_predict,
    )


def embed_spec(settings: Settings) -> EmbedSpec:
    return EmbedSpec(
        model=embed_model_for(settings),
        keep_alive=settings.keep_alive,
        num_gpu=profile_spec(settings).embed_num_gpu,
    )


def warmup_specs(settings: Settings) -> list[GenSpec]:
    """Chat models to preload at startup, largest first so the GPU fills in a stable order.

    Loading the 9B after the 4B and the embedder leaves it partly on the CPU; loading it
    first keeps all three resident. Duplicates (profile 'big') collapse to one entry.
    """
    seen: list[str] = []
    for tier in ("strong", "fast"):
        m = model_for(settings, tier)  # type: ignore[arg-type]
        if m not in seen:
            seen.append(m)
    return [
        GenSpec(model=m, num_ctx=settings.num_ctx, keep_alive=settings.keep_alive, think=False,
                temperature=0.0, num_predict=1)
        for m in seen
    ]
