"""local_models_status: what the server will use and what Ollama has loaded."""

from __future__ import annotations

import logging

from ..app import App
from ..routing import profile_models, profile_spec

log = logging.getLogger(__name__)


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


async def local_models_status(app: App) -> str:
    settings = app.settings
    spec = profile_spec(settings)
    models = profile_models(settings)
    lines = [
        f"profile: {settings.profile} — {spec.note}",
        f"ollama: {settings.ollama_host}",
        "",
        "tier -> model",
        f"  strong : {models['strong']}",
        f"  fast   : {models['fast']}",
        f"  embed  : {models['embed']}",
        f"num_ctx={settings.num_ctx} keep_alive={settings.keep_alive} max_concurrency={settings.max_concurrency}",
        "",
    ]
    warnings: list[str] = []

    try:
        version = await app.backend.version()
        lines.append(f"ollama version: {version}")
    except Exception as exc:  # noqa: BLE001
        return "\n".join(lines + [f"error: cannot reach Ollama at {settings.ollama_host}: {exc}"])

    try:
        available = set(await app.backend.list_models())
    except Exception as exc:  # noqa: BLE001
        available = set()
        warnings.append(f"could not list models: {exc}")
    for tier, name in models.items():
        if available and name not in available and f"{name}:latest" not in available:
            warnings.append(f"{tier} model {name} is not pulled — run: ollama pull {name}")

    try:
        loaded = await app.backend.ps()
    except Exception as exc:  # noqa: BLE001
        loaded = []
        warnings.append(f"could not query loaded models: {exc}")

    lines.append("")
    if not loaded:
        lines.append("loaded: (none — first call will load the model, ~5-20 s)")
    else:
        lines.append("loaded (ollama ps):")
        for m in loaded:
            where = "GPU" if m.gpu_fraction >= 0.99 else f"{m.gpu_fraction:.0%} GPU / rest CPU"
            ctx = f" ctx={m.context_length}" if m.context_length else ""
            lines.append(f"  {m.name:28} {_gib(m.size):>9}  {where}{ctx}")
    wanted = set(models.values())
    foreign = [m.name for m in loaded if m.name not in wanted and m.name.split(":")[0] + ":latest" not in wanted]
    if foreign:
        warnings.append(
            "models loaded that this profile does not use: "
            + ", ".join(foreign)
            + " — they compete for VRAM and may cause reload thrash"
        )
    gpu_total = sum(m.size_vram for m in loaded)
    if loaded:
        lines.append(f"  total on GPU: {_gib(gpu_total)}")

    if warnings:
        lines.append("")
        lines.extend(f"warning: {w}" for w in warnings)
    return "\n".join(lines)
