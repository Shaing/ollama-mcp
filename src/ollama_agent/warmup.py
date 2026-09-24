"""Preload the profile's models at startup so VRAM fills in a predictable order."""

from __future__ import annotations

import logging
import time

from .app import App
from .routing import embed_spec, warmup_specs

log = logging.getLogger(__name__)


async def warmup(app: App) -> list[str]:
    """Load chat models largest-first, then the embedder. Returns the models loaded.

    Runs in the background after the MCP handshake, so a slow load never delays
    Claude Code's startup. Failures are logged, never raised.
    """
    loaded: list[str] = []
    async with app.semaphore:
        for spec in warmup_specs(app.settings):
            t0 = time.monotonic()
            try:
                await app.backend.load_model(spec)
                loaded.append(spec.model)
                log.info("warmup: %s loaded in %.1fs", spec.model, time.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001
                log.warning("warmup: could not load %s: %s", spec.model, exc)
        espec = embed_spec(app.settings)
        try:
            await app.backend.embed(espec, ["warmup"])
            loaded.append(espec.model)
            log.info("warmup: %s loaded", espec.model)
        except Exception as exc:  # noqa: BLE001
            log.warning("warmup: could not load %s: %s", espec.model, exc)
    return loaded
