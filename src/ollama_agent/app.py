"""Process-wide state shared by every tool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .backend import Backend
from .config import Settings
from .outputs import OutputStore


@dataclass
class App:
    settings: Settings
    backend: Backend
    outputs: OutputStore
    semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.semaphore = asyncio.Semaphore(max(1, self.settings.max_concurrency))

    def clamp_timeout(self, timeout_s: float | int | None, default: int) -> float:
        t = float(timeout_s) if timeout_s else float(default)
        return max(5.0, min(t, float(self.settings.max_timeout_s)))
