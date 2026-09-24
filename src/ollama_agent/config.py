"""Settings loaded from the environment (and an optional .env file).

Mirrors the pattern in ~/work/slack-agent/cc_slack/config.py: no third-party
dotenv dependency, existing environment variables win over the file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

Profile = Literal["trio", "big"]
PROFILE_NAMES: tuple[str, ...] = get_args(Profile)


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, '#' comments, no interpolation."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.sub(r"\s+#.*$", "", value).strip()
        os.environ.setdefault(key, value)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {value!r}") from exc


class ConfigError(Exception):
    pass


def default_data_dir() -> Path:
    """$XDG_CACHE_HOME/ollama-agent (~/.cache/ollama-agent): never inside the user's repos."""
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "ollama-agent"


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://127.0.0.1:11434"
    profile: Profile = "trio"
    strong_model: str | None = None  # overrides the profile's choice
    fast_model: str | None = None
    embed_model: str | None = None
    num_ctx: int = 32768
    keep_alive: str = "30m"
    max_return_chars: int = 10_000
    max_input_chars: int = 90_000
    max_timeout_s: int = 600
    max_concurrency: int = 2
    data_dir: Path = Path(".ollama-agent")
    log_level: str = "INFO"
    warmup: bool = True  # preload the profile's models (largest first) when the server starts

    @classmethod
    def from_env(cls, cwd: Path | None = None) -> "Settings":
        cwd = (cwd or Path.cwd()).resolve()
        profile = _env("OLLAMA_AGENT_PROFILE") or "trio"
        if profile not in PROFILE_NAMES:
            raise ConfigError(
                f"OLLAMA_AGENT_PROFILE must be one of {', '.join(PROFILE_NAMES)}, got {profile!r}"
            )
        data_dir_raw = _env("OLLAMA_AGENT_DATA_DIR")
        data_dir = Path(data_dir_raw).expanduser() if data_dir_raw else default_data_dir()
        return cls(
            ollama_host=_env("OLLAMA_HOST") or "http://127.0.0.1:11434",
            profile=profile,  # type: ignore[arg-type]
            strong_model=_env("OLLAMA_AGENT_STRONG_MODEL"),
            fast_model=_env("OLLAMA_AGENT_FAST_MODEL"),
            embed_model=_env("OLLAMA_AGENT_EMBED_MODEL"),
            num_ctx=_env_int("OLLAMA_AGENT_NUM_CTX", 32768),
            keep_alive=_env("OLLAMA_AGENT_KEEP_ALIVE") or "30m",
            max_return_chars=_env_int("OLLAMA_AGENT_MAX_RETURN_CHARS", 10_000),
            max_input_chars=_env_int("OLLAMA_AGENT_MAX_INPUT_CHARS", 90_000),
            max_timeout_s=_env_int("OLLAMA_AGENT_MAX_TIMEOUT_S", 600),
            max_concurrency=_env_int("OLLAMA_AGENT_MAX_CONCURRENCY", 2),
            data_dir=data_dir,
            log_level=(_env("OLLAMA_AGENT_LOG_LEVEL") or "INFO").upper(),
            warmup=_env_bool("OLLAMA_AGENT_WARMUP", True),
        )
