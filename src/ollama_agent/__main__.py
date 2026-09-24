"""Entry point: `ollama-agent` runs the MCP server over stdio.

stdout is the MCP transport, so all logging goes to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, Settings, load_dotenv
from .server import build_app, build_server
from .tools.status import local_models_status


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ollama-agent", description="Local Ollama models as MCP tools")
    p.add_argument("--check", action="store_true", help="print settings and Ollama status, then exit")
    p.add_argument("--version", action="version", version=__version__)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    load_dotenv(Path.cwd() / ".env")
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"ollama-agent: {exc}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Ollama request otherwise
    log = logging.getLogger("ollama_agent")
    log.info("profile=%s data_dir=%s ollama=%s", settings.profile, settings.data_dir, settings.ollama_host)

    if args.check:
        report = asyncio.run(local_models_status(build_app(settings)))
        print(report, file=sys.stderr)
        return 0 if "error:" not in report else 1

    build_server(settings).run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
