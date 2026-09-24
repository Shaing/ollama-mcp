#!/usr/bin/env bash
# Install ollama-agent as a uv tool and register it with Claude Code at user scope,
# so the local tools are available in every repo (not only this one via .mcp.json).
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
# Pin a profile only when asked (OLLAMA_AGENT_PROFILE=big scripts/register.sh). A pinned --env
# value overrides the shell, so leaving it out keeps `OLLAMA_AGENT_PROFILE=big claude` working;
# the server defaults to trio.
ENV_ARGS=()
if [[ -n ${OLLAMA_AGENT_PROFILE:-} ]]; then
  ENV_ARGS=(--env "OLLAMA_AGENT_PROFILE=$OLLAMA_AGENT_PROFILE")
fi

uv tool install --editable "$HERE" --force
# Not `command -v`: an activated repo .venv would shadow the uv tool install.
BIN="$(uv tool dir --bin)/ollama-agent"
"$BIN" --check

claude mcp remove --scope user ollama-agent >/dev/null 2>&1 || true
claude mcp add --scope user --transport stdio ollama-agent \
  "${ENV_ARGS[@]}" -- "$BIN"

echo
claude mcp list
echo
echo "Done. Start a new Claude Code session; tools appear as mcp__ollama-agent__<tool>."
if [[ ${#ENV_ARGS[@]} -eq 0 ]]; then
  echo "Switch profile per session with: OLLAMA_AGENT_PROFILE=big claude"
else
  echo "Profile pinned to $OLLAMA_AGENT_PROFILE; re-run without OLLAMA_AGENT_PROFILE to allow per-session switching."
fi
