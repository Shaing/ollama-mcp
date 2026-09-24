#!/usr/bin/env bash
# Install ollama-agent as a uv tool and register it with Claude Code at user scope,
# so the local tools are available in every repo (not only this one via .mcp.json).
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE="${OLLAMA_AGENT_PROFILE:-trio}"

uv tool install --editable "$HERE" --force
BIN="$(command -v ollama-agent || echo "$HOME/.local/bin/ollama-agent")"
"$BIN" --check

claude mcp remove --scope user ollama-agent >/dev/null 2>&1 || true
claude mcp add --scope user --transport stdio ollama-agent \
  --env "OLLAMA_AGENT_PROFILE=$PROFILE" \
  -- "$BIN"

echo
claude mcp list
echo
echo "Done. Start a new Claude Code session; tools appear as mcp__ollama-agent__<tool>."
echo "Switch profile per session with: OLLAMA_AGENT_PROFILE=big claude"
