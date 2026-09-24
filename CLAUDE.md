# ollama-agent

MCP server (Python, `mcp` 2.x) that exposes local Ollama models to Claude Code as tools.

## Using the local tools (`mcp__ollama-agent__*`)

Prefer them for work where a local model is good enough and cloud context is precious:
- `summarize` for long logs, dumps, generated files, long docs — instead of reading them whole.
- `review_diff` as a first-pass review; verify every finding before acting on it.
- `search_code` for "where is the code that…" questions by meaning; Grep for exact identifiers.
- `delegate_task` for boilerplate, tests for a given file, docstrings, mechanical rewrites.

Do not use them for final correctness or security decisions, whole-repo reasoning, or tasks
needing more than ~32K tokens of context. Treat their output as a suggestion from a junior
engineer. If a tool is slow or fails, `local_models_status` shows what is loaded on the GPU.

## Developing

- `uv sync --all-groups`, then `uv run pytest` (fast, fake backend) and
  `OLLAMA_AGENT_INTEGRATION=1 uv run pytest tests/integration -s` (real Ollama).
- `uv run ollama-agent --check` prints the resolved settings and Ollama status.
- stdout is the MCP transport: log to stderr only, never `print()` in server code.
- Routing lives in `src/ollama_agent/routing.py`; the profile is fixed per process on purpose
  (switching between the 35B model and the 9B/4B pair costs a 15–20 s reload on a 16 GB GPU).
- Tool descriptions are the docstrings in `src/ollama_agent/server.py`; they are what Claude
  reads to decide when to call a tool, so keep the "NOT for" lines.

## Service management

Registered with `~/work/ops/svc` as `ollama-agent` (`KIND=mcp`, see `~/work/ops/services.d/ollama-agent.conf`)
and depends on `ollama` (system unit). It is spawned per Claude Code session, so there is nothing to
start or stop: `svc status ollama-agent` lists the running copies and the session that owns each,
`svc check ollama-agent` runs `--check`, `svc logs ollama-agent` tails the newest MCP log.
If this project ever grows a long-running part (e.g. a shared HTTP MCP server), register it as its
own service per `~/work/ops/README.md`.
