# ollama-agent

Local Ollama models exposed to **Claude Code** as MCP tools. Claude Code stays the brain
(cloud); the RTX 4080 does the bulk work: summarising big files, first-pass code review,
semantic code search, boilerplate and tests. A launcher also runs Claude Code fully offline
against Ollama.

## Tools

| Tool | What it does | Local model |
|---|---|---|
| `delegate_task` | Run a self-contained subtask (tests, boilerplate, rewrites, log triage) | strong or fast tier |
| `review_diff` | Structured second-opinion review of a diff (`git_range="--staged"`, `main...HEAD`, or raw diff) | strong, thinking on |
| `summarize` | Map-reduce summary of large files/logs, optionally focused on a question | fast (map) + strong (reduce) |
| `index_codebase` / `search_code` | Local embedding index + semantic search by meaning | qwen3-embedding |
| `local_models_status` | Profile, tier→model map, `ollama ps` with GPU/CPU split, thrash warnings | – |

Full outputs are always written to `~/.cache/ollama-agent/outputs/*.md`; Claude gets the head plus the path.

## Profiles (16 GB GPU)

| Profile | strong | fast | embed | Measured |
|---|---|---|---|---|
| `trio` (default) | qwen3.5:latest (9B) | qwen3.5:4b | qwen3-embedding:0.6b | 72 / 97 tok/s, all resident (15.0 GB) |
| `big` | qwen3.6:35b-a3b | same model | embed on CPU | 36–38 tok/s, 12 GiB on GPU + 9 GiB RAM |

The profile is fixed per server process (`OLLAMA_AGENT_PROFILE`); mixing the 35B with the
9B/4B pair would cost a 15–20 s reload on every switch. On startup the server preloads the
profile's models in the background, largest first — loading the 9B last leaves ~15% of it
on the CPU (`OLLAMA_AGENT_WARMUP=0` disables this).

## Setup

```bash
ollama pull qwen3.5:latest qwen3.5:4b qwen3-embedding:0.6b   # trio
ollama pull qwen3.6:35b-a3b                                    # big (optional)

uv sync --all-groups
uv run ollama-agent --check          # settings + Ollama reachability

# Claude Code, this repo only: .mcp.json is already here (approve it on first start).
# Claude Code, every repo:
scripts/register.sh                  # uv tool install + claude mcp add --scope user
```

Then in Claude Code: *"call local_models_status"*, *"summarize build.log with the local
model"*, *"use local-reviewer on my staged changes"* (`.claude/agents/local-reviewer.md`).

Environment variables are listed in `.env.example`.

## Offline fallback

```bash
ollama create qwen3.6-cc -f Modelfile.qwen3.6-cc   # once: 35B with 64K context
bin/claude-local                                    # Claude Code -> Ollama /v1/messages
MODEL=qwen3.5:latest bin/claude-local -p "explain this repo"
```

The launcher points every Claude Code model slot and the MCP tools at the same model so
only one LLM is loaded.

## Development

```bash
uv run pytest                                              # unit tests, fake backend
OLLAMA_AGENT_INTEGRATION=1 uv run pytest tests/integration -s   # real Ollama
uv run scripts/bench.py trio moe big64k                    # VRAM placement + tok/s
```

stdout is the MCP transport — log to stderr only.
