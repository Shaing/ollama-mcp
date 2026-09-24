# ollama-agent — formal specification and verification status

This is the specification of ollama-agent (the MCP server in `src/ollama_agent`), extracted from
what the project promises in `README.md`, the tool docstrings in `server.py`, `.env.example`,
`CLAUDE.md` and the module docstrings, stated precisely, and checked. `formal/README.md`
explains the machinery and how to run it; this file is the contract.

Status legend

| Status | Meaning |
|---|---|
| **PROVED** | Z3 proof over all inputs (`formal/smt/verify.py`, check id in parentheses) |
| **CHECKED** | TLC exhaustively checked every reachable state / behaviour of the model for the listed configurations (`formal/tla`) |
| **BOUNDED** | CrossHair confirmed the property on the real function for every input inside the contract's bounds (`formal/crosshair/contracts.py`) |
| **PBT** | Hypothesis property test on the real code, randomized (`tests/test_properties.py`) |
| **STATIC** | AST / schema check that holds for the checked-in source (`tests/test_spec_static.py`) |
| **TESTED** | covered by an ordinary example test in `tests/` (listed for completeness) |
| **VIOLATED** | the code does not satisfy the property; see the finding number (Fn) below. The check is pinned as an expected counterexample / strict xfail so a fix is noticed. |

Assumptions the whole analysis rests on

- **A1** Every call to Ollama sits inside `async with app.semaphore` (checked: STATIC X3). The TLA+ model takes this as given.
- **A2** Ollama keeps a model resident while the resident set fits the card and evicts otherwise. VRAM figures are the 2026-09-24 measurements (9B 6.6, 4B 3.4, embedder 1.1, 35B-A3B 12.2 GiB on GPU; ~15 GiB usable), not guarantees.
- **A3** "Fits the context" always means the code's own estimate `estimate_tokens` (⌊chars/3.5⌋+1). The real tokenizer can differ by ±20 %; the properties are about the code being consistent with its own estimate.
- **A4** Model output is not modelled. Nothing here says anything about the quality of a summary or a review.

---

## 1. Routing and configuration (`routing.py`, `config.py`)

| ID | Source of the promise | Property | Status |
|---|---|---|---|
| R1 | `routing.py` docstring; README "The profile is fixed per server process" | Within one process only the active profile's models (plus explicit `OLLAMA_AGENT_*_MODEL` overrides) are ever requested from Ollama. | CHECKED `OnlyProfileModels` (trio, big, conc1, nowarmup); PBT `test_routing_resolution` |
| R2 | `PROFILES["big"]` comment "never load a second LLM next to the 35B"; README | In profile `big`: strong = fast = the 35B, the embedder is pinned to the CPU (`num_gpu=0`), warmup loads exactly one LLM, and no behaviour of the process requests a second LLM. | CHECKED `BigSingleLLM` (big); PBT `test_big_profile_is_one_llm_without_overrides` |
| R3 | `.env.example` "Override individual tiers" | `strong_model` / `fast_model` / `embed_model` win over the profile; everything else (`num_gpu`, `keep_alive`) still comes from the profile/settings. | PBT `test_routing_resolution` — see F9: an override can defeat R2 |
| R4 | `warmup_specs` docstring; README "largest first" | Warmup order is strong, then fast (dropped if identical), then the embedder; each warmup spec is a 1-token, no-think request. | CHECKED `WarmupLargestFirst`; PBT (R4, R6) |
| R5 | `tests/test_routing_config.py` | Every profile names all three tiers. | TESTED |
| R6 | `.env.example` "must be the same for all calls or Ollama reloads" | Every `GenSpec`/`EmbedSpec` the process sends carries `settings.num_ctx` and `settings.keep_alive`: there is one `num_ctx` per process, so no call triggers a context-size reload. | STATIC X2 (specs are only built in `routing.py`); PBT (R6, R7) |
| K1–K3 | `load_dotenv` docstring | `KEY=VALUE` lines are loaded; `#` comments and blank lines skipped; an unquoted trailing ` # comment` is stripped; quotes protect `#`; an existing environment variable is never overridden. | PBT `test_dotenv_parses_values_and_never_overrides` |
| K4 | `Settings.from_env` | An unknown profile or a non-integer numeric variable raises `ConfigError` (exit code 2 in `__main__`). | TESTED |

## 2. Concurrency, warmup and VRAM (`app.py`, `warmup.py`, `server.py`, tools)

Model: `formal/tla/OllamaAgent.tla`; configurations `trio` (2 permits, 3 calls, 14 174 states), `big` (8 367), `trio_conc1` (1 permit, 734), `trio_nowarmup` (5 750), `trio_foreign` (stops at the first violation of W5, deliberately).

| ID | Source | Property | Status |
|---|---|---|---|
| S1 | design ("Semaphore" in `app.py`) | Every Ollama request (`run_generation`, `embed_documents`, `embed_query`, `load_model`, `stream_chat`, `embed`) executes inside `async with app.semaphore`. | STATIC X3 |
| S2 | `.env.example` `OLLAMA_AGENT_MAX_CONCURRENCY` | At most `max_concurrency` Ollama conversations are open at once, warmup counting as one; the semaphore is never leaked or over-released. | CHECKED `ConcBound`, `SemInv` |
| S3 | consequence | With `max_concurrency = 1`, no tool can run while warmup holds the permit: the first tool call after startup waits for the whole warmup. | CHECKED `WarmupExcludesCallsWhenConc1` — F7 (undocumented) |
| S4 | — | Every tool call eventually completes; the semaphore cannot deadlock or starve a caller (weak fairness). | CHECKED `AllCallsTerminate` (liveness) |
| W5 | README "all resident … zero reloads" (trio), "12 GiB on GPU" (big) | Within one process, with the measured sizes, Ollama never has to evict a model: no reload. | CHECKED `NoReload`, `VramFits` (trio, big). `trio_foreign` shows an unrelated client's model does cause one, which is what `local_models_status`'s "reload thrash" warning is for. |
| W6 | `warmup` docstring "Failures are logged, never raised", "Runs in the background" | Warmup terminates and returns its permit; the MCP handshake does not wait for it (`lifespan` creates a task). | CHECKED `WarmupTerminates` (liveness); TESTED (server construction) |

## 3. Chunking (`chunking.py`)

| ID | Source | Property | Status |
|---|---|---|---|
| C1 | docstring "nothing is lost at a seam" | The chunks cover lines 1…n in order without gaps: first starts at 1, last ends at n, and each next chunk starts at or before the previous end + 1. | PROVED (C4, for `overlap_lines ≥ 0`); PBT `test_chunk_lines_covers_every_line_in_order` |
| C2 | — | `chunk.text` is exactly lines `start…end` joined with `\n`. | PBT; BOUNDED (CrossHair, not confirmed within the time limit — no counterexample) |
| C3 | docstring "at most max_chars … a single line longer than max_chars becomes its own chunk" | A chunk with two or more lines is shorter than `max_chars`; a chunk longer than `max_chars` is exactly one line. | PROVED (C1, C2: any lengths, up to 8 lines); PBT |
| C4 | — | Termination: the window start strictly increases. | PROVED (C3) |
| C5 | docstring "Consecutive chunks share overlap_lines lines" | Exact rule: `next.start = max(prev.end − overlap_lines + 1, prev.start + 1)`, i.e. `min(overlap_lines, lines − 1)` shared lines. | PBT |
| C5′ | docstring | For `overlap_lines < 0` the function either rejects the argument or still covers every line. | **VIOLATED** — F5. PROVED counterexample (C5); xfail `test_chunk_lines_negative_overlap_is_rejected_or_harmless` |
| C6 | — | When a chunk holds ≤ `overlap_lines` lines the window advances by exactly one line. | PROVED (C6) — F6: input amplification for long lines |
| C7 | — | Empty text gives no chunks. | PBT; TESTED |
| C8 | implied by "chunks" | Every chunk contributes at least one line not in its predecessor. | **VIOLATED** — F6. xfail `test_chunk_lines_every_chunk_adds_a_line` |

## 4. Outputs (`outputs.py`)

| ID | Source | Property | Status |
|---|---|---|---|
| O1 | `.env.example` `MAX_RETURN_CHARS` "clipped to this many chars" | `len(head) ≤ max_chars`. | PROVED (O1); BOUNDED (CrossHair, confirmed over all paths for `max_chars ≤ 12`, `len ≤ 24`); PBT |
| O2 | `clip` docstring | `truncated` is exactly `len(text) > max_chars`. | PROVED (O2); BOUNDED; PBT |
| O3 | docstring "Cuts at a line boundary when possible" | A truncated head keeps ≥ 60 % of `max_chars`, is a prefix of the text, and when shorter than `max_chars` the cut is immediately before a `\n`. | PROVED (O3); BOUNDED; PBT |
| O4 | README "Full outputs are always written … Claude gets the head plus the path" | The rendered reply carries the model/token/time footer, a `[truncated …]` marker when clipped, the output path and every warning. | TESTED `test_store_and_render` |

## 5. Budgets and context fitting (`files.py`, `tools/delegate.py`, `tools/review.py`, `tools/summarize.py`, `app.py`)

| ID | Source | Property | Status |
|---|---|---|---|
| F1 | `read_paths` docstring "Stops adding files once the budget is exhausted" | Σ chars of returned blocks ≤ `budget_chars`. | PBT `test_read_paths_budget_and_accounting` |
| F2 | docstring "notes explain every skip" | `len(blocks) + len(notes) == len(paths)`: every path is either read or explained. | PBT |
| F3 | — | Input order is preserved; missing / binary files are skipped with a note. | PBT |
| D1 | `delegate_task` docstring `max_tokens` | `num_predict` is clamped to `[64, 8192]`. | PROVED (D1) |
| D2 | docstring "~90k chars total" + "NOT for … more than ~32K tokens" | Every input within the 90 000-char budget (+2 000 chars of file headers) is accepted at the default `max_tokens=4096`, `num_ctx=32768`. | PROVED (D2) |
| D3 | same | …and at any `max_tokens` up to the cap. | **VIOLATED** — F4: holds only for `max_tokens ≤ 6383`; at 8192 inputs above ~85.6 k chars are refused. PROVED counterexample (D3) |
| D4 | — | A refused input makes no backend call. | TESTED `test_delegate_rejects_oversized_input` |
| V3 | `review_diff` note "diff further cut to ~N chars to fit the num_ctx-token context" | After the cut, `estimate_tokens(diff) + 400 + 4096 ≤ num_ctx`. | **VIOLATED** by one token — F2. PROVED counterexamples (R1, R1b); xfail `test_review_cut_fits_the_context_exactly`. The −1-token variant is PROVED (R2). |
| V4 | — | The cut length is non-negative for every `num_ctx`. | **VIOLATED** for `num_ctx < 4497` — F3. PROVED counterexample (R3) |
| M1 | `summarize` map-reduce design | The single-pass prompt (≤ 45 000 chars) and each map chunk (12 000 chars) fit the default 32 K context. | PROVED (S1, S2) |
| M2 | `.env.example` "`OLLAMA_AGENT_NUM_CTX` … must be the same for all calls" (implies it is configurable) | `summarize`'s prompts fit whatever `num_ctx` is configured. | **VIOLATED** — F1: `summarize.py` never reads `settings.num_ctx`; its constants need `num_ctx ≥ 15 038` (16 384 fine, 8 192 / 4 096 silently overflow). PROVED counterexample (S3) |
| M3 | `tests/test_summarize.py` | ≤ 45 000 chars → exactly one strong-tier call; above → fast-tier map calls then one strong-tier reduce. | TESTED |
| T1 | `App.clamp_timeout` | The effective timeout is ≥ 5 s and, for `max_timeout_s ≥ 5`, ≤ `max_timeout_s`. | PROVED (T1, T2) |
| T2 | `.env.example` `MAX_TIMEOUT_S` | …for every `max_timeout_s`. | **VIOLATED** for `max_timeout_s < 5` — F8. PROVED counterexample (T3) |

## 6. Review, generation, token estimate (`tools/review.py`, `generate.py`)

| ID | Source | Property | Status |
|---|---|---|---|
| V1 | `_normalize_verdict` docstring | The returned verdict is never `approve` when a finding is `critical` or `major`. | PBT `test_normalize_verdict_never_approves_serious_findings` |
| V2 | — | Only the verdict may change, and then exactly one note is appended. | PBT |
| V5 | `review_diff` docstring; `tests/test_review.py` | Invalid JSON is retried once (plain, deterministic), then falls back to a `comment` verdict carrying the raw text; an empty diff short-circuits with `approve` and no model call. | TESTED |
| G1 | `run_generation` docstring "on timeout return what arrived so far" | Everything streamed before the stop is returned. | PBT `test_run_generation_reports_the_length_cap`; TESTED (timeout) |
| G2 | — | `truncated` ⇔ timeout or `done_reason == "length"`; `reason` names which. | PBT; TESTED |
| G3 | — | Exactly one warning is attached iff truncated. | PBT |
| E1 | `estimate_tokens` docstring "~3.5 chars/token" | `estimate ≥ 1` and `3.5·(estimate−1) ≤ len < 3.5·estimate`. | PBT; BOUNDED (CrossHair, not confirmed within the time limit — no counterexample) |
| E2 | — | Monotone in the text length. | PBT |

## 7. Search index (`index/store.py`, `tools/search.py`)

| ID | Source | Property | Status |
|---|---|---|---|
| I1 | `replace_file` | After `replace_file(path, …)` the store holds exactly the given chunks/vectors for `path` and `files.nchunks` equals their count. | PBT (stateful `IndexStoreMachine`) |
| I2 | `remove_files` | Removing a path drops its chunks and its `files` row together. | PBT (stateful) |
| I3 | — | Invariant: `stats()` = (#files, Σ nchunks); every chunk row belongs to a file row (no orphans). | PBT (stateful invariant) |
| I4 | — | The cached similarity matrix is invalidated by every mutation (no stale search results). | PBT (stateful invariant) |
| I5 | `search` | Returns exactly `min(top_k, n)` rows, best first, whose scores are the true top-k cosine similarities (zero vectors score 0). | PBT (stateful) |
| I6 | `index_codebase` docstring "Incremental: only files whose mtime/hash changed are re-embedded" | Unchanged mtime → no work; changed mtime + same hash → mtime touched, no embedding; changed hash → re-embedded; vanished files removed. | TESTED `test_index_is_incremental` |
| I7 | docstring "Stored in the server's data dir, not in the repo" | Index and outputs live under `data_dir` (`~/.cache/ollama-agent`), never under the indexed root. | TESTED |

## 8. MCP contract and hygiene (`server.py`, `__main__.py`)

| ID | Source | Property | Status |
|---|---|---|---|
| X1 | `CLAUDE.md` "stdout is the MCP transport … never `print()`" | Every `print()` in the package writes to `sys.stderr`. | STATIC |
| X2 | (R6) | `GenSpec`/`EmbedSpec` are constructed only in `routing.py`. | STATIC |
| X3 | (S1) | Every Ollama call is under the semaphore. | STATIC |
| X4 | `CLAUDE.md` "keep the NOT for lines"; README tool table | Exactly six tools; dedented descriptions; `delegate_task`, `summarize`, `search_code` keep their "NOT for" line; all tools closed-world; all read-only except `index_codebase`; the server instructions' "~32K tokens" matches the default `num_ctx`. | STATIC |
| X5 | — | `pyproject.toml`, `__version__` and the MCP `version` agree. | STATIC |

---

## Findings

Ordered by how likely a user is to hit them. None are reachable with the default settings except F6 and F7. Fixing one flips its pinned check (see `formal/README.md`).

| # | Where | What | Effect | Suggested fix |
|---|---|---|---|---|
| F1 | `tools/summarize.py` constants | `SINGLE_PASS_CHARS`, `CHUNK_CHARS`, `REDUCE_PREDICT` assume a 32 K context; `settings.num_ctx` is ignored (unlike `delegate_task` and `review_diff`). Safe only for `num_ctx ≥ 15 038`. | With `OLLAMA_AGENT_NUM_CTX=8192` (a documented knob) a 45 000-char input is sent as one ~13 K-token prompt; Ollama truncates it silently, the summary covers the tail only. | Derive the thresholds from `settings.num_ctx` (e.g. `single_pass = (num_ctx − REDUCE_PREDICT − 500) · 3.5`), or refuse with an error when `num_ctx` is below the minimum. |
| F2 | `tools/review.py` "further cut" | `int((num_ctx − 4096 − 400) · 3.5)` chars re-estimate to `num_ctx + 1` tokens. | Off by one token; harmless in practice (the estimate is heuristic) but the note's claim is not exact. | Subtract one more token: `int((num_ctx − MAX_PREDICT − 401) · 3.5)` — proved to fit (R2). |
| F3 | `tools/review.py` | For `num_ctx < 4497` the cut length is negative and `diff[:keep]` only trims the tail. | Config edge: such a context cannot hold the system prompt plus 4 096 output tokens anyway; the request is doomed. | Return a `comment` result explaining the context is too small (like `delegate_task` does). |
| F4 | `delegate_task` docstring | "~90k chars total" is guaranteed only for `max_tokens ≤ 6 383`; at the 8 192 cap the effective input cap is ~85.6 k chars (the error message already explains it). | Documentation nuance. | Say "~90k chars at the default max_tokens" or compute the budget as `(num_ctx − num_predict − system) · 3.5`. |
| F5 | `chunking.chunk_lines` | A negative `overlap_lines` skips lines silently. | Latent: all callers pass 0, 3 or 8. | `if overlap_lines < 0: raise ValueError`, next to the existing `max_chars` guard. |
| F6 | `chunking.chunk_lines` | When lines are long the window advances one line per chunk (`lines per chunk ≤ overlap_lines`), and a chunk can be a strict sub-window of its predecessor. `summarize`: lines ≥ 1 333 chars → up to 8× the input goes through the map phase; index: lines ≥ 375 chars → up to 3× the embeddings and duplicate hits. | Cost and latency on minified files, JSON-lines logs, long tracebacks; duplicate search results. Reachable with defaults. | Cap the overlap to a fraction of the window (e.g. `min(overlap_lines, (j − i) // 2)`) and skip a chunk whose `end` does not exceed the previous `end`. |
| F7 | `warmup.py` + `max_concurrency = 1` | Warmup holds the only permit for its whole duration; the first tool call waits for it (up to ~20 s in profile `big`). | Design consequence, not documented; the docstring only promises the MCP handshake is not delayed. | Document it in `.env.example`, or warm up per model (release between loads). |
| F8 | `App.clamp_timeout` | `OLLAMA_AGENT_MAX_TIMEOUT_S < 5` is not honoured (the 5 s floor wins). | Cosmetic config edge. | Validate the setting (`≥ 5`) in `Settings.from_env`. |
| F9 | `routing.py` overrides in profile `big` | `OLLAMA_AGENT_FAST_MODEL=qwen3.5:4b` with `OLLAMA_AGENT_PROFILE=big` re-creates the 35B + second-LLM reload thrash that the profile exists to prevent; `local_models_status` does not warn. | Footgun for anyone tuning tiers by hand. | Warn in `local_models_status` when the profile is `big` and `strong != fast`. |
