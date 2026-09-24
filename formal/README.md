# formal/ — the specification of ollama-agent, and the machinery that checks it

`SPEC.md` is the specification: every property the README, the tool docstrings, `.env.example`
and `CLAUDE.md` promise, written down precisely, with how it is verified and whether it holds.
The rest of this directory (plus two test modules) is the machinery.

| Layer | Where | What it establishes |
|---|---|---|
| SMT proofs (Z3) | `smt/verify.py` | Arithmetic invariants over **all** inputs: timeout clamping, token budgets, the context-fit cuts in `review_diff` and `delegate_task`, `summarize`'s thresholds, `clip`, the chunker's window rule. Constants are imported from the package. |
| Model checking (TLA+/TLC) | `tla/OllamaAgent.tla`, `tla/*.cfg`, `tla/run.sh` | One server process as a state machine: fixed profile, the semaphore, warmup order, tool calls, Ollama's VRAM. Exhaustive for each configuration, safety and liveness. One configuration is meant to fail (a foreign model evicts ours) to show the invariant is not vacuous. |
| Symbolic execution (CrossHair) | `crosshair/contracts.py` | The **real** `clip`, `estimate_tokens` and `chunk_lines` run on symbolic inputs against pre/postconditions; bounded by the preconditions. |
| Property-based + static tests | `tests/test_properties.py`, `tests/test_spec_static.py` | Bind the spec to the implementation: Hypothesis properties (incl. a model-based state machine for the SQLite index) and AST checks (every Ollama call holds the semaphore, specs are only built in `routing.py`, no stdout prints, the MCP tool contract). Known violations are `xfail(strict=True)`. |

## Running

```bash
uv sync --all-groups                      # adds hypothesis (dev) and z3-solver, crosshair-tool (formal)
formal/run.sh                             # all four layers; exit 1 if anything disagrees with SPEC.md

uv run python formal/smt/verify.py        # ~2 s
formal/tla/run.sh                         # ~1 min; needs Java 11+, fetches tla2tools.jar into ~/.cache once
uv run crosshair check formal/crosshair/contracts.py --analysis_kind=PEP316 --per_condition_timeout=30
uv run pytest tests/test_properties.py tests/test_spec_static.py
```

## Reading the results

- `PROVED` (Z3) holds for every input, not just tested ones. `COUNTEREXAMPLE` prints one violating input.
- TLC `pass` means every reachable state of that configuration satisfies the invariants and every
  behaviour satisfies the temporal properties; the number of distinct states is printed.
- CrossHair `Confirmed over all paths` is a proof within the precondition bounds; `Not confirmed`
  means no counterexample was found before the per-condition timeout.
- Hypothesis tests are randomized; failures are minimized and replayed from `.hypothesis/`.

## Keeping it honest

Some checks are *expected* to fail: the `COUNTEREXAMPLE` expectations in `verify.py`, the
`violation NoReload` line in `tla/OllamaAgent_trio_foreign.cfg`, and any strict xfails in
`tests/test_properties.py` (none at the moment: the four findings that had them are fixed). Each
maps to a finding in `SPEC.md`. When the code is fixed, the
check flips and the run fails until the expectation is removed and the SPEC.md row is updated,
so the spec cannot silently drift from the code in either direction.

What is *not* verified: the language models' output, the Ollama server itself, and the file
system. Every property about "fitting the context" is about the code's own token estimate
(`estimate_tokens`, ~3.5 chars/token), which is a heuristic; the real tokenizer may differ by
±20%. The TLA+ model's VRAM figures are measurements, not guarantees.
