#!/usr/bin/env python3
"""SMT proofs of the arithmetic invariants in ollama-agent.  Property IDs refer to formal/SPEC.md.

Run:  uv run python formal/smt/verify.py

Every check states a property over ALL inputs (unbounded Python ints unless the title says
otherwise) and asks Z3 for a counterexample to it:

  PROVED          the negation is unsatisfiable, so the property holds for every input
  COUNTEREXAMPLE  the property is false; one violating assignment is printed

Checks whose *expected* result is COUNTEREXAMPLE document known gaps between the code and
the spec it claims.  The script exits 1 only when a result differs from its expectation, so
fixing one of those gaps shows up here (flip the expectation, and update SPEC.md), the same
convention as the strict xfail tests in tests/test_properties.py.

The constants (MAX_PREDICT, SINGLE_PASS_CHARS, prompt lengths, defaults) are imported from the
package, so the proofs follow the code.
"""
from __future__ import annotations

import sys

from z3 import (
    And,
    Bool,
    BoolVal,
    If,
    Implies,
    Int,
    Ints,
    Not,
    Optimize,
    Solver,
    Sum,
    sat,
    unsat,
)

from ollama_agent.config import Settings
from ollama_agent.tools import delegate, review, summarize

PROVED, CE = "PROVED", "COUNTEREXAMPLE"
_results: list[tuple[str, str, bool]] = []


# --- modelling helpers ------------------------------------------------------------------
def zmax(a, b):
    return If(a >= b, a, b)


def zmin(a, b):
    return If(a <= b, a, b)


def est(chars):
    """generate.estimate_tokens(): int(len / 3.5) + 1  ==  floor(2*len / 7) + 1  for len >= 0."""
    return (2 * chars) / 7 + 1  # Z3 Int division floors for a positive divisor


def trunc_x35(x):
    """int(x * 3.5) with Python's truncation toward zero, for any integer x."""
    return If(x >= 0, (7 * x) / 2, -((7 * -x) / 2))


def check(cid: str, expect: str, title: str, assumptions, prop, *, show=(), note: str = "") -> None:
    s = Solver()
    s.add(*assumptions)
    s.add(Not(prop))
    r = s.check()
    got = PROVED if r == unsat else CE if r == sat else "UNKNOWN"
    ok = got == expect
    line = f"{'ok' if ok else 'XX'}  {cid:<3} {got:<14} {title}"
    if got == CE:
        m = s.model()
        vals = ", ".join(f"{v}={m.eval(v, model_completion=True)}" for v in show)
        line += f"\n                        e.g. {vals}"
    if note:
        line += f"\n                        {note}"
    print(line)
    _results.append((cid, got, ok))


def info(text: str) -> None:
    print(f"    {'':<3} {'info':<14} {text}")


S = Settings()  # documented defaults: num_ctx=32768, max_input_chars=90000, max_timeout_s=600

# --- T: App.clamp_timeout -----------------------------------------------------------------
# t = float(timeout_s) if timeout_s else float(default);  max(5.0, min(t, max_timeout_s))
t, mx, d = Ints("timeout_s max_timeout_s default")
t0 = If(t != 0, t, d)
clamp = zmax(5, zmin(t0, mx))
check("T1", PROVED, "clamp_timeout never returns less than 5 s", [d > 0], clamp >= 5)
check("T2", PROVED, "clamp_timeout never exceeds max_timeout_s when max_timeout_s >= 5",
      [d > 0, mx >= 5], clamp <= mx)
check("T3", CE, "clamp_timeout never exceeds max_timeout_s for ANY max_timeout_s >= 1",
      [d > 0, mx >= 1], clamp <= mx, show=(t, mx),
      note="the 5 s floor wins over a smaller OLLAMA_AGENT_MAX_TIMEOUT_S (config edge, not reachable with defaults)")

# --- D: tools/delegate.py -----------------------------------------------------------------
# num_predict = max(64, min(int(max_tokens or 4096), MAX_PREDICT))
# reject when estimate_tokens(user) + estimate_tokens(SYSTEM_PROMPT) + num_predict > num_ctx
mt, U, n = Ints("max_tokens user_chars num_ctx")
np_ = zmax(64, zmin(If(mt != 0, mt, 4096), delegate.MAX_PREDICT))
check("D1", PROVED, f"delegate_task clamps num_predict to [64, {delegate.MAX_PREDICT}]", [],
      And(np_ >= 64, np_ <= delegate.MAX_PREDICT))

HEADERS = 2000  # '## Context files' + '### FILE: <path>' fences for a handful of files
sys_est = est(len(delegate.SYSTEM_PROMPT))
accepted = est(U) + sys_est + np_ <= n
budget = [U >= 0, U <= S.max_input_chars + HEADERS, n == S.num_ctx]
check("D2", PROVED, f"delegate_task accepts every input inside the {S.max_input_chars}-char budget "
      f"(+{HEADERS} header chars) at the default max_tokens=4096, num_ctx={S.num_ctx}",
      budget + [mt == 4096], accepted)
check("D3", CE, "... and also at the MAX_PREDICT cap max_tokens=8192",
      budget + [mt == 8192], accepted, show=(U,),
      note="the docstring's '~90k chars total' holds only up to the max_tokens reported below")
o = Optimize()
o.add(mt >= 64, mt <= delegate.MAX_PREDICT, est(S.max_input_chars + HEADERS) + sys_est + mt <= S.num_ctx)
o.maximize(mt)
o.check()
info(f"D: the full {S.max_input_chars}-char budget is always accepted iff max_tokens <= {o.model()[mt]}")

# --- R: tools/review.py -------------------------------------------------------------------
# est = estimate_tokens(diff) + 400; if est + MAX_PREDICT > num_ctx:
#     keep = int((num_ctx - MAX_PREDICT - 400) * 3.5); diff = diff[:keep]
D = Int("diff_chars")
MAXP = review.MAX_PREDICT
need_cut = est(D) + 400 + MAXP > n
keep = trunc_x35(n - MAXP - 401)  # the code's formula (F2 fixed: one token of margin)
kept = If(keep >= 0, zmin(D, keep), zmax(0, D + keep))  # Python slice semantics, incl. a negative stop
D_sent = If(need_cut, kept, D)
fits = est(D_sent) + 400 + MAXP <= n
check("R1", PROVED, "review_diff: after 'diff further cut to fit the context' the estimate fits num_ctx "
      "(any diff, any num_ctx >= MAX_PREDICT + 401)", [D >= 0, n >= MAXP + 401], fits)
check("R1b", PROVED, "review_diff: same, at the .env example OLLAMA_AGENT_NUM_CTX=16384 with a 90000-char diff",
      [D == S.max_input_chars, n == 16384], fits)
check("R3", CE, "review_diff: the cut length is non-negative for every num_ctx >= 1",
      [n >= 1], keep >= 0, show=(n,),
      note=f"num_ctx < {MAXP + 401}: even an empty diff cannot fit; the code slices with a negative "
           "index instead of refusing (config edge)")

# --- S: tools/summarize.py ----------------------------------------------------------------
# The thresholds are module constants; settings.num_ctx is never consulted.
SP, CH = summarize.SINGLE_PASS_CHARS, summarize.CHUNK_CHARS
RED, MAP = summarize.REDUCE_PREDICT, summarize.MAP_PREDICT
hdr = 200  # 'Summarize the following.' / 'Section: <path> lines a-b' / question / '### path' headings
single_fits = est(SP + len(summarize.REDUCE_SYSTEM) + hdr) + RED <= n
map_fits = est(CH + len(summarize.MAP_SYSTEM) + hdr) + MAP <= n
check("S1", PROVED, f"summarize: the single-pass prompt (<= {SP} chars) fits the default num_ctx={S.num_ctx}",
      [n == S.num_ctx], single_fits)
check("S2", PROVED, f"summarize: a map chunk ({CH} chars) + {MAP} output tokens fits the default num_ctx",
      [n == S.num_ctx], map_fits)
# budgets(num_ctx, question=""): min(constant, int((num_ctx - predict - est(system) - est("") - HEADER_TOKENS - 1) * 3.5))
def budget(system_len: int, predict: int):
    return trunc_x35(n - predict - est(system_len) - est(0) - summarize.HEADER_TOKENS - 1)


single_n = zmin(SP, budget(len(summarize.REDUCE_SYSTEM), RED))
chunk_n = zmin(zmin(CH, budget(len(summarize.MAP_SYSTEM), MAP)), single_n)  # a chunk never exceeds the reduce input
single_fits_n = est(single_n + len(summarize.REDUCE_SYSTEM) + hdr) + RED <= n
map_fits_n = est(chunk_n + len(summarize.MAP_SYSTEM) + hdr) + MAP <= n
check("S3", PROVED, "summarize: with budgets() the single-pass prompt and every map chunk fit ANY num_ctx the tool "
      f"accepts (chunk budget >= MIN_CHUNK_CHARS={summarize.MIN_CHUNK_CHARS}); F1 fixed",
      [n >= 1, chunk_n >= summarize.MIN_CHUNK_CHARS], And(single_fits_n, map_fits_n))
check("S4", PROVED, "summarize: at the default num_ctx budgets() returns the module constants unchanged",
      [n == S.num_ctx], And(single_n == SP, chunk_n == CH))
check("S5", PROVED, "summarize: whenever it accepts num_ctx, the fold loop's chunk fits its reduce input (convergence)",
      [n >= 1, chunk_n >= summarize.MIN_CHUNK_CHARS], chunk_n <= single_n)
o = Optimize()
o.add(n >= 1, chunk_n >= summarize.MIN_CHUNK_CHARS)
o.minimize(n)
o.check()
info(f"S: summarize accepts num_ctx >= {o.model()[n]} and refuses below (was: silent overflow below 15038)")

# --- O: outputs.clip ----------------------------------------------------------------------
# if len <= max: unchanged. else head = text[:max]; nl = head.rfind('\n'); if nl > max*0.6: head = head[:nl]
L, M, nl = Ints("text_len max_chars last_newline")
cut = Bool("cut_at_newline")
trunc = L > M
h = If(trunc, If(cut, nl, M), L)
clip_hyp = [L >= 0, M >= 1,
            Implies(trunc, And(nl >= -1, nl <= M - 1)),  # rfind over a max_chars-long prefix
            Implies(cut, 5 * nl >= 3 * M)]  # the float test `nl > max*0.6` implies this exactly (nl is an int)
check("O1", PROVED, "clip: the returned head never exceeds max_chars", clip_hyp, h <= M)
check("O2", PROVED, "clip: the truncated flag is exactly len(text) > max_chars", clip_hyp, (h < L) == trunc)
check("O3", PROVED, "clip: a truncated head keeps at least 60% of max_chars", clip_hyp,
      Implies(trunc, 5 * h >= 3 * M))

# --- C: chunking.chunk_lines --------------------------------------------------------------
# inner loop: take line j while size + len + 1 <= max_chars, or unconditionally when j == i
K = 8
mc = Int("max_chars")
lens = [Int(f"len_{k}") for k in range(K)]
acc = [BoolVal(True)]
size = [lens[0] + 1]
for k in range(1, K):
    a = And(acc[-1], size[-1] + lens[k] + 1 <= mc)
    acc.append(a)
    size.append(If(a, size[-1] + lens[k] + 1, size[-1]))
n_lines = Sum([If(a, 1, 0) for a in acc])
text_len = size[-1] - 1  # '\n'.join adds n_lines-1 separators
c_hyp = [mc >= 1] + [ln >= 0 for ln in lens]
check("C1", PROVED, f"chunk_lines: a chunk holding >= 2 lines is shorter than max_chars (any lengths, <= {K} lines)",
      c_hyp, Implies(n_lines >= 2, text_len <= mc - 1))
check("C2", PROVED, "chunk_lines: a chunk longer than max_chars is exactly one (long) line",
      c_hyp, Implies(text_len > mc, n_lines == 1))

# outer loop: i = max(j - min(overlap_lines, (j - i) // 2), i + 1); overlap_lines < 0 raises ValueError (F5 fixed)
i, j, ov = Ints("i j overlap_lines")
c_lines = j - i
nxt = zmax(j - zmin(ov, c_lines / 2), i + 1)
c_hyp2 = [i >= 0, j > i, ov >= 0]
check("C3", PROVED, "chunk_lines: the window always advances (termination)", c_hyp2, nxt > i)
check("C4", PROVED, "chunk_lines: no line is skipped between consecutive chunks", c_hyp2, nxt <= j)
check("C5", PROVED, "chunk_lines: consecutive chunks share at most min(overlap_lines, half the chunk) lines",
      c_hyp2, And(j - nxt <= ov, 2 * (j - nxt) <= c_lines))
check("C6", PROVED, "chunk_lines: every chunk advances the window by at least half its lines, so the chunks' "
      "total text is at most ~2x the input (F6 fixed; was 1 line per chunk for long lines)",
      c_hyp2, 2 * (nxt - i) >= c_lines)

# --- summary --------------------------------------------------------------------------------
bad = [cid for cid, _, ok in _results if not ok]
proved = sum(1 for _, got, _ in _results if got == PROVED)
ces = sum(1 for _, got, _ in _results if got == CE)
print(f"\n{len(_results)} checks: {proved} proved, {ces} counterexamples "
      f"({'all as expected' if not bad else 'UNEXPECTED: ' + ', '.join(bad)})")
sys.exit(1 if bad else 0)
