---------------------------- MODULE OllamaAgent ----------------------------
(***************************************************************************)
(* One ollama-agent server process (src/ollama_agent), abstracted:         *)
(*                                                                         *)
(*  - the profile is fixed for the life of the process (routing.py);       *)
(*  - warmup() takes one permit of app.semaphore, loads the profile's chat *)
(*    models largest-first, then the embedder, and releases (warmup.py);   *)
(*  - every tool takes a permit before talking to Ollama and releases it   *)
(*    when done (tools/*.py; tests/test_spec_static.py::test_X3 checks the *)
(*    code really does this, which is the assumption this model rests on); *)
(*  - Ollama keeps a model resident while it fits in VRAM and evicts       *)
(*    otherwise: a "reload", the 15-20 s cost README.md talks about.       *)
(*                                                                         *)
(* Sizes are measured 0.1 GiB units from PROGRESS.md (9B 6.6, 4B 3.4,      *)
(* embedder 1.1, 35B-A3B 12.2 on a 16 GB card with ~15 GB usable).         *)
(* Checked with TLC via formal/tla/run.sh; properties are listed in        *)
(* formal/SPEC.md (W*, S* rows).                                           *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Profile,    \* "trio" | "big"           OLLAMA_AGENT_PROFILE
    MaxConc,    \* >= 1                     OLLAMA_AGENT_MAX_CONCURRENCY
    Warmup,     \* BOOLEAN                  OLLAMA_AGENT_WARMUP
    NumCalls,   \* tool calls the Claude Code session will make
    Foreign     \* BOOLEAN: another Ollama client may load an unrelated model

ASSUME Profile \in {"trio", "big"} /\ MaxConc \in Nat \ {0} /\ NumCalls \in Nat

Tools == {"delegate", "review", "summarize", "search", "status"}
Calls == 1..NumCalls

\* --- routing.py -----------------------------------------------------------
Strong == IF Profile = "trio" THEN "qwen9b" ELSE "qwen35b"
Fast   == IF Profile = "trio" THEN "qwen4b" ELSE "qwen35b"
Embed  == "embed"
LLMs   == {"qwen9b", "qwen4b", "qwen35b"}
ProfileModels == {Strong, Fast, Embed}

\* warmup_specs(): [strong, fast] with duplicates collapsed, then the embedder
WarmSeq == IF Strong = Fast THEN <<Strong, Embed>> ELSE <<Strong, Fast, Embed>>

\* which models each tool asks Ollama for
ToolModels(t) == CASE t = "delegate"  -> {Strong, Fast}   \* model_tier is the caller's choice
                   [] t = "review"    -> {Strong}
                   [] t = "summarize" -> {Fast, Strong}    \* map on fast, reduce on strong
                   [] t = "search"    -> {Embed}
                   [] t = "status"    -> {}

\* --- Ollama and the GPU -----------------------------------------------------
Capacity == 150
Size(m) == CASE m = "qwen9b"  -> 66
             [] m = "qwen4b"  -> 34
             [] m = "qwen35b" -> 122
             [] m = "embed"   -> 11
             [] m = "foreign" -> 50
\* profile 'big' pins the embedder to the CPU (embed_num_gpu = 0)
GpuSize(m) == IF m = "embed" /\ Profile = "big" THEN 0 ELSE Size(m)

RECURSIVE Used(_)
Used(S) == IF S = {} THEN 0
           ELSE LET m == CHOOSE x \in S : TRUE IN GpuSize(m) + Used(S \ {m})

VARIABLES
    sem,        \* free permits of app.semaphore
    warm,       \* "off" | "idle" | "loading" | "done"
    warmIdx,    \* next position in WarmSeq
    warmReq,    \* models warmup has requested so far, in order
    call,       \* Calls -> [tool, phase];  phase: "waiting" | "running" | "done"
    loaded,     \* models resident on the GPU
    requested,  \* every model this process has asked Ollama for
    reloaded    \* TRUE once Ollama had to evict something to satisfy a request
vars == <<sem, warm, warmIdx, warmReq, call, loaded, requested, reloaded>>

Running == {c \in Calls : call[c].phase = "running"}
Holding == Cardinality(Running) + (IF warm = "loading" THEN 1 ELSE 0)

\* Ollama loads m; when it does not fit, resident models are evicted (a reload)
LoadEffect(m) ==
    IF m \in loaded THEN [loaded |-> loaded, reload |-> FALSE]
    ELSE IF Used(loaded) + GpuSize(m) <= Capacity
         THEN [loaded |-> loaded \cup {m}, reload |-> FALSE]
         ELSE [loaded |-> {m}, reload |-> TRUE]

Request(m) ==
    LET e == LoadEffect(m) IN
    /\ loaded' = e.loaded
    /\ reloaded' = (reloaded \/ e.reload)
    /\ requested' = requested \cup {m}

Init ==
    /\ sem = MaxConc
    /\ warm = IF Warmup THEN "idle" ELSE "off"
    /\ warmIdx = 1
    /\ warmReq = <<>>
    /\ call = [c \in Calls |-> [tool |-> "none", phase |-> "waiting"]]
    /\ loaded = {}
    /\ requested = {}
    /\ reloaded = FALSE

\* --- warmup.py: `async with app.semaphore: for spec in warmup_specs(): load` ----
WarmupAcquire ==
    /\ warm = "idle" /\ sem > 0
    /\ warm' = "loading" /\ sem' = sem - 1
    /\ UNCHANGED <<warmIdx, warmReq, call, loaded, requested, reloaded>>

WarmupLoad ==
    /\ warm = "loading" /\ warmIdx <= Len(WarmSeq)
    /\ Request(WarmSeq[warmIdx])
    /\ warmReq' = Append(warmReq, WarmSeq[warmIdx])
    /\ warmIdx' = warmIdx + 1
    /\ UNCHANGED <<sem, warm, call>>

WarmupRelease ==
    /\ warm = "loading" /\ warmIdx > Len(WarmSeq)
    /\ warm' = "done" /\ sem' = sem + 1
    /\ UNCHANGED <<warmIdx, warmReq, call, loaded, requested, reloaded>>

\* --- tools/*.py: pick a tool, `async with app.semaphore:` talk to Ollama, return ----
CallStart(c) ==
    /\ call[c].phase = "waiting" /\ sem > 0
    /\ \E t \in Tools : call' = [call EXCEPT ![c] = [tool |-> t, phase |-> "running"]]
    /\ sem' = sem - 1
    /\ UNCHANGED <<warm, warmIdx, warmReq, loaded, requested, reloaded>>

CallRequest(c) ==
    /\ call[c].phase = "running"
    /\ \E m \in ToolModels(call[c].tool) : Request(m)
    /\ UNCHANGED <<sem, warm, warmIdx, warmReq, call>>

CallDone(c) ==
    /\ call[c].phase = "running"
    /\ call' = [call EXCEPT ![c].phase = "done"]
    /\ sem' = sem + 1
    /\ UNCHANGED <<warm, warmIdx, warmReq, loaded, requested, reloaded>>

\* --- the environment: `ollama run llama3`, another agent, ... -------------------
ForeignLoad ==
    /\ Foreign /\ "foreign" \notin loaded
    /\ LET e == LoadEffect("foreign") IN
         loaded' = e.loaded /\ reloaded' = (reloaded \/ e.reload)
    /\ UNCHANGED <<sem, warm, warmIdx, warmReq, call, requested>>

AllDone == warm \in {"done", "off"} /\ \A c \in Calls : call[c].phase = "done"

Next ==
    \/ WarmupAcquire \/ WarmupLoad \/ WarmupRelease
    \/ \E c \in Calls : CallStart(c) \/ CallRequest(c) \/ CallDone(c)
    \/ ForeignLoad
    \/ (AllDone /\ UNCHANGED vars)

Fairness ==
    /\ WF_vars(WarmupAcquire) /\ WF_vars(WarmupLoad) /\ WF_vars(WarmupRelease)
    /\ \A c \in Calls : WF_vars(CallStart(c)) /\ WF_vars(CallDone(c))

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- safety ----------------------------------------------------------------------
TypeOK ==
    /\ sem \in 0..MaxConc
    /\ warm \in {"off", "idle", "loading", "done"}
    /\ warmIdx \in 1..(Len(WarmSeq) + 1)
    /\ loaded \subseteq LLMs \cup {"embed", "foreign"}
    /\ \A c \in Calls : call[c].tool \in Tools \cup {"none"}
                     /\ call[c].phase \in {"waiting", "running", "done"}

\* S1  the semaphore is never leaked or over-released
SemInv == sem = MaxConc - Holding
\* S2  at most max_concurrency Ollama conversations at once (warmup counts as one)
ConcBound == Holding <= MaxConc
\* W1  "the profile is fixed per process": only its models are ever requested
OnlyProfileModels == requested \subseteq ProfileModels
\* W2  profile 'big' never loads a second LLM next to the 35B
BigSingleLLM == Profile = "big" => Cardinality(requested \cap LLMs) <= 1
\* W3  warmup requests models in warmup_specs() order: largest first, embedder last
WarmupLargestFirst == warmReq = SubSeq(WarmSeq, 1, Len(warmReq))
\* W4  what is resident always fits the card
VramFits == Used(loaded) <= Capacity
\* W5  zero reloads within one process (README: "all resident", "zero reloads")
NoReload == ~reloaded
\* S3  with max_concurrency = 1 no tool can run while warmup holds the permit
WarmupExcludesCallsWhenConc1 == MaxConc = 1 => ~(warm = "loading" /\ Running # {})

\* --- liveness ----------------------------------------------------------------------
\* W6  warmup finishes and gives its permit back
WarmupTerminates == Warmup => <>(warm = "done")
\* S4  every tool call completes (no deadlock or starvation on the semaphore)
AllCallsTerminate == <>AllDone
=============================================================================
