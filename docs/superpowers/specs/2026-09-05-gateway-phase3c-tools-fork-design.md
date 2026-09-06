# sous — Cross-Session Prompt-Cache Reuse: the Tools-Boundary Fork (Phase 3c)

*2026-09-05. Continues issue #41 after Phase 3a (PR #71, keyed prompt-cache
slots with header forks and a resident budget). Corrects a Phase 3a finding.
Grounded in the runtime-landscape research of 2026-09-05 (MTPLX, mlx-serve,
llama.cpp, oMLX, mlx-dspark read against sous), a tokenizer test against the
pinned Qwen3.8-27B template, a recording of seven real Claude Code subagent
requests across six fresh sessions and two projects, and the gateway's own
daemon log. This is PR 1 of two; the small items it deliberately excludes are
listed under Non-goals.*

## What Phase 3a got wrong

Phase 3a recorded two Claude Code subagent request bodies from different
`claude` processes and found the tools arrays byte-identical (114 tools, same
order) while the system texts differed at a per-session scratchpad path. It
concluded that the path sits "before ~56K tokens of tool schemas", so two
sessions share only ~1K rendered tokens and cross-session forks are
structurally impossible. That reasoning used the *request-body* order —
`system` precedes `tools` in the JSON — and not the rendered order.

Qwen3.8's chat template renders the other way round. With tools present the
system block is (`chat_template.jinja`, lines 57–85 in the
`mlx-community/Qwen3.8-27B-4bit` snapshot):

```
<|im_start|>system\n
# Tools\n\nYou have access to the following functions:\n\n<tools>{…every tool…}</tools>\n\n
If you choose to call a function … <IMPORTANT> … </IMPORTANT>
\n\n<client system text>
<|im_end|>\n
```

The tool block comes first; the client's system text — with its volatile
session path — comes last. Measured with the real tokenizer (120 synthetic
fat tools, two system texts differing at a UUID): 49,225 of 49,653 rendered
tokens are common to both "sessions" (99.1%), and the common prefix ends at
`…</IMPORTANT>\n\n`. An independent adversarial check reproduced this
against the pinned snapshot with 114 tools and 29 different first characters
of the system text: zero misalignments, because the Qwen2 pretokenizer keeps
`\n\n` as its own piece.

sous never saw that reuse because its only fork sits at the *end* of the
system block. `_header_probe` (`src/sous/engine/vlm.py`, `lm.py`) takes the
common prefix of two renders that differ only in the first user turn, which
is the whole system block including the volatile text, so no other session's
render is ever a strict prefix of that slot's ids.

The daemon log shows what that costs. Of 16 locally served gateway turns in
`~/.sous/daemon.err.log`, seven of eight conversations paid a full cold first
turn (`cache=miss`, ~168–170 s for ~57K tokens), every follow-up turn hit its
turn slot (`cache=hit`, 1.3 s), and the one same-session fork reused 57,400 of
57,658 tokens. The first turn of each new session is the entire cold cost, and
nothing inside conversations needs fixing.

## What the recording established

Seven subagent requests were recorded through a throwaway loopback probe that
answered `sous-local` requests itself and kept only tool names, per-tool
schema hashes, and lengths (no bodies, no headers):

| sessions | agent | tools | tool-array hash |
|---|---|---|---|
| four fresh `claude -p` sessions in the sous repo | Explore | 112 | identical |
| one fresh session in a different project (mlx-dspark) | Explore | 112 | identical to the above |
| one session, first spawn | general-purpose | 117 | Explore set + Agent, Artifact, Edit, NotebookEdit, Write |
| same session, second spawn | general-purpose | 105 | the 117 minus twelve deferred tools (Cron*, DesignSync, ListAgents, ListMcpResources*, PushNotification, ReadMcpResource*, RemoteTrigger, ReportFindings, ShareOnboardingGuide) |

Every tool that appears in more than one request has a byte-identical schema
in all of them; 90 of the 112 Explore tools are MCP tools. So the tool array
is stable per *tool set*, across sessions and across projects, and a session
can present more than one tool set (deferred-tool loading changes the
general-purpose agent's list between spawns). Reuse is all-or-nothing on the
whole array — one added or reordered tool means a different tool set and its
own fork.

## Goal

The first locally served turn of a new Claude Code session, for a tool set the
daemon has already seen, starts from a fork taken at the tools boundary
instead of prefilling ~57K tokens cold. Same-session behaviour (the header
fork, the turn slots) is unchanged.

Expected effect on Kyle's M5 Pro: ~168 s → 10–35 s for that turn. The residual
is the client system text (~3–5K tokens ≈ 10–15 s at the measured ~340 tok/s)
plus the user turn plus decode. It applies to every new session in the same
daemon lifetime and idle window, per tool set, per model.

## Non-goals (PR 2 and later)

Deliberately not in this change: the `lcp=` longest-common-prefix counter on
misses, the `speculative_block_size` default and `≤ 5` validation, disjoint
`cache_read_input_tokens` in `usage`, anchors (mlx-dspark-style forks planted
at an observed divergence point), an SSD tier, message-boundary checkpoints
(the log shows nothing inside conversations to bookmark), and any change to
what the model is shown (no rewriting of the client's system text).

## Design

### Boundaries: a list, computed by two probe pairs

The engine's probe returns an ascending list of candidate boundaries instead
of one:

- **Tools boundary** (new): `os.path.commonprefix` of the renders of
  `[system("0"), user("0")]` and `[system("1"), user("0")]` with the turn's
  tools. The renders differ at the first character of the system text, so the
  common prefix is the tool block plus the `\n\n` that precedes the system
  text. Encoded once and memoized in a new `PromptMemo` slot `"tools"`
  (`_MEMO_SLOTS` grows from three to four; reusing `"header"` would make the
  two texts evict each other every cold turn).
- **Header boundary** (existing): common prefix of `[system, user("0")]` and
  `[system, user("1")]` — the end of the whole system block.

Each candidate goes through the existing `fork_point`: it must be a strict
token prefix of this turn's stable render and clear `FORK_MIN_TOKENS` (4096),
else it is 0 and dropped. Equal boundaries are deduplicated (a request with no
system text renders no `\n\n`, so the tools candidate is not a prefix and
drops out on its own; the worker's short prompts never clear the floor). The
result is `[tools, header]` or a subset.

`_header_probe` becomes `_fork_probe` in `VLMEngine` and `LMEngine`, still
returning a lazy callable so a warm turn that cannot use it never pays for
it; its type becomes `Callable[[], list[int]]`. The added cost on a cold turn
is two renders (~1 ms each) and one ~45K-token encode (~25 ms), under the
existing `_tokenize_lock`. The docstrings that assert reuse "in practice means
subagents of one type inside one Claude Code session" are rewritten.

### PrefixCache: fork at every boundary the prefill passes

`PrefixCache.generate(..., fork_at=...)` accepts the callable (or a list of
ints, for tests). `_fork_wanted` is replaced by `_fork_boundaries(owner,
stable_ids, fork_at, reuse) -> list[int]`:

1. Refuse everything when `max_bytes <= 0` (the pre-3a single-slot behaviour
   is untouched).
2. Resolve the callable **regardless of `reuse`**. Today's `if reuse: return 0`
   guard is correct for one boundary and wrong for two: once a tools fork
   exists, every later session's first turn starts warm at it (`reuse > 0`),
   and under the old guard would never publish its own header fork, so that
   session's *second* subagent would reuse ~45K tokens instead of ~57K and
   the measured 8.3 s same-session result would regress. The probe is ~35 ms;
   resolving it on warm turns is free.
3. Keep boundaries `b` with `reuse < b < len(stable_ids)`; drop any for which
   this owner already holds a `fork` slot with exactly `stable_ids[:b]`; sort
   ascending. The scan takes the bookkeeping lock; the probe never runs under
   it.

`_run` then walks the list in order. For each boundary: prefill
`stable_ids[reuse:b]`, set `reuse = b`, price the copy with `slot_bytes(cache)`
(recomputed at each stop), and if it fits the budget call `_make_room`, take
`fork_copy`, and `_publish` a `Slot(..., kind="fork")`. `_make_room` and the
caps eviction take a `protect` set carrying every slot published earlier in
this same turn, so charging the second copy cannot evict the first (today
that survives only because the newest slot has the highest `last_used`). A
copy that cannot fit even on an empty map is skipped, as today. After the
last boundary the turn continues exactly as now (fused decode when every
layer trims, otherwise prefill to the anchor, snapshot, decode, restore).

Lookup is unchanged: `_take` picks the longest slot owned by the current
thread whose ids are a strict prefix of the new stable render. A new session
whose render begins with a resident tools fork's ids starts warm at it; the
same session's next subagent of the same type starts warm at the deeper
header fork. Turn slots are unchanged.

### Ownership

The gateway keeps one `GenerationSession` (`src/sous/gateway/turn.py`,
`_session_for`) across every turn it serves until a stall or an engine change,
so fork slots it publishes are already visible to later Claude Code sessions.
No ownership change is needed; the Phase 3a owner-scoping rules stand.

### Memory

A cold turn on a new tool set now takes two copies (~3.4–3.6 GiB each at
~57K tokens, bf16 KV of the 16 full-attention layers plus the 48 recurrent
states) instead of one, both charged before allocation. With the three tool
sets observed today the steady state is up to six fork slots, ~20 GiB against
the ~27 GiB automatic budget on a 64 GB machine; LRU and `MAX_SLOTS` (16) cap
it and `_evict_pressure` still answers live memory pressure. A cold turn's
peak is the live cache (covered by the budget's one-window reserve) plus two
copies (~7 GiB, each charged to the budget before allocation, evicting LRU
slots if needed). README states the per-tool-set cost so a 48 GB machine can
size `prompt_cache_gb` or disable forks (`prompt_cache_gb = 0` keeps today's
single-slot behaviour).

### Degradation

Nothing here can fail a turn. A probe that raises produces no boundaries and a
warning; a boundary that is not a strict prefix is 0; a copy that fails is
skipped with a warning (existing behaviour, now per boundary); a publish
refused by a reset or a retired owner drops the copy. `PromptMemo` stays
text-keyed, so a stale `tools` entry can never yield wrong ids.

### Observability

No new counters in this PR. The existing `cache=fork reused_tokens=N` log line
in `routes.py:_log_turn` and `stats.forks` distinguish a tools-fork start
(~45–56K reused) from a header-fork start (~57K) by the reused count; the
`lcp=` miss diagnostic is PR 2.

## Invariants preserved

- Never rewind a cache to make a slot; forks are copies taken while prefilling
  past a boundary. Boundaries are visited ascending, so no copy is ever needed
  behind a prefix already prefilled.
- Slots are owned by the building thread and looked up only by it.
- A copy is charged to the budget before it is allocated; a slot is never
  evicted by its own publish, nor by a later publish in the same turn.
- `max_bytes <= 0` means no forks at all.
- The gateway still never rewrites, reorders or logs the client's prompt.

## Testing

Unit tests (`tests/test_promptcache.py`, existing fakes):
- A cold turn with two boundaries publishes two `fork` slots with the expected
  ids, prefilled in three segments, ascending.
- A turn that starts warm at the tools fork (`reuse` = tools boundary) still
  publishes the header fork; a turn that starts warm at the header fork
  publishes nothing.
- A second owner-matching conversation with a different system text starts
  from the tools fork (reused = tools boundary), and a third with the same
  system text starts from the header fork.
- Three tool sets' forks coexist under a budget that holds them; the second
  same-turn copy never evicts the first (`protect`); under a budget that holds
  one, LRU behaves as before.
- Boundaries equal, out of range, or below `FORK_MIN_TOKENS` are dropped; a
  probe that raises yields no forks and a warning; `max_bytes = 0` forks nothing.
- No-doubling: with the budget at exactly one copy, the second boundary is
  skipped, not evicted into.

Engine tests (`tests/test_engine_forkpoint.py`, mlx-free fake tokenizer):
- The tools probe's common prefix ends at the tool block for a template that
  renders tools first; a template with no tools yields no tools boundary; a
  render with no system text yields no tools boundary.
- Both engines pass the same list; `_MEMO_SLOTS` has `tools`; the memo is hit
  on the second cold turn of the same tool set.

Model tests (local, `@pytest.mark.model`, 0.6B and the hybrid 9B):
- A fork taken at the tools boundary and continued reproduces a split prefill
  at the same boundaries bit-for-bit (like-for-like prefill shapes, per the
  Phase 3a ULP finding).
- Two conversations with different system texts and the same tools: the second
  starts from the tools fork and its output matches its cold run token-for-
  token.

Runtime gate (Kyle's machine, real daemon, `sous claude`):
- Session A (sous repo) runs an Explore subagent: turn 1 `cache=miss` publishes
  two forks. Session B (another project) runs an Explore subagent: its turn 1
  logs `cache=fork` with 45–56K reused tokens and completes in 10–35 s. A
  second Explore in session B reuses ≥ 57K (header fork). A general-purpose
  subagent produces a third fork set; `server_status` shows the slot count and
  resident bytes within budget; no evictions of same-turn forks.
- The four CI jobs green; one real delegated task after the change (the
  mlx-thread rule).

## Documentation

README (prompt cache section: forks per tool set, cross-session and
cross-project reuse, memory expectations, the idle-unload window), CLAUDE.md
gotcha (replace the "cross-session impossible" sentence), both engines'
probe docstrings, `_fork_boundaries`'s docstring (why warm turns resolve the
probe), and a Phase 3c note on #41 correcting the Phase 3a comment.

## Risks and open questions

- **Tool-set churn.** Deferred tools loading mid-conversation (Claude Code's
  tool search) changes the tools array for that conversation and misses every
  slot from token 0; the new fork does not help or hurt that case. A Claude
  Code upgrade that edits any tool description invalidates the tools forks
  once. Both are visible in the log as misses with no reuse; PR 2's `lcp=`
  makes them diagnosable.
- **Idle unload.** `idle_unload_minutes` (default 30) drops every slot with the
  weights, so "every new session" means every new session inside the idle
  window. Acceptable; noted in README. An SSD tier would remove it (later).
- **Two copies per cold turn.** ~7 GiB more resident after the first turn of a
  new tool set, taken from the budget (LRU evicts older slots to make room);
  a 48 GB machine should set `prompt_cache_gb` lower or to 0.
- **Prefill split points.** One more split in a cold prefill, one more
  bf16-ULP divergence point between cold and warm runs (Phase 3a finding).
  Not a correctness issue for a chat model; bit-exact tests compare like
  shapes.
- **Probe cost on the tokenize lock.** ~35 ms per cold turn on the lock shared
  with `count_tokens`. Noise against a multi-second turn.
