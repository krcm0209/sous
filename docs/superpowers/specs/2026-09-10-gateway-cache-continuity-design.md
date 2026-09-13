# sous — Gateway Cache Continuity: in-place system messages, retained turn slots, preload and hold

*2026-09-10. Continues issue #41 after Phase 3c (PR #72) and its two follow-ups
(PR #74 `lcp=` diagnostics, PR #75 kernel-level pressure valve). Grounded in a
forensic reconstruction of the first real `sous claude` session Kyle ran outside
our gates (a 13m14s general-purpose subagent, seven gateway turns, reconciled to
the token against Claude Code's own transcript and cost state), a byte-level
reading of the Claude Code 2.1.268 binary Kyle runs, and a tokenizer experiment
against the pinned Qwen3.8-27B template. Two PRs: **A** (in-place system
messages + retained turn slots, both in the request→render→cache path) and **B**
(preload and hold, launcher + engine manager). A third item, an SSD tier for
cache slots, is scoped here only as a spike; it gets its own spec if the spike
passes. The companion spec `2026-09-10-gateway-observability-design.md` ships
first: its PR 1 is the measurement every claim below is verified against.*

## What the 13-minute session established

The seven `sous-local` turns in `~/.sous/daemon.err.log` sum to 794.3 s, which
is Claude Code's reported 13m14s to within 0.1 s. Prefill was ~70% of it,
decode ~28%; 171,258 tokens were prefilled to produce 3,304. About 61% of the
session (~483 s) was prefill a perfect cache would not have paid, from three
distinct causes:

| turn | outcome | seconds | cause |
|---|---|---|---|
| T1 | miss, `lcp=0` | 201.0 | model reload + 63.7K cold: the idle unload had dropped every slot, tools fork included |
| T2 | fork @62856 | 9.6 | an auxiliary call (see below), served from the header fork |
| T3 | miss, `lcp=62851` | 202.8 | the system block grew by exactly 593 tokens; header fork and turn slot both invalid |
| T4 | hit | 2.1 | auxiliary call, took T3's turn slot |
| T5 | fork @63449 | 87.5 | should have extended T3's slot; T4 had consumed it — 19,198 tokens re-prefilled |
| T6 | hit | 2.5 | auxiliary call, took T5's turn slot |
| T7 | fork @63449 | 288.8 | same as T5 (20,879 re-prefilled) + the 2,887-token summary at ~14.7 tok/s |

**Cause 1 — idle unload.** `EngineManager.unload_if_idle` fires after
`[model].idle_unload_minutes` (default 30) and `VLMEngine.unload` begins with
`reset_prompt_cache(None)`, so every slot goes with the weights. The engine had
been idle for hours; T1 paid the load and a from-zero prefill (~185 s of it
avoidable). `lcp=0` is the empty-candidates early-out, not a measured zero:
every Claude Code render shares the 18-token `<|im_start|>system\n# Tools…`
prefix, and the smallest real `lcp` ever logged is 32.

**Cause 2 — a mid-conversation `role: "system"` message.** The 593 tokens are
Claude Code's `agent_listing_delta` attachment ("Available agent types for the
Agent tool: …" plus the concurrency note). Read from the binary: Claude Code's
message assembler routes text attachments into **a new message with
`role: "system"` inserted into the `messages` array** after the preceding user
message (its "mid-conversation system" feature, enabled for every first-party
model id not on an exclusion list — `sous-local` included). It is not a change
to the top-level `system` field; both `prompt_snapshot` records in the
transcript are byte-identical. On Anthropic's API this costs almost nothing:
tools, system and every message before the insertion stay cached. On sous it
cost everything, because `convert._system_text` hoists every inline system
message into the template's single system slot — after `# Tools`, before the
whole conversation — since Qwen's template raises on a system message anywhere
but index 0. The hoist is our workaround, and the hoist is what moved the
header boundary from 62,856 to 63,449 and broke every prefix behind it. Claude
Code's own fallback for models without the feature renders the same text
**into the preceding user message as a `<system-reminder>` block**. The
subagent got its listing one turn late because its bootstrap attachment scan
is narrower than the per-turn one; every general-purpose subagent that runs
more than one turn pays this. `mcp_instructions_delta` (a server connecting
late) and `deferred_tools_delta` travel the same way.

**Cause 3 — background-agent progress summaries.** T2, T4 and T6 are absent
from the subagent transcript but billed to it (`cost-state.modelUsage`
reconciles to the token). They are Claude Code's `agent_summary` calls: every
30 s while a *background* agent runs, it forks the agent's conversation,
appends a fixed "Describe your most recent action in 3-5 words…" user turn,
and asks for a label. The prompt inherits the agent's model, so it lands on
`sous-local`; no environment variable reroutes it. T4/T6 are +138/+155 tokens
over the real conversation and answered in 8/7 tokens; T2's 103-token
`tool_use` reply is the local model ignoring "Do not use tools". The damage is
mechanical: `PrefixCache._take` **removes** the turn slot it serves (the cache
is about to be mutated in place), the summary call extends it down a branch the
real conversation never takes and republishes that, and the next real turn —
a strict +16,533 / +1,681 extension of the previous real one — finds only the
header fork. ~108 s here (14%). The summary calls also hold `TurnRunner._lock`
while the real request waits, so the client saw 212.4 s for T3 against a
logged 202.8.

**Rejected alternative for cause 2: move the header fork boundary to the end
of the system content.** Validated on the pinned tokenizer (nine system-text
endings, zero boundary shrink; Qwen2's vocabulary has no token mixing a newline
with a non-whitespace byte, and the template's `|trim` guarantees the content
never ends in whitespace). It rescues the header fork (T3 → ~3.3K tokens
prefilled) but not the turn slot, publishes one more ~4 GiB fork per append,
and regresses system-text-first templates (Qwen3-0.6B's header fork falls below
the 4096 floor). Once inline system messages stay in the body there is nothing
left for it to fix; mid-text edits to the top-level system prompt (Claude Code's
`currentDate`, `gitStatus` sections) are unfixable by any boundary.

## Goal

A subagent conversation on the local model stays append-only from its first
turn to its last, so every real turn extends its own slot: a Claude Code
attachment arriving mid-conversation costs its own tokens, a progress-summary
call costs its own tokens, and neither costs a re-prefill of what was already
resident. And a `sous claude` session never pays a model load or a from-zero
prefill because the daemon went idle between two of its subagents.

Measured against the session above, with the observability PR 1 fields: T3
`cache=hit` prefilling ~3K tokens (~10 s, not 202.8); T5/T7 `cache=hit`
prefilling 16.5K/1.7K (not 19.2K/20.9K); T1 `load_s=0` on a second session
after the first has been idle for an hour.

## Non-goals

- Rerouting or suppressing the `agent_summary` calls. They are Claude Code's;
  the fix below makes them harmless, and PR 1's `took=` field makes them
  visible. (They remain a waste of local work — T6 prefilled 82.8K tokens for
  7 — but 2 s on a warm hit.)
- Surviving a change in the *middle* of the system text or a shrink. `lcp=`
  and PR 1's `lcp_region=` diagnose it; nothing saves it.
- The SSD tier itself. Scoped as a spike at the end of this document.
- Batching (Phase 3b, parked), Claude Desktop (not pursued).
- Changing what the model *sees*: the in-place rendering below reproduces
  Claude Code's own legacy shape for models without mid-conversation system.

## Design

### A1. Inline system messages render in place (`gateway/convert.py`)

The template's system turn is built from the **top-level `system` field
only** (`string` or text blocks, billing header dropped, `strip_volatile`
applied, `\n\n`-joined as today).

An inline message with `role: "system"` is handled by position, in a pre-pass
over `messages` before turn conversion:

1. **Leading** (no user or assistant message before it): folded into the
   system prompt exactly as today. Claude Code never sends this shape (its
   assembler emits an inline system message only after a user message), but
   another client may, and the template accepts a system turn at index 0 only.
2. **After a user message** (Claude Code's shape; its validator guarantees the
   previous message is `user` and the next is `assistant`, another system
   message, or the end): its text parts, `\n\n`-joined and passed through
   `strip_volatile`, become one more **text block appended to that preceding
   user message's content**, wrapped as
   `<system-reminder>\n{text}\n</system-reminder>` unless the text already
   begins with `<system-reminder>` (Claude Code wraps it itself for one model
   id). Consecutive system messages merge into the same user message in order.
   A message whose text strips to nothing (a bare `<total_tokens>` reminder)
   contributes no block, so the render is byte-identical to a request without
   it — the property `strip_volatile` already provides for the hoisted case.
3. **After an assistant message** (not a shape Claude Code produces): rendered
   as its own user turn with the same wrapping — the equivalent of Claude
   Code's own `isMeta` user-message fallback.

Because the appended block goes through `_user_turns` like any other text
block, the existing rule "text after tool results is its own user turn" places
it exactly where Claude Code's legacy merge would: after the tool turns, as a
user turn, before the next assistant turn. `_ROLES` keeps `"system"`; the
`role == "system"` skip in `chat_messages` goes away because the pre-pass has
consumed them.

Effect on the incident: T3's render is T1's render + assistant turn + tool
turns + one user turn of 593+ tokens. T1's turn slot is a strict prefix;
`_take` serves it; ~3K tokens are prefilled.

### A2. Retained turn slots (`engine/promptcache.py`)

Today a `turn` slot is *moved* into the turn that extends it: `_take` removes
it from the map and the turn adopts its arrays. A `fork` slot is *copied* and
stays. The change: **a turn slot is copied and stays whenever the budget can
hold the copy, and moved only when it cannot.**

In `PrefixCache.generate`, after `_take` returns a `turn` slot:

- Under the lock, `_make_room(slot.bytes, protect=(slot,))`. If it returns
  True, the slot stays in the map (and is protected through this turn's
  pre-turn `_evict_caps` pass, as a taken fork already is); the turn copies it
  with `fork_copy` into a fresh cache exactly as a fork hit does. If it returns
  False — the budget cannot hold a second copy beside the live cache, which is
  always the case at `prompt_cache_gb = 0` — the slot is removed and adopted as
  today. `_take` itself no longer removes anything; the decision moves to
  `generate`, where the budget is visible.
- Stats: a copied turn-slot take counts as `hits` + `reused_tokens` (the log's
  `cache=hit` keeps meaning "this conversation's own slot"; `fork_hits` keeps
  meaning "a shared boundary"). Two new counters: `retained` (turn-slot takes
  that left the slot in place) and `moved` (turn-slot takes that did not). The
  observability spec's `took=` field prints `turn@N` / `turn-moved@N`.
- **Lineage bounds the accumulation.** `Slot` gains `parent: Slot | None`, set
  on the new turn slot to the slot it was copied from. When a turn publishes
  its turn slot, and the slot it copied has a still-resident `turn`-kind
  parent, that grandparent is dropped (charged to `evictions`, not
  `pressure_evictions`). A linear conversation therefore holds at most its
  current and previous lengths; a branch (the summary call) holds its own
  slot beside the real conversation's until LRU takes it. `fork`-kind slots
  are never dropped by this rule. `parent` is a plain reference cleared when
  the parent leaves the map (so a dropped grandparent's arrays are freed, not
  pinned by lineage).
- Cost: one `fork_copy` of the taken slot per warm turn (~4 GiB at 63K tokens;
  the VLM path already snapshots the whole cache once per turn for the
  restore-after-decode, so this is a second copy of the same order). Memory:
  one extra slot per live conversation under the auto budget (~23 GiB on the
  M5 Pro), none at `prompt_cache_gb = 0`.

Effect on the incident: T4 copies T3's slot and publishes its own (garbage,
LRU-evicted later); T5 copies T3's slot — still resident — and extends it,
prefilling 16,533 tokens; publishing T5's slot drops T3's parent (T1's slot)
if still present. T6/T7 likewise: T7 prefills 1,681.

### B. Preload and hold (`engine/base.py`, `cli.py`, a new daemon-level route)

**Engine manager.** `EngineManager` gains a holder registry
`{pid: create_time}` and three behaviours:

- `hold(pid, create_time) -> dict`: records the holder; if no engine is loaded
  and no preload is running, starts one daemon thread that calls `get()` and
  then `release_mlx_thread_state()` (the CLAUDE.md invariant: every thread that
  touched mlx releases its state before exit). Load failures are logged
  (`sous.engine: preload failed (<type>)`), never raised to the caller — a hold
  is an optimization and the first turn's own `get()` reports the real error.
  Returns `{"loaded", "loading", "holders"}`.
- `unload_if_idle`: before the idle test, prune holders whose PID is gone or
  whose live `create_time` differs from the recorded one by more than one
  second (PID reuse); `psutil` is already a dependency. If any holder remains,
  return False. If the *last* holder was pruned in this sweep, `touch()` first
  so the idle clock restarts — a user who quits `sous claude` and relaunches it
  within `idle_unload_minutes` does not pay a reload. The sweep already runs
  on the worker's loop every poll interval when no task is claimed.
- `status()` adds `holders` (count) and `loading` (bool).

The hold does not touch `_leases` (a lease is a per-call pin around `get()` →
`count_tokens()`; a hold is a cross-call pin) and does not change
`generation_in_flight`.

**Route.** A new module `sous/monitor.py` hosts the daemon-level loopback
routes under `/sous/` (mounted in `create_server` regardless of the gateway
flag, guarded by the same Host/Origin check as the gateway's routes, and
registered before the gateway's catch-all so `/sous/*` never forwards
upstream). This PR adds:

- `POST /sous/hold` — JSON body `{"pid": int, "create_time": float}` (cap
  1 KiB; anything else 400, Anthropic-shaped error body like the gateway's) →
  200 with `hold()`'s dict.
- `GET /sous/status` — the `server_status` dict as JSON. The observability
  spec extends it; here it exists so the launcher's preflight can stop
  speaking MCP.

**Launcher.** `sous claude`'s preflight fetches `GET /sous/status` with
`httpx` (already a dependency) instead of the MCP `server_status` tool; the
MCP client code in `cli.py` (`_server_status`, its function-local `mcp`
imports) is deleted. A 404 means the running daemon predates this CLI: the
preflight prints `sous claude: the running daemon predates this CLI; restart
it (sous stop; sous serve)` and exits 1 — the same rule the pre-routing check
already applies, no compatibility mode (the CLI and daemon ship in one
package). After the existing gateway checks pass, it `POST`s
`/sous/hold` with `os.getpid()` and `psutil.Process().create_time()`, prints
one line — `sous claude: preloading <model_id>; held while this session runs`
(or `already loaded`) — and continues without waiting. `execve` keeps the PID,
so the holder *is* the Claude Code process; when it exits, the next sweep
releases the hold. A hold request that fails for any other reason warns and
launches anyway.

**Not in scope, stated:** a `release` route (nothing runs after `execve` to
call it; the PID watch is the release); re-registering after a daemon restart
(holders are in-memory; a restart drops them, the model unloads after
`idle_unload_minutes`, and the next subagent turn reloads it — documented);
holding on behalf of the MCP-only `sous mcp` path (delegated tasks have the
worker's own `get()`/`touch()` cadence).

### Observability hooks this spec adds

Log lines (metadata only): `sous.engine: hold pid=N (holders=M)`,
`sous.engine: hold released pid=N (holders=M)`, `sous.engine: preloading
<model_id>`, `sous.engine: model loaded in N.N s`. PIDs are not secrets. The
prompt-cache counters `retained`/`moved` appear in `prompt_cache_stats` and,
via the observability spec, on the turn line.

## Invariants preserved

- Slots are owned by the thread that built them and looked up only by it
  (#34); the copy on a retained take happens on the owning session thread, as
  fork copies do. The preload thread never builds a slot.
- Never rewind a cache to make a slot; the copy-then-extend shape is the fork
  path's, unchanged.
- The template's rule (one system turn, at index 0) is honoured by
  construction: nothing after index 0 renders with `role: "system"`.
- The gateway never logs bodies, header values or query strings; the new
  routes log nothing about their bodies beyond the PID.
- `sous claude` sets no `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` or tier
  variable; the hold request carries no credential.
- `_make_room`'s feasibility rule (a copy that cannot fit even on an empty map
  is not taken) applies to the retained copy: at `prompt_cache_gb = 0` the path
  is byte-for-byte today's move.

## Testing

**A1 (`tests/test_gateway_convert.py`, fake render):**
- A `role: "system"` message after a user message with tool results renders as
  a user turn after the tool turns, wrapped in `<system-reminder>`; the
  rendered message list of the request without it is a strict prefix.
- Leading system message folds into the system prompt (today's behaviour).
- Two consecutive inline system messages merge in order into one block each.
- A pre-wrapped text is not double-wrapped.
- A `<total_tokens>`-only inline system message renders nothing.
- After an assistant message: its own user turn.
- Top-level `system` still drops the billing header and strips volatile text.

**A1 end to end (`tests/test_gateway_routes.py`, fake engine):** three requests
shaped like T1→T3→T5 (the third adds a system message after the tool results)
are `miss`, `hit`, `hit` with `reused_tokens` equal to the previous stable
render each time.

**A2 (`tests/test_promptcache.py`):**
- Retained take: after a turn-slot hit with room, the slot is still in the map,
  `retained == 1`, and a second conversation branching from the same prefix
  hits it too (the incident's shape: real → summary → real, three hits).
- No room: `moved == 1`, slot gone, arrays adopted (today's assertions).
- `prompt_cache_gb = 0`: always moved; resident count never exceeds one.
- Lineage: after three linear turns exactly two `turn` slots of that
  conversation remain (current and previous), the grandparent's bytes are
  freed, and `fork` slots are untouched by the rule.
- The taken slot is protected through the pre-turn cap pass.
- The pressure valve can still take a retained slot (it is unprotected after
  the turn).

**B (`tests/test_engine_base.py`, `tests/test_monitor.py`, `tests/test_cli.py`):**
- `hold()` starts exactly one preload thread, which calls the factory once and
  releases mlx thread state (monkeypatched); a second `hold()` while loading
  starts none.
- The sweep prunes a dead PID and a reused PID (fake `psutil`), keeps the model
  while any holder remains, and restarts the idle clock when the last leaves.
- `POST /sous/hold` accepts the shape, rejects a non-loopback Host, a foreign
  Origin, a malformed body and an oversized one; `GET /sous/status` returns the
  dict; both are reachable with the gateway disabled; `/sous/anything-else`
  is not forwarded upstream when the gateway is enabled.
- The preflight: 404 on `/sous/status` → exit 1 with the restart message; a
  refused hold warns and still execs (assert the env and argv handed to a
  fake `execve`).

**Model tests (local, `@pytest.mark.model`):** replay the T1→T3 shape on the
real 27B with the real template: T3 is a `hit` and token-for-token equal to a
fresh cold generation with the same seed; the retained slot serves a second
branch bit-exactly.

**Real-daemon gate (before merge, recorded in the PR):** a `sous claude`
session with one background general-purpose subagent of ≥3 tool-using turns:
every real turn after the first logs `cache=hit`, no `lcp` above the header
appears, and `holders=1` during the session and `0` after `claude` exits with
the model still loaded for `idle_unload_minutes`.

## Documentation

- README "Gateway mode": the turn-slot sentence ("a subagent's consecutive
  turns reuse their own slot") gains the retained/moved rule and the
  two-lengths bound; a paragraph on what `sous claude` now does at launch
  (preload, hold, release on exit, the restart caveat); the mid-conversation
  system-message rendering under "What a locally served turn gives up" (it is
  a fidelity note, not a loss).
- CLAUDE.md gotchas: replace "A `turn` slot is *moved* into the turn that
  extends it" with the copy-when-room rule and its reason (Claude Code
  branches a conversation with its progress-summary calls); add the
  inline-system-message rule and why the hoist was wrong.
- Config docs: `idle_unload_minutes` note that a live `sous claude` pins the
  model.

## Risks and open questions

- **Retained copies double per-turn copy traffic** on the VLM path (fork copy
  + snapshot). Tens of milliseconds per GiB on unified memory; PR 1's
  `prefill_s`/`decode_s` will show it if it is not.
- **Lineage is by reference, not by content.** A conversation that branches
  twice (two summary calls between real turns) leaves two garbage slots until
  LRU. Bounded by `MAX_SLOTS` and the byte cap; acceptable.
- **`MAX_SLOTS = 16` now counts retained predecessors and branch garbage.**
  If the observability spec's `evicted=` field shows the slot count, not the
  byte cap, doing the evicting, raise it (32); the byte budget is the real
  limit and the count is a sanity bound.
- **The wrapper text is Claude Code's, not the model's training data.** Qwen
  has not seen `<system-reminder>` blocks, but the frontier fallback path
  shows the same text to older Claude models the same way; the model tests
  compare generations, not just caches.
- **PID watch cadence** equals the worker's poll interval (seconds); a hold can
  outlive its process by one sweep. Harmless.
- **A daemon restart mid-session** drops the hold silently. The launcher could
  re-register from a Claude Code hook; not worth it until it bites.

## Follow-up spike: an SSD tier for cache slots

Not part of PR A or B. Run after both specs land; its result decides whether a
disk-tier spec is written.

**Question.** Can a resident slot be written to the SSD and read back into a
usable, bit-exact cache faster than recomputing it, for the hybrid 27B model?

**Why it might pay.** KV state is 64 KiB per token on this model (16
full-attention layers × 4 KV heads × head dim 256 × K and V × bf16), so a 63K
slot is ~4 GiB; the linear-attention layers add a small fixed-size recurrent
state per layer. Recomputing that slot costs ~190 s; reading 4 GiB from the
internal SSD at an assumed 5–7 GB/s costs ~1 s. Causal attention makes a longer
slot's KV a strict extension of a shorter one's, so a conversation could be
mirrored as append-only chunks (a 2K-token turn writes ~128 MiB), with only
the recurrent state saved per boundary. Eviction would become demotion; a
tools fork would survive idle unload and daemon restarts. macOS swap is not a
substitute: Metal allocations are effectively wired, which is why the pressure
valve exists.

**Method.** In-process, on the real model, no daemon: prefill one real ~63K
Claude Code-shaped prompt; walk the cache the way `snapshot`/`fork_copy` do
and serialize every layer's `state`/`meta_state` (`mlx_vlm.models.cache`:
`KVCache`, `ArraysCache`) with `mx.save_safetensors`; time the write; drop
the arrays; reconstruct via `hooks.new_cache()` and set the states; time the
read; generate 300 tokens from the restored cache and from a fresh prefill
with the same seed and compare token-for-token; measure the 256-step padding
and offset handling of `KVCache` on restore; repeat with a 2K-token extension
to measure the incremental write.

**Pass criteria.** Token-for-token equality; restore (read + array
materialization) under 5 s for a 63K slot; incremental write proportional to
new tokens. If it passes: a spec covering a disk budget knob with its own LRU,
eager mirroring on the session thread after publish, demotion instead of
deletion in the pressure valve, and restore-on-miss before a cold prefill. If
it fails on equality, the spike report names which layer type is not
round-trippable and stops there.
