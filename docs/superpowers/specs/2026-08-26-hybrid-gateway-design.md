# sous — Hybrid Gateway Design Spec

*2026-08-26. Addresses issue #41 (alt proxy mode: local model as a first-class
Claude Code subagent) and sequences issue #44 behind it. Grounded in a deep
read of oMLX (github.com/jundot/omlx @ v0.6.3rc3, Apache 2.0) — the only
shipping Anthropic-compatible MLX server — and of sous v0.4.0. Full research
synthesis: https://claude.ai/code/artifact/140227fa-2f12-4e21-b1e5-fd4f9b3a9385*

sous today is an MCP daemon: Claude writes a spec, a sandboxed worker executes
it, Claude reviews the diff. Issue #41 proposes a second, complementary mode —
an Anthropic Messages endpoint on the same daemon, so Claude Code subagents
run on the local model with the session's tool surface while the main loop
stays frontier. This spec commits the design for that mode.

## What the research changed about #41

Three of #41's open questions are now settled; two remain and stay as gates.

**The routing predicate is solved by an env var #41 didn't know about.**
`CLAUDE_CODE_SUBAGENT_MODEL` pins the model id of every Task-tool request,
independently of the three `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL` tier
vars (oMLX sets all four: `omlx/integrations/claude.py:129-144`, tested in its
`tests/test_integrations.py:1408,1752,1792`). Subagent requests therefore
arrive self-identified — `request.model == "sous-local"` — while the main loop
keeps real `claude-*` ids. #41's spike gate 3 (internal Haiku traffic matching
the predicate) is retired by construction; gate 1 confirms by observation.

**Honest model ids are mandatory, not merely preferable.** Claude Code ignores
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` for model ids that canonicalize to
`claude-*` (live-verified caveat, `omlx/integrations/claude.py:148-158`), so
impersonating a Claude model forfeits context-window control. Claude Code also
refuses models under a 48K context window (oMLX gates at 49152); sous's
shipped `max_context_tokens = 32768` is below that floor and gateway mode must
run above it.

**Concurrent subagents do not force serialization.** Verified empirically
against the installed mlx-lm 0.31.3: the default model's cache classes
(16 × `KVCache`, whose `merge` upgrades to `BatchKVCache` and handles unequal
history, plus 48 × `ArraysCache` with the full batch method set) merge two
rows of different lengths and extend to a third. oMLX serializes only llama4.
Batching is a build decision, not a research risk.

**Still unknown, and still the kill criteria:** whether subscription (OAuth)
billing survives a transparent localhost proxy — no codebase can answer this;
and whether the 27B can hold a real harness prompt (large system prompt, rich
tool schemas) as a competent subagent. These are spike gates 1 and 2,
unchanged from #41, and every line of production code in this spec is
sequenced behind them.

## Why build, not wrap oMLX

oMLX has solved exactly the local-serving half of this problem and none of
the routing half — it has zero forwarding code, never touches
`api.anthropic.com`, and its "cloud/local" mode is a dashboard copy-paste
affordance, not a router. Wrapping it (sous as thin router, oMLX as backend)
was considered and rejected as the primary path: it adds a second daemon on
Python 3.11–3.13 whose model residency competes with sous's for the same
unified memory, and it discards sous's engine work (#33/#34/#35 prompt cache,
auto-context). oMLX's bulk is breadth sous does not need — ~10 tool-call
dialect parsers, a multi-model pool, SSD cache tiers, cluster mode. sous pins
one model family and one client, so the needed core is small. oMLX remains
the reference: specific small pieces are ported with attribution, and its
Claude-Code accommodations become our checklist. The wrap option stays a
documented fallback if endpoint quality becomes a time sink.

One porting trap, recorded so nobody falls in: oMLX's
`api/adapters/anthropic.py` is dead code with a broken tool-use stream. The
live implementation is inline in its `server.py:5158-6030` plus
`api/anthropic_utils.py`. Port from those.

## Goals

- A Claude Code session where the main loop runs on the user's subscription,
  untouched, and Task-tool subagents are served by the local model — one
  `ANTHROPIC_BASE_URL`, routing on `request.model`.
- Subagents get first-class citizenship: the session's client-side tool
  surface (built-ins, MCP servers, skills), hooks, permission rules, and
  native UI (the #41 motivation). Anthropic *server-side* tools (web search,
  code execution) cannot execute on a local endpoint and are dropped — a
  deliberate narrowing of #41's "full tool surface" claim, documented in
  checklist item 7 and in the Phase 4 mode comparison.
- The MCP delegation mode is unchanged and remains the sandboxed,
  diff-reviewed path.
- Off by default, opt-in via config, labelled experimental.

## Non-goals

- Full-local mode (#44). It becomes "widen the predicate" once this ships,
  and is explicitly deferred.
- Serving arbitrary model families. The gateway serves the configured sous
  model; the tool-call dialect is Qwen3.5's (both wire formats already parsed
  by `protocol.py`).
- Any change to the worker protocol, `toolexec.py`, or the MCP task queue.
- Multi-model residency, model swapping, or serving non-loopback clients.
- Anthropic API surface beyond `/v1/messages` and
  `/v1/messages/count_tokens` (no batches, no OpenAI shape).

## Core decisions

| Decision | Choice |
|---|---|
| Where the endpoint lives | `MCPServer.custom_route` on the existing daemon, same `127.0.0.1:8383` (SDK applies no auth to custom routes; starlette/uvicorn/sse-starlette already ship with `mcp` 2.0). |
| Routing predicate | `request.model` ∈ `[gateway].local_models` → local engine; everything else forwards upstream. |
| Local model id | Honest, non-`claude-*` (default `sous-local`). Required for context-window control, and it keeps the README's honesty posture. |
| Client configuration | A `sous claude` launcher verb adapting oMLX's env contract **minus its credential lines**: it must not set `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` (either one replaces subscription OAuth with API-credit billing — the load-bearing #41 fact; oMLX sets both because it never forwards), and it warns if the user's env already sets them. Sets `CLAUDE_CODE_SUBAGENT_MODEL=sous-local`; tier vars left unset. |
| Upstream auth | End-to-end headers forwarded unmodified — auth headers, `anthropic-beta`, `anthropic-version`, and anything unrecognized — while `Host` is regenerated for the upstream origin and hop-by-hop headers (RFC 9110 §7.6.1: `Connection` and the headers it names, `Transfer-Encoding`, etc.) are stripped. sous stores no credential and never logs bodies or headers. |
| Tool execution | Never. The gateway returns `tool_use` blocks; Claude Code executes tools under its own permission system. `toolexec.py` is not in this path and the docs say so plainly. |
| Concurrency, initially | Serialized behind the existing generation lock, with SSE keepalive pings holding queued clients. Batching is Phase 3b, built only if measurement demands it. |
| Prompt cache | Strict-prefix reuse (existing machinery), made effective by server-side prompt stabilization; keyed multi-conversation slots in Phase 3a. |
| Usage reporting | Real token counts, always. oMLX tried inflating them to fake a bigger window; it broke auto-compact and was reverted (their #2400). The window is communicated via env vars instead. |
| Sequencing | Nothing merges before spike gates 1 and 2 pass with observed wire data. |

## The Claude Code accommodation checklist

Each item below is scar tissue oMLX earned from a real breakage, with its
provenance. Phase 1 implements all of them.

1. **System prompts arrive inline in `messages[]`** (Claude Code ≥ 2.1.154),
   not only in the `system` field. Merge both, canonical field first.
   (`omlx/api/anthropic_utils.py:730-772`)
2. **SSE keepalive pings during long prefills.** Claude Code disconnects on
   read timeout when no events arrive during a 90k-token prefill. Emit
   `event: ping` immediately and every 10s. (`omlx/server.py:2124,2198-2234`)
3. **Strip the volatile `<total_tokens>N tokens left</total_tokens>` marker**
   from system text. Claude Code appends a freshly-decremented copy every
   request and keeps the stale ones — measured as full ~60k-token re-prefills
   per turn until stripped. This single item is what lets the strict-prefix
   cache work at all. (`omlx/api/anthropic_utils.py:689-706`)
4. **Drop `x-anthropic-billing-header:` system blocks** — per-request random
   values, same cache-destroying effect. (`omlx/api/anthropic_utils.py:687`)
5. **Thinking blocks need a non-empty signature.** An empty string makes the
   SDK's block parser error; a constant placeholder passes. sous ships with
   thinking disabled, so this only matters if #28 lands — record it anyway.
   (`omlx/api/anthropic_utils.py:1004-1019`)
6. **Suppress the phantom empty text block** before tool_use blocks, and keep
   content-block indices contiguous. (`omlx/server.py:5310-5315,5476-5483`)
7. **Accept and drop Anthropic server-side tools** (types prefixed
   `web_search_`, `code_execution_`, `bash_`, `text_editor_`, `computer_`)
   with an INFO log. (`omlx/api/anthropic_utils.py:884-898`) Consequence,
   stated rather than hidden: a locally-served subagent has no WebSearch or
   other server-executed capability, because those run inside Anthropic's
   API, not the client. Bridging them locally is out of scope; the gateway
   docs must list them as unavailable on local routes.
8. **Tolerate unknown fields.** Unknown top-level request fields are ignored;
   unknown *content-block types* get a tolerant catch-all rather than oMLX's
   422 — their sharpest forward-compat cliff, and one we fix rather than
   inherit. Error bodies are Anthropic-shaped (theirs are OpenAI-shaped, a
   known deviation).
9. **Launcher side:** `API_TIMEOUT_MS=3000000`;
   `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and `CLAUDE_CODE_AUTO_COMPACT_WINDOW`
   both set to the served window; `--disallowedTools LSP` injected by
   default unless the user passes their own `--disallowedTools` (a language
   server connecting mid-session appends its schema to the tools array and
   re-prefills the conversation). (`omlx/integrations/claude.py`)

## Memory budget (measured, Kyle's 64 GB M5 Pro)

- KV cost of the default model: 64 KiB/token (16 full-attention layers only).
- Flat GDN recurrent state: 146.8 MiB per conversation — charged at zero by
  today's `kv_bytes_per_token`; the gateway's accounting must charge it once
  per conversation at admission (never per chunk).
- One subagent conversation at the 48K floor: 3.14 GiB. Post-load Metal
  headroom ≈ 26 GiB. Three concurrent ≈ 9.4 GiB — feasible; memory is not
  the binding constraint at N≤3.

## Phases

**Phase 0 — spike gates (throwaway code, ~2 days).**
Gate 1: a trivial transparent proxy logging method, path, `model`, and sizes
— never bodies or headers. Launch Claude Code against it with the oMLX env
recipe **minus its credential lines** — no `ANTHROPIC_AUTH_TOKEN`, no
`ANTHROPIC_API_KEY` — because setting either switches Claude Code off the
OAuth path this gate exists to observe. Answers: does subscription billing
survive; does
`CLAUDE_CODE_SUBAGENT_MODEL` reach the wire verbatim; does anything
unexpected match. If billing falls back to API credits, the hybrid is dead —
stop. Gate 2: capture one real Explore-subagent request (opt-in body dump for
one run), replay offline against the 27B via `apply_chat_template(tools=…)`;
judge well-formed tool_use across a multi-turn loop. If the model cannot hold
the harness prompt, stop. Exit: both gates documented on #41 with observed
wire data.

**Phase 1 — Anthropic endpoint, serialized (~1.5–2.5k lines + tests).**
Opt-in `[gateway]` config; `/v1/messages` + `count_tokens` mounted on the
daemon; streaming threaded through `GenerationSession` (the reply queue
carries chunks instead of one string). Two Phase 1 requirements the current
internals don't meet: `parse_tool_calls` validates against a request-scoped
tool set instead of the module-level `WORKER_TOOLS` names
(`protocol.py:173` rejects anything else today), and **client disconnect
must never wedge the session** — today's one-slot reply queue with
`_gen_lock` held means an undrained producer blocks every later generation,
so the route drains the session to turn completion even after the client is
gone (the GPU finishes the turn; true mid-generation abort arrives in 3b).
Also: client tool schemas injected via the chat template and returned as
`tool_use` blocks; the accommodation checklist above; prompt stabilization
(items 3–4) from day one; 48K floor enforced.
Exit: one full-local Claude Code session (no routing yet) completes a real
bounded task against sous — which also answers most of #44's gate 1 for free.

**Phase 2 — hybrid routing + launcher (~400–700 lines).**
Routing on `request.model`; passthrough via httpx (promoted from transitive
to explicit dependency) with end-to-end headers forwarded, `Host`
regenerated, and hop-by-hop headers stripped (see the upstream-auth
decision); `sous claude` launcher, which sets no credential vars;
SECURITY.md gains the gateway boundary. Exit: one session, frontier main
loop on verified subscription billing, subagents local, task completed.

**Phase 3a — keyed prefix cache (~300 lines).**
The single `(cache, held)` slot becomes an LRU map keyed by conversation
prefix, with a byte budget reserved out of the generation budget (oMLX's
`_hot_cache_reserved_bytes` pattern: charge `min(cap, used + slack)`, shrink
under pressure, protect the in-flight conversation). Each slot keeps the
existing epoch/strict-prefix correctness argument.

**Phase 3b — batched serving (~600–900 lines), only if measurement demands.**
mlx-lm `BatchGenerator` decode; always-chunked prefill ("Metal cannot preempt
a running kernel, so bounding chunk duration IS the interleave mechanism");
one decode-fairness debt scalar (fair share 0.5); deferred aborts drained at
step top; client-disconnect → true mid-generation abort (upgrading Phase 1's
drain-to-completion, which stops the wedge but still spends the GPU on an
abandoned turn). Empirical constants worth keeping
verbatim: the 64-token chunk grid, and descending through chunk tiers rather
than dropping straight to the floor. Known hazards, all evidenced: sous's
`snapshot`/`restore`/`trim_to` need a rewrite for batched cache objects
(offsets become per-row arrays); `ArraysCache.merge` drops padding metadata
when a prefilling row joins a decoding batch; merge transiently allocates a
second copy of the batch KV. Ship 3a first; measure serialized latency at
N=2–3 before paying for any of this.

**Phase 4 — positioning (docs only).**
"How sous compares" becomes a menu of modes with guidance — MCP delegation
(sandboxed, diff-reviewed, async) versus gateway (first-class subagents,
Claude Code's permission system, synchronous) — the deliberate editorial
change both #41 and #44 call for. State plainly what gateway mode gives up:
no toolexec sandbox, no diff-review gate; the shipped default routes
subagents only, so a frontier model always reviews local output.

## Security posture

The gateway handles subscription credentials in transit. In scope for
SECURITY.md from Phase 2: loopback-only binding; request bodies and auth
headers never logged at any level, including debug; no credential storage —
sous never sees a token it did not receive to forward; the gateway bypasses
the toolexec sandbox by design and the docs must never imply otherwise.

## Risks that survive the research

- **Gate 1 and gate 2 are existential** and cannot be de-risked by more
  reading. Everything is sequenced behind them.
- **Claude Code moves.** Three of oMLX's accommodations were reactions to
  point releases. The gateway is a maintained surface, not a finished one;
  oMLX's commit log is a cheap early-warning feed.
- **Trust miscalibration.** A weaker model inherits whatever permissiveness
  the user configured for frontier subagents. The mitigation is positioning
  (subagent-only default, experimental label), not code.
