# How it works

The [README](../README.md) is the short version. This is the whole of it:
the launcher, the daemon's own routes, the forwarder, what a locally served
turn gives up, and the prompt cache.

sous stands between Claude Code and `api.anthropic.com`: the daemon serves
Anthropic's Messages API on `127.0.0.1:8383`, answers requests for the model
id `sous-local` with the local model, and forwards everything else — the main
loop's requests, the startup probe, usage and telemetry calls — to the real
API untouched. That is the hybrid: a frontier main loop on your subscription,
Task-tool subagents on the local model, one `ANTHROPIC_BASE_URL`, and a
frontier model always reviewing the local model's work.

## `sous claude`

```bash
sous claude                            # Claude Code, subagents served locally
sous claude -p "summarize README.md"   # every argument passes through to claude
```

`sous claude` asks the running daemon for its effective settings — the config
file may have been edited since it started, and only a restart applies it —
and refuses if no daemon answers or if the daemon predates routing. It says
so when the file and the daemon disagree, and uses the daemon's values. Then
it replaces itself with `claude`, having set:

| Variable | Value | Why |
|---|---|---|
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:8383` | one endpoint; sous routes on the requested model id |
| `CLAUDE_CODE_SUBAGENT_MODEL` | `sous-local` (the first `local_models` entry the daemon reports) | the default model for Task-tool subagents; the main loop keeps its `claude-*` id |
| `CLAUDE_CODE_SUBAGENT_MODEL_FORCE` | `1` | the override: since Claude Code 2.1.26x a built-in agent's own `model:` (Explore, for one) or a per-spawn model beats the default above; this applies the default to every subagent regardless |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | the daemon's `max_context_tokens` | Claude Code has no built-in size for `sous-local`; it honours this variable only for non-`claude-*` ids, so the main loop is unaffected |
| `API_TIMEOUT_MS` | `3000000` | a cold model load plus a long prefill takes minutes |

plus `--disallowedTools LSP` unless you pass your own `--disallowedTools` (a
language server connecting mid-session appends its schema to every request
and re-prefills the conversation). It sets **no** `ANTHROPIC_AUTH_TOKEN`,
`ANTHROPIC_API_KEY` or `ANTHROPIC_DEFAULT_*_MODEL`: either credential
variable switches Claude Code from your subscription login to API-credit
billing, and the tier variables would pull the main loop onto the local
model. If your shell already exports a credential variable, `sous claude`
warns and launches anyway — that is your billing decision, not sous's (the
same for an inherited `ANTHROPIC_DEFAULT_*_MODEL`, which would pull that tier
off the upstream). It
also leaves `CLAUDE_CODE_AUTO_COMPACT_WINDOW` alone: that setting is global,
and pinning it to the local window would make the frontier main loop compact
far too early; the subagent's window is bounded by
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` instead.

Before it execs, `sous claude` asks the daemon to keep the model loaded for
the session: it `POST`s `/sous/hold` with its own process id and start time,
and since `exec` keeps the process id, the holder *is* the Claude Code
process. If nothing is loaded the daemon starts loading right away, on its
own thread — the launcher prints `sous claude: preloading <model>; held
while this session runs` (or `already loaded`) and does not wait, so the
first subagent turn finds the weights resident or the load already under
way instead of paying it. The daemon's idle sweep — one thread inside the
daemon, every 15 s — checks its holders on every poll: when the `claude`
process exits, the hold goes with it at the next sweep, and the idle clock
restarts from that moment, so quitting and relaunching inside
`[model].idle_unload_minutes` never reloads either. A hold the daemon
refuses is a warning, not a stop. Two caveats: a daemon restart forgets its
holders (the model then unloads after `idle_unload_minutes` and the next
subagent turn reloads it); and `sous claude --help`, `-h`, `--version` and
`-v` skip the hold, while any other invocation that exits at once holds like
a session — the load it may start is one nobody waits for, and the weights
then stay resident for a fresh `idle_unload_minutes`.

## The `/sous/` routes

The preflight itself is plain HTTP. The daemon's own routes live under
`/sous/` on the same port, loopback-only like the Messages routes:

- `GET /sous/status` — one JSON document: `engine` (`loaded`, `loading`,
  `unloading`, `model_id`, `idle_seconds`, `holders`, `memory_gb`, the `prompt_cache`
  counters, and on the VLM backend `positions`, the load line's
  `engine|model`), `inflight` (the turn the model is serving right now — its
  `msg_` id, phase, tokens so far, rate and ETA — usually empty or one
  entry, ordered with the turn on the pass first and then the queue in
  arrival order), `config` and `recent_turns` (the last 50, every field of
  the [turn line](observability.md#the-turn-line)).
- `GET /sous/events` — the same document as a Server-Sent Events stream:
  once at connect, then whenever something on it changed — the turn in
  flight (at most ten times a second), a load, an unload, a hold or its
  release, an edit of the config file — and, only while a turn is in
  flight or a load or unload is under way, at
  least once a second; idle, nothing but a `ping` every 10 s. The idle clock is not
  refreshed between frames: a client advances `engine.idle_seconds` from
  the time since the frame arrived, the way `sous top` does; `memory_gb`
  likewise waits for the next frame.
- `POST /sous/hold` — what `sous claude` posts (above).
- `POST /sous/unload` — free the weights on request; what `sous tune` asks
  for before it measures ([Tuning](tuning.md)).

A `404` from `/sous/status` means the running daemon predates this CLI (the
route did not exist); `sous claude` says so and exits 1 — restart the daemon
from the same install (`sous stop`, then `sous serve` or `sous
install-launchd`). There is no compatibility mode: the CLI and the daemon
ship as one package.

## Forwarding

Forwarding is a plain HTTP/1.1 pass-through to `[server].upstream_url`
(default `https://api.anthropic.com`): the request body goes up byte for
byte; `Authorization`, `anthropic-beta`, `anthropic-version` and every header
sous does not recognise travel unmodified; only `Host` and the hop-by-hop
headers change — and, on the two Messages routes whose body sous had to read,
`Content-Length` is recomputed from the exact bytes — and responses carry the
upstream's headers minus the
hop-by-hop set, plus a `Via: 1.1 sous`, with `Date` and `Server` being the
daemon's own rather than the upstream's. sous stores no
credential — it never sees a token it did not receive to forward — honours
no `HTTPS_PROXY` or `~/.netrc`, follows no redirect, and retries nothing
(Claude Code retries). When the upstream is unreachable, forwarded requests
get a `502` (`504` on timeout) and the local model keeps working. WebSocket
features (voice) are not forwarded.

## Whole-session-local runs

A whole-session-local run — every tier pinned to `sous-local` — remains
possible for exercising the endpoint; the recipe is in
[CONTRIBUTING.md](../CONTRIBUTING.md#verifying-the-endpoint). It is a
verification setup, not a mode: it is exactly the trade the
[README](../README.md) argues against.

## What a locally served turn gives up

Claude Code executes the tool calls itself, under its own permission mode;
the local model only decides what to call. What a locally served turn gives
up, stated plainly:

- **No sandbox of sous's own.** The endpoint returns `tool_use` blocks and
  never runs a tool; Claude Code's permission mode is the boundary (see
  [Security model](../README.md#security-model)).
- **No Anthropic server-side or built-in tool types (no client-supplied
  schema).** `WebSearch`, `WebFetch`-as-server-tool and code execution run
  inside Anthropic's API; `bash_*`, `text_editor_*` and the other built-ins
  run on the client but arrive with their schema implied by the type. Both are
  dropped from a locally served request (logged as `dropped N tool(s) with no
  client-supplied schema`) — the local chat template can only offer a tool it
  has an explicit schema for. Claude Code sends custom-typed equivalents when
  it drives a non-claude model, so a local subagent keeps its file and shell
  tools. The main loop, forwarded upstream, keeps everything.
- **No thinking, no request-level sampling.** `thinking`, `temperature`,
  `top_p`, `top_k`, `stop_sequences` and `tool_choice` are accepted and
  ignored; the daemon's `[model]` sampler applies. Images and documents in
  messages become a one-line `[image omitted: sous serves text only]`
  placeholder.
- **A shorter window, and Claude Code compacts inside it.** A frontier
  subagent has a 200K-token window; a local one has
  `[model].max_context_tokens` (131072 by default), which `sous claude`
  passes as `CLAUDE_CODE_MAX_CONTEXT_TOKENS`. Claude Code auto-compacts a
  conversation when *its own* token count reaches the window minus its
  output reserve (20K) minus a 13K buffer — 98K at the default window — and
  its "precompute" variant fires earlier still, at a fraction of that it
  takes from remote configuration. In practice a general-purpose subagent
  compacts after four or five tool turns at 131072: one call to write the
  summary (a warm hit, but a ~3.4K-token generation at the model's decode
  speed — about 3.5 minutes on the default model) and one to continue from
  it (the summary plus every re-attached file, ~30K tokens of prefill from
  the tools fork), and the agent then reports from a summary of itself.
  Nothing on the sous side changes this — both calls are served as cheaply
  as their content allows — and none of Claude Code's compaction switches
  (`CLAUDE_CODE_AUTO_COMPACT_WINDOW`, `DISABLE_AUTO_COMPACT`,
  `autoCompactEnabled`) is per-model: each would also change the frontier
  main loop, so `sous claude` sets none of them. The lever is the window:
  the default model's native context is 262144, and `max_context_tokens =
  262144` moves the classic threshold to ~229K at the cost of one more
  window of KV reserved out of the auto prompt-cache budget (8 GiB on the
  default model — the ~27 GiB auto budget described under
  [`prompt_cache_gb`](configuration.md) becomes ~19 GiB on a 64 GB machine, still room for a tools fork
  beside a retaining conversation). Where the
  earlier precompute trigger lands at that window is not known until
  measured: its fraction comes from Claude Code's remote configuration,
  keyed by window size. Restart the daemon after the edit; `sous claude`
  passes the running daemon's value.
- **A client that disconnects does not stop the model.** A local turn runs to
  completion (so the next request never waits on a wedged lock); aborting
  mid-generation comes with batching, later. A forwarded stream, by contrast,
  is closed upstream the moment the client hangs up.

## Prompt cache and continuity

Local turns are serialized: each one takes the endpoint's own lock, and every
generation runs on the one session thread behind it. The prompt cache keeps
one slot per resident conversation (bounded by `[model].prompt_cache_gb`), so
a subagent's consecutive turns reuse their own slot — copied and left in
place when the budget can hold the copy, so a conversation that Claude Code
branches (its progress-summary calls for a background agent take the same
prefix down a different last turn) still finds it, and moved into the
extending turn when it cannot, which is every hit at `prompt_cache_gb = 0`; a
linear conversation holds at most its current and previous lengths, and two
subagents interleaving reuse theirs. Reuse is owner-scoped — a slot is used
only by the thread that built it — and the budget and memory-pressure
eviction ([`prompt_cache_gb`](configuration.md)) apply across the daemon, so a new conversation's slot can
displace a least-recently-used one (and at `prompt_cache_gb = 0`, where only
one slot fits, it will).

A new subagent starts from a *fork* when it can: a copy of an earlier
conversation's cache taken at a boundary its own prompt shares. Two
boundaries are kept, each when it is long enough to clear the 4096-token
floor. The *tools* boundary is where the template's tool block ends —
Qwen3.5/3.8 render the `# Tools` block *before* the client's system text,
and Claude Code's tool array is byte-identical across sessions and
projects for a given subagent type (Explore, general-purpose, …), so a
new `claude` process's first subagent turn starts ~45–56K tokens warm
instead of paying ~170 s cold. The *header* boundary is where the whole
system block ends, so a same-type subagent inside the same session starts
~57K tokens warm and prefills only its own brief. Each fork is a full copy
of the KV at its boundary (~3.5 GiB at 57K tokens on the default model):
one tools fork per tool set, plus one header fork per session that has
used it, so several live `claude` sessions accumulate more than the
tool-set count alone suggests. Budgeted by
[`[model].prompt_cache_gb`](configuration.md). The *tools* fork is also kept on disk: at the tools boundary a cold
turn writes the live cache to `~/.sous/forks/` (3.7 GiB at 58K tokens,
0.3–0.4 s on the M5 Pro, where the write lands in the page cache) before
taking its resident copy, and a miss whose render extends a stored file
reads it back in under a second (0.5–0.7 s, from a file none of which was
in memory) — across an idle unload, a daemon restart, an upgrade that
leaves the engine's numerics alone, and a reboot. The header fork lives
only as long as the weights (`[model].idle_unload_minutes` drops it with
the model): measured on the maintainer's log, keeping it on disk too
would have saved about eight seconds in nine days for a second 3.5 GiB
write per cold turn. A
file is used only when its ids are exactly a prefix of the render, whole
or not at all, and only by a daemon whose backend and its version
(mlx-vlm or mlx-lm), mlx version, GPU, weights snapshot, engine sources,
positions owner and int8 state match the ones that wrote it; anything else
is a natural miss and ages out of the disk budget
([`prompt_cache_disk_gb`](configuration.md)). Two subagents still
run one at a time; batching is a later phase.

Usage is split the way Anthropic's is. `cache_read_input_tokens` is what
the turn served from a resident cache slot and `input_tokens` the rest, so a
warm subagent turn shows a few hundred input tokens and ~57K cache reads.
`message_start` carries the whole count (it is sent before the cache
decision); `message_delta` and the non-streaming body carry the split,
which is where the SDKs read the input-side fields from when present. No
`cache_creation_input_tokens`: every prompt stays resident, so it would only
double-count the uncached tokens.

Mid-conversation system messages become `<system-reminder>` blocks.
Claude Code delivers attachments that arrive after the first turn (agent
listings, MCP instructions, deferred tools) as a `role: "system"` message
after the preceding user message. The chat template takes one system turn,
at index 0, so sous renders such a message the way Claude Code's own
fallback for models without the feature does — a `<system-reminder>` block
in the user turn before it (after that turn's tool results). The model sees
the same text in the same place a frontier model without the feature would;
nothing the model needs is lost, and the conversation stays a strict
extension of the previous turn's, which is what keeps it warm.
