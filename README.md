# sous

The sous-chef for Claude's kitchen: serve Claude Code's subagents from a
local MLX model on your Mac, so heavy Claude Code use stretches further on
the same plan. Claude designs the menu, the local model does the prep, and
Claude Code runs every tool itself under its own permission system.

## Why

Heavy Claude Code use runs into plan usage limits, and most of what consumes
them is generated output. sous exists to stretch that budget: the mechanical,
volume-heavy output comes from a local model at zero marginal cost, while
Claude spends its much cheaper input-side attention writing instructions and
reviewing the resulting diff. Claude stays the head chef, and the same plan
carries further into the week.

This is a hybrid local + cloud arrangement, deliberately not an
all-or-nothing switch to a local model. sous serves the subagents — the
volume-heavy half of the work — from a local model and leaves the main loop
on the frontier model you are paying for.

## How sous compares

- **Full-local replacements** point Claude Code (or a fork of it) at a local
  endpoint: every task drops to local-model quality, including the ones you
  wanted a frontier model for. sous keeps the main loop on the frontier model
  and serves only the subagents locally.
- **Bare routers and proxies** swap models per request and stop there. sous is
  one at its core — a byte-for-byte transparent proxy that stores no
  credential — and adds what makes a local subagent usable: a prompt cache
  whose forks are shared across sessions and projects (a new session's first
  local turn went from 170 s to 16 s on the default model), retained turn slots
  so the conversation Claude Code branches every 30 s keeps its cache, a live
  terminal and attributed turn lines, `sous tune` to settle the model and its
  window for your machine, and model holds with idle unload.
- **What gates the local model is Claude Code, not sous.** Claude Code executes
  every tool call under its permission mode: in auto mode a frontier
  classifier reviews shell commands, network access and protected-path writes,
  reads and edits inside the working directory are approved as they are for
  any subagent, and the optional sandbox adds an OS-level filesystem and
  network boundary. sous runs nothing. See [Security model](#security-model).

## Requirements

- Apple silicon Mac (tested on M-series with 64 GB; the default model needs
  ~18 GB free unified memory — for 16 GB machines, see
  [Smaller machines](#smaller-machines))
- Python 3.14 (standard build) via [uv](https://docs.astral.sh/uv/)
- Claude Code with a Pro/Max plan

## Install

    uv tool install sous-mcp

The package is `sous-mcp` and the command is `sous`. The name dates from the
MCP server sous shipped as before 0.7.0; the bare name `sous` on PyPI is held
by an abandoned placeholder and a PEP 541 request to claim it is open
(pypi/support#11881). When it is granted the package will be `sous`.

From a checkout, `uv tool install .` works instead of the PyPI package.

### Managing the daemon

```bash
sous status             # the daemon, its engine and the recent turns
sous top                # watch it live (alias: sous status --watch); q quits
sous statusline         # one line for Claude Code's status bar (below)
sous claude             # Claude Code with its subagents served locally
sous stop               # stop it (see below)
sous install-launchd    # start at login, keep alive (recommended)
sous uninstall-launchd  # stop it starting at login, and remove the agent
sous tune               # grade the candidates on this machine and propose settings (see Tuning)
sous tune --quick       # the throughput stage only, in about 15 minutes
```

`sous stop` deliberately refuses when launchd is managing the daemon, because
`KeepAlive` would restart it a second later and the command would look like it
did nothing. It tells you which command you actually want. Stopping is for
daemons nothing is supervising.

The daemon writes one log, `~/.sous/daemon.log`, both streams. Every line
sous emits reads `2026-09-10T19:26:14.025Z INFO sous.api: …` — UTC
timestamp with milliseconds, then `INFO` (served or forwarded), `WARNING`
(a request sous refused: a 4xx, a 529, a client gone while queued, tools it
had to drop) or `ERROR` (a failure sous produced), then which part spoke.
Library lines (`uvicorn.error`, `huggingface_hub…`) take the same
shape, and `warnings.warn` arrives as `WARNING py.warnings`.
Earlier releases split stdout and stderr into `daemon.log` and
`daemon.err.log`; re-run `sous install-launchd` once — it boots the old job
out, waits up to 30 s for its daemon to free the lock, folds `daemon.err.log`
onto `daemon.log` (or only removes the old name when it was already a link to
`daemon.log`), removes it, writes the new plist and bootstraps it. That re-run
needs the daemon stopped first if it was started by hand (`sous serve`, the
normal case on a machine with no launchd job installed yet): finding the lock
still held, the command refuses and writes nothing — `sous stop`, then retry.
Once the old job is unloaded, a failure before the new one loads (a daemon
still exiting after the wait, a fold or plist write that fails) says the job
is unloaded and exits nonzero; fix the cause and re-run. A failed
`launchctl bootstrap` after the plist is written prints the bootstrap command
to run by hand and exits nonzero rather than claiming success.

## Quick start

    sous serve            # or: sous install-launchd, to start at login
    sous claude           # Claude Code with its subagents served locally

`sous claude` launches Claude Code with `ANTHROPIC_BASE_URL` pointing at the
daemon, `CLAUDE_CODE_SUBAGENT_MODEL=sous-local` (forced) and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` set to the served window, and holds the model
loaded while the session lives. Every argument after `claude` passes through
to Claude Code. The first run downloads the model (~16 GB for the default),
once. Claude Desktop is not supported: it cannot point Claude Code at a local
base URL.

## How it works

sous stands between Claude Code and `api.anthropic.com`: the daemon serves
Anthropic's Messages API on `127.0.0.1:8383`, answers requests for the model
id `sous-local` with the local model, and forwards everything else — the main
loop's requests, the startup probe, usage and telemetry calls — to the real
API untouched. That is the hybrid: a frontier main loop on your subscription,
Task-tool subagents on the local model, one `ANTHROPIC_BASE_URL`, and a
frontier model always reviewing the local model's work.

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

The preflight itself is plain HTTP. The daemon's own routes live under
`/sous/` on the same port, loopback-only like the Messages routes:

- `GET /sous/status` — one JSON document: `engine` (`loaded`, `loading`,
  `unloading`, `model_id`, `idle_seconds`, `holders`, `memory_gb`, the `prompt_cache`
  counters, and on the VLM backend `positions`, the load line's
  `engine|model`), `inflight` (the turn the model is serving right now — its
  `msg_` id, phase, tokens so far, rate and ETA — usually empty or one
  entry, ordered with the turn on the pass first and then the queue in
  arrival order), `config` and `recent_turns` (the last 50, every field of
  the turn line below).
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
  for before it measures ([Tuning](#tuning)).

A `404` from `/sous/status` means the running daemon predates this CLI (the
route did not exist); `sous claude` says so and exits 1 — restart the daemon
from the same install (`sous stop`, then `sous serve` or `sous
install-launchd`). There is no compatibility mode: the CLI and the daemon
ship as one package.

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

A whole-session-local run — every tier pinned to `sous-local` — remains
possible for exercising the endpoint; the recipe is in
[CONTRIBUTING.md](CONTRIBUTING.md#verifying-the-endpoint). It is a
verification setup, not a mode: it is exactly the trade "Why" and "How sous
compares" above argue against.

Claude Code executes the tool calls itself, under its own permission mode;
the local model only decides what to call. What a locally served turn gives
up, stated plainly:

- **No sandbox of sous's own.** The endpoint returns `tool_use` blocks and
  never runs a tool; Claude Code's permission mode is the boundary (see
  [Security model](#security-model)).
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
  default model — the ~27 GiB auto budget described under `prompt_cache_gb`
  below becomes ~19 GiB on a 64 GB machine, still room for a tools fork
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
eviction below apply across the daemon, so a new conversation's slot can
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
tool-set count alone suggests. Budgeted by `[model].prompt_cache_gb`
below. Forks live as long as the weights: `[model].idle_unload_minutes`
drops them with the model. Two subagents still run one at a time; batching
is a later phase.

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

## Observability

**`sous status`** prints the status document once: the engine, the turns in
flight and the last five turns.

**`sous top`** (or `sous status --watch`) is the event stream in a terminal —
open it beside a `sous claude` session and watch a subagent's order go from
the rail to the plate — a perforated slip with a timer dial, a pixel chef
whose face is the state, the walk-in's counters, and the recent orders:

![sous top at 100×30: the order slip mid-decode with a second order queued, THE LINE with the chef, and the recent orders](tests/snapshots/sous-top-100x30.svg)

*`sous top` at 100×30, mid-decode with a second order queued — the render the test suite pins, so the picture is always the current one.*

`q`, `Esc` or `Ctrl-C` leave the screen exactly as it was; `enter` opens
the order under the cursor — its ticket, with the turn's numbers; `l` the
legend, `?` the About card, `m` turns the motion off. The vocabulary is
glossed on the screen itself (`SEAR ▐███▌ prefill`, `REHEAT 41 hit`) and
on its legend line; the numbers are the turn line's, in the terminal's own
foreground, and nothing paints over the terminal's background. It reconnects
with backoff if the daemon restarts and says so if none is running. It is
the one command in sous that imports [Textual](https://textual.textualize.io);
nothing else (the daemon included) loads it. It needs a terminal: piped or
run over a session with no pty, it prints a message on stderr and exits 2
instead of painting the pass into a pipe and waiting for a key that cannot
come.

**`sous statusline`** prints one line for Claude Code's `statusLine`
setting — `sous: decode 612 tok · 14.7 tok/s · eta 18s` during a turn,
`sous: idle · 5 slots · held` between turns, `sous: daemon down` when
nothing answers — and reads (and ignores) the JSON Claude Code pipes to it.
Add to `~/.claude/settings.json`:

```json
{"statusLine": {"type": "command", "command": "sous statusline", "refreshInterval": 1}}
```

`refreshInterval` matters: without it Claude Code re-runs the command only on
session events — a new assistant message, a compaction, a mode change — so the
line goes quiet for a whole subagent turn; with it the command runs every
second as well. The command loads none of the heavy dependencies (no
Textual, httpx or psutil); it drains stdin and fetches the status on a
thread joined to a half-second budget, printing `sous: daemon down` if that
runs out, so it costs the status bar nothing.

To bracket a subagent from the outside as well, a `SubagentStart` /
`SubagentStop` hook can append its own record — `{ts, agent_id,
agent_type}` — to a file; it sees the subagent's start and end, not the
progress-summary calls Claude Code makes in between, which only the turn
line and `sous top` show.

Each `/v1/messages` turn served locally logs one metadata-only line, for
example (wrapped here):

```
2026-09-10T19:26:14.025Z INFO sous.api: POST /v1/messages id=msg_… model=sous-local
  stream=1 status=200 input_tokens=84335 output_tokens=2887 stop=end_turn
  cache=hit took=turn@82647 reused_tokens=82647 prefilled_tokens=1681 forks=0 evicted=1 pressure=0
  load_s=0.0 queue_s=2.5 engine_wait_s=0.0 tokenize_s=1.1 ttft_s=9.8 prefill_s=7.9 decode_s=196.6
  prefill_tps=212.8 decode_tps=14.7 seconds=207.4 tools=1b21cd75 system=9f8e7d6c
```

`id` is the response's message id (the same one in `message_start`, so the
line joins to Claude Code's transcript). `cache` is `hit` (this
conversation's own slot), `fork` (a copy of a shared boundary — ~45–56K
reused tokens is a tools fork, ~57K a header fork) or `miss`; `took` names
the slot and its length — `turn@N` for this conversation's own slot copied
and left in place, `turn-moved@N` for one the budget could not hold a copy
of, or whose copy failed (removed and extended in place — every hit at
`prompt_cache_gb = 0`), `fork@N` for a shared boundary, `none` on a miss, and
on a hit whose warm attempt failed and was rebuilt cold (a
`WARNING py.warnings: … retrying cold` line comes first), where
`prefilled_tokens` and the phases below describe that cold attempt.
`prefilled_tokens` is what the turn had to prefill;
`forks`/`evicted`/`pressure` are what it published and what was dropped under
it — `evicted` counts every drop — budget, pressure, or the lengths below
the slot a turn takes, which the take retires — and `pressure` is the
subset of those `evicted` the pressure valve forced (not a second, disjoint
count). `load_s` (a model load, ≈0 when resident — including one this turn
only waited out, started by another request), `queue_s` (the wait for the
endpoint lock — Claude Code's small background calls hold it too),
`engine_wait_s` (the wait for the engine's own generation lock, which the
endpoint lock in front of it normally keeps at zero — an abandoned stalled
generation is what still holds it; part of `seconds` and `ttft_s`, in no
phase below),
`tokenize_s`, `prefill_s` and `decode_s` say where the time went, as far as
the prompt cache measures it — not a strict partition of `seconds`: a turn
that starts from a copied fork slot times that copy into neither phase, so
the phases can sum to less than the total. `ttft_s` is not one more slice
alongside them: it spans from the start of generation to the first token,
overlapping `prefill_s` and however much of `decode_s` ran before that token,
so adding every field this way can just as easily run past `seconds` as fall
short of it. `prefill_tps`/`decode_tps` are that same phase attribution
expressed as a rate, not a throughput measurement — `prefill_s` also carries
fork copies and the snapshot, so a turn that publishes forks reports a lower
`prefill_tps` than its real prefill speed. `seconds` is the total, running
from the moment the turn takes the endpoint lock, so client-visible latency is
`seconds` plus `queue_s`. On a miss the line adds `lcp=` (how many leading
tokens the render shared with the closest resident slot), `lcp_region=`
(`tools` below the tools boundary — the tool array changed; `system` below
the header — the system text did; `tools-or-system` below the header when it
is the only boundary, since a tool block under the floor, or one a template
renders inside the system block, cannot be told apart from the text there;
`conversation` otherwise; `-` when the probe found no boundary at all, whether
for lack of fork budget or because nothing cleared the 4096-token floor, or
when the session held no slot to compare with, `lcp=0`) and
`bounds=[tools,header]`, the boundaries the probe verified — either one is
left out when it never cleared the 4096-token floor, so this can also read
`bounds=[57123]` (only the header boundary) or `bounds=[]` (no probe ran). `tools=` and `system=` are
8-hex-character hashes of the rendered tool array and system text: comparable
across lines, not reversible. Refused requests log the same way at `WARNING`
with `status=` and `seconds=` — but there `seconds` is measured from
request receipt, not from the endpoint lock, so it already *is*
client-visible latency, with nothing to add. A streaming turn that fails
after its SSE headers already went out logs `status=200` — the status the
client actually received — with an `error=` naming the failure; alerting on
`status=5..` alone misses these, so watch `error=` on streamed turns too.
Once a request parses, its `529`, `499` and failure lines carry `id=` too —
on a streamed failure, the id `message_start` already delivered. A client that
disconnects mid-turn does not stop the turn, and its line is still written
once the turn drains. A client that disconnects while its turn is still
queued behind another logs `status=499 error=abandoned` the same way — a
*local* 499, distinct from
the forwarder's own synthesized `499` for a client gone mid-forward
(below). A locally served `count_tokens` logs its `input_tokens`, `load_s`
(the model load it paid for), `count_s` (time inside the runner) and
`seconds` (client-visible, from request receipt — it includes any wait for
a free counting thread that `count_s` does not); the engine logs `model_load
seconds=N.N model=<model_id>` when it loads, plus `positions=engine|model` on
the VLM backend (which side supplies the rotary positions behind a warm
cache). One more line names the Anthropic tool *types* a turn dropped, when
any.
Each forwarded request logs one line too: `upstream`, method, path, the
model id when the body named one, the upstream's status, and seconds to
its headers — at `INFO` whatever the status, since that is the upstream's
verdict; a `502`/`504`/`499` the forwarder itself produced (unreachable
upstream, a timeout, a client gone) logs at `ERROR` or `WARNING` instead.
The daemon also disables uvicorn's access log, which would otherwise print
every request target — query string included — at INFO. Nothing else is
logged — not a request body, not a header value, not a query string, not a
response — at any level. Errors sous produces itself are Anthropic-shaped
(`{"type": "error", "error": {"type": ..., "message": ...}}`): an oversized
body (over 32 MiB) on `/v1/messages` is a `413 request_too_large`, a prompt
that fills the local window an `invalid_request_error` saying `prompt is
too long`, an unreachable upstream a `502 api_error`. Errors from the real
API come back exactly as it sent them.

## Configuration — `~/.sous/config.toml`

```toml
[server]
port = 8383
upstream_url = "https://api.anthropic.com"
local_models = ["sous-local"]
generation_timeout_minutes = 30

[model]
id = "mlx-community/Qwen3.8-27B-4bit"
idle_unload_minutes = 30
max_context_tokens = 131072
temperature = 0.7
top_p = 0.8
top_k = 20
prompt_cache = true
# prompt_cache_gb = 8
speculative_draft_id = "z-lab/Qwen3.8-27B-DFlash2"
speculative_block_size = 3
int8_prefill = false
```

Every value is optional. Swap `[model].id` for any MLX text or vision model
(e.g. the `-8bit`/`-4bit` conversions, or a fast MoE coder via mlx-lm).

`[server].local_models` are the model ids served locally; every other id is
forwarded upstream. Never `claude-*`: Claude Code ignores its context-window
environment variables for those ids (the config rejects them).
`[server].upstream_url` is where non-local requests go — an https origin, no
path, ASCII hostname or IP literal; plain http is accepted for a loopback
host only.

`[model].max_context_tokens` is the server-side limit on prompt + reply
tokens for local turns. A Claude Code subagent's prompt — its agent prompt
plus the session's tool schemas — is ~50-58K tokens before it does anything,
and it asks for 32K of output; 65536 was too small in practice. Positive
values below 49152 are raised to it. `sous claude` sets
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the running daemon's value (restart it
after an edit); without the launcher, set it yourself or a long subagent
conversation grows past it and fails with `prompt is too long`. Claude Code
auto-compacts a local subagent once its own count nears this minus ~33K
(sooner with its precompute trigger); 262144 — the default model's native
length — pushes that out at the cost of 8 GiB of prompt-cache budget.

`[model].idle_unload_minutes` frees the weights after that long with nothing
using them; a live `sous claude` session pins the model, and the clock
restarts when its last one exits.

`[model].prompt_cache` (default `true`) reuses a KV cache across the turns of a
conversation, prefilling only what the conversation gained instead of the
whole thing every turn. Every local generation runs on the daemon's one
session thread so the cache survives between turns; measured on the default
model in one process, six growing turns took 29.5s warm against 77s cold,
with per-turn time flat instead of growing. Set it to `false` to prefill
every turn from scratch.

`[model].prompt_cache_gb` (default `"auto"` — a number sets the GiB, `0`
keeps a single slot) bounds the caches kept resident
*beyond* the turn that is running: each conversation's current length, its
previous one when the budget holds both (see below), plus *fork*
slots at every boundary long enough to be worth copying (4096 tokens or
more): one at the end of the tool block, shared by every Claude Code session
and project that presents the same tool array, and one at the end of the
whole system block, shared by same-type subagents of one session (see
above). Prefixes must match token for token — one added, removed or
reordered tool is a different tool set with its own tools fork and header forks. Each fork
is a copy of the KV at its boundary, ~3.4–3.6 GiB at ~57K tokens on the
default model: one tools fork per tool set, plus one header fork per
session that has used it, so a daemon that has seen the usual three tool
sets across one or two live `claude` sessions holds roughly five to nine
forks, ~17–31 GiB (the tools forks stay most-recently-used, since every new
session touches them). `"auto"` is what Metal's recommended working set has
left once the weights, one full context window of KV
(`[model].max_context_tokens`) and 2 GiB of slack are paid for — about
27 GiB on a 64 GB machine with the default model and the default window, room
for those forks and several conversations; each live conversation under the
auto budget also keeps its previous length resident (the slot a branch of it
starts from), so a live conversation is two slots, ~8 GiB at 63K tokens,
beyond the forks; a 48 GB machine should set it to `0` (forks off, one slot), or to
more than twice one conversation slot at the length its conversations reach
— a little over 8 GiB with the default model at ~57K tokens — because the
copy that keeps a conversation's previous length resident is taken only
when that length and the longer one the turn publishes both fit; below that
every hit moves its slot, and a branch of the conversation (a
progress-summary call) starts from the fork instead. The copy's room comes
from least-recently-used slots,
forks included, so keeping a fork resident beside a retaining conversation
wants about four slots' worth, ~14 GiB. A value below ~14 GiB makes every
cold turn take a fork copy that a later turn then evicts to make room for
its own copy, so it pays the fork and never reuses it. Slots are
evicted least-recently-used first when the budget, a count of 16, or memory
pressure says so. Pressure is two readings: Metal's own headroom (room for one
more window of KV), and the kernel's memory-pressure level — at *warn* each
publish drops one least-recently-used slot (a cold turn publishes up to three
times: two forks and its turn slot; a warm turn once), at *critical* every
slot but the one that just ran. The
kernel's level is used rather than free RAM because a freshly loaded model
leaves ~17 GB of its weight files in the page cache, which reads as "used"
for a while and would evict the forks a cold turn had just made. The
conversation that just ran is never evicted by
its own turn, and a cold turn's second fork copy never evicts its first, so
`0` means exactly one slot and a 32 GB machine
degrades to that on its own. Forks live as long as the weights do:
`idle_unload_minutes` drops them with
the model, so "every new session" means every new session inside that
window.
`GET /sous/status` (and `sous status`, `sous top`) reports the engine —
`holders` (live `sous claude` sessions pinning the model), `loading` (a load
in progress, a preload included), `memory_gb` and `prompt_cache` — slots,
resident bytes, hits, fork hits, retained and moved turn-slot takes,
evictions and the subset the pressure valve took — counts only — and
`inflight`, the turn being served right now with its phase, tokens and rate.

`[model].int8_prefill` (default `false`) runs the prefill matmuls as INT8 activations
against the checkpoint's packed 4-bit weights on the M5 GPU's neural accelerators
(Apache-2.0 kernel derived from oMLX, compiled at model load — no build step). Measured
on an M5 Pro with the default model: 492 → 695 tok/s at 4K tokens, 410 → 570 tok/s at
32K (MLP and linear-attention projections; attention projections are not routed, see
#76). Decode and speculative verify are untouched. It changes prefill numerics (KL 0.033
vs the stock path on a code prompt; 4-bit weights alone are 0.052 vs 8-bit), which is
why it ships off. Needs an M5-family or newer GPU and macOS 26.2+; anywhere else the
status document reports `int8_prefill: unavailable` with the reason and prefill
runs stock. Only dense Qwen3.5-family models (`model_type` `qwen3_5`, as the default
model is) with affine 4-bit, group-size-64 weights route, and the MoE variant is refused;
a checkpoint with no eligible projection warns once; in a mixed checkpoint, ineligible
projections fall through per projection.

`[server].generation_timeout_minutes` bounds a turn at both ends: how long it
may wait for a generation slot before the endpoint answers `529`, and how long
one generation may run before the turn abandons it and answers `500` (the
generation itself cannot be interrupted and runs on to completion).

`temperature`/`top_p`/`top_k` control the local model's sampler (Qwen's own
documented non-thinking-mode defaults). Greedy decoding (temperature 0)
sounds safer but isn't: it gives the model no way to escape a bad
completion once it happens, since a near-identical prompt plus a nudge
still argmaxes to the same wrong output every time.

Speculative decoding (`speculative_draft_id`, `speculative_block_size`) is
~1.8x decode on the default model with the shipped sampling; `""` disables
it. The greedy path's speedup is unmeasured: the "~2.4x greedy" figure
earlier versions quoted ran an argmax sampler through the sampled
speculative walk, before the engine reached mlx-vlm's greedy branch, and
that branch also drafts differently (#87). Block size 3 measured best on an
M5 Pro (+3% on prose, +13% on code re-emission over the drafter's adaptive
policy); 0 lets that policy pick the depth; anything above 5 is clamped,
because mlx's fused attention kernel takes at most 5 verify rows on this
model and 6–8 rows run 5–6x slower per layer. It auto-disables with a
warning when the drafter can't serve the configured model.

## Tuning

`sous tune` chooses the model and the `[model]` settings for the machine it
runs on, so you never read a tok/s table or a quantization format. It ends
with a report, a unified diff of `~/.sous/config.toml`, and a question:

```
Apply these changes to ~/.sous/config.toml? [y/N]
```

Nothing is written before that yes; a backup is kept beside the config file,
and the tune prints the daemon-restart command every applied change needs
(`sous stop`, then `sous serve`, or `launchctl kickstart -k gui/<uid>/<label>`
for a managed daemon — the daemon reads `[model]` once at startup).

**`sous tune --quick`** (about 15 minutes on an M5 Pro for three fitting
candidates) detects the chip, the Metal working set and whether the GPU has
tensor units, fits every curated candidate to memory (printing the
arithmetic for each one it refuses), asks about every download it would need
one by one, then measures prefill and decode throughput of every arm — a
model, a drafter or none, a block size — through sous's own engine
(`--repeat` sets the attempts per measurement, default 2; the best wins and
the spread is printed). It proposes only the settings that cannot change
what the model says: the drafter, its block size, and a window that fits.
Other models are measured and reported with a "quality untested" label; a
quick run never changes the model.

**`sous tune`** (about three hours on an M5 Pro for three fitting models; the
estimate is printed after the quick stage from the measured speeds and covers
the model stage; the winner stage adds up to two arms of the same size) does
all of the above, then grades the candidates: for each model's fastest
quality-neutral arm, and for your current configuration, it runs a suite of
eight mechanical coding tasks — implement a module from its spec, write
tests for one, a docstring sweep, a cross-file rename, a bug fix, a dataclass
from a JSON schema, a CLI flag, a config-format migration — through an agent
loop of its own over a scratch copy of each task's project, running only that
task's verify commands and the suite's test runners, and denying nothing
because there is nothing to approve. Each run is scored by a
hidden grader (`--runs` sets the runs per task, default 2). The rule, printed
in full with every number:

- the **reference** is your current configuration when it fits this machine,
  else the fastest arm of the largest tier that does;
- an arm is **eligible** when its mean grade is within 0.05 of the
  reference's, it completed at least as many runs as the reference minus one
  task's worth, and it looped on a tool no more often;
- the **winner** is the eligible arm with the lowest total suite wall time
  (a tie goes to the smaller memory footprint).

On the winner, the same rule then judges one extra arm per quality-affecting
setting: INT8 prefill (where the tensor units and the checkpoint allow it)
and greedy sampling (`temperature = 0`, which also lets the drafter's
exact-match verify run). A setting lands in the diff only when its own
measured arm is eligible and faster — that is why there is no `--greedy`
flag to understand. The full run may therefore change `[model].id`, the
drafter and block size, the window, `int8_prefill` and `temperature`.

The daemon is asked to release the model first (`POST /sous/unload`) and
refuses while a `sous claude` session holds it, a turn is in flight, or a
load or unload is under way — the tune waits for none of them, it tells you,
and it asks again right before its first bench load and before every model the
suite loads. Results (`results.jsonl` with every bench row and suite run,
`hardware.json`, `report.md`, and each suite run's project and transcript under
`suite/`) land in `~/.sous/tune/<run-id>/`; `--resume <run-id>` continues an
interrupted run from the rows it already has; `--models ID ...` measures ids of
your own; `--yes` answers every prompt for scripted use, `--apply` skips only
the final one. Adding a suite task or a curated candidate is described in
[docs/tuning.md](docs/tuning.md).

## Smaller machines

`sous tune` tells you what fits and grades it here (`--quick` measures
throughput only); the table below is the fallback for a machine that cannot
reach the Hub. The alternative shares the default's `qwen3_5` architecture,
so it loads through the exact same mlx-vlm path — edit `[model].id` in
`~/.sous/config.toml` and the next local turn downloads and uses it.

| Unified memory | `[model].id` | Weights |
|---|---|---|
| 64 GB / 32 GB (default) | `mlx-community/Qwen3.8-27B-4bit` | ~16.1 GB |
| 16 GB | `mlx-community/Qwen3.5-9B-MLX-4bit` | ~6 GB |

The default is the affine 4-bit Qwen3.8-27B: half the footprint of the
8-bit-class quants with no measured tool-loop quality loss. It also keeps
mlx-vlm's speculative-decoding fast path available, which requires affine
quantization (4-, 5-, or 8-bit); mxfp quants fall into a much slower
per-token verify fallback (krcm0209/sous#58 has the measurements behind
both claims). The 16 GB pick
drops to the 9B tier because an 8-bit 9B (~11 GB) would crowd the ≈10.7 GB
Metal working-set limit of a 16 GB machine once the KV cache lands on top of
the weights; on 16 GB, also consider `[model].max_context_tokens = 49152`
(the floor) if you see memory pressure. (`mlx-community/Qwen3.5-9B-mxfp8`/`-mxfp4` look
like the obvious picks, but as of 2026-08 they are empty placeholder repos
with no weights.)

The 16 GB alternative passed the same validation as the default
(see [Validation status](#validation-status)), run on the 64 GB test
machine — which validates the models and quants through sous's whole stack,
not the memory fit on physical 32 GB / 16 GB hardware (that remains
arithmetic: weights plus KV-cache headroom). Smaller models still fail more
tasks in general and make reviewing the diff matter more — but a reviewed
draft from a small local model costs your plan nothing.

## Security model

sous executes nothing. The daemon binds to `127.0.0.1`, refuses foreign
`Host`/`Origin` values and cross-site fetch metadata on every route, forwards
the `Authorization` header Claude Code sends to `[server].upstream_url`
unmodified and nowhere else, stores no credential, and never logs a request
body, header value or query string.

The boundary around the local model is Claude Code's permission system, the
same one that governs a frontier subagent:

- **Auto mode** (the default starting mode on Pro, Max and Team) sends the
  local subagent's shell commands, network operations and protected-path
  writes, plus its spawn description and its final report, to a frontier
  classifier (Claude Sonnet 5 by default). Through sous each check is one
  request forwarded upstream, about 1.5 s. Reads and edits inside the working
  directory are approved without a check. Three consecutive or twenty total
  blocks fall back to prompting you.
- **Manual mode** prompts for every non-read action the local model takes;
  under `-p`, prompts become denials unless you pass `--permission-mode auto`.
- **The sandbox** (`/sandbox` in Claude Code) adds an OS-level filesystem and
  network boundary around every shell command, local or frontier.

These are Claude Code behaviours, read from its documentation on permission
modes and classifier billing at version 2.1.27x; sous adds no permission flag
and prints no notice about them.

## Validation status

Validated end to end on an M5 Pro / 64 GB with the default model
(`mlx-community/Qwen3.8-27B-4bit`) and its DFlash drafter:

- **Cross-session prompt-cache forks** (2026-09-06): a new Claude Code
  session's first local turn went from 170 s to 16 s once another session had
  built the tools fork, across processes and projects.
- **Same-session forks** (2026-09-04): 8.3 s against 168 s cold for the
  second subagent of a session.
- **Continuity** (2026-09-13/14): retained turn slots kept a subagent's cache
  through the progress-summary branches Claude Code makes every 30 s, at a
  262144-token window.
- **Auto mode** (2026-09-14): with a `sous-local` subagent under Claude Code's
  auto mode, each classified action was one Sonnet 5 request through sous,
  1.2–1.8 s; reads and working-directory edits were approved without one.

Tool-call parsing accepts both the XML-ish `<function=…>/<parameter=…>`
format Qwen3 emits and the hermes JSON format other MLX models use.

## Limitations

- MLX generation cannot be aborted mid-stream. Generations are serialized by
  a per-engine lock (and the engine is never idle-unloaded while one is in
  flight), so a truly wedged generation delays subsequent turns until the
  daemon is restarted. Running the model in a separate process (process
  isolation) is the future fix.
- sous serves one local turn at a time on a
  keyed prompt cache (a conversation's last two lengths plus tools and header forks, budgeted
  by `[model].prompt_cache_gb`), drops Anthropic server-side and built-in tool
  types from local turns (the ones that carry no client-supplied schema),
  ignores request-level sampling and thinking, and finishes a local turn
  even after the client hangs up. It serves the model ids in
  `[server].local_models` locally and forwards everything else to
  `[server].upstream_url` over HTTP/1.1 only — no WebSocket upgrade, no
  HTTP/2, no proxy environment.

## Upgrading from 0.6

0.7.0 removed the MCP server, the task queue, the worker and its sandbox.

- Remove the MCP registration: `claude mcp remove sous`, and any `sous mcp`
  entry in `claude_desktop_config.json`. Claude Desktop is no longer supported.
- Config: `[gateway]` keys moved — `local_models`, `upstream_url` and
  `generation_timeout_minutes` to `[server]`, `max_context_tokens` to
  `[model]`; `enabled` is gone (the endpoint is always on); `[budgets]`,
  `[commands]`, `[context]` and `[tasks]` are gone. The daemon warns once at
  startup about any of them and starts anyway. A `[model].max_context_tokens`
  written for the worker (32768) is clamped to the 49152 floor; set it to the
  window you want.
- `~/.sous/tasks/` and `~/.sous/tasks.db` are no longer read and can be
  deleted.
- `sous wait` and `sous mcp` are gone; `sous status` now prints the engine and
  the turns.

## Development

Setup, the checks CI runs, and the pull-request process are in
[CONTRIBUTING.md](CONTRIBUTING.md).

Manual end-to-end with the real model: `sous serve`, then `sous claude`, ask
for something a subagent will do, and watch `sous top`.
`uv run python scripts/api_smoke.py` is the cheap version: one served turn
against a tiny model.

Design specs: `docs/superpowers/specs/2026-08-26-hybrid-gateway-design.md`,
amended by `docs/superpowers/specs/2026-09-19-remove-mcp-path-design.md`.
