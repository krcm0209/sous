# Observability

The [README](../README.md#watching-it) shows `sous top` and the status line.
This is the reference: the daemon's log, the three status commands, and the
one line every served turn writes.

## The daemon log

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

## `sous status`, `sous top` and `sous statusline`

**`sous status`** prints the status document once: the engine, the turns in
flight and the last five turns.

**`sous top`** (or `sous status --watch`) is the event stream in a terminal —
open it beside a `sous claude` session and watch a subagent's order go from
the rail to the plate — a perforated slip with a timer dial, a pixel chef
whose face is the state, the walk-in's counters, and the recent orders:

![sous top at 100×30: the order slip mid-decode with a second order queued, THE LINE with the chef, and the recent orders](../tests/snapshots/sous-top-100x30.svg)


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

## The turn line

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
reused tokens is a tools fork, ~57K a header fork) or `miss` — `disk` when
the prefix came off disk; `took` names
the slot and its length — `turn@N` for this conversation's own slot copied
and left in place, `turn-moved@N` for one the budget could not hold a copy
of, or whose copy failed (removed and extended in place — every hit at
`prompt_cache_gb = 0`), `fork@N` for a shared boundary, `disk@N` for a
fork read off disk (`cache=disk`; the turn also republishes it as a
resident fork, so its line reads `forks=1`), `none` on a miss, and
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
the phases can sum to less than the total. A fork written to disk
(`persist_s`) or read from it (`restore_s`) is timed into neither phase
either — both are their own fields, so `prefill_tps` stays a prefill rate
on the turn that restores. `ttft_s` is not one more slice
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
cache) and `verify_attention=active|unavailable|off` (whether speculative
verify runs its attention as grouped exact calls; with
`verify_attention_probe_s=` when the load-time exactness probe ran). One
more line names the Anthropic tool *types* a turn dropped, when any.
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
