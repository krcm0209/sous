# Configuration — `~/.sous/config.toml`

The daemon reads `[server]` and `[model]` once at startup: restart it after
an edit (`sous stop`, then `sous serve`, or `launchctl kickstart -k
gui/<uid>/<label>` for a managed daemon). [`sous tune`](tuning.md) proposes
these values for your machine.

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
# prompt_cache_disk_gb = "auto"
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
[How it works](how-it-works.md#prompt-cache-and-continuity)). Prefixes must match token for token — one added, removed or
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
degrades to that on its own.

`[model].prompt_cache_disk_gb` (default `"auto"` — a number sets the GiB,
`0` keeps nothing on disk) bounds the tools forks kept under
`~/.sous/forks/`, one file per tool array, least-recently-used first when
the budget says so. `"auto"` is a quarter of the volume's free space at
load (counting what the store already holds, so a restart never shrinks
it), capped at 16 GiB — four to five forks of the default model; a write
is skipped, with one warning, when it would leave the volume with less
than 10 GiB (or 5 %) free. The daemon logs the directory, the budget and
what it found at every load; `sous status` and `sous top` show the count
and bytes, or the reason the store is off — the count is this model's
forks, while the byte total (and the budget it is measured against) spans
every key directory under `~/.sous/forks/`, one per model and backend
combination that has ever written there. `rm -rf ~/.sous/forks` is the
eraser, safe under a running daemon. Exclude the directory from Time
Machine: its files are worthless the moment the weights or a version
change, and churn with every new tool array. Requires `prompt_cache =
true`; a daemon that cannot size the model's KV keeps forks in memory
only.

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
(see [Validation status](../README.md#validation-status)), run on the 64 GB test
machine — which validates the models and quants through sous's whole stack,
not the memory fit on physical 32 GB / 16 GB hardware (that remains
arithmetic: weights plus KV-cache headroom). Smaller models still fail more
tasks in general and make reviewing the diff matter more — but a reviewed
draft from a small local model costs your plan nothing.
