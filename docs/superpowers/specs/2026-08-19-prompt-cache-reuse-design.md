# sous — Prompt Cache Reuse Design Spec

*2026-08-19. Addresses issue #27. Replaces an earlier unreviewed draft of this
file whose central design measurement disproved; the failed approaches are kept
below so nobody repeats them.*

Every worker turn calls `generate()` with the whole conversation, so turn N pays
prefill again for turns 1..N-1. This spec makes the worker carry one KV cache
across the turns of one task, anchored to a boundary that provably survives from
turn to turn, and restored to that boundary after every generation.

Every number below was measured on this machine, against **mlx-lm 0.31.3**,
**mlx-vlm 0.6.13**, and **mlx 0.32.0**. Models: `Qwen3.5-9B-MLX-4bit` for the
hybrid path (same `model_type: qwen3_5` and cache layout as the default
`Qwen3.8-27B-mxfp8`) and `Qwen3-0.6B-4bit` for the text-only path. These are
young internals; a dependency bump must re-verify them.

## The measurement that motivates this

Prefill is the binding cost, and its share grows with the conversation. Six
turns on the 9B, cold:

| turn | prompt tokens | wall | prefill | prefill share |
|---|---|---|---|---|
| 1 | 888 | 1.38s | 0.63s | 46% |
| 3 | 5,672 | 4.68s | 3.64s | 75% |
| 6 | 12,858 | 9.68s | 8.92s | **95%** |

Across the six turns prefill is **85%** of wall clock. Issue #27 was right about
where the time goes.

## What the architecture actually permits

The default model is a linear-attention hybrid. `make_cache` returns
`[ArraysCache(size=2) if l.is_linear else KVCache() ...]`
(`mlx_vlm/models/qwen3_5/language.py:2886`). The two layer kinds behave
oppositely, and conflating them is what sank the earlier draft:

| layers (27B) | state | size in sequence length | rewindable |
|---|---|---|---|
| 16 `KVCache` | attention K/V | **O(n)** — 768 MiB at 12k tokens, 16 GiB at the 262k window | yes, `trim` rewinds `offset` |
| 48 `ArraysCache` | Gated-DeltaNet recurrent state | **O(1)** — `mx.zeros((B, Hv, Dv, Dk))` | no |

The part that cannot be rewound is the part that does not grow. So a fork of a
hybrid cache costs a constant, not a copy of the KV. Measured snapshot sizes:

- `Qwen3-0.6B-4bit` (28 `KVCache`, no recurrent layers): **0.00 MiB** — 28 integers.
- `Qwen3.5-9B-MLX-4bit` (8 `KVCache`, 24 `ArraysCache`): **49.1 MiB**.
- `Qwen3.8-27B-mxfp8` (16 `KVCache`, 48 `ArraysCache`): **~147 MiB**, projected
  from `linear_num_value_heads=48`, `linear_value_head_dim=128`,
  `linear_key_head_dim=128`, fp32. Unverified; see Verification.

An `ArraysCache` carries no hidden live state — inspected as
`{'left_padding': None, 'lengths': None, 'cache': list of 2}`.

## The boundary that survives

sous re-renders the whole conversation through `apply_chat_template` every turn,
so the question is which render is stable across turns. Measured with the
tokenizer alone, no weights:

- **`add_generation_prompt=False` — the stable render — is a strict prefix every
  turn.** 866 → 906 → 946 → 986 → 1026, `strict_prefix=YES` throughout.
- **`add_generation_prompt=True` — the full prompt — never is.** The template
  appends a generation-only block (`<|im_start|>assistant\n<think>\n\n</think>\n\n`,
  7 tokens with `enable_thinking=False`; 5 with it `True`) that is absent when
  the same assistant turn is later re-rendered as history.

So the reusable anchor is the stable render, and the design keeps the cache
there. Note the divergence is not caused by `enable_thinking=False`: that flag
changes the block's length, not its presence.

## Goals

- A turn prefills only the tokens the conversation genuinely gained.
- Per-turn wall clock stops growing with conversation length.
- The saving is measurable per task, not once in a pull request.
- Reuse must not degrade output quality. **Not** token equality with a cold
  run — see Numerical characteristics.
- The snapshot rule is testable in CI, where the engine tests cannot run.
- A cache bug degrades to today's behaviour instead of failing a task.

## Non-goals

- Cross-task reuse. A cache belongs to one task and dies with it.
- Reuse across a model unload, a daemon restart, or a config change.
- Cache persistence to disk.
- Any change to elision policy, which stays "oldest elidable tool result first".
- Multimodal prompts. Text-only, as in v1.

## Core decisions

| Decision | Choice |
|---|---|
| Anchor | The **stable render** (`add_generation_prompt=False`), proven strict-prefix. |
| Restore mechanism | **Cheap snapshot**: record `offset` where `trim` exists, copy `state` where it does not. |
| Path split | By **`can_trim_prompt_cache()`**, not by backend — the property is architectural. |
| Cache ownership | **sous owns the cache list.** No `PromptCacheState`, no mRoPE priming. |
| Divergence response | Discard and rebuild. Elision is the only cause. |
| Cache location | Engine-held, task-scoped. `generate()` keeps its signature. |
| Kill switch | `[model] prompt_cache`, a boolean defaulting to `true`. |
| Measurement | `prompt_cache_stats()`, folded into every task report. |

## Why sous owns the cache

mlx-vlm primes Qwen mRoPE state before feeding a suffix
(`dispatch.py:595-633`), and I expected that to be mandatory. It is not, for
text-only prompts. Measured at two split points, one chunk-aligned and one not:

```
  primed  vs unprimed
    ArraysCache  tensors= 48 bit_exact= 48/48  max_abs=0.000e+00
    KVCache      tensors= 16 bit_exact= 16/16  max_abs=0.000e+00
```

Bit-identical. So the design passes `prompt_cache` plus `input_ids` directly and
never touches `PromptCacheState`. That removes the one fragile coupling the
earlier draft had at its centre, and it makes both backends the same shape.

## The per-turn algorithm

All of it lives inside `Engine.generate`, so `worker.py` still sees one call,
one thread, and one timeout. `_generate_with_timeout` needs no change.

```
stable_text = template(messages, add_generation_prompt=False)
stable_ids  = encode(stable_text)
full_ids    = encode(template(messages, add_generation_prompt=True))
suffix      = full_ids[len(stable_ids):]          # the 7-token generation block

reuse = reuse_length(held_ids, stable_ids)
if reuse == 0:
    cache, held_ids = fresh_cache(), []
else:
    restore(cache, snapshot)                       # back to the previous anchor
```

**Trimmable cache** (all `KVCache`; `can_trim_prompt_cache()` is `True`):
one generation call with `full_ids[reuse:]`, then `trim` the cache back to
`len(stable_ids)`. No snapshot is taken and none is needed.

**Non-trimmable cache** (any recurrent layer), two calls:

1. Prefill `stable_ids[reuse:]` with no decode, advancing the cache to the anchor.
2. Take the snapshot at the anchor.
3. Feed `suffix` and generate.
4. Restore, returning the cache to the anchor exactly.

Then `held_ids = stable_ids` either way.

Prefill-only in step 1 is `max_tokens=0` on the mlx-vlm path. `dispatch.py` has
an explicit `if not generated_tokens:` branch that yields a result and returns
without updating any state, so this is deliberate behaviour rather than an edge
case. On the mlx-lm path, `stream_generate` would raise on `max_tokens=0`
(`token` is unbound when its loop never runs), but the trimmable branch never
needs a prefill-only call.

### The reuse rule

```python
def reuse_length(cached_ids: Sequence[int], new_ids: Sequence[int]) -> int:
    """len(cached_ids) when cached_ids strictly prefixes new_ids, else 0."""
    if not cached_ids or len(new_ids) <= len(cached_ids):
        return 0
    for a, b in zip(cached_ids, new_ids):
        if a != b:
            return 0
    return len(cached_ids)
```

### The snapshot rule

```python
def snapshot(cache):
    """Offsets for layers that can rewind; state copies for layers that cannot."""
    out = []
    for c in cache:
        if hasattr(c, "trim"):
            out.append(("trim", int(getattr(c, "offset", 0) or 0)))
        else:
            out.append(("state", [None if a is None else mx.array(a) for a in c.state]))
    return out
```

`restore` trims back to the recorded offset and writes the copied state back.
Both rules are pure branching on `hasattr(c, "trim")`, so a fake cache object
tests them with no mlx present. That matters: the engine tests are
`model`-marked and never run in CI.

## Measured result

The design run against cold, identical prompts, six turns on the 9B:

| turn | prompt | cold | design | faster |
|---|---|---|---|---|
| 1 | 888 | 1.38s | 1.47s | −6% |
| 2 | 3,269 | 3.24s | 2.75s | 15% |
| 3 | 5,672 | 4.68s | 2.84s | 39% |
| 4 | 8,076 | 6.01s | 2.66s | 56% |
| 5 | 10,467 | 7.78s | 2.76s | 65% |
| 6 | 12,858 | 9.68s | **2.83s** | **71%** |

Total 32.77s → 15.30s, **53% saved**, with reuse of 12,851 of 12,858 tokens on
turn 6. Per-turn wall clock goes flat, so the gap widens with task length —
which is what the issue asked for. Turn 1 pays 6% for the extra call and the
snapshot; that is the fixed overhead.

Snapshot and restore are exact. After contaminating a 2,475-token cache with 63
tokens and restoring:

```
restored vs cold  : max_abs_diff=0.000e+00  exact 64/64 tensors  -> BIT-EXACT
```

The same on the text-only path: 56/56 tensors, bit-exact.

## Numerical characteristics

Two facts belong in the record, because one of them was a bad goal in the
earlier draft.

**sous output is already nondeterministic.** The worker samples at
`temperature=0.7, top_p=0.8, top_k=20`. Token-level equality with a cold run was
never a property sous had, so requiring it of reuse is meaningless. The goal is
that reuse must not degrade quality, and the test is a real delegated task that
still completes.

**Incremental prefill is not bit-identical to single-shot prefill on the hybrid
path.** The Gated-DeltaNet prefill is a chunked scan with `C=64` and zero
padding to a multiple of 64 (`models/qwen3_5/gated_delta.py:288`), whose own
docstring claims `rel-err < 1e-3 (fp32)` against the sequential reference.
Different chunk boundaries therefore give different rounding, and the attention
layers inherit it because they are interleaved with the recurrent ones. This is
inherent to any incremental prefill on this architecture, including mlx-vlm's own
chat, and cannot be designed away — the only way to avoid it is to have no
feature. Under greedy decoding the design matched cold on 4 of 6 turns. The
magnitude is **not** characterised: the statistics I gathered were max-only and
the behavioural probe was degenerate. Verification must characterise it properly
before this ships.

## Approaches that failed, with evidence

Recorded so they are not retried.

**Strict-prefix reuse on the full generation prompt: 0 hits in 6 turns.** The
template appends a generation-only think block and strips think blocks when
re-rendering history, so turn N's prompt is never a prefix of turn N+1's. 873 of
891 cached tokens were a valid prefix and none was usable, because trimming a
hybrid cache is impossible. Four attempts to make the re-render match — strip the
EOS, re-insert the think block, both, neither — produced byte-identical results,
proving the template normalises assistant content rather than passing it through.

**mlx-vlm's APC exact mode: works, then freezes.** It is built for this
architecture, and its docstring says so. But `dispatch.py` stores a snapshot only
when `reused_prefix_len == 0`, so after the first hit no fresher one is ever
stored. Reuse stayed pinned at 872 tokens while the prompt grew to 12,858: 27% →
7% of the prompt, 12% overall, **decaying** with task length. Verified: 5 exact
hits, `exact_stores=2`.

**Hand-driving `PromptCacheState`: unnecessary.** See Why sous owns the cache.

## Cache lifetime, invalidation, and the stall path

`run_task` calls `reset_prompt_cache()` at task start, so no prefix survives from
a previous task, and again in a `finally`, so a finished task leaves nothing
resident.

**A generation that raises.** The cache is mutated in place, so each engine
invalidates its carrier *before* it generates and restores it only after
success. A raise therefore never leaves a cache whose token record understates
its contents.

**An abandoned stalled generation.** `_generate_with_timeout` abandons a wedged
thread rather than killing it, and that thread reaches the write-back after
`run_task` moved on. A monotonic epoch closes the race: `reset_prompt_cache`
bumps it, and a late write whose captured epoch is stale drops itself. This is
what lets `reset_prompt_cache` stay lock-free, which is what keeps it from
wedging behind a stalled generation — `ManagedEngine._gen_lock` is still held by
the abandoned thread.

Elision gets no special case. The engine's prefix check is the single authority
on divergence. `run_task` counts elisions for the report and nothing more.

## Degrade, never break

A warm generation that raises retries once with a fresh cache, warns, and counts
the retry. `decide_context` already sets this rule for auto sizing: an
optimization bug must never fail a task. The retry fires only when a cache was
in use, so a genuine engine error still surfaces at once. It can push a turn past
its deadline, which `_generate_with_timeout` reports as a stall or as budget
exhaustion — the existing shape for any slow generation.

## The prompt memo

`count_tokens` and `generate` both build and tokenize the prompt. Each engine
gains a one-slot memo keyed by the exact prompt string, cleared by
`reset_prompt_cache`. Correctness rests on string equality alone, never on call
order, so a stale slot cannot yield wrong ids even if an abandoned thread
overwrites it. It does not help inside `_elide_if_needed`'s loop, where every
iteration has a different prompt.

The design needs both renders per turn, so the memo holds two entries: the
stable render and the full prompt.

## Configuration

`[model] prompt_cache`, a boolean defaulting to `true`.

- `SousConfig.prompt_cache: bool = True`
- `_KNOWN["model"]` gains `"prompt_cache"`
- `from_toml` reads `model.get("prompt_cache", True)`
- `_default_factory` and both engine constructors take it
- `EngineManager`'s factory lambda passes `config.prompt_cache`

`false` restores today's behaviour with no downgrade and no code change.

## Memory accounting

The working KV cache costs 64 KiB per token on the default model
(2 × 16 full-attention layers × 4 KV heads × 256 head_dim × 2 bytes). The
snapshot adds a fixed ~147 MiB, independent of context length — negligible
beside a working cache that reaches 16 GiB at the 262k window. There is no
doubling.

Peak memory does not move: a generation already held the whole cache. Residency
tracks the tokens a task actually uses, not the window, since the window is a cap
and not an allocation. What changes is duration — the cache now stays resident
across tool calls, so a `run_command` subprocess competes with it where it
previously did not. `[context] fraction` therefore stops bounding a peak and
starts bounding sustained residency, and the config documentation must say so.

`decide_context` runs once per task before any cache exists, so nothing
double-counts. The module docstring in `src/sous/context.py` says the KV cache is
transient per generation. That sentence becomes wrong and this change must
rewrite it.

## Interaction with auto context sizing (issue #25)

Elision fires only when the prompt exceeds the window, and elision is the only
thing that discards the cache. So a window the task never reaches means the cache
never dies, and every turn after the first reuses it. Auto sizing does not soften
the cost of cache loss — it removes the cause.

The regime to avoid is the opposite one: near the cap, each turn pushes back over
and elision fires repeatedly, so reuse dies every turn while the task still pays
full prefill. Today's default is `context_mode = "fixed"` at 32,768 tokens, so
the shipped default lands in the weaker regime. The README and config
documentation must state that reuse pays most in `auto` mode. This spec does not
change the default, which belongs to issue #25.

## Observability

`prompt_cache_stats()` returns `hits`, `misses`, `reused_tokens`,
`snapshot_bytes`, and `cold_retries`. `run_task` adds its own `elisions` count
and folds the merged block into the task report under `prompt_cache`, beside
`budget`. `_failure_extra` carries the same block, so a failed or cancelled task
still reports what reuse did. The elision count is what makes a miss legible:
hits and misses say reuse failed, elisions say why.

## Testing strategy

CI cannot load a model, so the model-free tests carry the weight.

- **`tests/test_promptcache.py`** — `reuse_length` against an empty cache, an
  exact match, a strict extension, divergence at position 0, divergence in the
  middle, and a cache longer than the new prompt. Then `snapshot`/`restore`
  against fake layer objects: one with `trim` and an `offset`, one without and
  with a `state` list. Both rules branch only on `hasattr(c, "trim")`, so no mlx
  is needed.
- **Engine internals without mlx** — build engines with `object.__new__`, the
  pattern `tests/test_engine_unloaded.py` already uses, and cover the two-entry
  memo, the invalidate-before-generate step, and the epoch guard dropping a
  stale write-back.
- **`tests/test_engine_base.py`** — `ManagedEngine` forwards both new methods,
  and `reset_prompt_cache` returns promptly while another thread holds
  `_gen_lock`.
- **`tests/test_worker.py`** — `run_task` resets at start and in the `finally`;
  the report carries the `prompt_cache` block; `_failure_extra` carries it too;
  the elision count appears when the window forces elision.
- **`tests/test_config.py`** — the new flag's default, an override, and an
  unaffected unknown-key warning.
- **`model`-marked** — a bit-exactness test on both paths: prefill, snapshot,
  contaminate, restore, and compare tensor-by-tensor against a cold prefill of
  the same prefix. This is the test that would have caught the earlier draft's
  error, and it must assert `max_abs_diff == 0`.

## Verification before this ships

Three items, all local-only, none of which CI can do.

1. **The 27B.** Confirm the projected ~147 MiB snapshot, and re-measure the
   wall-clock table. The mechanism is architectural and will hold; the ratios
   will differ.
2. **Characterise the incremental-prefill difference properly.** Distribution
   rather than maxima, and a non-degenerate behavioural comparison — not a
   next-token probe on an already-complete render. This is the one number the
   spec currently declines to state.
3. **One real delegated task on the default model**, per the mlx thread-state
   rule in CLAUDE.md. Confirms the task still completes, and that no new thread
   leaks mlx state.

`scripts/e2e_smoke.py` gains a second turn and prints the `prompt_cache` block,
which is the reproducible before-and-after the issue asks for.

## Future work

- **Upstream: refresh APC's exact snapshot on every turn, not only on cold
  turns.** A small change in `dispatch.py` that would make APC an alternative to
  this design rather than a decaying one.
- **Upstream: guard the trim in `dispatch.py:857-859`.** It calls `c.trim(n_drop)`
  without consulting `is_trimmable()`, so any linear-attention hybrid raises
  `AttributeError` there on a diverged prefix. Reproduced on the 9B. mlx-lm's
  `trim_prompt_cache` guards the same operation, so the fix has a sibling
  precedent. This design never triggers it, because it owns the cache directly.
- **Cheaper elision.** `_elide_if_needed` tokenizes once per iteration. An
  estimate-then-confirm pass would cut that, independent of this work.
