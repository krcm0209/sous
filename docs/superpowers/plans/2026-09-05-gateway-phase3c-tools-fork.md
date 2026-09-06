# Gateway Phase 3c (Tools-Boundary Fork) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first locally served turn of a *new* Claude Code session start warm from a fork slot taken at the end of the rendered tool block, so it prefills ~3–5K tokens instead of ~57K (~168 s → 10–35 s on the M5 Pro), while same-session behaviour (the header fork, the turn slots) is unchanged.

**Architecture:** The engines' single-boundary header probe becomes a two-pair probe (`promptcache.probe_boundaries`) that returns an ascending list of boundaries: the *tools* boundary (common prefix of two renders that differ at the first character of the system text — the tool block, on a template that renders tools before the client's text, as Qwen3.5/3.8 do) and the existing *header* boundary (common prefix of two renders that differ in the first user turn). `PrefixCache._fork_wanted` becomes `_fork_boundaries`, which resolves the probe on warm turns too and returns every boundary this turn is itself prefilling past; `_run` walks the list ascending, prefilling to each boundary and publishing a `fork` copy there, charging each copy to the budget before allocating it and protecting this turn's earlier copies from being evicted to make room for later ones. Lookup, ownership, turn slots, the budget and pressure eviction are untouched. Everything decisive stays in `promptcache.py`, mlx-free and exercised by fakes.

**Tech Stack:** Python 3.14, uv, mlx 0.32.2 / mlx-lm / mlx-vlm 0.6.16 (only in model-marked tests), pytest, ty, ruff. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-05-gateway-phase3c-tools-fork-design.md` — read it first; this plan argues from it. Its "What Phase 3a got wrong" section is the correction this change ships; its Non-goals (the `lcp=` counter, `speculative_block_size`, `cache_read_input_tokens`, anchors, SSD, checkpoints) are PR 2 and later — do not add them here. Background: the Phase 3a plan (`docs/superpowers/plans/2026-09-04-gateway-phase3a-keyed-cache.md`) records the slot/fork/budget design this extends; its decision 2 ("the fork boundary is the header") is what this plan supersedes.

## Global Constraints

- Python >= 3.14; everything runs via `uv run`. Never pip. `except A, B:` without parentheses is valid 3.14 syntax (PEP 758) — don't "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`. `uv run ty check` covers tests and scripts too.
- **mlx / mlx_lm / mlx_vlm imports stay function-local** in the engines (absent on non-macOS; the lint job runs on ubuntu). `promptcache.py` imports no mlx at all — every decision there is exercised by fakes in CI. The `os` import it gains is stdlib.
- **Any thread that touches mlx calls `sous.engine.base.release_mlx_thread_state()` before it exits** (ml-explore/mlx#4327); CI cannot catch a violation, so the runtime gate (Task 6) includes one real delegated task. Nothing in this plan adds a thread.
- **Cache arrays are usable only from the thread whose streams built them** (#34). Slots are owned by the building thread and looked up only by it — unchanged. The gateway keeps one `GenerationSession` across every turn it serves (`gateway/turn.py`, `_session_for`), which is why forks it publishes are visible to later Claude Code sessions with no ownership change.
- **Never rewind a cache to make a slot.** A fork is a copy taken *while* prefilling past a boundary; boundaries are visited ascending so no copy is ever wanted behind a prefix already prefilled.
- **A copy is charged to the budget before it is allocated.** A fork slot is never evicted by its own publish, nor is a cold turn's first fork copy evicted to make room for (or by the publish of) its second. `max_bytes <= 0` means no forks at all and no probe resolved (the pre-3a single-slot behaviour).
- **Nothing here can fail a turn.** A probe that raises → warning, no boundaries. A boundary that is not a strict prefix, is below `FORK_MIN_TOKENS` (4096), or leaves nothing to prefill → dropped. A copy that fails → warning, skipped (per boundary). A publish refused by a reset or a retired owner → the copy is dropped.
- **The gateway still never rewrites, reorders or logs the client's prompt.** No new log fields; the existing `cache=fork reused_tokens=N` line and `stats.forks` are the observability.
- Engine `on_delta` callbacks fire on the generation thread inside the decode loop — never block or raise in one (untouched here). `PrefixCache` refuses its cold retry once any non-`ReplaySafe` delta has reached the client — untouched.
- Tests never touch the real `~/.sous`. `model`-marked tests download weights and run locally only (Task 4 and the gate); `slow`-marked tests are run, not skipped.
- `docs/superpowers/**` are point-in-time records: never edit an existing spec or plan (this plan is a new file). The spec this plan implements is already committed (83f04d9).
- Conventional Commits, imperative lowercase subject, *why* in the body. Commit trailer: exactly `Co-Authored-By: Claude <noreply@anthropic.com>` (model-less, per the user's global CLAUDE.md; that instruction overrides any harness default). Branch `feat/gateway-phase3c` (exists, holds the spec commit); `main` is protected. CI is exactly: `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`. Every commit step runs `uv run ruff format .` *before* `uv run ruff check .`.
- Comments explain non-obvious *why*; never restate what code does.

---

## Decisions this plan locks (beyond the spec)

Settled while writing the plan against the installed templates and the current code; implementers should not reopen them.

1. **`fork_at` takes a sequence or a callable returning one — no bare `int`.** `PrefixCache.generate(..., fork_at: Sequence[int] | Callable[[], Sequence[int]] = ())`. Tests that passed `fork_at=FORK` pass `fork_at=[FORK]`. Two accepted shapes instead of three keeps `_fork_boundaries` a straight `isinstance(fork_at, Sequence)` split, which is the narrowing ty follows (the prior code and tests already avoided `callable()` for that reason).
2. **The probe pairs live in `promptcache.py` as `probe_boundaries(render, encode, memo, system, stable_ids)`.** Both engines had a near-identical closure; with two pairs it would be ~25 duplicated lines each, and the rule (which pairs, which memo slot, dedupe, `fork_point` per candidate) is a cache decision that belongs with the other cache decisions where fakes reach it. The engines keep a thin `_fork_probe` (the guards, the tokenize lock, the closure).
3. **The tools pair is `[system("0"), user("0")]` vs `[system("1"), user("0")]`, rendered with the turn's tools; the header pair is `[messages[0], user("0")]` vs `[messages[0], user("1")]`** (unchanged). Both pairs are rendered and encoded under the engine's `_tokenize_lock`, in one closure, on every turn `_fork_boundaries` decides to resolve. Memo slot `"tools"` for the tools text, `"header"` for the header text (`_MEMO_SLOTS` grows to four); the texts are memoized by content, so a warm turn of a known tool set pays two renders and a string compare, not an encode.
4. **Which templates yield a tools boundary — verified against the snapshots in the HF cache:** `mlx-community/Qwen3.8-27B-4bit` (the default) and `mlx-community/Qwen3.5-9B-MLX-4bit` (`HYBRID_VLM`) render `# Tools … </IMPORTANT>` first, then `\n\n` + the trimmed system text *only when the text is non-empty*, so the tools boundary ends at `</IMPORTANT>\n\n`. `mlx-community/Qwen3-0.6B-4bit` (`TINY`) renders the system text first and the tool block after it, so its tools pair shares only `<|im_start|>system\n` and the header is its only boundary. `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`TINY_VLM`) has no tools handling at all. The model tests assert exactly these facts per model.
5. **A request whose system text is empty gets no tools boundary on the default template, and that needs no special case:** with empty text the template emits no `\n\n`, so the tools text (block + `\n\n`) is not a prefix of the render and `fork_point` returns 0. The fake tokenizer in `tests/test_engine_forkpoint.py` reproduces this shape (separator only when there is text) so the rule is tested without a model.
6. **`_fork_boundaries` also enforces the floor** (`b >= FORK_MIN_TOKENS`) in addition to `reuse < b < len(stable_ids)`. The engine's `fork_point` already does; repeating the cheap check here means the list form used by tests cannot smuggle a boundary the callable form would refuse.
7. **The existing-fork filter compares whole id lists**, once per candidate boundary: `stable_ids[:b] not in [s.held for owner's fork slots]`. Two slices per turn, C-speed list compares — the same cost class as `_take`.
8. **`protect` becomes a sequence of slots compared by identity** (`not any(s is p for p in protect)`), never by `==`: `Slot` is a dataclass, and equality would compare two ~50K-token lists and two caches per candidate. `_evictable`, `_drop_lru`, `_evict_caps`, `_make_room`, `_publish` and `_evict_pressure` all take it; the pre-turn `_evict_caps` passes `(slot,)` for a fork hit and `()` otherwise; `_publish` protects `(slot, *protect)`.
9. **`_make_room` returns whether the copy now fits, and the turn skips a copy it cannot make room for without evicting a protected slot.** This is the spec's "no-doubling" rule made precise: a budget with room for exactly one fork copy publishes the tools fork (the one every later session can use) and never allocates the header copy. The old `price <= self.max_bytes` pre-check is subsumed (on an empty map the two conditions coincide) and removed.
10. **The turn-slot publish at the end of `generate` protects only itself, as today.** A same-turn fork is fresher than any older slot, so LRU already spares it unless the map holds nothing else — and then the budget cannot hold the fork plus the turn slot, and dropping the fork *now* is exactly what the next turn's pre-turn cap pass would do anyway (the turn slot serves this conversation's next turn, worth ~168 s; the fork serves a hypothetical later session). Protecting it there would only leave the map over budget for one turn boundary. The spec's invariant is therefore implemented as: *a fork copy is never evicted by a later fork copy's charge or publish in the same turn*.
11. **The list of this turn's published forks is dropped after the boundary loop**, before decode. It exists only to protect the copies from each other while they are charged; holding it across the decode would pin a fork that pressure or a reset evicted mid-turn until the turn ended.
12. **The probe-failure warning reads `fork probe failed`** (was `header probe failed`); two tests' `match=` strings change with it. The `fork copy failed` and `fork clone failed` warnings are unchanged.
13. **Model tests use a deterministic sampler for the token-for-token check** (`temperature=0.0`) and compare a warm run against a cold run *of the same engine after `reset_prompt_cache()`*, so both runs split the prefill at the same boundaries and the comparison tests the fork, not bf16 ULP drift between prefill shapes (the Phase 3a measurement: a 440-token prefill and a 220+220 one differ by one ULP from layer one).
14. **The runtime gate (Task 6) is Kyle's machine only** and includes the one real delegated task the mlx-thread rule demands after any change on the generation path. Posting the Phase 3c note on issue #41 is outward-facing: draft it in the task, post only after Kyle confirms the text.

---

## File Structure

Modified:

| File | Change |
|---|---|
| `src/sous/engine/promptcache.py` | `import os`; `_MEMO_SLOTS` gains `"tools"`; `_HEADER_PROBES`, `_TOOLS_PROBES`, `probe_boundaries()`; `PrefixCache.generate` `fork_at` type; `_fork_wanted` → `_fork_boundaries`; `_run` walks a list with a `published` protect list; `_evictable`/`_drop_lru`/`_evict_caps`/`_make_room`/`_publish`/`_evict_pressure` take `protect: Sequence[Slot]`; `_make_room -> bool`. |
| `src/sous/engine/vlm.py` | `_header_probe` → `_fork_probe` returning `Callable[[], list[int]] | list[int]`, body delegates to `probe_boundaries`; `_PROBES`, `os`, `fork_point` import removed; docstrings rewritten. |
| `src/sous/engine/lm.py` | Same as `vlm.py`, plus the `_ids` docstring's "third slot" sentence. |
| `tests/test_promptcache.py` | `fork_at=[FORK]` everywhere; `_probe` returns a list; two probe-resolution tests flipped (warm turns resolve); new section "forks at more than one boundary" (Task 1) and its budget/protect tests (Task 2); `test_memo_accepts_the_tools_slot`. |
| `tests/test_engine_forkpoint.py` | Fake tokenizer renders tools first (with the separator rule) and the system *content*; `Recording` resolves through `_fork_boundaries` and records lists; `SystemFirst` fake; tests for the tools boundary, no-tools, empty text, second session memo, both backends. |
| `tests/test_engine_lm.py` (model) | `_header_probe` → `_fork_probe` in the existing probe test; new `_fat_tools`; new system-first-template test. |
| `tests/test_engine_vlm.py` (model) | New `_fat_tools`; tools-boundary probe test on `HYBRID_VLM`; two-boundary bit-exact fork test; second-session end-to-end test; token-for-token warm-vs-cold test. |
| `README.md` | Gateway bullet (~lines 236–243), `cache=` log sentence (~line 253), `prompt_cache_gb` paragraph (~lines 368–382), the one-line summary (~line 520). |
| `CLAUDE.md` | The prompt-cache gotcha's fork sentence. |

Not modified: `gateway/*` (the log line and `TurnResult.forked` already distinguish the cases), `config.py`, `base.py`, `manager.py`, `worker.py`.

---

## Task 1: `PrefixCache` forks at a list of boundaries

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`generate` signature ~line 561; `_fork_wanted` ~lines 419–478 → `_fork_boundaries`; `_run` ~lines 744–800)
- Modify: `src/sous/engine/vlm.py:259-303`, `src/sous/engine/lm.py:194-238` (minimal: the probe returns a one-element list)
- Modify: `tests/test_promptcache.py` (~lines 1147–1400, the forks section)
- Modify: `tests/test_engine_forkpoint.py` (`Recording`, expected values — minimal; Task 3 rewrites the file)

**Interfaces:**
- Consumes: `fork_point`, `slot_bytes`, `fork_copy`, `Slot`, `FORK_MIN_TOKENS` (all existing in `promptcache.py`).
- Produces: `PrefixCache.generate(stable_ids, full_ids, max_tokens, on_delta=None, fork_at: Sequence[int] | Callable[[], Sequence[int]] = ()) -> str`; `PrefixCache._fork_boundaries(owner: threading.Thread, stable_ids: list[int], fork_at: Sequence[int] | Callable[[], Sequence[int]], reuse: int) -> list[int]`; the warning text `sous prompt cache: fork probe failed (<ExcType>); not forking this turn`. Task 2 changes `_run`'s protect handling; Task 3 makes the engines return real lists.

- [ ] **Step 1: Mechanically move the existing tests to the list form**

```bash
cd /Users/kyle.krcmaric/code/personal/sous
sed -i '' 's/fork_at=FORK)/fork_at=[FORK])/g; s/fork_at=len(C1) + 5)/fork_at=[len(C1) + 5])/' tests/test_promptcache.py
grep -n "fork_at=" tests/test_promptcache.py | grep -v "fork_at=\[" | grep -v "fork_at=probe\|fork_at=_probe\|fork_at=0\b" || true
```

Expected: the grep prints nothing (every literal `fork_at=` is now a list or a callable). Also apply these three edits by hand in `tests/test_promptcache.py`:

In `_probe` (~line 1335):

```python
def _probe(value: int, calls: list[int]):
    """A fork probe like the engines': it costs renders and a tokenize, so the
    count of calls is what these tests are about."""

    def probe() -> list[int]:
        calls.append(value)
        return [value]

    return probe
```

Replace `test_a_callable_fork_at_is_not_resolved_when_a_turn_slot_serves_the_turn` and `test_a_callable_fork_at_is_not_resolved_on_a_fork_hit` with:

```python
def test_a_callable_fork_at_is_resolved_on_a_turn_hit_and_forks_nothing():
    """Warm turns resolve the probe now (see _fork_boundaries): with two
    boundaries, a turn that started at another session's tools fork has to
    learn its own header. A turn-slot hit is past every boundary, so the
    answer is dropped whole."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=[FORK])
    calls: list[int] = []
    pc.generate(C1_NEXT, C1_NEXT_FULL, 16, fork_at=_probe(FORK, calls))
    assert calls == [FORK]
    assert pc.stats()["forks"] == 1
    assert h.prefilled == [H]  # only the cold turn ever stopped at the boundary


def test_a_callable_fork_at_is_resolved_on_a_fork_hit_and_forks_nothing_new():
    # The header is already inside this turn's cache and no layer rewinds to
    # it: the one boundary the probe answers is `reuse` itself, and is dropped.
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=[FORK])
    calls: list[int] = []
    pc.generate(C2, C2_FULL, 16, fork_at=_probe(FORK, calls))
    assert pc.stats()["fork_hits"] == 1
    assert calls == [FORK]
    assert pc.stats()["forks"] == 1
```

In `test_a_failing_header_probe_warns_and_leaves_the_turn_alone` (~line 1385): rename to `test_a_failing_fork_probe_warns_and_leaves_the_turn_alone`, change the probe's return annotation to `-> list[int]`, and change `match="header probe"` to `match="fork probe"`.

- [ ] **Step 2: Add the new unit tests (multi-boundary behaviour)**

Append to `tests/test_promptcache.py`, after `test_a_failing_fork_probe_warns_and_leaves_the_turn_alone` and before the `# ---- budget ----` section:

```python
# ---- forks at more than one boundary ----------------------------------------
#
# What Claude Code sessions look like to the gateway once the template renders
# the tool block before the client's system text: every session presenting one
# tool set shares the tool block T; each session adds its own system text (HA,
# HB — they differ at a per-session path); each subagent adds its brief.
T = list(range(1, FORK_MIN_TOKENS + 1))  # the tool block: one tool set
TOOLS_AT = len(T)
HA = [*T, 8001, 8002]  # session A's header: the tools, then its system text
HB = [*T, 8101, 8102, 8103]  # session B's header: a different system text
BOUNDS_A = [TOOLS_AT, len(HA)]
BOUNDS_B = [TOOLS_AT, len(HB)]
AX1, AX1_FULL = [*HA, 501, 502], [*HA, 501, 502, 90, 91]  # session A, subagent 1, turn 1
AX1_NEXT, AX1_NEXT_FULL = [*AX1, 503], [*AX1, 503, 90, 91]  # ... its turn 2
AX2, AX2_FULL = [*HA, 601], [*HA, 601, 90, 91]  # session A, subagent 2
BX1, BX1_FULL = [*HB, 701, 702], [*HB, 701, 702, 90, 91]  # session B, subagent 1
BX2, BX2_FULL = [*HB, 801], [*HB, 801, 90, 91]  # session B, subagent 2


@pytest.mark.parametrize("trimmable", [True, False])
def test_a_cold_turn_forks_at_every_boundary_ascending(trimmable):
    h = FakeHooks(trimmable=trimmable)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    kinds = sorted((s.kind, s.held) for s in pc.slots())
    assert kinds == [("fork", T), ("fork", HA), ("turn", AX1)]
    assert pc.stats()["forks"] == 2
    # One prefill segment per boundary, in order, whatever the layer kinds:
    # each copy has to be taken exactly at its boundary.
    assert h.prefilled[:2] == [T, HA[TOOLS_AT:]]
    if trimmable:
        assert h.decoded[0] == AX1_FULL[len(HA) :]  # the rest still fuses into decode
    else:
        assert h.prefilled[2] == AX1[len(HA) :]


def test_a_turn_warm_at_the_tools_fork_still_forks_at_its_own_header():
    """Session B's first turn starts from session A's tools fork. Under the
    one-boundary rule ("a warm turn never forks") it would publish nothing and
    B's second subagent would reuse the tool block instead of B's whole
    header. It must publish B's header fork."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    s = pc.stats()
    assert (s["fork_hits"], s["forks"]) == (1, 3)
    assert sorted(x.held for x in pc.slots() if x.kind == "fork") == sorted([T, HA, HB])
    # B prefilled only its own system text before the copy, then decoded its brief.
    assert h.prefilled[-1] == HB[TOOLS_AT:]
    assert h.decoded[-1] == BX1_FULL[len(HB) :]


def test_a_turn_warm_at_the_header_fork_forks_nothing():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    pc.generate(AX2, AX2_FULL, 16, fork_at=BOUNDS_A)  # same session: header-fork hit
    pc.generate(AX1_NEXT, AX1_NEXT_FULL, 16, fork_at=BOUNDS_A)  # turn-slot hit
    s = pc.stats()
    assert (s["hits"], s["fork_hits"], s["forks"]) == (2, 1, 2)
    assert h.decoded[-2] == AX2_FULL[len(HA) :]
    assert h.decoded[-1] == AX1_NEXT_FULL[len(AX1) :]


def test_a_second_session_starts_from_the_tools_fork_and_its_next_subagent_from_its_header():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    before = pc.stats()["reused_tokens"]
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    assert pc.stats()["reused_tokens"] - before == TOOLS_AT
    before = pc.stats()["reused_tokens"]
    pc.generate(BX2, BX2_FULL, 16, fork_at=BOUNDS_B)
    assert pc.stats()["reused_tokens"] - before == len(HB)
    assert pc.stats()["fork_hits"] == 2


def test_equal_out_of_range_and_short_boundaries_are_dropped():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    given = [len(HA), TOOLS_AT, TOOLS_AT, 0, 10, len(AX1), len(AX1) + 5, -1]
    pc.generate(AX1, AX1_FULL, 16, fork_at=given)
    kinds = sorted((s.kind, s.held) for s in pc.slots())
    assert kinds == [("fork", T), ("fork", HA), ("turn", AX1)]
    assert h.prefilled[:2] == [T, HA[TOOLS_AT:]]  # ascending, whatever order was given


def test_the_probe_is_resolved_on_warm_turns_too():
    """Two boundaries make the warm-turn probe worth its renders: it is how a
    session whose first turn started at another session's tools fork gets a
    header fork of its own."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    calls: list[list[int]] = []

    def probe(bounds: list[int]):
        def run() -> list[int]:
            calls.append(bounds)
            return bounds

        return run

    pc.generate(AX1, AX1_FULL, 16, fork_at=probe(BOUNDS_A))  # cold
    pc.generate(BX1, BX1_FULL, 16, fork_at=probe(BOUNDS_B))  # warm at the tools fork
    pc.generate(AX1_NEXT, AX1_NEXT_FULL, 16, fork_at=probe(BOUNDS_A))  # warm at a turn slot
    assert calls == [BOUNDS_A, BOUNDS_B, BOUNDS_A]
    assert pc.stats()["forks"] == 3
```

- [ ] **Step 3: Run the forks tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -k "fork or probe or boundar" -q 2>&1 | tail -15`

Expected: FAIL — `TypeError` (`_fork_wanted` receives a list where it expects `int | Callable`), `AttributeError: 'PrefixCache' object has no attribute '_fork_boundaries'` is not yet hit because nothing calls it; the new multi-boundary tests fail on `forks == 2` etc.

- [ ] **Step 4: Change `generate`'s signature and docstring**

In `src/sous/engine/promptcache.py`, replace the `generate` signature and docstring (~lines 554–566):

```python
    def generate(
        self,
        stable_ids: list[int],
        full_ids: list[int],
        max_tokens: int,
        on_delta: OnDelta | None = None,
        fork_at: Sequence[int] | Callable[[], Sequence[int]] = (),
    ) -> str:
        """`fork_at` names the boundaries this turn may leave a fork slot at —
        token counts into `stable_ids`, in any order — or a callable that
        computes them. The callable form is the engines' probe: it renders and
        tokenizes, so it is deferred until `_fork_boundaries` knows there is a
        budget to fork into at all."""
```

Also update the two `_run` parameter annotations (`fork_at: int | Callable[[], int]` → `fork_at: Sequence[int] | Callable[[], Sequence[int]]`).

- [ ] **Step 5: Replace `_fork_wanted` with `_fork_boundaries`**

Replace the whole `_fork_wanted` method (~lines 419–478) with:

```python
    def _fork_boundaries(
        self,
        owner: threading.Thread,
        stable_ids: list[int],
        fork_at: Sequence[int] | Callable[[], Sequence[int]],
        reuse: int,
    ) -> list[int]:
        """The boundaries this turn will stop at to leave a fork slot,
        ascending: those it is itself prefilling past (`reuse < b` — a hit
        that started at or beyond one holds it inside a cache that cannot
        rewind), that leave something to prefill after them
        (`b < len(stable_ids)`), that clear the fork floor, and for which this
        owner holds no fork with exactly those ids yet. Empty when there is no
        budget to charge a copy to — and then the probe is not even resolved,
        so at `max_bytes == 0` (the pre-3a single-slot behaviour) nothing about
        the turn changes.

        The probe is resolved on warm turns too. With one boundary that was
        waste — a turn that started at or past it could not fork there — but
        with two it is the point: a session's first turn starts warm at
        another session's tools fork (`reuse` is the tools boundary) and must
        still publish its own header fork, or that session's next subagent
        reuses ~45K tokens instead of ~57K. The memo makes a warm probe two
        renders and a string compare; the encode it saves is the cost.

        Resolved before the lock is taken: the probe tokenizes, and nothing
        that slow may run under the bookkeeping lock that reset() promises
        never to wait behind. Takes the lock itself, for the scan.
        """
        if self.max_bytes <= 0:
            return []
        if isinstance(fork_at, Sequence):
            candidates = list(fork_at)
        else:
            try:
                candidates = list(fork_at())
            except Exception as e:
                # An optimization must never fail a turn — the same rule the
                # warm-retry path follows. A chat template is free to refuse
                # a probe conversation; the turn just does not fork.
                warnings.warn(
                    f"sous prompt cache: fork probe failed ({type(e).__name__}); "
                    "not forking this turn",
                    stacklevel=2,
                )
                return []
        wanted = sorted(
            {b for b in candidates if b >= FORK_MIN_TOKENS and reuse < b < len(stable_ids)}
        )
        if not wanted:
            return []
        with self._lock:
            forks = [s.held for s in self._slots if s.owner is owner and s.kind == "fork"]
        return [b for b in wanted if stable_ids[:b] not in forks]
```

ty narrows the `else` branch to the callable (verified against this exact union while writing the plan) — do not restructure around `callable()`, which ty does not narrow on.

- [ ] **Step 6: Make `_run` walk the list**

Replace the fork block in `_run` (from `# A separate name, not fork_at reassigned` through the `finally: copy = None` — ~lines 758–800) with:

```python
        for boundary in self._fork_boundaries(owner, stable_ids, fork_at, reuse):
            # Stop at the boundary and continue from there: a fork is taken
            # while prefilling past it because no layer can be rewound to it
            # afterwards. Ascending order means no copy is ever wanted behind
            # a prefix already prefilled.
            hooks.prefill(cache, list(stable_ids[reuse:boundary]))
            reuse = boundary
            # What the cache holds at the boundary is what the copy will hold,
            # so it is also the price — re-read at every stop, since the cache
            # has grown. A copy that cannot fit under the budget even on an
            # empty map is not taken at all — better one prefill than a second
            # cache the machine has no room for.
            price = slot_bytes(cache)
            if price > self.max_bytes:
                continue
            copy: list | None = None
            try:
                self._make_room(price)
                copy = hooks.new_cache()
                fork_copy(cache, copy, hooks.copy_array)
                if self._publish(
                    Slot(copy, list(stable_ids[:boundary]), owner, "fork", slot_bytes(copy)),
                    epoch,
                ):
                    stats.forks += 1
            except Exception as e:
                # The fork is an optimization on a turn that is already
                # prefilled to the boundary and needs nothing more from it.
                # Saying so explicitly matters here: a cold attempt forks with
                # `reuse` 0, and `generate`'s handler re-raises instead of
                # retrying there — an exception escaping this block would fail
                # a viable generation.
                warnings.warn(
                    f"sous prompt cache: fork copy failed ({type(e).__name__}); "
                    "continuing without a fork",
                    stacklevel=2,
                )
            finally:
                # A publish can still refuse the slot — a reset since the turn
                # began, or the owner retired mid-turn — and neither a refused
                # nor a half-built copy may stay pinned by this frame for the
                # whole decode that follows. (An accepted one is never evicted
                # by its own publish: _publish protects the slot it adds.)
                copy = None
```

The `if all_trimmable(cache):` block and everything after it are unchanged.

- [ ] **Step 7: Minimal engine and engine-test compatibility edits**

The engines' probe closures return an `int`; `list(fork_at())` on an `int` raises and every engine test would see a `fork probe failed` warning. In both `src/sous/engine/vlm.py` (`_header_probe`, ~line 259) and `src/sous/engine/lm.py` (~line 194), change the return annotation to `-> Callable[[], list[int]] | list[int]`, the guard's `return 0` to `return []`, the inner `def probe() -> int:` to `def probe() -> list[int]:`, and its last line to:

```python
            at = fork_point(header_ids, stable_ids)
            return [at] if at else []
```

In `tests/test_engine_forkpoint.py`: in `Recording`, change `self.fork_ats: list[int] = []` to `self.fork_ats: list[list[int]] = []`, the `generate` default to `fork_at=()`, the resolver call to `self._resolver._fork_boundaries(threading.current_thread(), list(stable_ids), fork_at, 0)`, and `self.fork_ats.append(fork_at)` to `self.fork_ats.append(list(fork_at))`. Then:

```bash
sed -i '' 's/rec.fork_ats == \[0\]/rec.fork_ats == [[]]/g; s/rec.fork_ats == \[FORK_MIN_TOKENS + 1\]/rec.fork_ats == [[FORK_MIN_TOKENS + 1]]/g; s/rec.fork_ats == \[FORK_MIN_TOKENS + 1, FORK_MIN_TOKENS + 1\]/rec.fork_ats == [[FORK_MIN_TOKENS + 1], [FORK_MIN_TOKENS + 1]]/g; s/match="header probe"/match="fork probe"/g' tests/test_engine_forkpoint.py
```

In `tests/test_engine_lm.py::test_lm_header_probe_matches_a_real_conversation_render` (model-marked, but ty checks it): change `assert probe() >= FORK_MIN_TOKENS` to `assert probe() and probe()[0] >= FORK_MIN_TOKENS`.

- [ ] **Step 8: Run the whole suite and the checks**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest -m "not model" -q 2>&1 | tail -5
```

Expected: all green. If `test_the_fork_copy_makes_room_before_it_is_allocated` or `test_a_fork_larger_than_the_whole_budget_is_never_copied` fail, the `price > self.max_bytes` guard or `_make_room` order was changed — restore the order in Step 6.

- [ ] **Step 9: Commit**

```bash
git add src/sous/engine/promptcache.py src/sous/engine/vlm.py src/sous/engine/lm.py tests/test_promptcache.py tests/test_engine_forkpoint.py tests/test_engine_lm.py
git commit -m "feat(engine): fork at a list of boundaries, resolved on warm turns too

PrefixCache.generate takes a sequence of fork boundaries (or a callable
returning one) instead of a single header boundary, and _run stops at each
in ascending order to leave a fork slot there. The probe is now resolved on
warm turns as well: with one boundary that was waste, but once a tools
fork exists every later session's first turn starts warm at it and must
still publish its own header fork, or that session's second subagent would
reuse ~45K tokens instead of ~57K. The engines still answer with the header
alone; the tools boundary follows.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Same-turn copies protect each other; a budget for one copy skips the second

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`_evictable`, `_drop_lru`, `_evict_caps`, `_make_room`, `_publish`, `_evict_pressure` ~lines 361–460; the pre-turn `_evict_caps` call in `generate` ~line 612; the loop in `_run` from Task 1)
- Test: `tests/test_promptcache.py`

**Interfaces:**
- Consumes: Task 1's `_run` loop and `_fork_boundaries`.
- Produces: `_evictable(protect: Sequence[Slot]) -> list[Slot]`; `_drop_lru(protect: Sequence[Slot]) -> None`; `_evict_caps(protect: Sequence[Slot]) -> None`; `_make_room(nbytes: int, protect: Sequence[Slot] = ()) -> bool`; `_publish(slot: Slot, epoch: int, protect: Sequence[Slot] = ()) -> bool`; `_evict_pressure(protect: Sequence[Slot]) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_promptcache.py`, at the end of the "forks at more than one boundary" section (before `# ---- budget ----`):

```python
def test_a_budget_for_one_copy_skips_the_second_boundary_rather_than_evicting_into_it():
    """The second copy of a cold turn is charged with the first protected: a
    budget with room for exactly one fork keeps the tools fork (the one every
    later session can use) and never allocates the header copy. Without the
    protection LRU would pick the tools fork — the only evictable slot — and
    the turn would end with the less shareable of the two."""
    seen: list[list[list[int]]] = []
    h = FakeHooks(trimmable=True)
    h.on_new_cache = lambda: seen.append([s.held for s in pc.slots() if s.kind == "fork"])
    pc = PrefixCache(h, max_bytes=TOOLS_AT * 8 * 2)  # exactly the tools copy, two layers
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    assert pc.stats()["forks"] == 1
    # Two allocations only — the turn's own cache and the tools copy. No
    # header copy was allocated, let alone evicted into.
    assert len(h.caches) == 2
    assert seen == [[], []]
    assert h.prefilled[:2] == [T, HA[TOOLS_AT:]]  # the turn still stopped at both
    # The turn slot's own publish then takes the budget for itself, exactly as
    # in 3a: a turn slot that alone exceeds the budget lands protected and
    # evicts what it can. Not this test's subject, but say what happens.
    assert [s.kind for s in pc.slots()] == ["turn"]


def test_three_tool_sets_forks_coexist_under_a_budget_that_holds_them():
    U = list(range(20001, 20001 + FORK_MIN_TOKENS))  # a second tool set (general-purpose)
    HU = [*U, 8201]
    UX1, UX1_FULL = [*HU, 901, 902], [*HU, 901, 902, 90, 91]
    V = list(range(30001, 30001 + FORK_MIN_TOKENS))  # a third (general-purpose, deferred tools)
    HV = [*V, 8301, 8302]
    VX1, VX1_FULL = [*HV, 911], [*HV, 911, 90, 91]
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)
    pc.generate(UX1, UX1_FULL, 16, fork_at=[len(U), len(HU)])
    pc.generate(VX1, VX1_FULL, 16, fork_at=[len(V), len(HV)])
    forks = sorted(s.held for s in pc.slots() if s.kind == "fork")
    assert forks == sorted([T, HA, HB, U, HU, V, HV])
    assert pc.stats()["forks"] == 7
    assert pc.stats()["evictions"] == 0


def test_across_turns_the_least_recently_used_fork_goes_first():
    """Protection is within a turn only. Across turns LRU decides as in 3a: an
    earlier session's header fork, untouched since, goes before the tools
    fork the current turn just started from."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(AX1, AX1_FULL, 16, fork_at=BOUNDS_A)
    pc.generate(BX1, BX1_FULL, 16, fork_at=BOUNDS_B)  # touches T, adds HB
    # One byte short of everything resident: the next turn evicts, LRU first.
    pc.max_bytes = pc.stats()["resident_bytes"] - 1
    pc.generate(BX2, BX2_FULL, 16, fork_at=BOUNDS_B)  # hits HB (touched), adds a turn slot
    held = [s.held for s in pc.slots()]
    assert HA not in held  # session A's header fork: the oldest untouched slot
    assert AX1 not in held  # then session A's finished conversation
    assert T in held and HB in held
    assert pc.stats()["evictions"] == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -k "one_copy or three_tool_sets or least_recently_used_fork" -q 2>&1 | tail -12`

Expected: the first fails (`forks == 1` is false — the header copy evicted the tools copy, `len(h.caches) == 3`); the other two may already pass (they pin behaviour this task must not break).

- [ ] **Step 3: Make `protect` a sequence**

In `src/sous/engine/promptcache.py`, replace `_evictable`, `_drop_lru`, `_evict_caps` and `_make_room` (~lines 361–402) with:

```python
    def _evictable(self, protect: Sequence[Slot]) -> list[Slot]:
        # Identity, not equality: Slot is a dataclass, and `==` would compare
        # two ~50K-token id lists (and two caches) per candidate.
        return [s for s in self._slots if not any(s is p for p in protect)]

    def _drop_lru(self, protect: Sequence[Slot]) -> None:
        victim = min(self._evictable(protect), key=lambda s: s.last_used)
        self._slots.remove(victim)
        self._stats_for(victim.owner).evictions += 1
        # `victim` dies with this frame: the slot's cache is freed the moment
        # nothing else references it, which is what the pressure re-read in
        # _evict_pressure relies on.

    def _evict_caps(self, protect: Sequence[Slot]) -> None:
        """Bring the map under its count and byte caps, never touching a slot
        in `protect`. On its own before a turn starts, `protect` is empty:
        nothing here is spoken for."""
        while self._evictable(protect) and (
            len(self._slots) > MAX_SLOTS or self._resident() > self.max_bytes
        ):
            self._drop_lru(protect)

    def _make_room(self, nbytes: int, protect: Sequence[Slot] = ()) -> bool:
        """Evict LRU until a slot of `nbytes` would fit under the byte cap, and
        say whether it now does.

        For the fork copy, which is charged before it is allocated rather than
        after: publishing first and evicting afterwards would put the copy and
        whatever it displaces on the machine at the same time — the transient
        doubling the spec promises never happens. `protect` is the forks this
        same turn already published: a second copy is never made room for by
        evicting the first, and when the budget cannot hold both the answer
        is False and the caller skips the copy rather than evicting into it.
        The turn's own cache is not in the map, and a fork slot this turn
        copied from has already been copied, so neither needs protecting.

        Takes the lock itself.
        """
        with self._lock:
            while self._evictable(protect) and self._resident() + nbytes > self.max_bytes:
                self._drop_lru(protect)
            return self._resident() + nbytes <= self.max_bytes
```

Replace `_publish` and `_evict_pressure` (~lines 419–460) with:

```python
    def _publish(self, slot: Slot, epoch: int, protect: Sequence[Slot] = ()) -> bool:
        """Add `slot` unless the world moved on: a full reset since the turn
        began, or the owner retired (its task ended) while it generated.

        The caps are applied under the same lock as the append — nothing must
        ever observe the map over budget — but the pressure pass runs after
        the lock is released, because it asks the engine for a reading. Both
        protect `slot` and `protect` (the forks this turn published before
        it): a slot is never evicted by its own publish, so on a machine with
        room for exactly one slot that one survives — and a cold turn's second
        fork never evicts its first.
        """
        keep = (slot, *protect)
        with self._lock:
            if epoch != self._epoch or slot.owner in self._retired:
                return False
            self._slots.append(slot)
            self._evict_caps(protect=keep)
        self._evict_pressure(protect=keep)
        return True

    def _evict_pressure(self, protect: Sequence[Slot]) -> None:
        """Shrink the map until the machine has `reserve_bytes` free again,
        never touching a slot in `protect` — the slot just published and this
        turn's earlier forks.

        Takes the lock one drop at a time instead of holding it across the
        loop, because `headroom()` is an engine call (an mlx memory query, on
        a real engine). The bookkeeping lock covers list and dict work plus
        the deallocation a drop triggers, all bounded; that is the whole
        reason reset() can promise never to wait on a generation, and it would
        stop being true the moment an engine call ran underneath it. Dropping
        the lock between readings is safe: every iteration re-reads both the
        headroom and the map, so a concurrent publish or reset just changes
        what the next pass sees.
        """
        while (headroom := self._hooks.headroom()) is not None and headroom < self.reserve_bytes:
            with self._lock:
                if not self._evictable(protect):
                    return
                self._drop_lru(protect)
```

In `generate`, change the pre-turn call (~line 612) to:

```python
            self._evict_caps(protect=(slot,) if slot is not None and slot.kind == "fork" else ())
```

- [ ] **Step 4: Protect same-turn copies in `_run`**

Replace the loop from Task 1 Step 6 with:

```python
        published: list[Slot] = []
        for boundary in self._fork_boundaries(owner, stable_ids, fork_at, reuse):
            # Stop at the boundary and continue from there: a fork is taken
            # while prefilling past it because no layer can be rewound to it
            # afterwards. Ascending order means no copy is ever wanted behind
            # a prefix already prefilled.
            hooks.prefill(cache, list(stable_ids[reuse:boundary]))
            reuse = boundary
            # What the cache holds at the boundary is what the copy will hold,
            # so it is also the price — re-read at every stop, since the cache
            # has grown.
            price = slot_bytes(cache)
            copy: list | None = None
            try:
                # Charged before it is allocated, with this turn's earlier
                # copies protected: a budget with room for one copy skips the
                # second rather than evicting the first into it, and a copy
                # that cannot fit even on an empty map is not taken at all —
                # better one prefill than a second cache the machine has no
                # room for.
                if self._make_room(price, protect=published):
                    copy = hooks.new_cache()
                    fork_copy(cache, copy, hooks.copy_array)
                    slot = Slot(copy, list(stable_ids[:boundary]), owner, "fork", slot_bytes(copy))
                    if self._publish(slot, epoch, protect=published):
                        stats.forks += 1
                        published.append(slot)
            except Exception as e:
                # The fork is an optimization on a turn that is already
                # prefilled to the boundary and needs nothing more from it.
                # Saying so explicitly matters here: a cold attempt forks with
                # `reuse` 0, and `generate`'s handler re-raises instead of
                # retrying there — an exception escaping this block would fail
                # a viable generation.
                warnings.warn(
                    f"sous prompt cache: fork copy failed ({type(e).__name__}); "
                    "continuing without a fork",
                    stacklevel=2,
                )
            finally:
                # A publish can still refuse the slot — a reset since the turn
                # began, or the owner retired mid-turn — and neither a refused
                # nor a half-built copy may stay pinned by this frame for the
                # whole decode that follows. (An accepted one is never evicted
                # by its own publish: _publish protects the slot it adds.)
                copy = None
        # The list existed to protect the copies from each other while they
        # were charged. Dropping it here means a fork that pressure or a reset
        # evicts during the decode below is freed then, not pinned by this
        # frame until the turn ends.
        published = []
```

- [ ] **Step 5: Run the suite and the checks**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest -m "not model" -q 2>&1 | tail -5
```

Expected: all green, including `test_a_fork_larger_than_the_whole_budget_is_never_copied` (`_make_room` returns False on an empty map when `nbytes > max_bytes`, so `len(h.caches) == 1` still holds) and `test_the_default_budget_refuses_the_fork_before_it_splits_the_prefill` (`_fork_boundaries` returns `[]` at `max_bytes == 0`, so the loop body never runs).

- [ ] **Step 6: Commit**

```bash
git add src/sous/engine/promptcache.py tests/test_promptcache.py
git commit -m "fix(engine): never evict a cold turn's first fork to make room for its second

A turn that forks at two boundaries charges two copies. The second copy's
_make_room and _publish now protect the first — until now it survived only
because the newest slot has the highest last_used, which fails the moment
it is the only evictable slot. _make_room reports whether the copy fits so
a budget with room for exactly one copy publishes the tools fork (the one
every later session can use) and skips the header copy instead of evicting
into it.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: The two-pair probe — `probe_boundaries`, the `tools` memo slot, the engines' `_fork_probe`

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`import os`; `_MEMO_SLOTS`, `PromptMemo` docstring ~lines 237–262; new `_HEADER_PROBES`, `_TOOLS_PROBES`, `probe_boundaries` after `fork_point`)
- Modify: `src/sous/engine/vlm.py` (imports ~lines 5–16; `generate` ~line 256; `_header_probe` ~lines 259–303)
- Modify: `src/sous/engine/lm.py` (imports ~lines 5–15; `_ids` docstring ~lines 89–92; `generate` ~line 191; `_header_probe` ~lines 194–238)
- Rewrite: `tests/test_engine_forkpoint.py`
- Modify: `tests/test_promptcache.py` (one memo test), `tests/test_engine_lm.py:174-196` (rename)

**Interfaces:**
- Consumes: Task 1's `_fork_boundaries`, `fork_point`, `PromptMemo`.
- Produces: `promptcache.probe_boundaries(render: Callable[[list[dict]], str], encode: Callable[[str], list[int]], memo: PromptMemo, system: dict, stable_ids: Sequence[int]) -> list[int]`; `promptcache._HEADER_PROBES`, `promptcache._TOOLS_PROBES`; `_MEMO_SLOTS == ("stable", "full", "header", "tools")`; `VLMEngine._fork_probe(messages, tools, stable_ids) -> Callable[[], list[int]] | list[int]` and `LMEngine._fork_probe` with the same signature (Task 4's model tests call them).

- [ ] **Step 1: Write the failing memo test**

Append after `test_memo_accepts_the_header_slot` in `tests/test_promptcache.py`:

```python
def test_memo_accepts_the_tools_slot_independently_of_the_header():
    # Two texts, two slots: in a shared slot the tools text and the header
    # text would evict each other on every cold turn.
    m = PromptMemo()
    m.put("tools", "tools", [1])
    m.put("header", "tools + sys", [1, 2])
    assert m.get("tools", "tools") == [1]
    assert m.get("header", "tools + sys") == [1, 2]
```

- [ ] **Step 2: Rewrite `tests/test_engine_forkpoint.py`**

Replace the whole file with:

```python
"""The engines' fork boundaries: what two probe pairs' renders have in common
— the tool block, and the whole system block — each verified as a token
prefix of the whole render; exercised without mlx by faking the tokenizer,
the way tests/test_engine_unloaded.py builds engines."""

import threading
from typing import cast

import pytest

from sous.engine.lm import LMEngine
from sous.engine.promptcache import FORK_MIN_TOKENS, CacheHooks, PrefixCache, PromptMemo
from sous.engine.vlm import VLMEngine

# The fake's tool block, whenever a conversation has tools: long enough to be
# a fork boundary on its own.
TOOL_BLOCK = "T" * FORK_MIN_TOKENS


def _render(messages, tools, add_generation_prompt):
    """The default model's shape, one token per character: with tools, the
    system turn is the tool block, then a separator and the client's text —
    the separator only when there is text, as Qwen3.5/3.8's template does;
    without tools, just the text."""
    out = []
    for m in messages:
        if m["role"] == "system":
            block = TOOL_BLOCK if tools else ""
            sep = "\n\n" if tools and m["content"] else ""
            out.append(block + sep + m["content"])
        else:
            out.append(m["content"])
    if add_generation_prompt:
        out.append("G")
    return "|".join(out)


class FakeTokenizer:
    """Renders messages as a fixed-width token per character so a text prefix
    is also a token prefix, and vice versa.

    Refuses a message list with no user turn, exactly as the default model's
    template does (`raise_exception("No user query found in messages.")`) —
    the whole reason the boundaries are probed rather than rendered directly."""

    bos_token = None

    def __init__(self):
        self.encoded: list[str] = []

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        if not any(m["role"] == "user" for m in messages):
            raise Exception("No user query found in messages.")
        return _render(messages, tools, add_generation_prompt)

    def encode(self, text, add_special_tokens):
        self.encoded.append(text)
        return [ord(c) for c in text]


class Tolerant(FakeTokenizer):
    """A template that renders a system-only list instead of refusing it, the
    way Qwen3-0.6B's does. Both kinds are in the wild; only this one lets a
    system-only conversation reach the engine at all."""

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        return _render(messages, tools, add_generation_prompt)


class SystemFirst(FakeTokenizer):
    """A template that renders the client's text BEFORE the tool block, the
    way Qwen3's does: the tools probe pair then shares nothing above the
    system text, and the header is the only boundary."""

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        if not any(m["role"] == "user" for m in messages):
            raise Exception("No user query found in messages.")
        out = []
        for m in messages:
            if m["role"] == "system":
                out.append(m["content"] + ("\n\n" + TOOL_BLOCK if tools else ""))
            else:
                out.append(m["content"])
        if add_generation_prompt:
            out.append("G")
        return "|".join(out)


class Rewriting(FakeTokenizer):
    """A template that changes the system turn once the conversation has
    history. The probe renders still agree with each other — a probe can only
    ever vary the first user turn or the system text — so what catches this is
    fork_point's per-turn check that each boundary is a token prefix of the
    real render."""

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        text = super().apply_chat_template(
            messages, tools, add_generation_prompt, tokenize, enable_thinking
        )
        return text.replace("S", "s", 1) if len(messages) > 2 else text


class RefusingProbes(FakeTokenizer):
    """A template that renders the real conversation but refuses the probes —
    whatever a probe render happens to contain, it raises."""

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, enable_thinking
    ):
        if any(m["role"] == "user" and m["content"] in ("0", "1") for m in messages):
            raise Exception("no probing")
        return super().apply_chat_template(
            messages, tools, add_generation_prompt, tokenize, enable_thinking
        )


class Recording:
    """A PrefixCache stand-in recording the fork_at it was handed.

    A callable is resolved through the real `PrefixCache._fork_boundaries` —
    with a budget and a cold miss — so these tests see the shipped rule
    (including its warn-and-empty on a probe that raises) rather than a second
    copy of it here. That resolver's hooks are never reached: nothing below
    `generate` runs."""

    enabled = True

    def __init__(self):
        self.fork_ats: list[list[int]] = []
        self._resolver = PrefixCache(cast("CacheHooks", None), max_bytes=1 << 40)

    def generate(self, stable_ids, full_ids, max_tokens, on_delta=None, fork_at=()):
        if callable(fork_at):
            fork_at = self._resolver._fork_boundaries(
                threading.current_thread(), list(stable_ids), fork_at, 0
            )
        self.fork_ats.append(list(fork_at))
        return "text"

    def stats(self, owner=None):
        return {}

    def reset(self, owner=None):
        pass


class Spy(Recording):
    """Records the owner every passthrough was called with."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def stats(self, owner=None):
        self.seen.append(("stats", owner))
        return {}

    def reset(self, owner=None):
        self.seen.append(("reset", owner))


def _engine(tokenizer) -> tuple[LMEngine, Recording]:
    engine = object.__new__(LMEngine)
    engine.model_id = "test/model"
    engine._model = object()
    engine._tokenizer = tokenizer
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    rec = Recording()
    engine._cache = rec  # ty: ignore[invalid-assignment]
    return engine, rec


def _vlm_engine(tokenizer) -> tuple[VLMEngine, Recording]:
    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    # _loaded() only checks for None; the fake never reaches the model.
    engine._model = object()  # ty: ignore[invalid-assignment]
    # VLMEngine._tokenizer reads `processor.tokenizer` when there is one, and
    # falls back to the processor itself — which is what the fake is.
    engine._processor = tokenizer
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    rec = Recording()
    engine._cache = rec  # ty: ignore[invalid-assignment]
    return engine, rec


SYSTEM_TEXT = "S" * FORK_MIN_TOKENS
SYSTEM = {"role": "system", "content": SYSTEM_TEXT}
# Another session's system text: the same length, different at the first
# character — where a per-session scratchpad path would differ.
OTHER_SYSTEM = {"role": "system", "content": "s" + SYSTEM_TEXT[1:]}
SHORT_SYSTEM = {"role": "system", "content": "S" * 10}
EMPTY_SYSTEM = {"role": "system", "content": ""}
USER = {"role": "user", "content": "hello"}
ASSISTANT = {"role": "assistant", "content": "hi"}
TOOLS = [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}]

# What the probes find on the default-shaped fake. Without tools the header is
# the system text plus the separator before the first user turn's content.
# With tools the tool block plus its separator is the tools boundary, and the
# header runs on through the system text to that same user separator.
HEADER = SYSTEM_TEXT + "|"
HEADER_AT = len(HEADER)  # FORK_MIN_TOKENS + 1
TOOLS_TEXT = TOOL_BLOCK + "\n\n"
TOOLS_AT = len(TOOLS_TEXT)  # FORK_MIN_TOKENS + 2
TOOLED_HEADER = TOOLS_TEXT + SYSTEM_TEXT + "|"
TOOLED_HEADER_AT = len(TOOLED_HEADER)


def test_a_leading_system_turn_long_enough_is_a_header_boundary():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM, USER], [], 8)
    # One past the system render: the probes share the separator the template
    # emits before the first user turn's content, and everything they share is
    # header. No tools, so no tools boundary.
    assert rec.fork_ats == [[HEADER_AT]]


def test_with_tools_rendered_first_the_tool_block_is_a_second_boundary():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]]


def test_two_sessions_with_different_system_text_share_the_tools_boundary():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    engine.generate([OTHER_SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_a_template_that_renders_the_system_text_first_has_no_tools_boundary():
    engine, rec = _engine(SystemFirst())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    # The two sessions' renders diverge inside the first line, so the tools
    # pair shares only what precedes the text: nothing here, a few tokens on
    # Qwen3. The header is the whole system turn, tool block included.
    assert rec.fork_ats == [[len(SYSTEM_TEXT) + 2 + len(TOOL_BLOCK) + 1]]


def test_a_system_turn_with_no_text_has_no_tools_boundary():
    # The template renders no separator before empty text, so the tools text
    # (block plus separator) is not a prefix of this render; the header — the
    # block plus the user separator — still is. No special case needed.
    engine, rec = _engine(FakeTokenizer())
    engine.generate([EMPTY_SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[len(TOOL_BLOCK) + 1]]


def test_a_short_system_turn_does_not_fork():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([SHORT_SYSTEM, USER], [], 8)
    assert rec.fork_ats == [[]]


def test_no_system_turn_means_no_fork():
    engine, rec = _engine(FakeTokenizer())
    engine.generate([USER], TOOLS, 8)
    assert rec.fork_ats == [[]]


def test_a_system_only_prompt_does_not_fork():
    # Nothing would follow the header, but _fork_probe's `len(messages) < 2`
    # guard returns first: no probe is rendered and fork_point is never asked.
    # A tolerant template, because a system-only conversation cannot reach the
    # engine at all under one that refuses to render it.
    engine, rec = _engine(Tolerant())
    engine.generate([SYSTEM], [], 8)
    assert rec.fork_ats == [[]]


def test_a_template_whose_header_is_not_a_token_prefix_does_not_fork():
    engine, rec = _engine(Rewriting())
    engine.generate([SYSTEM, USER, ASSISTANT], [], 8)
    assert rec.fork_ats == [[]]


def test_a_template_that_refuses_the_probe_warns_and_does_not_fork():
    engine, rec = _engine(RefusingProbes())
    with pytest.warns(UserWarning, match="fork probe"):
        assert engine.generate([SYSTEM, USER], TOOLS, 8) == "text"
    assert rec.fork_ats == [[]]


def test_the_boundary_texts_are_encoded_once_across_turns():
    tokenizer = FakeTokenizer()
    engine, rec = _engine(tokenizer)
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert engine._memo.get("tools", TOOLS_TEXT) == [ord(c) for c in TOOLS_TEXT]
    assert engine._memo.get("header", TOOLED_HEADER) == [ord(c) for c in TOOLED_HEADER]
    engine.generate([SYSTEM, {"role": "user", "content": "a different brief"}], TOOLS, 8)
    # The renders repeat (they are cheap and text-keyed); the two ~50K-token
    # encodes are what the memo slots have to save.
    assert tokenizer.encoded.count(TOOLS_TEXT) == 1
    assert tokenizer.encoded.count(TOOLED_HEADER) == 1
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_a_second_session_re_encodes_only_its_own_header():
    tokenizer = FakeTokenizer()
    engine, rec = _engine(tokenizer)
    engine.generate([SYSTEM, USER], TOOLS, 8)
    engine.generate([OTHER_SYSTEM, USER], TOOLS, 8)
    assert tokenizer.encoded.count(TOOLS_TEXT) == 1  # same tool set: memo hit
    assert tokenizer.encoded.count(TOOLED_HEADER) == 1
    assert tokenizer.encoded.count(TOOLS_TEXT + OTHER_SYSTEM["content"] + "|") == 1
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_the_probe_is_skipped_when_the_cache_is_disabled():
    engine, rec = _engine(FakeTokenizer())
    rec.enabled = False
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[]]
    assert engine._memo.get("tools", TOOLS_TEXT) is None
    assert engine._memo.get("header", TOOLED_HEADER) is None


def test_owner_scoped_stats_and_reset_pass_through():
    engine, _ = _engine(FakeTokenizer())
    spy = Spy()
    engine._cache = spy  # ty: ignore[invalid-assignment]
    me = threading.current_thread()
    engine.prompt_cache_stats(owner=me)
    engine.reset_prompt_cache(owner=me)
    assert spy.seen == [("stats", me), ("reset", me)]


# ---- the same, for the VLM backend -----------------------------------------
#
# _encode is monkeypatched away because its real body imports mlx_vlm; it is
# routed to the fake tokenizer's own encode() — the one the LM tests exercise
# — so the encode count is taken the same way on both backends, and the file
# stays runnable on a machine without mlx.


def _mlx_free_vlm(monkeypatch, tokenizer) -> tuple[VLMEngine, Recording]:
    monkeypatch.setattr(VLMEngine, "_encode", lambda self, text: self._tokenizer.encode(text, True))
    return _vlm_engine(tokenizer)


def test_vlm_a_leading_system_turn_long_enough_is_a_header_boundary(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SYSTEM, USER], [], 8)
    assert rec.fork_ats == [[HEADER_AT]]


def test_vlm_with_tools_rendered_first_the_tool_block_is_a_second_boundary(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]]


def test_vlm_two_sessions_with_different_system_text_share_the_tools_boundary(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    engine.generate([OTHER_SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_vlm_a_template_that_renders_the_system_text_first_has_no_tools_boundary(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, SystemFirst())
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[len(SYSTEM_TEXT) + 2 + len(TOOL_BLOCK) + 1]]


def test_vlm_a_system_turn_with_no_text_has_no_tools_boundary(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([EMPTY_SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[len(TOOL_BLOCK) + 1]]


def test_vlm_a_short_system_turn_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([SHORT_SYSTEM, USER], [], 8)
    assert rec.fork_ats == [[]]


def test_vlm_no_system_turn_means_no_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    engine.generate([USER], TOOLS, 8)
    assert rec.fork_ats == [[]]


def test_vlm_a_system_only_prompt_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, Tolerant())
    engine.generate([SYSTEM], [], 8)
    assert rec.fork_ats == [[]]


def test_vlm_a_template_whose_header_is_not_a_token_prefix_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, Rewriting())
    engine.generate([SYSTEM, USER, ASSISTANT], [], 8)
    assert rec.fork_ats == [[]]


def test_vlm_a_template_that_refuses_the_probe_warns_and_does_not_fork(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, RefusingProbes())
    with pytest.warns(UserWarning, match="fork probe"):
        assert engine.generate([SYSTEM, USER], TOOLS, 8) == "text"
    assert rec.fork_ats == [[]]


def test_vlm_the_boundary_texts_are_encoded_once_across_turns(monkeypatch):
    tokenizer = FakeTokenizer()
    engine, rec = _mlx_free_vlm(monkeypatch, tokenizer)
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert engine._memo.get("tools", TOOLS_TEXT) == [ord(c) for c in TOOLS_TEXT]
    assert engine._memo.get("header", TOOLED_HEADER) == [ord(c) for c in TOOLED_HEADER]
    engine.generate([SYSTEM, {"role": "user", "content": "a different brief"}], TOOLS, 8)
    assert tokenizer.encoded.count(TOOLS_TEXT) == 1
    assert tokenizer.encoded.count(TOOLED_HEADER) == 1
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_vlm_a_second_session_re_encodes_only_its_own_header(monkeypatch):
    tokenizer = FakeTokenizer()
    engine, rec = _mlx_free_vlm(monkeypatch, tokenizer)
    engine.generate([SYSTEM, USER], TOOLS, 8)
    engine.generate([OTHER_SYSTEM, USER], TOOLS, 8)
    assert tokenizer.encoded.count(TOOLS_TEXT) == 1
    assert tokenizer.encoded.count(TOOLED_HEADER) == 1
    assert tokenizer.encoded.count(TOOLS_TEXT + OTHER_SYSTEM["content"] + "|") == 1
    assert rec.fork_ats == [[TOOLS_AT, TOOLED_HEADER_AT]] * 2


def test_vlm_the_probe_is_skipped_when_the_cache_is_disabled(monkeypatch):
    engine, rec = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    rec.enabled = False
    engine.generate([SYSTEM, USER], TOOLS, 8)
    assert rec.fork_ats == [[]]
    assert engine._memo.get("tools", TOOLS_TEXT) is None
    assert engine._memo.get("header", TOOLED_HEADER) is None


def test_vlm_owner_scoped_stats_and_reset_pass_through(monkeypatch):
    engine, _ = _mlx_free_vlm(monkeypatch, FakeTokenizer())
    spy = Spy()
    engine._cache = spy  # ty: ignore[invalid-assignment]
    me = threading.current_thread()
    engine.prompt_cache_stats(owner=me)
    engine.reset_prompt_cache(owner=me)
    assert spy.seen == [("stats", me), ("reset", me)]
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_engine_forkpoint.py tests/test_promptcache.py -k "tools or forkpoint or memo" -q 2>&1 | tail -15`

Expected: `test_memo_accepts_the_tools_slot_independently_of_the_header` fails with `KeyError: 'tools'`; the forkpoint tests with `TOOLS` fail (the header-only probe answers `[TOOLED_HEADER_AT]`, the fake's system content is now rendered so `HEADER_AT` cases pass already).

- [ ] **Step 4: Add `probe_boundaries` and the `tools` memo slot to `promptcache.py`**

Add `import os` to the stdlib imports (alphabetical: after `import dataclasses`). Change `_MEMO_SLOTS` and the `PromptMemo` docstring's second paragraph:

```python
_MEMO_SLOTS = ("stable", "full", "header", "tools")


class PromptMemo:
    """One slot per render, keyed by the exact prompt text.

    The header and tools slots hold the two fork boundaries' texts: what the
    header probe pair's renders have in common (the whole system block) and
    what the tools pair's have (the tool block, on a template that renders
    tools first) — see `probe_boundaries`. Two slots, because the two texts
    would evict each other from one on every cold turn.

    Each turn needs both the stable render and the full prompt, and
    `count_tokens` asks for one of them before `generate` asks again. Keying on
    text equality rather than on call order means a stale slot can never yield
    the wrong ids, even if an abandoned generation thread overwrites it.
    """
```

Insert after `fork_point` (before `slot_bytes`):

```python
# Probe conversations, each rendered with the turn's tools. The header pair
# keeps the real system turn and differs only in the first user turn's
# content, so their renders share exactly the system block — the header. The
# tools pair keeps the user turn fixed and differs at the first character of
# the system text, so their renders share whatever the template emits BEFORE
# the client's text: on a template that renders tools first (Qwen3.5/3.8) the
# whole tool block — the ~45–56K tokens every Claude Code session presenting
# the same tool set shares, across sessions and projects — and on one that
# renders the system text first (Qwen3) a few tokens, which fork_point drops.
_HEADER_PROBES = ({"role": "user", "content": "0"}, {"role": "user", "content": "1"})
_TOOLS_PROBES = (
    ({"role": "system", "content": "0"}, {"role": "user", "content": "0"}),
    ({"role": "system", "content": "1"}, {"role": "user", "content": "0"}),
)


def probe_boundaries(
    render: Callable[[list[dict]], str],
    encode: Callable[[str], list[int]],
    memo: PromptMemo,
    system: dict,
    stable_ids: Sequence[int],
) -> list[int]:
    """Every boundary a turn may fork at, ascending: the tools boundary and the
    header boundary, each the common prefix of a probe pair's renders, encoded
    once (memoized by text, one slot each) and verified by `fork_point`
    against this turn's own render. A candidate that fails the check is
    dropped and equal candidates merge. A request with no system text renders
    no separator before it on the default template, so the tools text is then
    not a prefix and drops out on its own.

    Found from probe pairs rather than from partial renders: [model].id
    accepts any chat template, and a template may refuse a message list with
    no user turn outright (the default model's does:
    `raise_exception("No user query found in messages.")`). The caller holds
    its tokenize lock — `render` and `encode` both run here.
    """
    found: set[int] = set()
    pairs = (
        ("tools", [list(conversation) for conversation in _TOOLS_PROBES]),
        ("header", [[system, user] for user in _HEADER_PROBES]),
    )
    for slot, conversations in pairs:
        text = os.path.commonprefix([render(conversation) for conversation in conversations])
        ids = memo.get(slot, text)
        if ids is None:
            ids = encode(text)
            memo.put(slot, text, ids)
        found.add(fork_point(ids, stable_ids))
    found.discard(0)
    return sorted(found)
```

- [ ] **Step 5: Replace the engines' probes**

`src/sous/engine/vlm.py`: remove `import os`; change the promptcache import to `from sous.engine.promptcache import FORK_MIN_TOKENS, PrefixCache, PromptMemo, probe_boundaries`; delete the `_PROBES` tuple and its two comment lines. In `generate`, change `fork_at=self._header_probe(messages, tools, stable_ids)` to `fork_at=self._fork_probe(messages, tools, stable_ids)`. Replace the whole `_header_probe` method with:

```python
    def _fork_probe(
        self, messages: list[dict], tools: list[dict], stable_ids: list[int]
    ) -> Callable[[], list[int]] | list[int]:
        """The boundaries this turn may fork at — the end of the tool block and
        the end of the whole system block (`promptcache.probe_boundaries`) —
        as a closure PrefixCache resolves only when there is a budget to fork
        into.

        The tools fork is what one Claude Code session hands the next: the
        template renders the tool block before the client's system text, and
        a subagent type's tool array is byte-identical across sessions and
        projects, so a new `claude` process's first subagent turn starts
        ~45–56K tokens warm instead of prefilling ~57K cold. The header fork
        serves the same session's next subagent of that type, ~57K warm. The
        worker's short system prompt never clears the floor, and a render
        below it cannot contain a boundary above it — so it never pays the
        probe."""
        if (
            len(messages) < 2
            or messages[0].get("role") != "system"
            or len(stable_ids) < FORK_MIN_TOKENS
        ):
            return []

        def probe() -> list[int]:
            with self._tokenize_lock:
                return probe_boundaries(
                    lambda conversation: self._prompt(conversation, tools, generation=False),
                    self._encode,
                    self._memo,
                    messages[0],
                    stable_ids,
                )

        return probe
```

`src/sous/engine/lm.py`: the same three import edits (`import os` removed, `probe_boundaries` imported instead of `fork_point`, `_PROBES` deleted), the same `generate` change, and the same `_fork_probe` method verbatim. Also change the `_ids` docstring's last sentence to: `The memo's other two slots, header and tools, are filled by the fork probe (promptcache.probe_boundaries), which builds their texts rather than rendering a message list.`

- [ ] **Step 6: Rename in the model test**

In `tests/test_engine_lm.py`, rename `test_lm_header_probe_matches_a_real_conversation_render` to `test_lm_fork_probe_matches_a_real_conversation_render`, change `probe = e._header_probe(msgs, [], stable)` to `probe = e._fork_probe(msgs, [], stable)`, and the two assertions to:

```python
    # not isinstance(..., list) rather than callable(): the same narrowing, and
    # the one ty follows.
    assert not isinstance(probe, list), "a long system turn must be probed, not refused"
    bounds = probe()
    assert bounds == [bounds[0]] and bounds[0] >= FORK_MIN_TOKENS  # no tools: the header only
```

- [ ] **Step 7: Run the suite and the checks**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest -m "not model" -q 2>&1 | tail -5
```

Expected: all green. `ruff check` must report no unused imports in the engines (`os`, `fork_point`) — remove any it names.

- [ ] **Step 8: Commit**

```bash
git add src/sous/engine/promptcache.py src/sous/engine/vlm.py src/sous/engine/lm.py tests/test_engine_forkpoint.py tests/test_promptcache.py tests/test_engine_lm.py
git commit -m "feat(engine): probe the tools boundary so sessions share a fork

Qwen3.5/3.8's template renders the # Tools block before the client's
system text, so two Claude Code sessions whose tool arrays match share
~45-56K rendered tokens above the per-session path Phase 3a took to be
the first divergence (it read the request body's order, not the
render's). A second probe pair - two renders differing at the first
character of the system text - finds where that block ends; the header
pair is unchanged. Both pairs live in promptcache.probe_boundaries, with
the tools text in its own memo slot, and each engine keeps a thin
_fork_probe. On Qwen3's template (system text first) the tools pair
shares a few tokens and drops out, which the fake-tokenizer tests cover.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Model tests (local only)

**Files:**
- Modify: `tests/test_engine_lm.py` (append)
- Modify: `tests/test_engine_vlm.py` (append)

**Interfaces:**
- Consumes: `LMEngine._fork_probe`, `VLMEngine._fork_probe`, `VLMEngine._ids`, `VLMEngine.prefill`, `VLMEngine.copy_array`, `promptcache.fork_copy`, `promptcache.slot_bytes`, `FORK_MIN_TOKENS`.
- Produces: nothing downstream; these are the local-only proof that the boundary is where the spec says on real templates and that the copies are exact.

These tests are `model`-marked (both files set `pytestmark = pytest.mark.model`) and download weights: `TINY` (~350 MB, already cached), `HYBRID_VLM` (~5.6 GB, already cached on Kyle's machine). They run locally only; CI skips them.

- [ ] **Step 1: Add a shared tool array helper and the system-first test to `tests/test_engine_lm.py`**

Append:

```python
def _fat_tools(n: int = 80) -> list[dict]:
    """A tool array long enough to clear the fork floor on its own (~90 tokens
    per tool), in the shape convert.py hands the template."""
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Tool number {i}. "
                + "It does one specific, well-documented thing to a file. " * 4,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "An absolute path."}},
                    "required": ["path"],
                },
            },
        }
        for i in range(n)
    ]


def test_lm_fork_probe_finds_only_the_header_on_a_system_first_template():
    """Qwen3's template renders the client's system text BEFORE its # Tools
    block, so two sessions' renders diverge inside the first line: the tools
    pair shares only `<|im_start|>system\\n`, far below the floor, and the
    header is the one boundary. The default 27B model's template is the other
    way round — see the hybrid VLM test."""
    from sous.engine.lm import LMEngine
    from sous.engine.promptcache import FORK_MIN_TOKENS

    e = LMEngine(TINY, cache_budget=1 << 34)
    tools = _fat_tools()
    system = {"role": "system", "content": "Session /tmp/a. You are terse."}
    msgs = [system, {"role": "user", "content": "Say A."}]
    stable = e._ids("stable", msgs, tools)
    assert len(stable) >= FORK_MIN_TOKENS
    probe = e._fork_probe(msgs, tools, stable)
    assert not isinstance(probe, list)
    bounds = probe()
    assert len(bounds) == 1
    assert FORK_MIN_TOKENS <= bounds[0] < len(stable)
    # The one boundary is the whole system block: it ends right before the
    # user turn, i.e. the render up to it ends with the system turn's close.
    _, tokenizer = e._loaded()
    assert tokenizer.decode(stable[: bounds[0]]).endswith("<|im_end|>\n<|im_start|>user\n")
    e.unload()
```

- [ ] **Step 2: Add the hybrid-VLM tests to `tests/test_engine_vlm.py`**

Append:

```python
def _fat_tools(n: int = 80) -> list[dict]:
    """A tool array long enough to clear the fork floor on its own (~90 tokens
    per tool), in the shape convert.py hands the template."""
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Tool number {i}. "
                + "It does one specific, well-documented thing to a file. " * 4,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "An absolute path."}},
                    "required": ["path"],
                },
            },
        }
        for i in range(n)
    ]


def test_vlm_fork_probe_finds_the_tools_boundary_on_a_tools_first_template():
    """Qwen3.5/3.8's template renders # Tools before the client's system text,
    so two sessions differing only in that text share the whole tool block.
    The tools boundary is the first of two and ends at `</IMPORTANT>\\n\\n` —
    the separator the template emits only when there is system text."""
    from sous.engine.promptcache import FORK_MIN_TOKENS
    from sous.engine.vlm import VLMEngine

    e = VLMEngine(HYBRID_VLM, cache_budget=1 << 34)
    tools = _fat_tools()
    a = {"role": "system", "content": "Session /tmp/a. You are terse."}
    b = {"role": "system", "content": "Session /tmp/b. You are terse."}
    user = {"role": "user", "content": "Say A."}
    stable_a = e._ids("stable", [a, user], tools)
    probe = e._fork_probe([a, user], tools, stable_a)
    assert not isinstance(probe, list)
    bounds = probe()
    assert len(bounds) == 2
    assert FORK_MIN_TOKENS <= bounds[0] < bounds[1] < len(stable_a)
    # The tools boundary is a prefix of BOTH sessions' renders; the header of A's only.
    stable_b = e._ids("stable", [b, user], tools)
    assert stable_b[: bounds[0]] == stable_a[: bounds[0]]
    assert stable_b[: bounds[1]] != stable_a[: bounds[1]]
    assert e._tokenizer.decode(stable_a[: bounds[0]]).endswith("</IMPORTANT>\n\n")
    # With no system text there is no separator, so no tools boundary — the
    # header (the tool block plus the system turn's close) is the one left.
    empty = {"role": "system", "content": ""}
    stable_e = e._ids("stable", [empty, user], tools)
    probe_e = e._fork_probe([empty, user], tools, stable_e)
    assert not isinstance(probe_e, list)
    bounds_e = probe_e()
    assert len(bounds_e) == 1
    assert e._tokenizer.decode(stable_e[: bounds_e[0]]).endswith("<|im_end|>\n<|im_start|>user\n")
    e.unload()


def test_vlm_fork_copies_at_two_boundaries_match_a_split_prefill_bit_for_bit():
    """A cache continued from the tools copy, and one continued from the
    header copy taken after the source itself was continued past the tools
    copy, must both equal a cache prefilled cold over the same three segments
    — on the hybrid, whose recurrent layers cannot be rebuilt by replay.

    Reference is three prefill calls, not one: a quantized matmul's rounding
    depends on the row count, so a single call differs by a bf16 ULP from the
    split every warm turn already makes (measured in Phase 3a)."""
    import mlx.core as mx
    from mlx_vlm.models.cache import make_prompt_cache

    from sous.engine.promptcache import fork_copy, slot_bytes
    from sous.engine.vlm import VLMEngine

    def assert_identical(x, y):
        for p, q in zip(x, y, strict=True):
            # getattr, not p.offset: an ArraysCache layer has no offset at all
            # (nothing to rewind), and its state is compared whole below.
            off = int(getattr(p, "offset", 0) or 0)
            assert off == int(getattr(q, "offset", 0) or 0)
            for xa, xb in zip(p.state, q.state, strict=True):
                if xa is None or xb is None:
                    assert xa is None and xb is None, "one side is None, the other is not"
                    continue
                if hasattr(p, "trim") and xa.ndim >= 3 and off:
                    xa, xb = xa[..., :off, :], xb[..., :off, :]
                d = mx.max(mx.abs(xa.astype(mx.float32) - xb.astype(mx.float32)))
                mx.eval(d)
                assert d.item() == 0.0

    e = VLMEngine(HYBRID_VLM, cache_budget=0)
    model, _ = e._loaded()
    ids = e._encode("def f(x):\n    return x + 1\n" * 60)
    third = len(ids) // 3
    tools_seg, header_seg, tail = ids[:third], ids[third : 2 * third], ids[2 * third :]

    ref = make_prompt_cache(model.language_model)
    e.prefill(ref, tools_seg)
    e.prefill(ref, header_seg)
    e.prefill(ref, tail)

    src = make_prompt_cache(model.language_model)
    e.prefill(src, tools_seg)
    tools_fork = make_prompt_cache(model.language_model)
    fork_copy(src, tools_fork, e.copy_array)
    e.prefill(src, header_seg)  # the source turn continues past its first copy
    header_fork = make_prompt_cache(model.language_model)
    fork_copy(src, header_fork, e.copy_array)
    assert 0 < slot_bytes(tools_fork) <= slot_bytes(header_fork)

    # A later session from the tools fork: its own "system text" then the tail.
    e.prefill(tools_fork, header_seg)
    e.prefill(tools_fork, tail)
    assert_identical(tools_fork, ref)
    # The same session's next subagent, from the header fork: the tail only.
    e.prefill(header_fork, tail)
    assert_identical(header_fork, ref)
    # And the source, continued to the end itself, is the same cache too.
    e.prefill(src, tail)
    assert_identical(src, ref)
    e.unload()


def test_vlm_engine_serves_a_second_session_from_the_tools_fork():
    """End to end on the hybrid: two conversations with the same tools and
    different system texts. The second's first turn is a fork hit that reuses
    the tool block; that session's next conversation reuses its whole header."""
    from sous.engine.promptcache import FORK_MIN_TOKENS
    from sous.engine.vlm import VLMEngine

    e = VLMEngine(HYBRID_VLM, prompt_cache=True, cache_budget=1 << 34)
    tools = _fat_tools()
    a = {"role": "system", "content": "Session /tmp/a. You are terse."}
    b = {"role": "system", "content": "Session /tmp/b. You are terse."}
    e.generate([a, {"role": "user", "content": "Say A."}], tools, 4)
    s1 = e.prompt_cache_stats()
    assert s1["forks"] == 2  # the tools fork and A's header fork
    e.generate([b, {"role": "user", "content": "Say B."}], tools, 4)
    s2 = e.prompt_cache_stats()
    assert (s2["fork_hits"], s2["forks"]) == (1, 3)  # started from the tools fork; added B's header
    reused_by_b = s2["reused_tokens"] - s1["reused_tokens"]
    assert reused_by_b >= FORK_MIN_TOKENS
    e.generate([b, {"role": "user", "content": "Say C."}], tools, 4)
    s3 = e.prompt_cache_stats()
    assert s3["fork_hits"] == 2
    assert s3["reused_tokens"] - s2["reused_tokens"] > reused_by_b  # B's header is deeper
    e.unload()


def test_vlm_a_turn_from_the_tools_fork_matches_its_cold_run_token_for_token():
    """Greedy, so the comparison is exact. The cold run is the same engine
    after a reset, so both runs split the prefill at the same boundaries and
    the comparison tests the copy, not prefill-shape ULP drift."""
    from sous.engine.vlm import VLMEngine

    e = VLMEngine(HYBRID_VLM, temperature=0.0, prompt_cache=True, cache_budget=1 << 34)
    tools = _fat_tools()
    a = {"role": "system", "content": "Session /tmp/a. You are terse."}
    b = {"role": "system", "content": "Session /tmp/b. You are terse."}
    brief = {"role": "user", "content": "In one sentence, what does tool_7 do?"}
    e.generate([a, brief], tools, 8)  # publishes the tools fork
    warm = e.generate([b, brief], tools, 32)  # from the tools fork
    assert e.prompt_cache_stats()["fork_hits"] == 1
    e.reset_prompt_cache()
    cold = e.generate([b, brief], tools, 32)  # a miss: the same prefill, split the same way
    assert e.prompt_cache_stats()["misses"] == 1
    assert warm == cold
    e.unload()
```

- [ ] **Step 3: Run the model tests locally**

```bash
uv run pytest tests/test_engine_lm.py tests/test_engine_vlm.py -m model -k "fork or probe or second_session or token_for_token" -v 2>&1 | tail -25
```

Expected: all PASS. Known things to check if not: `_fat_tools()` must push `len(stable)` past 4096 tokens on each model's tokenizer (raise `n` to 120 if a floor assertion fails); the `endswith` strings are the exact template output on the pinned snapshots (decision 4) — if a snapshot is updated and one fails, inspect `tokenizer.decode(stable[: bounds[0]])[-80:]` and adjust the expectation, never the code. The VLM engine's `_tokenizer` property is what the tests decode with.

- [ ] **Step 4: Run the CI checks (the model files are type-checked and linted even though CI never runs them)**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv run pytest -m "not model" -q 2>&1 | tail -3
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_engine_lm.py tests/test_engine_vlm.py
git commit -m "test(engine): prove the tools boundary on real templates and exact copies at two boundaries

Local-only model tests: Qwen3.5-9B (the hybrid, tools-first template)
yields a tools boundary ending at </IMPORTANT>\\n\\n shared by two sessions
and a header boundary that is not; Qwen3-0.6B (system-first) yields the
header only; a session started from the tools fork produces the same
tokens as its cold run under greedy decoding; and copies taken at two
boundaries in one prefill equal a three-segment cold prefill bit for bit.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Documentation

**Files:**
- Modify: `README.md` (~lines 236–243, ~253, ~368–382, ~520)
- Modify: `CLAUDE.md` (~lines 47–58)

**Interfaces:** none. Text only; every number below comes from the spec.

- [ ] **Step 1: README gateway bullet**

In the "One turn at a time; keyed prompt-cache slots" bullet, replace the sentences from `A new subagent whose rendered header is identical to one already seen` through `share only the first ~1K tokens.` with:

```markdown
  A new subagent starts from a *fork* when it can: a copy of an earlier
  conversation's cache taken at a boundary its own prompt shares. Two
  boundaries are kept per conversation long enough to clear them. The
  *tools* boundary is where the template's tool block ends — Qwen3.5/3.8
  render the `# Tools` block *before* the client's system text, and Claude
  Code's tool array is byte-identical across sessions and projects for a
  given subagent type (Explore, general-purpose, …), so a new `claude`
  process's first subagent turn starts ~45–56K tokens warm instead of paying
  ~170 s cold. The *header* boundary is where the whole system block ends,
  so a same-type subagent inside the same session starts ~57K tokens warm
  and prefills only its own brief. Each fork is a full copy of the KV at its
  boundary (~3.5 GiB at 57K tokens on the default model), two per tool set
  the daemon has seen, budgeted by `[model].prompt_cache_gb` below. Forks
  live as long as the weights: `[model].idle_unload_minutes` drops them with
  the model.
```

Keep the sentence `Two subagents still run one at a time; batching is a later phase.` that follows.

- [ ] **Step 2: README log-line sentence**

Replace `(`fork`: the turn started from a copied header slot)` with `(`fork`: the turn started from a copied fork slot — ~45–56K reused tokens is a tools fork, ~57K a header fork)`.

- [ ] **Step 3: README `prompt_cache_gb` paragraph**

Replace the paragraph beginning `` `[model].prompt_cache_gb` (default `"auto"`) bounds the caches kept resident `` (through `on its own.`) with:

```markdown
`[model].prompt_cache_gb` (default `"auto"`) bounds the caches kept resident
*beyond* the turn that is running: one slot per conversation, plus *fork*
slots at every boundary long enough to be worth copying (4096 tokens or
more): one at the end of the tool block, shared by every Claude Code session
and project that presents the same tool array, and one at the end of the
whole system block, shared by same-type subagents of one session (see
above). Prefixes must match token for token — one added, removed or
reordered tool is a different tool set with its own pair of forks. Each fork
is a copy of the KV at its boundary, ~3.4–3.6 GiB at ~57K tokens on the
default model, so a daemon that has seen the usual three tool sets holds up
to six forks, ~20 GiB. `"auto"` is what Metal's recommended working set has
left once the weights, one full context window of KV (the larger of
`[model]`'s and `[gateway]`'s) and 2 GiB of slack are paid for — about
27 GiB on a 64 GB machine with the default model and gateway window, room
for those forks and several conversations; a 48 GB machine should set it
lower, or to `0` to keep forks off. Slots are evicted least-recently-used
first when the budget, a count of 16, or live memory pressure says so; the
conversation that just ran is never evicted by its own turn, and a cold
turn's second fork copy never evicts its first, so `0` means exactly one slot
(the pre-3a behaviour) and a 32 GB machine degrades to that on its own.
Forks live as long as the weights do: `idle_unload_minutes` drops them with
the model, so "every new session" means every new session inside that
window.
```

- [ ] **Step 4: README one-line summary**

Replace `keyed prompt cache (one slot per conversation plus header forks, budgeted` with `keyed prompt cache (one slot per conversation plus tools and header forks, budgeted`.

- [ ] **Step 5: CLAUDE.md gotcha**

In the prompt-cache gotcha, replace the text from `A `fork` slot is a *copy* taken while` through `recurrent layers cannot.` (keeping the preceding `which drops the gateway's slots too.`) with:

```markdown
  A `fork` slot is a *copy* taken while
  a turn prefills past a boundary (`fork_point`, 4096-token floor). There are
  two: the *tools* boundary — Qwen3.5/3.8 render `# Tools` *before* the
  client's system text, so sessions and projects presenting the same tool
  array share ~45–56K rendered tokens (Phase 3a's "cross-session reuse is
  impossible" read the request body's order, not the render's) — and the
  *header* boundary, the whole system block. Both come from probe pairs
  (`promptcache.probe_boundaries`), never from rendering the system turn
  alone, which Qwen3.8's template refuses. Warm turns resolve the probe too:
  a turn that starts at another session's tools fork must still publish its
  own header fork. A `turn` slot is *moved* into the turn that extends it.
  Never rewind a cache to make a slot — a hybrid model's recurrent layers
  cannot.
```

- [ ] **Step 6: Check the rendered text and run the full CI set**

```bash
grep -n "one Claude Code session\|not across sessions\|copied header slot\|structurally impossible\|_header_probe\|subagents of one type" README.md CLAUDE.md src/sous/engine/*.py || echo "no stale claims"
uv run ruff format . && uv run ruff check . && uv run ruff format --check . && uv run ty check && uv lock --check && uv run pytest -m "not model" -q 2>&1 | tail -3
```

Expected: `no stale claims`; every check green.

- [ ] **Step 7: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: describe the tools-boundary fork and its memory cost

README's gateway and prompt_cache_gb sections said a fork pays off only
within one Claude Code session; with the tools boundary it pays off across
sessions and projects for each tool set, at two copies (~7 GiB) per tool
set, inside the idle-unload window. CLAUDE.md's gotcha names both
boundaries, the probe-pairs helper and why warm turns probe.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Runtime gate on the real daemon (Kyle's machine), the mlx-thread check, and the #41 note

**Files:** none modified in the repo. Reads `~/.sous/daemon.err.log`.

**Interfaces:** consumes the shipped `cache=fork reused_tokens=N` log line (`gateway/routes.py:_log_turn`) and `server_status`'s `prompt_cache` block.

The daemon runs from the installed uv tool (`~/.local/bin/sous serve`, unmanaged — not launchd; confirm with `ps -o pid,ppid,command -p $(pgrep -f 'sous serve')`). The gateway is enabled in `~/.sous/config.toml`. The idle window is 30 min: run the sessions back to back.

- [ ] **Step 1: Install the branch build and restart the daemon**

```bash
cd /Users/kyle.krcmaric/code/personal/sous && git status --short && uv tool install --force --reinstall . && sous stop; sleep 2; nohup sous serve >> ~/.sous/daemon.log 2>> ~/.sous/daemon.err.log & disown; sleep 3; tail -3 ~/.sous/daemon.err.log
```

Expected: a clean tree, `sous serve` starts and logs its bind line. If the previous daemon was launched differently (e.g. via `sous install-launchd`), mirror how it was launched instead of the `nohup` form.

- [ ] **Step 2: Session A — the sous repo, an Explore subagent, twice**

```bash
cd /Users/kyle.krcmaric/code/personal/sous && sous claude -p "Use the Explore subagent to find the file that logs each gateway turn, then reply with only its path." --max-turns 6
```

Then in the same shell, a second Explore in a *new* session of the same project (this exercises the tools fork with the same system text — expect a header-fork hit if the system text is identical, a tools-fork hit if the per-session path differs):

```bash
cd /Users/kyle.krcmaric/code/personal/sous && sous claude -p "Use the Explore subagent to find where FORK_MIN_TOKENS is defined, then reply with only the path." --max-turns 6
```

- [ ] **Step 3: Session B — a different project, an Explore subagent, then a second Explore in that session**

```bash
cd /Users/kyle.krcmaric/code/personal/mlx-dspark && sous claude -p "Use the Explore subagent to find where the README describes speculative decoding, then use the Explore subagent again to find the benchmark command. Reply with both file paths only." --max-turns 10
```

- [ ] **Step 4: A general-purpose subagent — a third tool set**

```bash
cd /Users/kyle.krcmaric/code/personal/sous && sous claude -p "Use a general-purpose subagent to count the test files under tests/ and reply with the number only." --max-turns 6
```

- [ ] **Step 5: Read the log and the status**

```bash
grep "cache=" ~/.sous/daemon.err.log | tail -12 | sed -E 's/^[^ ]+ //'
sous status 2>/dev/null | grep -i -A12 "prompt_cache" || echo "use the server_status MCP tool from a Claude Code session and read its prompt_cache block"
```

Exit criteria (from the spec):
- Session A's first Explore turn logs `cache=miss reused_tokens=0`, ~168 s; `server_status` then shows `forks` ≥ 2 (tools + header).
- Session B's first Explore turn logs `cache=fork` with `reused_tokens` between 45,000 and 56,999 and `seconds` between 10 and 35.
- Session B's second Explore turn logs `cache=fork` with `reused_tokens` ≥ 57,000 (its own header fork).
- The general-purpose turn logs `cache=miss` (a new tool set) and adds two forks; `slots` and `resident_bytes` stay under the budget; `evictions` did not remove a same-turn fork (the counts of forks after each cold turn go up by exactly 2).
- No `fork probe failed` / `fork copy failed` warnings in the log.

If session B's first turn is a `miss`: record the `input_tokens` and check whether the Explore tool arrays differ between the two projects in this Claude Code build (the spec's recording found them identical; a Claude Code update may have changed a description). That is a finding to report, not a code bug — PR 2's `lcp=` diagnostic is for exactly this. Report it in the PR.

- [ ] **Step 6: One real delegated task (the mlx-thread rule)**

From a Claude Code session with the sous MCP server attached (the desktop app's session is fine), delegate a trivial task and wait for it:

> Use `delegate_to_local_model` with project_root `/Users/kyle.krcmaric/code/personal/sous`, instructions "Create hello_phase3c.txt in the project root containing the single line: hello from the worker. Do nothing else.", then `sous wait <task_id>`.

Expected: the task reaches `done`; the daemon is still alive afterwards (`pgrep -f 'sous serve'` prints the same pid); `~/.sous/daemon.err.log` shows no segfault. Delete `hello_phase3c.txt` afterwards (`rm /Users/kyle.krcmaric/code/personal/sous/hello_phase3c.txt`) — it is test litter, not a deliverable.

- [ ] **Step 7: Draft the Phase 3c note for issue #41 and ask Kyle before posting**

Draft (fill in the measured numbers from Step 5):

```markdown
## Phase 3c: cross-session forks at the tools boundary — correcting the 3a ruling

Phase 3a's exit note said a new `claude` process shares only ~1K rendered tokens with an earlier one because the per-session scratchpad path sits "before ~56K tokens of tool schemas". That read the request body's order (`system` before `tools`), not the render's: Qwen3.8's template emits the `# Tools` block first and the client's system text last, so two sessions presenting the same tool array share ~99% of the rendered system block.

Spec: `docs/superpowers/specs/2026-09-05-gateway-phase3c-tools-fork-design.md`. Plan: `docs/superpowers/plans/2026-09-05-gateway-phase3c-tools-fork.md`. PR: #<n>.

What changed: the engines probe two boundaries (tools, header) and `PrefixCache` forks at each; warm turns resolve the probe so a session that starts at another's tools fork still publishes its own header fork; a cold turn's second copy never evicts its first.

Measured on the M5 Pro / 64 GB, default 27B, gateway window 131072:
- Session A (sous repo), Explore, turn 1: `cache=miss`, <N> s — publishes two forks.
- Session B (mlx-dspark), Explore, turn 1: `cache=fork reused_tokens=<N>`, <N> s (was ~168 s).
- Session B, second Explore: `cache=fork reused_tokens=<N>` (its header fork).
- general-purpose: `cache=miss`, a third tool set; `server_status` prompt_cache: slots=<N> resident_bytes=<N> forks=<N> evictions=<N>.

Not in this PR (PR 2): `lcp=` on misses, `speculative_block_size` default 3 with a ≤ 5 guard, disjoint `cache_read_input_tokens`.
```

Ask Kyle: "Post this on #41 as-is, with edits, or not at all?" Post with `gh issue comment 41 --body-file <file>` only on an explicit yes.

---

## Task 7: Pull request

- [ ] **Step 1: Push and open the PR**

```bash
cd /Users/kyle.krcmaric/code/personal/sous && git push -u origin feat/gateway-phase3c && gh pr create --base main --title "feat(engine): cross-session prompt-cache reuse via a tools-boundary fork (Phase 3c)" --body-file - <<'EOF'
## Summary

Phase 3a's "cross-session fork reuse is impossible" read the request body's order, not the render's. Qwen3.8's template renders the `# Tools` block *before* the client's system text, so two Claude Code sessions presenting the same tool array share ~99% of the rendered system block. This PR forks there.

- **`promptcache.probe_boundaries`**: two probe pairs. The header pair (unchanged) differs in the first user turn; the new tools pair differs at the first character of the system text, so its common prefix is the tool block. Each candidate still passes `fork_point` (strict prefix, 4096-token floor) against this turn's render. New memo slot `tools`.
- **`PrefixCache._fork_boundaries`** replaces `_fork_wanted`: a list, resolved on warm turns too (a session that starts at another's tools fork must still publish its own header fork, or its second subagent regresses from ~57K to ~45K reused). `_run` walks the list ascending.
- **Same-turn protection**: `_make_room`/`_publish`/`_evict_*` take a `protect` sequence; the second copy never evicts the first, and a budget with room for one copy skips the second (`_make_room -> bool`).
- Engines: `_header_probe` → `_fork_probe`, thin wrapper over the shared helper. README/CLAUDE.md corrected.

Spec: `docs/superpowers/specs/2026-09-05-gateway-phase3c-tools-fork-design.md`. Plan: `docs/superpowers/plans/2026-09-05-gateway-phase3c-tools-fork.md`. Continues #41.

## Runtime gate (M5 Pro, 64 GB, default 27B)

<paste the Step 5 log lines and status from the plan's Task 6>

## Test plan

- `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check` — green.
- Model tests (local): Qwen3.5-9B tools boundary ends at `</IMPORTANT>\n\n` and is shared by two sessions; Qwen3-0.6B (system-first template) yields the header only; warm-from-tools-fork output equals the cold run token for token under greedy decoding; two-boundary fork copies equal a three-segment cold prefill bit for bit.
- One real delegated task after the change (mlx thread-state rule): daemon alive.

Not in this PR (PR 2): `lcp=` on misses, `speculative_block_size` default 3 with a ≤ 5 guard, disjoint `cache_read_input_tokens`.
EOF
```

- [ ] **Step 2: Watch CI and review rounds**

```bash
gh pr checks --watch
```

Expected: the four jobs green. Address Copilot review comments per `superpowers:receiving-code-review` (verify each against the code before acting; reply in-thread). Known review-bait to have answers ready for: the `Sequence` isinstance narrowing (decision 1), why warm turns pay the probe (decision 3 and the `_fork_boundaries` docstring), why the turn-slot publish does not protect same-turn forks (decision 10), and why `published` is dropped before decode (decision 11).
