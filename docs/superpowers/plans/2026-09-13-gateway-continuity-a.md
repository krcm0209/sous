# Gateway Cache Continuity A — In-Place System Messages, Retained Turn Slots — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A subagent conversation on the local model stays append-only from its first turn to its last: a Claude Code attachment arriving mid-conversation as a `role: "system"` message costs its own tokens instead of a re-prefill of everything behind it, and a progress-summary call that branches the conversation no longer consumes the slot the real conversation extends next.

**Architecture:** Two changes on the request→render→cache path. (A1) `gateway/convert.py` places every inline `role: "system"` message by position in a pre-pass — leading ones fold into the template's single system turn as today; one after a user message becomes a `<system-reminder>` text block appended to that user message (Claude Code's own legacy shape); one after an assistant message becomes a user message of its own — so nothing after index 0 ever renders as a system turn and the header boundary stops moving. (A2) `engine/promptcache.py` copies a taken `turn` slot and leaves it in the map whenever the byte budget can hold the copy, moving it only when it cannot (always at `prompt_cache_gb = 0`); `Slot.parent` lineage drops a linear conversation's grandparent at publish so a conversation holds at most its current and previous lengths. `TurnRunner`/`_log_turn` print `took=turn-moved@N` for the moved case so the log tells the two apart.

**Tech Stack:** Python 3.14, stdlib `weakref`, pytest with the existing `FakeHooks`/`FakeEngine` seams. No new dependencies. No mlx code changes.

**Spec:** `docs/superpowers/specs/2026-09-10-gateway-cache-continuity-design.md` — "What the 13-minute session established" (causes 2 and 3), "Design" A1 and A2, "Invariants preserved", "Testing" (A1, A1 end to end, A2, Model tests), "Documentation". Part B (preload and hold) is a separate plan. The turn-line fields this plan is verified with are the observability PR 1's (`docs/superpowers/specs/2026-09-10-gateway-observability-design.md`, merged as #85).

## Global Constraints

- Python `>=3.14`; `except A, B:` without parentheses is valid 3.14 syntax — do not "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`; `ty` flags an *unused* pragma as its own diagnostic — add one only when `ty check` names the rule.
- `mlx` / `mlx_lm` / `mlx_vlm` imports stay function-local. `engine/promptcache.py` imports no mlx at all; this plan keeps it that way (array copies go through the injected `copy_array`).
- The gateway never logs a request body, a header value or a query string. Nothing in this plan adds a log field that is not a count, a duration or a label from a fixed set.
- Never rewind a cache to make a slot; a hybrid model's recurrent layers cannot. The retained copy is `fork_copy` into a fresh cache, exactly the fork path's shape.
- Prompt-cache slots are owned by the thread that built them and looked up only by it; the retained copy is made on the owning session thread inside `generate`, where the fork copy already is.
- Tests never touch the real `~/.sous` or `~/.claude`: every path comes from `tmp_path`.
- Never edit, reformat or "sync" anything under `docs/superpowers/**` (this plan and its spec included) once committed.
- **No plan language in committed code.** Code and test comments/docstrings must never cite a spec, a plan, a task or a step ("Task 1's helper", "Spec:", "the spec promises", "docs/superpowers/…"). Where a code block below carries such a reference, restate the underlying fact instead. Sweep before the PR: `grep -rnE 'Task [0-9]|Step [0-9]|[Ss]pec[: ]|docs/superpowers' src tests`.
- Never loosen an existing assertion to make a test pass. When this plan changes a behaviour an existing test pinned (named per task below), restate the test's expected value and its docstring; if a test's premise is gone, delete it and say so in the commit body.
- Commits: Conventional Commits, imperative lowercase subject, *why* in the body, trailer exactly `Co-Authored-By: Claude <noreply@anthropic.com>` — model-less, whatever the harness suggests.
- Verification before every commit: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` (never pipe without `pipefail`; `pyproject.toml` already sets `-q`, so never add another `-q` — it suppresses the summary line), `uv run ty check`, `uv run ruff format .` then `uv run ruff check . && uv run ruff format --check .`. Line length is 100 and `E501` is enforced: `ruff format` reflows code but never splits a string literal, so a long expected string is written as adjacent literals over two lines (the code blocks below already do this where it matters — keep it that way when copying).
- `model`-marked tests download or load real weights and are not run by task implementers. The orchestrator runs them on the M5 Pro over `ssh m5pro` (checkout `~/code/personal/sous-remote`) after Task 7 and records the result in the ledger and the PR.
- Branch: `feat/gateway-continuity-a` off `main` at `1bfe6e0`. `main` is protected: PR with all four CI jobs green.
- Kyle's daemon (on the M5 Pro, unmanaged, pid in `~/.sous/daemon.lock`) is not restarted by any task. The real-daemon gate at the end is the orchestrator's, after Kyle's explicit go-ahead, since it installs branch code over his working `sous`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/sous/gateway/convert.py` | `_place_inline_system` (the pre-pass), `_reminder`, `_with_block`; `_system_text(system, leading)` takes the leading texts instead of the whole message list; `chat_messages` renders the placed list; `_user_turns.flush` strips each text block before joining. |
| `src/sous/engine/promptcache.py` | `PromptCacheStats.retained` / `.moved`; `Slot.parent` (a weak reference); `_take` never removes; `_make_room(require=)` refuses a slot no longer in the map; `generate` decides retain-or-move after a turn-slot take and runs that take's cap pass after the decision; `_publish` calls `_drop_grandparent`. |
| `src/sous/gateway/turn.py` | `TurnResult.moved`; `run()` reads it as a counter delta. |
| `src/sous/gateway/routes.py` | `_log_turn` prints `took=turn-moved@N`. |
| `tests/test_gateway_convert.py`, `tests/test_gateway_routes.py`, `tests/test_promptcache.py`, `tests/test_gateway_turn.py`, `tests/test_gateway_model.py` | Tests per task below. |
| `README.md`, `CLAUDE.md` | Docs per Task 6. |

## Decisions this plan makes beyond the spec's text

1. **`Slot.parent` is a `weakref.ref`, not a plain reference cleared by hand.** The spec's requirement is that a dropped grandparent's arrays are freed, not pinned by lineage. A weak reference gives exactly that with no bookkeeping in the four paths a slot can leave the map by (`_drop_lru`, `_drop_grandparent`, `reset`, `_sweep`); a plain reference would need each of them to walk the map and clear links. The lineage check dereferences the weakref and still confirms membership by identity.
2. **`_user_turns.flush` strips each text block before joining, not the joined text.** Claude Code emits the `<total_tokens>` marker as the last text block after every tool-result batch, and A1 appends the reminder block after it; stripping the joined text leaves the marker block's trailing newline and the join's newline in front of the reminder (`"\n\n<system-reminder>…"`). Per-block stripping drops a marker-only block entirely. A marker inside a longer block strips exactly as before. The one render that changes besides the new shape: an *empty* text block (`{"type": "text", "text": ""}`) or a marker-only block that was not last used to contribute the join's newline (`"a\n"`), and now contributes nothing; no test pinned that newline and no client is known to send it.
3. **`took=turn-moved@N` comes from the `moved` counter's before/after delta**, the same way `cache=fork` comes from `fork_hits`; no new gauge.
4. **The model test is parameterized over two models:** the tiny `mlx-community/Qwen3-0.6B-4bit` through `LMEngine` (what the file's existing test uses; runs anywhere the weights fit) and the real `mlx-community/Qwen3.8-27B-4bit` through `VLMEngine` (the hybrid model whose recurrent layers are the reason the copy-then-extend shape exists). `mx.random.seed` before each generation makes the token-for-token comparisons meaningful. The orchestrator runs both on the M5 Pro; the 27B run is the spec's model test, and the real-daemon gate is the incident replay on top of it.

---

### Task 1: Inline system messages render in place (`convert.py`)

**Files:**
- Modify: `src/sous/gateway/convert.py` (`_system_text` at lines 131–139, `_user_turns.flush` at 172–180, `chat_messages` at 248–264)
- Test: `tests/test_gateway_convert.py` (rewrite lines 65–86; `test_gate_capture_shape_converts_cleanly` at 537–584), `tests/test_gateway_routes.py` (`test_prompt_conversion_reaches_the_engine` at 545–566)

**Interfaces:**
- Consumes: `strip_volatile`, `_text_parts`, `_user_turns`, `_assistant_turn`, `_invalid` (all present).
- Produces: `convert._place_inline_system(messages: list[dict]) -> tuple[list[str], list[dict]]`; `convert._reminder(text: str) -> str`; `convert._with_block(msg: dict, block: dict) -> dict`; `convert._system_text(system: object, leading: list[str]) -> str` (signature change). `chat_messages(system, messages)` keeps its signature and is what Task 2 and Task 7 exercise end to end.

- [ ] **Step 1: Write the failing tests**

Replace `test_inline_system_messages_fold_into_the_leading_system_message` and `test_inline_system_without_a_canonical_field_still_leads` (lines 65–86 of `tests/test_gateway_convert.py`) with:

```python
def test_a_leading_inline_system_message_folds_into_the_system_prompt():
    """Nothing precedes it, so the template's one system turn (index 0) is
    where it belongs — canonical field first, as before."""
    messages = [
        {"role": "system", "content": "Inline instructions."},
        {"role": "system", "content": [{"type": "text", "text": "More."}]},
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
    ]
    out = chat_messages([{"type": "text", "text": "Canonical."}], messages)
    assert out == [
        {"role": "system", "content": "Canonical.\n\nInline instructions.\n\nMore."},
        {"role": "user", "content": "task"},
    ]


def test_a_leading_inline_system_without_a_canonical_field_still_leads():
    out = chat_messages(None, [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}])
    assert out[0] == {"role": "system", "content": "S"}


def _tool_exchange(*trailing: dict) -> list[dict]:
    """A user brief, one tool call, its result with the volatile marker after
    it (Claude Code's shape), then whatever the test appends."""
    return [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "A"},
                {"type": "text", "text": "<total_tokens>123 tokens left</total_tokens>\n"},
            ],
        },
        *trailing,
    ]


def _wrapped(*texts: str) -> str:
    """The user-turn content the converter renders for inline system text:
    one <system-reminder> block per message, newline-joined like any other
    text blocks in a user message."""
    return "\n".join(f"<system-reminder>\n{t}\n</system-reminder>" for t in texts)


LISTING = "Available agent types for the Agent tool:\n- Explore: read-only"
WRAPPED_LISTING = _wrapped(LISTING)


def test_an_inline_system_message_after_tool_results_renders_as_a_user_turn_after_them():
    """Claude Code inserts its attachments (agent listings, MCP instructions,
    deferred tools) as a `role: "system"` message after the preceding user
    message. Rendered where Claude Code's own fallback for models without that
    feature puts it — a <system-reminder> block in that user message — it lands
    after the tool turns, and the request without it renders a strict prefix,
    so the earlier turn's cache slot still serves this one."""
    without = chat_messages(None, _tool_exchange())
    with_it = chat_messages(None, _tool_exchange({"role": "system", "content": LISTING}))
    assert with_it[: len(without)] == without
    assert with_it[len(without) :] == [{"role": "user", "content": WRAPPED_LISTING}]
    assert without[-1] == {"role": "tool", "content": "A"}


def test_consecutive_inline_system_messages_merge_in_order_one_block_each():
    out = chat_messages(
        None,
        _tool_exchange(
            {"role": "system", "content": "First."},
            {"role": "system", "content": [{"type": "text", "text": "Second."}]},
        ),
    )
    assert out[-1] == {"role": "user", "content": _wrapped("First.", "Second.")}


def test_a_pre_wrapped_inline_system_message_is_not_wrapped_again():
    out = chat_messages(None, _tool_exchange({"role": "system", "content": WRAPPED_LISTING}))
    assert out[-1] == {"role": "user", "content": WRAPPED_LISTING}


def test_a_marker_only_inline_system_message_renders_nothing():
    """The bare <total_tokens> reminder arrives as its own system message on
    some paths; stripped, it is empty, and an empty block would still cost a
    <system-reminder> wrapper the next request does not repeat byte for byte."""
    marker = {"role": "system", "content": "<total_tokens>5 tokens left</total_tokens>"}
    assert chat_messages(None, _tool_exchange(marker)) == chat_messages(None, _tool_exchange())
    assert chat_messages("Sys.", [marker, {"role": "user", "content": "q"}]) == [
        {"role": "system", "content": "Sys."},
        {"role": "user", "content": "q"},
    ]


def test_an_inline_system_message_after_an_assistant_turn_is_its_own_user_turn():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "Working."},
        {"role": "system", "content": "Note."},
        {"role": "system", "content": "Note 2."},
    ]
    assert chat_messages(None, messages)[2:] == [
        {"role": "user", "content": _wrapped("Note.", "Note 2.")}
    ]


def test_an_inline_system_message_appends_to_a_string_user_message():
    out = chat_messages(None, [{"role": "user", "content": "hi"}, {"role": "system", "content": "S"}])
    assert out == [{"role": "user", "content": "hi\n" + _wrapped("S")}]
    empty = chat_messages(None, [{"role": "user", "content": ""}, {"role": "system", "content": "S"}])
    assert empty == [{"role": "user", "content": _wrapped("S")}]


def test_nothing_after_index_zero_renders_as_a_system_turn():
    """The chat template accepts a system message only at index 0."""
    out = chat_messages(
        "Sys.", _tool_exchange({"role": "system", "content": LISTING}, {"role": "user", "content": "go"})
    )
    assert [m["role"] for m in out[1:]].count("system") == 0
    assert out[0]["role"] == "system"


def test_the_request_body_is_not_mutated_by_placement():
    messages = _tool_exchange({"role": "system", "content": LISTING})
    before = json.dumps(messages, sort_keys=True)
    chat_messages(None, messages)
    assert json.dumps(messages, sort_keys=True) == before


def test_a_marker_block_followed_by_a_text_block_leaves_no_leading_newline():
    """Each text block is stripped on its own before the join: the marker is
    the last block Claude Code writes, and the block appended after it must
    not inherit the join's newline."""
    content = [
        {"type": "text", "text": "<total_tokens>1 tokens left</total_tokens>\n"},
        {"type": "text", "text": "after"},
    ]
    assert chat_messages(None, [{"role": "user", "content": content}]) == [
        {"role": "user", "content": "after"}
    ]
```

Add `import json` to the file's imports. Update `test_gate_capture_shape_converts_cleanly` (lines 575–584): the system turn no longer carries the inline text and the user turn gains it:

```python
    assert chat.messages == [
        {
            "role": "system",
            "content": (
                "You are Claude Code, Anthropic's official CLI for Claude.\n\nLong system prompt."
            ),
        },
        {
            "role": "user",
            "content": (
                "Agent prompt.\nEnvironment.\n"
                "<system-reminder>\nInline subagent system text.\n</system-reminder>"
            ),
        },
    ]
```

and its docstring's last clause to "then a plain-string inline system message, rendered into the user turn". In `tests/test_gateway_routes.py::test_prompt_conversion_reaches_the_engine` (lines 562–565), expect:

```python
    assert inner.calls[0] == [
        {"role": "system", "content": "Canonical."},
        {"role": "user", "content": "task\n<system-reminder>\nInline.\n</system-reminder>"},
    ]
```

and change its docstring to "Billing header and the volatile marker stripped, the inline system message rendered into the user turn: the engine sees one stable system message, so the prefix cache can hold across turns."

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_convert.py tests/test_gateway_routes.py::test_prompt_conversion_reaches_the_engine`
Expected: six new tests FAIL because the inline message is still hoisted into the system turn — `…after_tool_results_renders_as_a_user_turn…` (`out[-1]` is the tool turn), `…consecutive…`, `…pre_wrapped…`, `…after_an_assistant_turn…`, `…appends_to_a_string_user_message` — and `test_a_marker_block_followed_by_a_text_block_leaves_no_leading_newline` FAILS with content `"\n\nafter"` (the block's own trailing newline plus the join's). The two rewritten existing tests (`test_gate_capture_shape_converts_cleanly`, `test_prompt_conversion_reaches_the_engine`) FAIL on their new expected values. The other five new tests — the two `…leading…` tests, `…marker_only_inline_system_message_renders_nothing`, `…nothing_after_index_zero…`, `…request_body_is_not_mutated…` — PASS already: they pin what must not change and must still pass after Step 3.

- [ ] **Step 3: Implement the pre-pass**

In `src/sous/gateway/convert.py`, replace `_system_text` (lines 131–139) with:

```python
_REMINDER_OPEN = "<system-reminder>"


def _reminder(text: str) -> str:
    """System text delivered inside a user message, in the shape Claude Code
    itself uses for models without mid-conversation system support. Text it
    already wrapped (it does so for one model id) is left alone."""
    if text.startswith(_REMINDER_OPEN):
        return text
    return f"{_REMINDER_OPEN}\n{text}\n</system-reminder>"


def _with_block(msg: dict, block: dict) -> dict:
    """A copy of user message `msg` with `block` appended to its content. A
    string content becomes one text block first (an empty string, none)."""
    content = msg.get("content")
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}] if content else []
    elif isinstance(content, list):
        blocks = list(content)
    else:
        raise _invalid("content must be a string or a list of content blocks")
    return {**msg, "content": [*blocks, block]}


def _place_inline_system(messages: list[dict]) -> tuple[list[str], list[dict]]:
    """Consume every `role: "system"` message by position. Qwen's template
    accepts a system turn at index 0 only, and hoisting a later one there
    moves the whole system block: every cached prefix behind it dies on the
    turn the attachment arrives. So: before any user or assistant message it
    joins the system prompt; after a user message it becomes a text block on
    that message (rendered after that message's tool results, as a user
    turn); after an assistant message it becomes a user message of its own.
    Consecutive ones land in the same place, in order; one that strips to
    nothing lands nowhere, so the render equals the request's without it.
    Returns the leading texts and the rewritten list; `messages` is not
    mutated."""
    leading: list[str] = []
    placed: list[dict] = []
    # Index in `placed` of the user message the next system message appends
    # to; None while nothing precedes it or an assistant message was last.
    target: int | None = None
    for msg in messages:
        role = msg.get("role")
        if role != "system":
            placed.append(msg)
            target = len(placed) - 1 if role == "user" else None
            continue
        text = strip_volatile("\n\n".join(_text_parts(msg.get("content"), drop_billing=True)))
        if not text:
            continue
        if not placed:
            leading.append(text)
            continue
        block = {"type": "text", "text": _reminder(text)}
        if target is None:
            placed.append({"role": "user", "content": [block]})
            target = len(placed) - 1
        else:
            placed[target] = _with_block(placed[target], block)
    return leading, placed


def _system_text(system: object, leading: list[str]) -> str:
    """One system prompt from the canonical field plus the inline system
    messages that precede every user and assistant message, canonical first."""
    parts = _text_parts(system, drop_billing=True) if system is not None else []
    return strip_volatile("\n\n".join([*parts, *leading]))
```

In `_user_turns`, change `flush` (lines 172–180) so each block is stripped before the join:

```python
    def flush() -> None:
        # Claude Code appends the <total_tokens> marker as its own text block
        # after every tool-result batch. Stripped block by block, a marker-only
        # block contributes nothing — not even the join's newline in front of
        # a block that follows it — and an empty user turn right after the
        # tool responses would read to the model as "the user said nothing".
        stripped = [t for t in (strip_volatile(t) for t in texts) if t]
        texts.clear()
        if stripped:
            out.append({"role": "user", "content": "\n".join(stripped)})
```

Replace `chat_messages` (lines 248–264) with:

```python
def chat_messages(system: object, messages: list[dict]) -> list[dict]:
    """The chat-template message list for a request. Raises RequestError."""
    leading, placed = _place_inline_system(messages)
    out: list[dict] = []
    prompt = _system_text(system, leading)
    if prompt:
        out.append({"role": "system", "content": prompt})
    for msg in placed:
        role = msg.get("role")
        if role == "user":
            out.extend(_user_turns(msg.get("content")))
        elif role == "assistant":
            out.append(_assistant_turn(msg.get("content")))
        else:
            raise _invalid(f"messages: role must be one of {_ROLES}, got {role!r}")
    return out
```

`_ROLES` keeps `"system"` — `parse_messages_request` still validates it as a role. Update the module docstring's list of what is dropped (lines 3–7) to add: "and a mid-conversation system message is rendered into the user turn before it rather than hoisted into the system block".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gateway_convert.py tests/test_gateway_routes.py`
Expected: PASS. Then the whole suite: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` — any other test that asserted the hoist is updated per the Global Constraints rule (restate, never loosen).

- [ ] **Step 5: Type-check, lint, commit**

```bash
uv run ty check && uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add src/sous/gateway/convert.py tests/test_gateway_convert.py tests/test_gateway_routes.py
git commit -m "feat(gateway): render mid-conversation system messages in place

Claude Code delivers its attachments (agent listings, MCP instructions,
deferred tools) as a role:system message inserted after the preceding
user message. Hoisting it into the template's single system turn moved
the whole system block, so every cached prefix behind it died on the
turn the attachment arrived — a 200 s re-prefill on the first real
session. Rendered where Claude Code's own fallback for models without
mid-conversation system puts it, a <system-reminder> block in the
preceding user message, it costs its own tokens and nothing else.

Text blocks are now volatile-stripped one by one before the join, so
the reminder block appended after the marker block does not inherit
the join's newline.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: A1 end to end — three requests shaped like the incident stay warm

**Files:**
- Test: `tests/test_gateway_routes.py` (append after `test_the_log_carries_lcp_only_on_a_miss`, ~line 909)

**Interfaces:**
- Consumes: `chat_messages` behaviour from Task 1; `sous.engine.promptcache.reuse_length`; the file's `_app`, `_post`, `_body`, `_turn_lines`, `_fields`, `FakeEngine`.
- Produces: nothing later tasks use; pins that the routes layer feeds the engine a strictly extending render across the T1→T3→T5 shape.

- [ ] **Step 1: Write the failing test**

The scripted `FakeEngine` reports whatever `stats` it is given, so this test brings its own engine that measures reuse the way the real cache does — a strict-prefix test on the render it receives:

```python
class _PrefixReuseEngine(FakeEngine):
    """Reports a hit exactly when the render it receives strictly extends the
    previous one, with `reused_tokens` the previous render's length — the
    prompt cache's own rule, on a token-per-character stand-in."""

    def __init__(self, script: list[str]):
        super().__init__(script)
        self._held: list[int] = []
        self.lengths: list[int] = []  # each render's length, in the order served
        self.stats = {"hits": 0, "misses": 0, "fork_hits": 0, "reused_tokens": 0}

    def generate(self, messages, tools, max_tokens, on_delta=None):
        from sous.engine.promptcache import reuse_length

        ids = list(b"".join(json.dumps(m, sort_keys=True).encode() + b"\n" for m in messages))
        reused = reuse_length(self._held, ids)
        self.stats = {**self.stats, "reused_tokens": self.stats["reused_tokens"] + reused}
        self.stats["hits" if reused else "misses"] += 1
        self._held = ids
        self.lengths.append(len(ids))
        return super().generate(messages, tools, max_tokens, on_delta)


def test_an_attachment_and_a_summary_call_leave_the_conversation_warm(tmp_path: Path, capsys):
    """T1 the brief, T3 the same conversation plus a tool exchange and a
    role:system attachment after the tool result, T5 one more exchange: each
    render strictly extends the one before, so the second and third turns
    are hits that reuse the whole previous render. Hoisting the attachment
    into the system turn made T3 a miss at the header."""
    inner = _PrefixReuseEngine(["one", "two", "three"])
    app = _app(tmp_path, inner)
    system = [{"type": "text", "text": "You are Claude Code."}]
    t1 = [{"role": "user", "content": "Summarise the repo."}]
    t3 = [
        *t1,
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": "x"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "contents of x"},
                {"type": "text", "text": "<total_tokens>900 tokens left</total_tokens>\n"},
            ],
        },
        {"role": "system", "content": "Available agent types for the Agent tool:\n- Explore"},
    ]
    t5 = [
        *t3,
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "b", "name": "Read", "input": {"file_path": "y"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "b", "content": "contents of y"},
                {"type": "text", "text": "<total_tokens>800 tokens left</total_tokens>\n"},
            ],
        },
    ]
    for messages in (t1, t3, t5):
        r = _post(app, _body(system=system, tools=[READ_TOOL], messages=messages))
        assert r.status_code == 200
    lines = [_fields(line) for line in _turn_lines(capsys.readouterr().err)]
    assert [f["cache"] for f in lines] == ["miss", "hit", "hit"]
    # Each hit reused exactly the previous render — the whole of it.
    assert [f["reused_tokens"] for f in lines] == ["0", str(inner.lengths[0]), str(inner.lengths[1])]
    assert lines[1]["system"] == lines[0]["system"] == lines[2]["system"]
    # The attachment rendered as a user turn after the tool turn, not into the system block.
    assert inner.calls[1][-1] == {
        "role": "user",
        "content": "<system-reminder>\nAvailable agent types for the Agent tool:\n- Explore\n"
        "</system-reminder>",
    }
    assert inner.calls[1][: len(inner.calls[0])] == inner.calls[0]
    assert inner.calls[2][: len(inner.calls[1])] == inner.calls[1]
```

- [ ] **Step 2: Run it to verify it discriminates**

The converter change is already committed, so put `main`'s converter back for one run:

Run: `git checkout main -- src/sous/gateway/convert.py && uv run pytest tests/test_gateway_routes.py -k attachment_and_a_summary; git checkout HEAD -- src/sous/gateway/convert.py`
Expected: FAIL on `["miss", "hit", "hit"]` — with the hoist, the second render's system turn differs, so it is `["miss", "miss", "hit"]` and the system hashes differ. Then with the converter restored: PASS. Confirm `git status` shows `convert.py` unmodified before continuing.

- [ ] **Step 3: Run the file and commit**

```bash
set -o pipefail; uv run pytest tests/test_gateway_routes.py 2>&1 | tail -3
uv run ty check && uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add tests/test_gateway_routes.py
git commit -m "test(gateway): pin that an attachment keeps a subagent conversation warm end to end

The 13-minute session's T1→T3→T5 shape, through the route and the
converter, against an engine that reports reuse the way the prompt
cache does: every render strictly extends the previous one.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Retain a taken turn slot when the budget holds the copy (`promptcache.py`)

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`PromptCacheStats` at lines 310–358; `Slot` at 421–437; `_take` at 565–580; `generate` at 825–904 and the publish at 976)
- Test: `tests/test_promptcache.py` (`test_stats_as_dict_reports_every_counter` at 874–905; the keyed-slot tests at 983–1057; new tests after them)

**Interfaces:**
- Consumes: `fork_copy`, `_make_room`, `_evict_caps`, `Slot`, `PromptCacheStats` (present).
- Produces: `PromptCacheStats.retained: int`, `PromptCacheStats.moved: int` (counters — summed by `add`, not gauges); `Slot.parent: weakref.ref[Slot] | None` (default `None`, `compare=False`); `_take` leaves every slot in the map; `_make_room(nbytes, protect=(), *, require: Slot | None = None) -> bool`; `PrefixCache.generate` publishes the turn slot with `parent` set to the retained slot it copied. Task 4 builds the lineage rule on `Slot.parent`; Task 5 reads `moved` as a delta.

- [ ] **Step 1: Write the failing tests**

In `tests/test_promptcache.py`, extend `test_stats_as_dict_reports_every_counter` — construct with `retained=2, moved=1` and add `"retained": 2, "moved": 1` to the expected dict (after `"pressure_evictions": 2`). Add to the `add` tests:

```python
def test_stats_add_sums_retained_and_moved():
    a = PromptCacheStats(retained=1, moved=2)
    a.add(PromptCacheStats(retained=3, moved=1))
    assert (a.retained, a.moved) == (4, 3)
```

Delete `test_a_turn_slot_is_consumed_by_the_turn_that_extends_it` (line 997): its premise is gone at a budget that holds the copy, and the moved case is pinned below by the budget-of-zero test. Adjust `test_two_interleaved_conversations_both_reuse` (line 983) — its `len(h.caches) == 2` becomes `4`, with the comment "each warm turn works on a copy of its slot; the slot itself stays for a branch to find". Then add, after `test_a_fork_the_turn_selected_survives_the_pre_turn_eviction_pass`:

```python
def test_a_turn_slot_is_retained_when_the_budget_holds_the_copy():
    """Claude Code branches a running subagent's conversation with progress
    summary calls: the same prefix, a different last turn. A slot the
    extending turn consumed would leave the next real turn nothing to start
    from; copied and left in place, both branches start warm."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    pc.generate(A2, A2_FULL, 16)
    assert sorted(s.held for s in pc.slots()) == sorted([A1, A2])
    s = pc.stats()
    assert (s["hits"], s["retained"], s["moved"], s["reused_tokens"]) == (1, 1, 0, len(A1))
    assert h.decoded[-1] == A2_FULL[len(A1) :]
    turn1, turn2 = [c for c in h.caches]
    assert cast(FakeTrimmable, turn1[0]).offset == len(A1)  # the original is untouched
    assert cast(FakeTrimmable, turn2[0]).offset == len(A2)


def test_a_retained_slot_serves_a_branch_and_then_the_real_conversation():
    """real → summary → real → real: the summary call takes A1's slot and
    publishes its own; the real conversation still finds A1's, then its own."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    summary = [*A1, 500]
    pc.generate(A1, A1_FULL, 16)
    pc.generate(summary, [*summary, 90, 91], 16)
    pc.generate(A2, A2_FULL, 16)
    a3 = [*A2, 7]
    pc.generate(a3, [*a3, 90, 91], 16)
    s = pc.stats()
    assert (s["hits"], s["misses"], s["retained"]) == (3, 1, 3)
    assert h.decoded[1] == [500, 90, 91]
    assert h.decoded[2] == A2_FULL[len(A1) :]
    assert h.decoded[3] == [7, 90, 91]


def test_a_turn_slot_is_moved_when_the_budget_cannot_hold_the_copy():
    """The copy is charged before it is allocated, like a fork's: when the
    budget cannot hold it beside the original, the slot is removed and its
    arrays adopted — today's behaviour, and the only path at a budget of 0."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    (slot,) = pc.slots()
    pc.max_bytes = slot.nbytes  # room for the original, not for a copy beside it
    pc.generate(A2, A2_FULL, 16)
    assert [s.held for s in pc.slots()] == [A2]
    s = pc.stats()
    assert (s["hits"], s["retained"], s["moved"], s["evictions"]) == (1, 0, 1, 0)
    assert len(h.caches) == 1  # no copy: the arrays were adopted
    assert h.decoded[-1] == A2_FULL[len(A1) :]


def test_at_a_budget_of_zero_a_turn_slot_is_always_moved():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)  # max_bytes=0: exactly one slot
    pc.generate(A1, A1_FULL, 16)
    pc.generate(A2, A2_FULL, 16)
    a3 = [*A2, 7]
    pc.generate(a3, [*a3, 90, 91], 16)
    assert [s.held for s in pc.slots()] == [a3]
    s = pc.stats()
    assert (s["hits"], s["moved"], s["retained"]) == (2, 2, 0)
    assert len(h.caches) == 1


def test_a_failed_turn_slot_copy_falls_back_to_moving_it():
    """A copy failure is an optimization failure: the turn still runs warm,
    on the original arrays, and the slot is moved rather than retained."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    _fail_next_copy(h)
    with pytest.warns(UserWarning, match="turn-slot copy failed"):
        pc.generate(A2, A2_FULL, 16)
    assert [s.held for s in pc.slots()] == [A2]
    s = pc.stats()
    assert (s["hits"], s["retained"], s["moved"], s["cold_retries"]) == (1, 0, 1, 0)
    assert h.decoded[-1] == A2_FULL[len(A1) :]


def test_the_taken_turn_slot_survives_the_pre_turn_cap_pass():
    """Between the take and the retain-or-move decision the slot is still in
    the map, and a map can be over budget when a turn starts (a slot is never
    evicted by its own publish). The cap pass in that window must not drop
    the very slot the turn chose: its arrays live on in the turn either way,
    so dropping it would count an eviction the conversation never had."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    (slot,) = pc.slots()
    pc.max_bytes = slot.nbytes - 1  # over budget before the next turn starts
    pc.generate(A2, A2_FULL, 16)
    s = pc.stats()
    assert (s["hits"], s["moved"], s["evictions"]) == (1, 1, 0)
    assert [x.held for x in pc.slots()] == [A2]


def test_the_pressure_valve_can_still_take_a_retained_slot():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    h.pressure_value = KERNEL_PRESSURE_WARN  # one drop per publish, the new slot protected
    pc.generate(A2, A2_FULL, 16)
    assert [s.held for s in pc.slots()] == [A2]
    s = pc.stats()
    assert (s["retained"], s["pressure_evictions"]) == (1, 1)


def test_a_take_that_ends_in_a_move_does_not_charge_the_moved_bytes_against_other_slots():
    """A moved slot's bytes leave the map, so the cap pass for a turn take
    runs after the retain-or-move decision: a budget that holds exactly one
    slot must not evict another conversation's slot to make room for bytes
    that were about to leave anyway."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    pc.generate(B1, B1_FULL, 16)
    (a_slot,) = [s for s in pc.slots() if s.held == A1]
    pc.max_bytes = a_slot.nbytes  # one slot's worth: the copy is refused, so A1's slot is moved
    seen: dict = {}

    def look_while_the_turn_runs(hooks, cache, token_ids, max_tokens):
        seen["held"] = [s.held for s in pc.slots()]
        hooks.decoded.append(list(token_ids))
        return "text"

    h.decode_impl = look_while_the_turn_runs
    pc.generate(A2, A2_FULL, 16)
    assert seen["held"] == [B1]  # A1's slot moved out; B1's was not evicted to make room for it
    assert pc.stats()["moved"] == 1


def test_a_slot_another_owner_evicted_after_the_take_is_not_copied():
    """LRU is daemon-wide and the lock is released between the take and the
    copy decision, so another owner's publish can drop the taken slot in
    that window. The decision re-checks residency under its own lock and
    refuses the copy: a full copy of a slot nobody can take again, made
    under the very pressure that evicted it, is the wrong answer."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(A1, A1_FULL, 16)
    (slot,) = pc.slots()
    with pc._lock:
        pc._slots = []  # what another owner's eviction does to the map
    assert pc._make_room(slot.nbytes, protect=(slot,), require=slot) is False
    assert pc._make_room(slot.nbytes, protect=(slot,)) is True  # only the residency rule refused
```

`_fail_next_copy` (line 1323) and `KERNEL_PRESSURE_WARN` (import it from `sous.engine.promptcache` at the top if not already) exist. `cast` and `FakeTrimmable` are already used in the file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py`
Expected: the new and rewritten tests FAIL — `test_stats_as_dict_reports_every_counter` and `test_stats_add_sums_retained_and_moved` (no `retained` field); the retain tests find `[A2]` where `[A1, A2]` is expected; `…falls_back_to_moving_it` sees no warning; `…does_not_charge_the_moved_bytes…` sees `[]` where `[B1]` is expected; `…evicted_after_the_take_is_not_copied` fails on the `require` keyword. `test_at_a_budget_of_zero_a_turn_slot_is_always_moved` fails only on the missing `moved` key: its slot arithmetic is today's behaviour. Everything else in the file still passes.

- [ ] **Step 3: Implement**

`PromptCacheStats` (after `pressure_evictions`, before the gauges):

```python
    # Turn-slot takes that copied the slot and left it in the map, and those
    # that removed it and adopted its arrays because the budget could not
    # hold a copy beside the live cache. Both are hits; the split says
    # whether a branch of the conversation could still start warm.
    retained: int = 0
    moved: int = 0
```

`Slot` gains a field (import `weakref` is already there):

```python
    # The turn slot this one was copied from and extended, when that slot was
    # retained: a weak reference, so a predecessor that has left the map is
    # freed rather than pinned by its descendants. Read only by the lineage
    # rule at publish; never compared.
    parent: weakref.ref[Slot] | None = field(default=None, compare=False, repr=False)
```

`Slot`'s docstring (lines 427–430): rewrite the `kind` sentence to: '`kind` is "turn" (published when a turn ends; copied and left in place by the turn that extends it when the budget holds the copy, otherwise removed and its arrays adopted) or "fork" (a copy taken at a shared-prefix boundary; copied on every hit, left in place). `parent` is set only on the copied case.'

`_take` (lines 565–580): delete the two lines `if best.kind == "turn": self._slots.remove(best)` and rewrite the docstring: "The longest slot `owner` holds that `stable_ids` strictly extends. Nothing leaves the map here: a fork slot is copied by the caller, and whether a turn slot is copied or moved is decided in `generate`, where the budget is visible."

`_make_room` (lines 538–563) gains a keyword-only `require: Slot | None = None`. Signature: `def _make_room(self, nbytes: int, protect: Sequence[Slot] = (), *, require: Slot | None = None) -> bool:`. Under the lock, before the feasibility check, add:

```python
            if require is not None and not any(s is require for s in self._slots):
                return False
```

and append to the docstring: "`require`, when given, must still be in the map, else False before any eviction: LRU is daemon-wide and the caller released the lock after choosing it, so another owner's publish may have dropped the slot meanwhile — and a copy of a slot nobody can take again is not worth making room for."

In `generate`, replace lines 845 (the comment `# Whatever is still in the map is not going to serve this turn,`) through 904 (`slot = None`) with:

```python
            # Whatever is still in the map is not going to serve this turn,
            # and this turn's own cache is about to be prefilled to full size
            # beside it. Bring the map under its caps here rather than only at
            # publish, so the two are never outstanding at once: at the
            # default budget of 0 this is what releases the previous turn's
            # cache before a miss rebuilds one. Only the caps, not the
            # pressure reading — headroom is read once the turn's own cache is
            # real and its cost visible, which is at publish. A taken fork
            # slot is still in the map and must survive this pass: dropping
            # the very slot this turn just chose to share would defeat the
            # point of having forked it. A taken turn slot is in the map too,
            # but whether it stays is decided below, and its pass runs after
            # that decision — bytes about to leave the map are never charged
            # against slots that are staying.
            if slot is None or slot.kind == "fork":
                self._evict_caps(protect=(slot,) if slot is not None else ())
        warm: list | None = None
        parent: weakref.ref[Slot] | None = None
        if slot is not None and slot.kind == "fork":
            # (the existing fork-copy block, unchanged)
            ...
        elif slot is not None:
            # A turn slot is copied and left in place when the budget can hold
            # a second slot of its size beside it — the forecast for this
            # turn's own publish, evicting least-recently-used slots now to
            # make that room — so a branch of the conversation (Claude Code's
            # progress-summary call takes the same prefix down a different
            # last turn) does not consume the slot the real conversation
            # extends next. When it cannot — always at a budget of 0, where
            # the feasibility rule refuses a copy that would not fit even on
            # an empty map — or when another owner's eviction has already
            # taken the slot out (LRU is daemon-wide), the slot is moved:
            # removed from the map and its arrays adopted, the pre-retention
            # behaviour byte for byte. A copy failure is an optimization
            # failure and falls back to the move; the turn still runs warm on
            # the original arrays.
            if self._make_room(slot.nbytes, protect=(slot,), require=slot):
                try:
                    warm = hooks.new_cache()
                    fork_copy(slot.cache, warm, hooks.copy_array)
                except Exception as e:
                    warnings.warn(
                        f"sous prompt cache: turn-slot copy failed ({type(e).__name__}); "
                        "extending the slot in place",
                        stacklevel=2,
                    )
                    warm = None
                else:
                    stats.retained += 1
                    parent = weakref.ref(slot)
            retained = warm is not None
            with self._lock:
                if not retained:
                    # By identity: another thread's eviction may already have
                    # taken it out, and a dataclass `remove` would compare
                    # two ~50K-token id lists per candidate.
                    self._slots = [s for s in self._slots if s is not slot]
                self._evict_caps(protect=(slot,) if retained else ())
            if not retained:
                stats.moved += 1
        if slot is not None:
            reuse = len(slot.held)
            stats.took_len = reuse  # the slot this turn took; stays 0 on a miss
            if warm is None:
                warm = slot.cache
            stats.hits += 1
            stats.reused_tokens += reuse
        else:
            reuse = 0
            stats.misses += 1
            # Scanned here, after the clone, so a fork whose copy failed is
            # measured like any other miss instead of keeping an earlier one's.
            stats.miss_lcp = max(
                (common_prefix_length(held, stable_ids) for held in candidates), default=0
            )
            warm = hooks.new_cache()
        # Drop this frame's reference to the slot before anything else is
        # allocated: on a moved slot this is what keeps one conversation from
        # ever holding two copies of itself, and on a retained one the map's
        # reference is the only one that should outlive this point (a weak
        # `parent` link does not pin it). (A fork slot is referenced by the
        # map by design.)
        slot = None
```

Keep the fork block's body exactly as it is today (lines 858–882). Keep the hit branch's existing `if warm is None: warm = slot.cache` too: the moved path leaves `warm` unset on purpose and that fallback assigns it — which is also what lets `ty` narrow `warm` to `list` for the calls below (a version that assigned it inside the `elif` and deleted the fallback failed `ty check` with three `invalid-argument-type` errors).

In the cold-retry branch (line 964, `warm = hooks.new_cache()` before the second `_run`), add `parent = None` — a slot rebuilt cold extended nothing. Change the publish at line 976 to:

```python
        self._publish(
            Slot(warm, list(stable_ids), owner, "turn", slot_bytes(warm), parent=parent), epoch
        )
```

Update the `PrefixCache` class docstring (lines 449–453): replace the whole sentence 'A "turn" slot is taken over and extended in place, so a conversation never holds two copies of itself; a "fork" slot is copied, so the next conversation sharing that prefix finds it too.' with: 'A "turn" slot is copied and left in place when the budget can hold the copy — so a branch of the same conversation (a progress-summary call) still finds it — and moved, its arrays extended in place, when it cannot; a "fork" slot is always copied, so the next conversation sharing that prefix finds it too.'

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_promptcache.py`
Expected: PASS, including the rewritten tests. Then `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3`. `tests/test_engine_forkpoint.py` and the gateway tests use `max_bytes=ROOMY`-like budgets in places; any assertion that counted a moved slot (a `slots` count, a `len(h.caches)`, a `held == [...]` list) is restated for retention, never loosened.

- [ ] **Step 5: Type-check, lint, commit**

```bash
uv run ty check && uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add src/sous/engine/promptcache.py tests/test_promptcache.py
git commit -m "feat(engine): retain a taken turn slot when the budget holds the copy

Claude Code branches a running background subagent's conversation every
30 s with a progress-summary call: same prefix, different last turn. A
turn slot that the extending turn removed left the next real turn only
the header fork, and it re-prefilled ~20K tokens each time. The slot is
now copied — the fork path's own shape — and left in place whenever
_make_room can hold the copy beside the live cache, and moved as before
when it cannot, which is every turn at prompt_cache_gb = 0.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Lineage bounds the accumulation (`promptcache.py`)

**Files:**
- Modify: `src/sous/engine/promptcache.py` (`_publish` at lines 651–670; new `_drop_grandparent` beside it)
- Test: `tests/test_promptcache.py` (after Task 3's tests)

**Interfaces:**
- Consumes: `Slot.parent` from Task 3; `_stats_for`, `_evict_caps`.
- Produces: `PrefixCache._drop_grandparent(slot: Slot) -> None` (lock held by the caller). After this task a linear conversation holds at most two `turn` slots.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_linear_conversation_keeps_only_its_current_and_previous_lengths():
    """Each retained take leaves the predecessor in place; without a bound a
    ten-turn subagent would hold ten copies of itself. Publishing a turn slot
    drops its grandparent, charged to evictions (the budget did not ask for
    it, and neither did pressure)."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    a3 = [*A2, 7]
    pc.generate(A1, A1_FULL, 16)
    pc.generate(A2, A2_FULL, 16)
    assert sorted(s.held for s in pc.slots()) == sorted([A1, A2])
    pc.generate(a3, [*a3, 90, 91], 16)
    assert sorted(s.held for s in pc.slots()) == sorted([A2, a3])
    s = pc.stats()
    assert (s["evictions"], s["pressure_evictions"], s["retained"]) == (1, 0, 2)
    assert s["resident_bytes"] == sum(x.nbytes for x in pc.slots())


def test_a_dropped_grandparent_is_freed_not_pinned_by_lineage():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    a3 = [*A2, 7]
    pc.generate(A1, A1_FULL, 16)
    (first,) = pc.slots()
    gone = weakref.ref(first)
    del first
    pc.generate(A2, A2_FULL, 16)
    pc.generate(a3, [*a3, 90, 91], 16)
    gc.collect()
    assert gone() is None


def test_lineage_never_drops_a_fork():
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    pc.generate(C1, C1_FULL, 16, fork_at=[FORK])
    pc.generate(C1_NEXT, C1_NEXT_FULL, 16, fork_at=[FORK])
    c1_third = [*C1_NEXT, 702]
    pc.generate(c1_third, [*c1_third, 90, 91], 16, fork_at=[FORK])
    assert [s.held for s in pc.slots() if s.kind == "fork"] == [H]
    assert sorted(s.held for s in pc.slots() if s.kind == "turn") == sorted([C1_NEXT, c1_third])


def test_a_branch_keeps_its_own_slot_until_lru_takes_it():
    """Lineage is by reference: the summary call's slot descends from A1's,
    not from the real conversation's next turn, so nothing drops it but LRU."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h, max_bytes=ROOMY)
    summary = [*A1, 500]
    a3 = [*A2, 7]
    pc.generate(A1, A1_FULL, 16)
    pc.generate(summary, [*summary, 90, 91], 16)
    pc.generate(A2, A2_FULL, 16)
    pc.generate(a3, [*a3, 90, 91], 16)  # drops A1's slot, the grandparent
    assert sorted(s.held for s in pc.slots()) == sorted([summary, A2, a3])


def test_a_moved_slot_carries_no_lineage():
    """At a budget of 0 nothing is retained, so there is never a grandparent
    to drop and the evictions counter stays what the cap pass makes it."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    a3 = [*A2, 7]
    pc.generate(A1, A1_FULL, 16)
    pc.generate(A2, A2_FULL, 16)
    pc.generate(a3, [*a3, 90, 91], 16)
    assert pc.stats()["evictions"] == 0
```

Add `import gc` to the test file's imports (`weakref` is already imported at line 10 — a second import fails `ruff check`); `C1`, `C1_NEXT`, `H`, `FORK` are the file's existing fork fixtures (lines ~1215–1227).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_promptcache.py -k "linear_conversation or grandparent or never_drops_a_fork or branch_keeps or no_lineage"`
Expected: four FAIL — the linear test with three slots held (`[A1, A2, a3]`) and `evictions == 0`; the freed test with `gone()` still the slot; `test_lineage_never_drops_a_fork` with three turn slots; `test_a_branch_keeps_its_own_slot_until_lru_takes_it` with four slots (`A1` still resident). `test_a_moved_slot_carries_no_lineage` PASSES already — at a budget of 0 nothing is retained — and pins that this stays so.

- [ ] **Step 3: Implement**

In `_publish`, after `self._slots.append(slot)` and before `self._evict_caps(protect=keep)`, add `self._drop_grandparent(slot)`. Add the method right after `_publish`:

```python
    def _drop_grandparent(self, slot: Slot) -> None:
        """Keep a linear conversation at its current and previous lengths.
        Every retained take leaves the predecessor in place, so without this
        a ten-turn subagent would hold ten copies of itself: when `slot`
        extends a retained turn slot that itself extended one, that
        grandparent goes, charged to `evictions` — neither the budget nor
        pressure asked for it. A branch (a progress-summary call descends
        from the same parent as the real conversation's next turn) is not on
        this chain and stays until LRU takes it. Forks are shared and never
        dropped by lineage. Lock held by the caller."""
        parent = slot.parent() if slot.parent is not None else None
        if parent is None or parent.parent is None:
            return
        grand = parent.parent()
        if grand is None or grand.kind != "turn" or not any(s is grand for s in self._slots):
            return
        self._slots = [s for s in self._slots if s is not grand]
        self._stats_for(grand.owner).evictions += 1
```

Extend `_publish`'s docstring with one sentence: "The lineage drop runs under the same lock, before the caps, so the caps see the map the conversation actually needs."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_promptcache.py` then `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3`
Expected: PASS.

- [ ] **Step 5: Type-check, lint, commit**

```bash
uv run ty check && uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add src/sous/engine/promptcache.py tests/test_promptcache.py
git commit -m "feat(engine): drop a retained conversation's grandparent at publish

Retaining every taken turn slot would let a ten-turn subagent hold ten
copies of itself. A weak parent link on each turn slot names the slot it
was copied from; publishing a slot whose parent has a resident turn
parent drops that grandparent, so a linear conversation holds its
current and previous lengths and a branch holds its own until LRU.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: The turn line tells a retained take from a moved one

**Files:**
- Modify: `src/sous/gateway/turn.py` (`TurnResult` at lines 60–91; `run()` at 216–243), `src/sous/gateway/routes.py` (`_log_turn` at 776–810)
- Test: `tests/test_gateway_turn.py`, `tests/test_gateway_routes.py` (`_ALL_GAUGES` at 1611–1626; after `test_a_hit_retried_cold_prints_took_none` at 1743)

**Interfaces:**
- Consumes: the `moved` counter from Task 3 in the owner-scoped `prompt_cache_stats` dict.
- Produces: `TurnResult.moved: bool = False`; `took=turn-moved@N` on the turn line when the hit moved its slot, `took=turn@N` when it copied it.

- [ ] **Step 1: Write the failing tests**

In `tests/test_gateway_routes.py`, add `"retained": 0, "moved": 0` to `_ALL_GAUGES`, then after `test_a_hit_retried_cold_prints_took_none`:

```python
def test_a_moved_turn_slot_prints_took_turn_moved(tmp_path: Path, capsys):
    """Copied and left in place is `turn@N`; removed and adopted — the only
    path at prompt_cache_gb = 0 — is `turn-moved@N`, so the log says whether
    a branch of this conversation could still start warm."""
    inner = FakeEngine(["a", "b"])
    inner.stats = dict(_ALL_GAUGES)
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        if len(inner.calls) == 2:
            inner.stats = {
                **_ALL_GAUGES,
                "hits": 1,
                "moved": 1,
                "reused_tokens": 40,
                "took_len": 40,
            }
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    app = _app(tmp_path, inner)
    _post(app, _body())
    _post(app, _body())
    f = _fields(_turn_lines(capsys.readouterr().err)[1])
    assert f["cache"] == "hit" and f["took"] == "turn-moved@40"
```

In `tests/test_gateway_turn.py`, next to the existing `TurnResult` field tests (find the one asserting `forked` from a `fork_hits` delta, near the top of the file's stats section), add:

```python
def test_moved_is_a_delta_of_the_moved_counter(tmp_path: Path):
    inner = FakeEngine(["one", "two"])
    inner.stats = {"hits": 0, "moved": 3}
    runner, _ = _runner(tmp_path, inner)
    first = runner.run([{"role": "user", "content": "hi"}], [], 8, RecordingSink())
    assert first.moved is False  # 3 before and 3 after: nothing moved on this turn
    inner.stats = {"hits": 1, "moved": 4}

    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        inner.stats = {"hits": 2, "moved": 5}
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    second = runner.run([{"role": "user", "content": "hi again"}], [], 8, RecordingSink())
    assert second.moved is True and second.cache_hit is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_routes.py tests/test_gateway_turn.py -k "took_turn_moved or moved_is_a_delta"` (one `-k`: pytest keeps only the last one given)
Expected: both FAIL — `TurnResult` has no `moved`; the line prints `took=turn@40`.

- [ ] **Step 3: Implement**

`TurnResult` gains, after `forked`:

```python
    # The hit moved its turn slot (removed and adopted, because the budget
    # could not hold a copy beside the live cache) rather than copying it.
    moved: bool = False
```

In `run()`'s `TurnResult(...)` construction add `moved=delta_of("moved") > 0,` after `forked=...`. In `_log_turn` replace the `took` expression with (the label hoisted: inline, the f-string is 107 columns and `ruff format` cannot split it):

```python
        kind = "fork" if cache == "fork" else "turn-moved" if result.moved else "turn"
        # A hit retried cold took no slot in the end: its took_len is 0.
        took = "none" if cache == "miss" or not result.took_len else f"{kind}@{result.took_len}"
```

and delete the old `# A hit retried cold took no slot in the end` comment line above the old expression (it moves into the block above).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `set -o pipefail; uv run pytest tests/test_gateway_turn.py tests/test_gateway_routes.py 2>&1 | tail -3`
Expected: PASS.

- [ ] **Step 5: Type-check, lint, commit**

```bash
uv run ty check && uv run ruff format . && uv run ruff check . && uv run ruff format --check .
git add src/sous/gateway/turn.py src/sous/gateway/routes.py tests/test_gateway_turn.py tests/test_gateway_routes.py
git commit -m "feat(gateway): say on the turn line whether a hit moved or copied its slot

took=turn@N is a copied, retained slot; took=turn-moved@N is one the
budget could not hold a copy of — the reading that says whether the
next branch of this conversation can start warm.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md` (the "gives up" list at lines 226–262; the `prompt_cache_gb` paragraph at 498–520), `CLAUDE.md` (line 61)

**Interfaces:** none. Every sentence below is checked against the code as shipped in Tasks 1–5; an inaccuracy is the defect.

- [ ] **Step 1: README — "What a locally served turn gives up"**

In the bullet "**One turn at a time; keyed prompt-cache slots.**" (line 249), replace "so a subagent's consecutive turns reuse their own slot, two subagents interleaving reuse theirs," with:

> so a subagent's consecutive turns reuse their own slot — copied and left in place when the budget can hold the copy, so a conversation that Claude Code branches (its progress-summary calls for a background agent take the same prefix down a different last turn) still finds it, and moved into the extending turn when it cannot, which is every turn at `prompt_cache_gb = 0`; a linear conversation holds at most its current and previous lengths — two subagents interleaving reuse theirs,

Add a new bullet after "**Usage is split the way Anthropic's is.**":

> - **Mid-conversation system messages become `<system-reminder>` blocks.** Claude Code delivers attachments that arrive after the first turn (agent listings, MCP instructions, deferred tools) as a `role: "system"` message after the preceding user message. The chat template takes one system turn, at index 0, so the gateway renders such a message the way Claude Code's own fallback for models without the feature does — a `<system-reminder>` block in the user turn before it (after that turn's tool results). The model sees the same text in the same place a frontier model without the feature would; nothing is lost, and the conversation stays a strict extension of the previous turn's, which is what keeps it warm.

- [ ] **Step 2: README — `prompt_cache_gb`**

In the paragraph at lines 498–520, after "room for those forks and several conversations;" add "each live conversation under the auto budget also keeps its previous length resident (the slot a branch of it starts from), so count ~4 GiB per conversation at 63K tokens beyond the forks;". Then restate the 48 GB sentence ("a 48 GB machine should set it to `0` (forks off, one slot), or to at least one fork copy plus one conversation slot — about 8 GiB with the default model at ~57K tokens.") as: "a 48 GB machine should set it to `0` (forks off, one slot), or to at least one fork copy plus one conversation slot — about 8 GiB with the default model at ~57K tokens, which keeps a conversation warm but moves its slot each turn, so a branch of it (a progress-summary call) starts from the fork; retaining the previous length as well needs one fork copy plus two conversation slots, about 12 GiB." Leave the `0` sentence as is — at `0` nothing is retained.

- [ ] **Step 2b: README — the turn line's field descriptions and `server_status`**

In the field-by-field paragraph after the log example (lines ~300–312): where `took` is described ("`took` names the slot and its length — `none` on a miss, …") add, before the `none` clause, "`turn@N` for this conversation's own slot copied and left in place, `turn-moved@N` for one the budget could not hold a copy of (removed and extended in place — every hit at `prompt_cache_gb = 0`), `fork@N` for a shared boundary,". Where `evicted` is described ("`evicted` counts every drop, budget or pressure") make it "`evicted` counts every drop — budget, pressure, or a retained predecessor two turns back that lineage retires". In the `server_status` sentence at lines 525–526 ("slots, resident bytes, hits, fork hits, evictions and the subset the pressure valve took") add "retained and moved turn-slot takes" after "fork hits".

- [ ] **Step 3: CLAUDE.md gotcha**

Replace the sentence "A `turn` slot is *moved* into the turn that extends it." (line 61) with:

> A `turn` slot is *copied* and left in place by the turn that extends it when `_make_room` can hold the copy (Claude Code branches a background subagent's conversation every 30 s with a progress-summary call, and a consumed slot left the real conversation only the header fork), and *moved* — removed, its arrays adopted — when it cannot, always at `prompt_cache_gb = 0`; `Slot.parent` lineage drops the grandparent at publish. Inline `role:"system"` messages after the first user message render as `<system-reminder>` blocks in the preceding user turn (`gateway/convert._place_inline_system`), never hoisted into the system block: the hoist moved the header boundary and killed every prefix behind it on the turn an attachment arrived.

- [ ] **Step 4: Verify the wording against the code and commit**

Read `convert._place_inline_system`, `PrefixCache.generate`'s retain branch, `_make_room`'s feasibility rule and `_drop_grandparent` once more and check every claim above: the wrapper text, "after that turn's tool results", "always at `prompt_cache_gb = 0`", "current and previous lengths", the 8 GiB / 12 GiB arithmetic (a retained take needs `2 × slot.nbytes ≤ max_bytes` beside any fork it must not evict), and the three `took` labels.

```bash
git add README.md CLAUDE.md
git commit -m "docs: describe in-place system messages and retained turn slots

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Model test — the incident's shape on a real template, token for token

**Files:**
- Test: `tests/test_gateway_model.py` (append)

**Interfaces:**
- Consumes: the file's `_post`, `_events`, `TINY`, and the mounted-gateway construction of its existing test.
- Produces: a `model`-marked test, parameterized over the tiny model (`LMEngine`) and the 27B (`VLMEngine`), that the orchestrator runs on the M5 Pro. Not run by the implementer.

- [ ] **Step 1: Write the test**

```python
def _seeded(n: int) -> None:
    import mlx.core as mx

    mx.random.seed(n)


def _usage(r: httpx.Response) -> dict:
    return r.json()["usage"]


MODELS = [
    pytest.param(TINY, "lm", id="tiny"),
    # The hybrid model the copy-then-extend shape exists for: its recurrent
    # layers cannot rewind, so a retained slot is a real second cache.
    pytest.param("mlx-community/Qwen3.8-27B-4bit", "vlm", id="27b"),
]


def _engine(model_id: str, backend: str):
    # An explicit budget: the factory bypasses [model].prompt_cache_gb, and a
    # budget of 0 would move T1's slot instead of retaining it.
    if backend == "vlm":
        from sous.engine.vlm import VLMEngine

        return VLMEngine(model_id, prompt_cache=True, cache_budget=8 << 30)
    from sous.engine.lm import LMEngine

    return LMEngine(model_id, prompt_cache=True, cache_budget=8 << 30)


@pytest.mark.parametrize(("model_id", "backend"), MODELS)
def test_an_attachment_keeps_the_conversation_warm_and_bit_exact(
    tmp_path: Path, model_id: str, backend: str
):
    """T1 a brief; T3 the same conversation plus a tool exchange and a
    role:system attachment after the tool result. T3 must be served from
    T1's slot (cache_read_input_tokens covers T1's whole render) and produce
    exactly what a cold run of the same prompt produces with the same seed —
    the render is the same text in the same place, only warm. Then a branch
    of T1 (a summary-shaped last turn) is served from the retained slot and
    is bit-exact against its own cold run too."""
    cfg = SousConfig(
        data_dir=tmp_path / "data", config_path=tmp_path / "config.toml", gateway_enabled=True
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: _engine(model_id, backend))
    mcp = MCPServer("test")
    gateway = mount_gateway(mcp, engines, cfg)
    app = mcp.streamable_http_app()
    read_tool = {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
    }
    system = [{"type": "text", "text": "You are a terse assistant. " * 40}]
    t1 = [{"role": "user", "content": "Read the file notes.txt and tell me its first word."}]
    t3 = [
        *t1,
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": "notes.txt"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "banana bread recipe"},
                {"type": "text", "text": "<total_tokens>900 tokens left</total_tokens>\n"},
            ],
        },
        {"role": "system", "content": "Available agent types for the Agent tool:\n- Explore: read-only"},
    ]
    branch = [*t1, {"role": "assistant", "content": "Reading."}, {"role": "user", "content": "Describe your most recent action in 3-5 words."}]

    def body(messages: list[dict]) -> dict:
        return {"model": "sous-local", "max_tokens": 24, "tools": [read_tool], "system": system, "messages": messages}

    try:
        _seeded(1)
        first = _post(app, body(t1))
        assert first.status_code == 200
        _seeded(2)
        warm = _post(app, body(t3))
        assert warm.status_code == 200
        assert _usage(warm)["cache_read_input_tokens"] > 0
        _seeded(3)
        branch_warm = _post(app, body(branch))
        assert _usage(branch_warm)["cache_read_input_tokens"] == _usage(warm)["cache_read_input_tokens"]

        engines.get().reset_prompt_cache()
        _seeded(2)
        cold = _post(app, body(t3))
        assert _usage(cold)["cache_read_input_tokens"] == 0
        assert cold.json()["content"] == warm.json()["content"]
        engines.get().reset_prompt_cache()
        _seeded(3)
        branch_cold = _post(app, body(branch))
        assert branch_cold.json()["content"] == branch_warm.json()["content"]
    finally:
        gateway.close()
        assert gateway._runner._session is None
    engines.get().unload()
```

The teardown mirrors the file's existing test (close the gateway inside `finally`, unload the engine after). If an equality assertion fails while the `cache_read_input_tokens` assertions pass, do not loosen it: report both texts to the orchestrator — the warm and cold prefills take different chunk shapes through the model, and whether that is allowed to change a sampled token is a decision, not a flake.

- [ ] **Step 2: Verify it is collected and skipped without the marker, then commit**

Run: `uv run pytest tests/test_gateway_model.py --collect-only -q | tail -4` and `uv run pytest -m "not model" tests/test_gateway_model.py` (deselected).
Expected: three tests collected (the existing one plus `[tiny]` and `[27b]`); nothing runs without `-m model`. `uv run ty check` and ruff clean.

```bash
git add tests/test_gateway_model.py
git commit -m "test(gateway): replay the attachment shape on a real template, bit-exact

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 3 (orchestrator, not the implementer): run both parameters on the M5 Pro**

`ssh m5pro 'cd ~/code/personal/sous-remote && git fetch -q && git checkout -q feat/gateway-continuity-a && uv sync -q && uv run pytest -m model tests/test_gateway_model.py -k attachment 2>&1 | tail -6'`. The `[27b]` case loads a second copy of the 27B (~15 GiB) beside whatever the daemon holds; run it once the daemon's idle unload has dropped its model (`sous status`; `[model].idle_unload_minutes`, default 30) or accept the pressure that a resident daemon model adds. Record both results in the ledger and on the PR. A failure on an equality assertion is a finding for the branch, not a flake: both engines sample through mlx's global RNG, which `_seeded` fixes; report both texts to Kyle rather than loosening it (see the note under Step 1).

---

## Orchestrator steps after Task 7

1. **Whole-branch review** (fresh reviewer, the full diff against `main`): the plan-language sweep, the docs' every claim against the code, and specifically whether any path still removes a turn slot in `_take`, whether `parent` can pin arrays, and whether a request body is mutated.
2. **PR** `feat/gateway-continuity-a` → `main`, title `feat(gateway,engine): render system messages in place; retain turn slots`, body from the commits, footer `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and nothing beneath it.
3. **Real-daemon gate (needs Kyle's go-ahead — installs branch code over his daemon):** on the M5 Pro, `uv tool install --force --reinstall` from `sous-remote` at the branch, `sous stop`, restart detached, then one `sous claude -p` session inside the GUI `screen` session with one *background* general-purpose subagent of ≥3 tool-using turns. Pass: every real subagent turn after the first logs `cache=hit`, `took=turn@N`, and `lcp` never appears above the header on any line. Record the turn lines on the PR. (The `holders=` half of the spec's gate belongs to plan B.)
