# Gateway Phase 1 (Anthropic Endpoint, Serialized) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve Claude Code from the local model over an Anthropic-compatible `/v1/messages` (streaming and non-streaming) plus `count_tokens` and `HEAD /api/hello`, mounted on the existing daemon and off by default — so one full-local Claude Code session can complete a real bounded task against sous (the spec's Phase 1 exit criterion).

**Architecture:** A new `sous.gateway` package with four focused modules — `convert.py` (Anthropic request → chat messages/tools, all Claude Code accommodations), `response.py` (engine deltas → Anthropic content blocks and stream events, one code path for both response shapes), `turn.py` (one serialized turn on the shared engine via a long-lived `GenerationSession`, bridged from the event loop to threads), `routes.py` (Starlette handlers registered through `MCPServer.custom_route`, SSE via sse-starlette, body bound, Anthropic-shaped errors, metadata-only logging). Underneath, the engine stack gains an optional `on_delta` streaming callback threaded through `Engine`/`ManagedEngine`/`GenerationSession`/`PrefixCache`/`LMEngine`/`VLMEngine`, and `protocol.parse_tool_calls` gains a request-scoped `ToolSet`. The worker path is untouched in behavior.

**Tech Stack:** Python 3.14, uv, mcp 2.1.1 (`MCPServer.custom_route`, `streamable_http_app`), starlette 1.6 + sse-starlette 3.4 (promoted from transitive to explicit deps), mlx-lm 0.31.3 `stream_generate`, mlx-vlm 0.6.16 `stream_generate`, pytest (+ httpx `ASGITransport` in-process, in-process uvicorn for `slow` tests, `model`-marked real-model tests).

**Spec:** `docs/superpowers/specs/2026-08-26-hybrid-gateway-design.md` — Phase 1 section, the core-decisions table, the 9-item accommodation checklist, and "Security posture". Phase 0 results (both gates PASSED, wire facts) are on issue #41, comments 5440572099 and 5440844229.

## Global Constraints

- Python >= 3.14; everything runs via `uv run`. Never pip. `except A, B:` without parentheses is valid 3.14 syntax (PEP 758) — don't "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`. `uv run ty check` covers tests too.
- mlx / mlx_lm / mlx_vlm imports stay function-local (absent on non-macOS; lint CI runs on ubuntu). Any thread that touches mlx calls `sous.engine.base.release_mlx_thread_state()` before it exits (ml-explore/mlx#4327).
- **The gateway never executes a tool** (spec core decision "Tool execution: Never"). It returns `tool_use` blocks; Claude Code runs them under its own permission system. `toolexec.py` is not imported by anything under `src/sous/gateway/`.
- **Request bodies and header values are never logged, at any level.** Log lines carry only method, path, model id, stream flag, status, token counts, stop reason, cache hit/miss, duration, and dropped server-tool names.
- **Honest model id**: the served id is whatever `[gateway].local_models` lists (default `sous-local`), never `claude-*` — enforced in `_gateway_values` (Task 1): an entry containing `claude` is rejected with a warning and the list falls back to the default. Requests for any other model id get an Anthropic-shaped 404 `not_found_error` in this phase (Phase 2 forwards them upstream).
- **48K floor**: `[gateway].max_context_tokens` is clamped up to `GATEWAY_MIN_CONTEXT_TOKENS = 48 * 1024 = 49152` with a warning (Claude Code refuses smaller windows; oMLX gates at the same constant).
- **Client disconnect never wedges the session**: a turn always drains to completion on its generation thread; the gateway holds no lock on the event loop side.
- **Real token counts, always** (spec "Usage reporting"): `usage.input_tokens` = the full rendered prompt's token count from the engine's own tokenization; `usage.output_tokens` = tokens generated. Never inflated, never estimated from library `prompt_tokens` (which counts only the fed suffix on a warm cache).
- Prompt stabilization from day one: `x-anthropic-billing-header:` system blocks dropped; `<total_tokens>N tokens left</total_tokens>` markers stripped (spec checklist items 3–4); `enable_thinking=False` on every render (the template otherwise injects a reasoning-effort line into the system turn and changes the whole prefix).
- Tests never touch the real `~/.sous`: every `SousConfig` in tests is built with tmp_path-based `data_dir`/`config_path`. `model`-marked tests are local/manual; `slow`-marked tests run for real, never mocked.
- `docs/superpowers/**` are point-in-time records: never edit existing spec/plan files (this plan is a new file).
- Conventional Commits, imperative lowercase subject, *why* in the body. Commit trailer: `Co-Authored-By: Claude <noreply@anthropic.com>` (model-less, per the user's global CLAUDE.md). Work on a branch; `main` is protected. CI is exactly: `uv run pytest -m "not model"`, `uv run ty check`, `uv run ruff check . && uv run ruff format --check .`, `uv lock --check`. The code blocks in this plan are written for readability, not pre-wrapped: every commit step runs `uv run ruff format .` *before* `uv run ruff check .` — the formatter wraps the over-long lines, and CI's `ruff format --check` then passes.

---

## Decisions this plan locks (beyond the spec)

These were settled by the Phase 0 wire data and by reading the installed libraries; implementers should not reopen them.

1. **The default model runs on `VLMEngine`, not `LMEngine`.** `mlx-community/Qwen3.8-27B-4bit` has `vision_config`, so `select_backend` picks mlx-vlm. Streaming therefore goes through `mlx_vlm.stream_generate` (same kwargs `generate()` forwards today — `input_ids`, `prompt_cache`, `sampler`, `draft_model`/`draft_kind`/`draft_block_size`; skip `is_draft` rows; replicate `generate()`'s `stopping_criteria.reset`) **and** `mlx_lm.stream_generate` for the LM backend. Both yield one result per token with `finish_reason=None`, then a final result (possibly empty text) with `finish_reason` `"stop"` or `"length"`; EOS text is never emitted.
2. **Streaming is an `on_delta` callback**, not a rewrite of `GenerationSession`'s reply queue. The one-slot request/reply protocol and its stall semantics stay exactly as they are; the callback rides inside the request tuple and fires on the session thread mid-decode. A `Delta(text, output_tokens, finish_reason)` carries what the response needs.
3. **A warm-cache retry is refused after any delta was streamed.** `PrefixCache`'s cold retry would replay the turn to a client that already received part of it; the failure is raised instead. Nothing was streamed → the existing retry stands. The runner always installs `on_delta` (it needs the final delta for counts), so this also holds for non-streaming requests: a warm-cache failure after the first token is a 500 there too, and Claude Code's own retry covers it. The worker path keeps its cold retry.
4. **Text streams, tool calls buffer.** Prose before `<tool_call>` is sent as `text_delta`s (with a hold-back for a split tag and for trailing whitespace); from the first `<tool_call>` on, text is buffered, parsed once at the end with `protocol.parse_tool_calls`, and emitted as `content_block_start(tool_use, input: {})` → one `input_json_delta` carrying the whole JSON → `content_block_stop` (oMLX's shape; Claude Code has years of mileage on it). Trailing prose after a tool call is dropped (the template forbids it). A tool call the parser cannot read comes back as plain text with `end_turn` — the model's malformed turn is shown, not hidden.
5. **`ping` frames are safe anywhere**, including before `message_start`: both official SDKs `continue` on `event: ping` in the SSE iterator, before their accumulator sees anything (verified in the TS and Python SDK sources; oMLX ships ping-first). One ping is sent as the very first frame; sse-starlette's ping task repeats it every `PING_INTERVAL_SECONDS = 10` for the life of the stream.
6. **`EventSourceResponse` (sse-starlette), not a bare `StreamingResponse` — and the daemon drives uvicorn itself with a bounded graceful shutdown.** sse-starlette gives the built-in ping task (with a custom `event: ping` frame), canonical framing, and self-cancellation on uvicorn's exit signal (its import-time hook is already active in the daemon because mcp's transport imports it). But uvicorn owns SIGTERM while it serves — it swaps sous's handler out and re-raises only after its own shutdown, which by default waits for every open connection — so a non-streaming turn (Claude Code's retry shape) could still defer `sous stop`/launchd restarts by a whole generation. `mcp.run` exposes no shutdown timeout, so `server.serve()` builds the same app and uvicorn config with `timeout_graceful_shutdown = 5`; the cancelled handler leaves its turn draining on the executor thread. `starlette`, `sse-starlette` and `uvicorn` become explicit dependencies.
7. **Errors after the stream has started are in-band `event: error`** (Anthropic-shaped). Cheap validation (JSON shape, model routing, body size) returns real HTTP statuses before any bytes are sent; everything that needs the engine (lock wait, model load, tokenization, "prompt is too long", generation failure) happens behind the first ping, because header latency is otherwise unbounded by model load and queueing. Non-streaming requests get real statuses for everything (400/404/413/500/529).
8. **Content blocks**: text block first (only if non-blank), then `tool_use` blocks, indices contiguous from 0; an empty reply yields one empty text block. `tool_use.id` is `toolu_` + 24 hex, `message.id` is `msg_` + 24 hex. `stop_reason` is `tool_use` when calls parsed, else `max_tokens` for `finish_reason == "length"`, else `end_turn`; `stop_sequence` is always `null`.
9. **Ignored request fields** (tolerated, never rejected): `thinking`, `output_config`, `context_management`, `metadata`, `temperature`, `top_p`, `top_k`, `stop_sequences`, `tool_choice`, `service_tier`, `cache_control` anywhere. Sampling is the daemon's `[model]` sampler; thinking stays off (checklist item 5 is recorded as N/A while it is).
10. **Content conversion** (Qwen3.8 template facts): the system message must be `messages[0]`, so the canonical `system` field and every inline `role: "system"` message are joined into one system message (canonical first, `"\n\n"` separators); `tool_result` → `role: "tool"` messages (the template groups consecutive ones into one `<tool_response>` user turn and matches by order; text preceding a result is flushed as its own user turn to keep the order); assistant `tool_use` → `tool_calls[].function.arguments` as a **dict** (a JSON string raises inside the template); `thinking`/`redacted_thinking` dropped; `image`/`document` → a one-line `[image omitted: sous serves text only]` placeholder; unknown block types skipped; `is_error` ignored (the content already says so).
11. **Server-side tools**: any `tools[]` entry whose `type` is present and not `"custom"` is dropped with a log line (`web_search_*`, `web_fetch_*`, `code_execution_*`, `bash_*`, `text_editor_*`, `computer_*`, `memory_*`, `browser_toolset_*`, `tool_search_tool_*` — the spec's prefix list is a subset). Entries may lack `name` (toolsets), so read it with `.get`.
12. **One gateway session, shared single-slot cache.** All gateway turns run on one long-lived `GenerationSession` (recreated after a stall — which also resets the prompt cache, since a cache is usable only from the thread that built it — or after an idle-unload/reload) so the strict-prefix cache survives between a subagent's turns. The worker still resets the cache at task start/end and the cache has one slot, so while a delegated task and a gateway conversation overlap, or two conversations interleave, every turn on both sides is a cold prefill (each evicts the other's slot) — accepted for Phase 1 and stated in the README; keyed slots are Phase 3a.
13. **Gateway window is its own knob**, `[gateway].max_context_tokens` (default 65536), independent of the worker's `[model].max_context_tokens` and of `[context] mode = "auto"`. `max_tokens` from the request is clamped to `window - input_tokens`; a prompt that fills the window is `invalid_request_error: prompt is too long: N tokens > M maximum`.
14. **Request-size bound** is the gateway's own: `MAX_REQUEST_BYTES = 32 MiB` (Anthropic's cap), checked on `Content-Length` and again while reading — the MCP transport's 4 MiB limit covers only `/mcp`.
15. **The gateway's `ToolSet` is non-strict** (spec: "validates against a request-scoped tool set"): a tool name outside the request's tools still becomes a `tool_use` block, typed as strings, because Claude Code answers it with its own tool-not-found result and the model can recover; ending the turn here could not be undone. The worker's set stays strict. Pinned by `test_non_strict_toolset_passes_unknown_names_through_untyped`.
16. **`/api/hello` (GET and HEAD → 200) amends the spec's surface list** per gate-1 O3: Claude Code probes it at startup. Phase 2 forwards it upstream like any non-local request.
17. **Host header check**: the gateway rejects non-loopback `Host` values with a 403 `permission_error`, mirroring the DNS-rebinding protection the SDK applies to `/mcp` but not to custom routes.
18. **Usage carries `input_tokens` and `output_tokens` only**, in both response shapes; the cache accounting fields are omitted (SDKs read them as null) until keyed slots give a truthful per-turn split.
19. **Abandoned-while-queued turns never start.** Drain-to-completion covers a generation in progress; a request whose client left while it was still waiting for the gateway lock is skipped (`TurnAbandoned`), so it cannot delay live requests by a whole turn.

---

## File Structure

New package `src/sous/gateway/`:

| File | Responsibility |
|---|---|
| `__init__.py` | empty |
| `convert.py` | Anthropic `/v1/messages` body → `ChatRequest` (chat messages + OpenAI-style tool schemas): validation, `RequestError`, system folding, volatile-marker stripping, block conversion, server-tool dropping. Pure. |
| `response.py` | `TextSplitter` (safe-to-stream text), `TurnAssembler` (deltas + final text → Anthropic events and the non-streaming message), id minting, `stop_reason`. Pure. |
| `turn.py` | `TurnRunner`: gateway lock, long-lived `GenerationSession`, count/clamp/generate, `Sink` relay, `TurnResult`; `PromptTooLong`, `GatewayBusy`. Synchronous; runs on pool threads. |
| `routes.py` | `Gateway` handlers (`/v1/messages`, `/v1/messages/count_tokens`, `/api/hello`), body bound, SSE framing, error bodies, logging, `mount_gateway()`. |

Modified:

| File | Change |
|---|---|
| `src/sous/config.py` | `[gateway]` section: `gateway_enabled`, `gateway_local_models`, `gateway_max_context_tokens`, `gateway_generation_timeout_minutes`; `GATEWAY_MIN_CONTEXT_TOKENS`. |
| `src/sous/protocol.py` | `ToolSet` (names + parameter types + strictness); `parse_tool_calls(text, toolset=WORKER_TOOLSET)`; array/object coercion. `TOOL_NAMES`/`_PARAM_TYPES` removed (nothing outside protocol.py imports them). |
| `src/sous/engine/base.py` | `Delta`, `OnDelta`; `on_delta` on `Engine.generate`, `ManagedEngine.generate`, `GenerationSession.generate`. |
| `src/sous/engine/promptcache.py` | `on_delta` through `CacheHooks.decode`, `PrefixCache.generate`/`_run`; no cold retry after streamed deltas. |
| `src/sous/engine/lm.py`, `src/sous/engine/vlm.py` | `decode(..., on_delta)` emits `Delta`s; VLM decode moves from `generate()` to `stream_generate()`. |
| `src/sous/server.py` | `mount_gateway` when enabled; `server_status` reports gateway config; startup line. |
| `pyproject.toml`, `uv.lock` | `starlette>=1.6`, `sse-starlette>=3.4` runtime deps; `httpx>=0.28` dev dep. |
| `tests/fake_engine.py` | `FakeEngine.generate(..., on_delta=None)`; new `ChunkedFakeEngine`. |
| `tests/test_engine_base.py`, `tests/test_worker.py`, `tests/test_promptcache.py` | fakes accept the new optional parameter. |
| `README.md`, `CLAUDE.md` | gateway docs, config keys, security/limitations, gotchas. |

New tests: `tests/test_gateway_convert.py`, `tests/test_gateway_response.py`, `tests/test_gateway_turn.py`, `tests/test_gateway_routes.py`, `tests/test_gateway_http.py` (slow), `tests/test_gateway_model.py` (model).

---

### Task 1: `[gateway]` config section and status reporting

**Files:**
- Modify: `src/sous/config.py`
- Modify: `src/sous/server.py` (`SousService.server_status`)
- Test: `tests/test_config.py`, `tests/test_server.py`

**Interfaces:**
- Produces: `SousConfig.gateway_enabled: bool` (default `False`), `SousConfig.gateway_local_models: tuple[str, ...]` (default `("sous-local",)`), `SousConfig.gateway_max_context_tokens: int` (default `65536`, floor `GATEWAY_MIN_CONTEXT_TOKENS`), `SousConfig.gateway_generation_timeout_minutes: int` (default `30`); module constant `sous.config.GATEWAY_MIN_CONTEXT_TOKENS = 49152`. `server_status()["config"]["gateway"]` = `{"enabled", "local_models", "max_context_tokens"}`.

- [ ] **Step 1: Write the failing config tests**

Append to `tests/test_config.py`:

```python
# --- [gateway] -----------------------------------------------------------------


def test_gateway_defaults_are_off_and_above_the_claude_code_floor(tmp_path: Path):
    from sous.config import GATEWAY_MIN_CONTEXT_TOKENS

    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.gateway_enabled is False
    assert cfg.gateway_local_models == ("sous-local",)
    assert cfg.gateway_max_context_tokens == 65536
    assert cfg.gateway_max_context_tokens >= GATEWAY_MIN_CONTEXT_TOKENS == 49152
    assert cfg.gateway_generation_timeout_minutes == 30


def test_gateway_section_overrides_every_key_without_warnings(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[gateway]\n"
        "enabled = true\n"
        'local_models = ["sous-local", "sous-fast"]\n'
        "max_context_tokens = 131072\n"
        "generation_timeout_minutes = 5\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_config(p)
    assert cfg.gateway_enabled is True
    assert cfg.gateway_local_models == ("sous-local", "sous-fast")
    assert cfg.gateway_max_context_tokens == 131072
    assert cfg.gateway_generation_timeout_minutes == 5


def test_gateway_window_below_the_floor_clamps_up_with_a_warning(tmp_path: Path):
    """Claude Code refuses models under 48K of context, so a smaller window can
    only be a misjudged floor: serve the floor, not the default, and say so."""
    from sous.config import GATEWAY_MIN_CONTEXT_TOKENS

    p = tmp_path / "config.toml"
    p.write_text("[gateway]\nmax_context_tokens = 32768\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.gateway_max_context_tokens == GATEWAY_MIN_CONTEXT_TOKENS
    assert any("floor" in str(w.message) for w in caught)


def test_gateway_bad_values_degrade_to_defaults_with_warnings(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[gateway]\n"
        'enabled = "yes"\n'
        "local_models = []\n"
        "max_context_tokens = -1\n"
        "generation_timeout_minutes = 0\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.gateway_enabled is False
    assert cfg.gateway_local_models == ("sous-local",)
    assert cfg.gateway_max_context_tokens == 65536
    assert cfg.gateway_generation_timeout_minutes == 30
    messages = " ".join(str(w.message) for w in caught)
    for key in ("enabled", "local_models", "max_context_tokens", "generation_timeout_minutes"):
        assert key in messages, key


def test_gateway_local_models_rejects_non_string_entries(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[gateway]\nlocal_models = ["ok", 3]\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.gateway_local_models == ("sous-local",)
    assert any("local_models" in str(w.message) for w in caught)


def test_gateway_local_models_rejects_claude_ids(tmp_path: Path):
    """Claude Code ignores CLAUDE_CODE_MAX_CONTEXT_TOKENS for ids that
    canonicalize to claude-*, so an impersonating id silently forfeits the
    window control the gateway depends on — the spec makes honest ids
    mandatory, not preferable."""
    p = tmp_path / "config.toml"
    p.write_text('[gateway]\nlocal_models = ["sous-local", "Claude-haiku-4-5"]\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(p)
    assert cfg.gateway_local_models == ("sous-local",)
    assert any(
        "local_models" in str(w.message) and "claude" in str(w.message).lower() for w in caught
    )
```

Append to `tests/test_server.py`:

```python
def test_server_status_reports_gateway_config(svc):
    """The gateway is off by default and experimental; the MCP-visible status
    is how a user confirms which model ids the daemon would serve locally."""
    service, _, _ = svc
    gw = service.server_status()["config"]["gateway"]
    assert gw == {
        "enabled": False,
        "local_models": ["sous-local"],
        "max_context_tokens": 65536,
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k gateway tests/test_server.py::test_server_status_reports_gateway_config -q`
Expected: FAIL — `ImportError: cannot import name 'GATEWAY_MIN_CONTEXT_TOKENS'` / `AttributeError: 'SousConfig' object has no attribute 'gateway_enabled'` / `KeyError: 'gateway'`.

- [ ] **Step 3: Implement the config section**

In `src/sous/config.py`, add `"gateway"` to `_KNOWN`:

```python
    "tasks": {"retention"},
    "gateway": {"enabled", "local_models", "max_context_tokens", "generation_timeout_minutes"},
}

# Claude Code refuses to run against a model advertising less than 48K of
# context (oMLX gates on the same 48 * 1024). A smaller gateway window would
# never be used, so the config clamps up to this instead of serving it.
GATEWAY_MIN_CONTEXT_TOKENS = 48 * 1024
```

Add the fields to `SousConfig`, after `context_min_tokens`:

```python
    # Anthropic-compatible endpoint on the daemon (issue #41), off by default
    # and experimental. It serves Claude Code with the local model and never
    # touches the toolexec sandbox: Claude Code executes the tools under its
    # own permission system. The window is the gateway's own — Claude Code's
    # prompts are far larger than the worker's, and the worker's cap stays put.
    gateway_enabled: bool = False
    gateway_local_models: tuple[str, ...] = ("sous-local",)
    gateway_max_context_tokens: int = 65536
    gateway_generation_timeout_minutes: int = 30
```

Add the validator after `_speculative_block_size`:

```python
def _gateway_values(gateway: dict) -> tuple[bool, tuple[str, ...], int, int]:
    """Validated [gateway] values, each degrading to its default with a
    warning — except the window, which is clamped UP to the Claude Code floor:
    a smaller value can only be a misjudged floor, and the floor is the
    closest thing to what the user asked for that would actually work."""
    enabled = gateway.get("enabled", False)
    if not isinstance(enabled, bool):
        warnings.warn(
            f"sous config: [gateway].enabled {enabled!r} must be true or false; using false",
            stacklevel=3,
        )
        enabled = False
    models = gateway.get("local_models", ["sous-local"])
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(m, str) and m for m in models)
    ):
        warnings.warn(
            f"sous config: [gateway].local_models {models!r} must be a non-empty list of "
            "model ids; using ['sous-local']",
            stacklevel=3,
        )
        models = ["sous-local"]
    # Spec: honest ids are mandatory. Claude Code ignores
    # CLAUDE_CODE_MAX_CONTEXT_TOKENS for any id that canonicalizes to claude-*
    # and trusts its built-in window instead, so an impersonating id silently
    # forfeits the window control the gateway relies on (and, once routing
    # lands, would pull real Claude traffic onto the local model). Substring,
    # not prefix: canonicalization strips provider prefixes, and no honest
    # local id has any reason to contain the word at all.
    if any("claude" in m.lower() for m in models):
        warnings.warn(
            f"sous config: [gateway].local_models {models!r} impersonates a Claude model; "
            "Claude Code ignores its context-window env vars for claude-* ids, so use an "
            "honest id like 'sous-local'; using ['sous-local']",
            stacklevel=3,
        )
        models = ["sous-local"]
    window = gateway.get("max_context_tokens", 65536)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        warnings.warn(
            f"sous config: [gateway].max_context_tokens {window!r} must be a positive "
            "integer; using 65536",
            stacklevel=3,
        )
        window = 65536
    elif window < GATEWAY_MIN_CONTEXT_TOKENS:
        warnings.warn(
            f"sous config: [gateway].max_context_tokens {window} is below Claude Code's "
            f"{GATEWAY_MIN_CONTEXT_TOKENS}-token floor; using {GATEWAY_MIN_CONTEXT_TOKENS}",
            stacklevel=3,
        )
        window = GATEWAY_MIN_CONTEXT_TOKENS
    timeout = gateway.get("generation_timeout_minutes", 30)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        warnings.warn(
            f"sous config: [gateway].generation_timeout_minutes {timeout!r} must be a "
            "positive integer; using 30",
            stacklevel=3,
        )
        timeout = 30
    return enabled, tuple(models), window, timeout
```

In `load_config`, read the section and pass the values:

```python
    tasks = _section(raw, "tasks")
    gateway_enabled, gateway_models, gateway_window, gateway_timeout = _gateway_values(
        _section(raw, "gateway")
    )
    return SousConfig(
        ...
        context_min_tokens=context_min_tokens,
        gateway_enabled=gateway_enabled,
        gateway_local_models=gateway_models,
        gateway_max_context_tokens=gateway_window,
        gateway_generation_timeout_minutes=gateway_timeout,
        data_dir=(path.parent if path.parent != Path(".") else DEFAULT_DATA_DIR),
        config_path=path,
    )
```

In `src/sous/server.py`, `SousService.server_status`, add a sibling of `"context"` inside `"config"`:

```python
                "gateway": {
                    "enabled": self.config.gateway_enabled,
                    "local_models": list(self.config.gateway_local_models),
                    "max_context_tokens": self.config.gateway_max_context_tokens,
                },
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_server.py -q`
Expected: all PASS (the pre-existing `test_server_status_reports_context_policy` still passes: it asserts only the `context` sub-dict).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/config.py src/sous/server.py tests/test_config.py tests/test_server.py
git commit -m "feat(config): add the opt-in [gateway] section

The Anthropic-compatible endpoint (issue #41, hybrid gateway spec Phase 1)
is off by default and experimental. Its context window is a separate knob
from the worker's because Claude Code refuses models under 48K of context
and its prompts dwarf the worker's; a configured value below that floor is
clamped up to it rather than served, since it could never be used.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 2: Request-scoped tool validation in `protocol.py`

**Files:**
- Modify: `src/sous/protocol.py`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Produces: `protocol.ToolSet` (frozen dataclass: `names: frozenset[str]`, `param_types: dict[str, dict[str, str]]`, `strict: bool`), `ToolSet.from_tools(tools: list[dict], *, strict: bool = True) -> ToolSet` (takes OpenAI-style `{"type": "function", "function": {"name", "parameters": {"properties": ...}}}` dicts — the shape `WORKER_TOOLS` already uses and Task 4 produces), `ToolSet.check(name) -> dict[str, str]`, `protocol.WORKER_TOOLSET`, and `parse_tool_calls(text: str, toolset: ToolSet = WORKER_TOOLSET) -> list[ToolCall]`. XML-ish parameters typed `array`/`object` in the schema are parsed as JSON.
- Consumes: nothing new. The worker keeps calling `parse_tool_calls(text)` unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_protocol.py`:

```python
# --- request-scoped tool sets (gateway) ----------------------------------------

from sous.protocol import WORKER_TOOLSET, ToolSet  # noqa: E402 — grouped with its tests

CLIENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "offset": {"type": "number"},
                    "limit": {"type": ["integer", "null"]},
                    "mode": {"enum": ["a", "b"]},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"todos": {"type": "array", "items": {"type": "object"}}},
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "run_in_background": {"type": "boolean"},
                    "meta": {"type": "object"},
                },
                "required": ["command"],
            },
        },
    },
]


def test_default_toolset_is_the_strict_worker_set():
    assert WORKER_TOOLSET.names == EXPECTED_TOOLS
    assert WORKER_TOOLSET.strict is True
    assert WORKER_TOOLSET.param_types["read_file"]["offset"] == "integer"
    assert WORKER_TOOLSET.param_types["list_dir"] == {"path": "string"}


def test_request_scoped_toolset_accepts_client_tool_names():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = '<tool_call>{"name": "Read", "arguments": {"file_path": "a.py"}}</tool_call>'
    assert parse_tool_calls(text, ts) == [ToolCall("Read", {"file_path": "a.py"})]


def test_strict_request_scoped_toolset_rejects_worker_names():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=True)
    with pytest.raises(ParseError, match="unknown tool"):
        parse_tool_calls('<tool_call>{"name": "read_file", "arguments": {}}</tool_call>', ts)


def test_non_strict_toolset_passes_unknown_names_through_untyped():
    """Claude Code answers a hallucinated tool with its own tool-not-found
    result, which the model can recover from; ending the turn here could not
    be undone. With no schema to coerce against, every value stays a string."""
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Imaginary>\n<parameter=count>\n5\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert parse_tool_calls(text, ts) == [ToolCall("Imaginary", {"count": "5"})]


def test_xml_number_union_and_untyped_parameters_coerce_from_the_client_schema():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=Read>\n"
        "<parameter=file_path>\na.py\n</parameter>\n"
        "<parameter=offset>\n10\n</parameter>\n"
        "<parameter=limit>\n20\n</parameter>\n"
        "<parameter=mode>\na\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = parse_tool_calls(text, ts)
    assert call.arguments == {"file_path": "a.py", "offset": 10.0, "limit": 20, "mode": "a"}
    assert isinstance(call.arguments["limit"], int)


def test_xml_array_and_object_parameters_parse_as_json():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=TodoWrite>\n"
        '<parameter=todos>\n[{"content": "x", "status": "pending"}]\n</parameter>\n'
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=Bash>\n"
        "<parameter=command>\nls\n</parameter>\n"
        '<parameter=meta>\n{"a": [1, 2]}\n</parameter>\n'
        "<parameter=run_in_background>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text, ts)
    assert calls[0].arguments == {"todos": [{"content": "x", "status": "pending"}]}
    assert calls[1].arguments == {"command": "ls", "meta": {"a": [1, 2]}, "run_in_background": True}


def test_xml_malformed_array_parameter_raises():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        "<tool_call>\n<function=TodoWrite>\n<parameter=todos>\nnot json\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    with pytest.raises(ParseError, match="todos"):
        parse_tool_calls(text, ts)


def test_xml_wrong_json_shape_for_array_parameter_raises():
    ts = ToolSet.from_tools(CLIENT_TOOLS, strict=False)
    text = (
        '<tool_call>\n<function=TodoWrite>\n<parameter=todos>\n{"not": "a list"}\n'
        "</parameter>\n</function>\n</tool_call>"
    )
    with pytest.raises(ParseError, match="array"):
        parse_tool_calls(text, ts)


def test_json_call_with_non_string_name_raises():
    with pytest.raises(ParseError):
        parse_tool_calls('<tool_call>{"name": 5, "arguments": {}}</tool_call>')


def test_toolset_tolerates_tools_without_properties():
    ts = ToolSet.from_tools(
        [{"type": "function", "function": {"name": "Ping", "parameters": {"type": "object"}}}],
        strict=True,
    )
    assert ts.param_types == {"Ping": {}}
    assert parse_tool_calls("<tool_call>\n<function=Ping>\n</function>\n</tool_call>", ts) == [
        ToolCall("Ping", {})
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_protocol.py -q`
Expected: FAIL at import — `ImportError: cannot import name 'WORKER_TOOLSET' from 'sous.protocol'`.

- [ ] **Step 3: Implement `ToolSet` and thread it through the parser**

In `src/sous/protocol.py`, replace the `TOOL_NAMES` and `_PARAM_TYPES` definitions (lines 104–115) with:

```python
def _schema_type(prop: object) -> str:
    """The JSON-schema type coercion should target. A union (`["integer",
    "null"]`) resolves to its first non-null member; a property without a
    `type` (enum, anyOf, free-form) stays a raw string."""
    if not isinstance(prop, dict):
        return "string"
    typ = prop.get("type", "string")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), "string")
    return typ if isinstance(typ, str) else "string"


@dataclass(frozen=True)
class ToolSet:
    """The tools one generation may call, and how to type their arguments.

    The XML-ish format writes string arguments raw and everything else via
    tojson, so the schema is the only way to know that e.g. read_file's offset
    must become an int — hence the per-tool parameter types.

    `strict` is the worker's contract: a name outside the set is a malformed
    turn, handled by FORMAT_REMINDER. The gateway is not strict — Claude Code
    answers a hallucinated tool with its own tool-not-found result, which the
    model can recover from, whereas ending the turn here could not be undone.
    An unknown tool then coerces nothing: every parameter stays a string.
    """

    names: frozenset[str]
    param_types: dict[str, dict[str, str]]
    strict: bool = True

    @classmethod
    def from_tools(cls, tools: list[dict], *, strict: bool = True) -> ToolSet:
        param_types = {
            t["function"]["name"]: {
                key: _schema_type(prop)
                for key, prop in (
                    (t["function"].get("parameters") or {}).get("properties") or {}
                ).items()
            }
            for t in tools
        }
        return cls(frozenset(param_types), param_types, strict)

    def check(self, name: str) -> dict[str, str]:
        """Parameter types for `name`; ParseError when strict and unknown."""
        if name in self.param_types:
            return self.param_types[name]
        if self.strict:
            raise ParseError(f"unknown tool: {name!r}")
        return {}


WORKER_TOOLSET = ToolSet.from_tools(WORKER_TOOLS)
```

`ParseError` is defined further down today; move `class ParseError(Exception): pass` above `_schema_type` so `ToolSet.check` can reference it (keep the `ToolCall` dataclass where it is).

Change the parser entry point and both branches:

```python
def parse_tool_calls(text: str, toolset: ToolSet = WORKER_TOOLSET) -> list[ToolCall]:
    calls: list[ToolCall] = []
    pos = 0
    while (match := _OPEN_RE.search(text, pos)) is not None:
        start = match.end()
        first = text[start : start + 1]
        if first == "{":
            call, pos = _parse_json_call(text, start, toolset)
        elif first == "<":
            call, pos = _parse_xml_call(text, start, toolset)
        else:
            raise ParseError(
                "tool_call must contain a JSON object or a <function=...> "
                f"block, got {text[start : start + 20]!r}"
            )
        calls.append(call)
    return calls


def _parse_json_call(text: str, start: int, toolset: ToolSet) -> tuple[ToolCall, int]:
    """Hermes JSON: {"name": ..., "arguments": {...}}. Returns (call, end)."""
    try:
        payload, end = _DECODER.raw_decode(text, start)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON in tool_call: {e}") from e
    if not isinstance(payload, dict):
        raise ParseError("tool_call payload must be a JSON object")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ParseError(f"tool_call name must be a non-empty string, got {name!r}")
    toolset.check(name)
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ParseError("tool_call arguments must be a JSON object")
    return ToolCall(name, arguments), end


def _parse_xml_call(text: str, start: int, toolset: ToolSet) -> tuple[ToolCall, int]:
    """Qwen3 XML-ish: <function=NAME><parameter=KEY>...</parameter></function>.

    Returns (call, end) where end is past the closing </tool_call> so tag
    text embedded in argument values is never re-scanned as a new call.
    """
    fn = _FUNCTION_RE.match(text, start)
    if fn is None:
        raise ParseError("expected <function=NAME> after <tool_call>")
    name = fn.group(1)
    param_types = toolset.check(name)
    # From here down — `arguments: dict = {}`, the parameter loop, through
    # `return ToolCall(name, arguments), pos` — the function is unchanged.
```

In `_parse_xml_call`, delete exactly two things from today's body: the `if name not in TOOL_NAMES: raise ParseError(...)` check and the `param_types = _PARAM_TYPES[name]` lookup; `toolset.check(name)` replaces both. Everything from `arguments: dict = {}` to the final `return` stays byte for byte (it already reads `param_types.get(key, "string")`).

Extend `_coerce` with the JSON-typed cases, before the final `return raw`:

```python
    if typ in ("array", "object"):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            raise ParseError(
                f"parameter {key!r} of {tool} must be a JSON {typ}, got {raw!r}"
            ) from None
        if (typ == "array" and not isinstance(value, list)) or (
            typ == "object" and not isinstance(value, dict)
        ):
            raise ParseError(f"parameter {key!r} of {tool} must be a JSON {typ}, got {raw!r}")
        return value
    return raw
```

Update the module docstring's first paragraph to mention that names are validated against a `ToolSet` (the worker's by default, the request's tools in the gateway).

- [ ] **Step 4: Confirm nothing else imported the removed names, then run the tests**

Run: `grep -rn "TOOL_NAMES\|_PARAM_TYPES" src tests scripts`
Expected: no output (the only users were inside `protocol.py`).

Run: `uv run pytest tests/test_protocol.py tests/test_worker.py -q`
Expected: all PASS (`test_unknown_tool_raises`, `test_xml_unknown_tool_raises`, `test_missing_name_raises` still raise through the strict default).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/protocol.py tests/test_protocol.py
git commit -m "feat(protocol): validate tool calls against a request-scoped ToolSet

The gateway parses calls to whatever tools Claude Code sent, not the
worker's eight, and needs their schemas to type XML-ish parameters —
including array/object values, which the template writes via tojson.
Strictness stays the worker's contract (an unknown name is a malformed
turn); the gateway passes unknown names through so Claude Code can answer
with its own tool-not-found result instead of the turn ending here.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 3: Stream generation deltas through the engine stack

**Files:**
- Modify: `src/sous/engine/base.py`, `src/sous/engine/promptcache.py`, `src/sous/engine/lm.py`, `src/sous/engine/vlm.py`
- Modify: `tests/fake_engine.py`, `tests/test_engine_base.py`, `tests/test_worker.py`, `tests/test_promptcache.py`
- Test: `tests/test_engine_base.py`, `tests/test_promptcache.py`, `tests/test_engine_lm.py` (model), `tests/test_engine_vlm.py` (model)

**Interfaces:**
- Produces: `sous.engine.base.Delta` (frozen dataclass: `text: str`, `output_tokens: int` cumulative, `finish_reason: str | None` — `None` mid-stream, `"stop"` or `"length"` on the last delta, whose `text` may be empty), `sous.engine.base.OnDelta = Callable[[Delta], None]`; `Engine.generate(messages, tools, max_tokens, on_delta: OnDelta | None = None) -> str`, same on `ManagedEngine.generate`; `GenerationSession.generate(messages, tools, max_tokens, timeout, on_delta=None) -> str`; `CacheHooks.decode(cache, token_ids, max_tokens, on_delta)`; `PrefixCache.generate(stable_ids, full_ids, max_tokens, on_delta=None)`. `tests.fake_engine.FakeEngine.generate(..., on_delta=None)` emits one `Delta(text, max(1, len(text.split())), "stop")`; new `tests.fake_engine.ChunkedFakeEngine(script, delay=0.0)` streams each scripted reply split on `|`, one `Delta` per piece, sleeping `delay` between pieces, and sets `.finished` (a `threading.Event`) when a generation completes.
- Contract: `on_delta` runs on the generating thread inside the decode loop; it must return quickly and never raise. After a stall, an abandoned session thread keeps calling it — consumers tolerate late deltas.

- [ ] **Step 1: Write the failing tests (base and promptcache)**

Append to `tests/test_engine_base.py`:

```python
# ---- streaming deltas (gateway) ---------------------------------------------


def test_session_relays_deltas_on_the_session_thread():
    """Deltas are emitted from inside the engine's decode loop, i.e. on the
    session thread — the consumer bridges them to wherever it lives."""
    from sous.engine.base import Delta

    seen: list[tuple[Delta, threading.Thread]] = []
    inner = FakeEngine(["hello world"])
    session = ManagedEngine(inner).session()
    text = session.generate(
        _msgs(), [], 8, timeout=5, on_delta=lambda d: seen.append((d, threading.current_thread()))
    )
    session.close()
    session._thread.join(5)
    assert text == "hello world"
    assert [d for d, _ in seen] == [Delta("hello world", 2, "stop")]
    assert seen[0][1] is session._thread


def test_managed_engine_forwards_on_delta():
    from sous.engine.base import Delta

    seen: list[Delta] = []
    managed = ManagedEngine(FakeEngine(["x y z"]))
    assert managed.generate(_msgs(), [], 8, on_delta=seen.append) == "x y z"
    assert seen == [Delta("x y z", 3, "stop")]


def test_chunked_fake_engine_streams_pieces_with_cumulative_counts():
    from sous.engine.base import Delta
    from tests.fake_engine import ChunkedFakeEngine

    seen: list[Delta] = []
    e = ChunkedFakeEngine(["a|b|c"])
    assert e.generate(_msgs(), [], 8, on_delta=seen.append) == "abc"
    assert seen == [Delta("a", 1, None), Delta("b", 2, None), Delta("c", 3, "stop")]
    assert e.finished.is_set()
```

Append to `tests/test_promptcache.py`:

```python
# ---- streaming deltas ----------------------------------------------------------


def test_on_delta_reaches_decode_on_every_path():
    from sous.engine.base import Delta

    seen: list[Delta] = []
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16, seen.append)  # cold
    pc.generate(STABLE_2, FULL_2, 16, seen.append)  # warm
    pc.generate(STABLE_2 + [7], FULL_2[:-2] + [7, 90, 91], 16)  # no consumer
    assert seen == [Delta("text", 1, "stop"), Delta("text", 1, "stop")]
    # The cache wraps the callback to count emissions, so decode sees a
    # callable (not the very object) when one was given, and None otherwise.
    assert [cb is not None for cb in h.on_deltas] == [True, True, False]
    disabled = PrefixCache(FakeHooks(trimmable=True), enabled=False)
    disabled.generate(STABLE_1, FULL_1, 16, seen.append)
    assert len(seen) == 3


def test_a_warm_failure_after_streamed_deltas_is_not_retried():
    """A cold retry would replay text the consumer already forwarded; the
    failure is surfaced instead, and the counters say no retry happened."""
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    h.fail_once = True
    h.stream_before_fail = True
    with pytest.raises(RuntimeError, match="not retrying cold"):
        pc.generate(STABLE_2, FULL_2, 16, lambda d: None)
    assert pc.stats()["cold_retries"] == 0
    # The warm attempt raised before FakeHooks recorded it and no cold retry
    # ran: the only decode on record is the first turn's, decode was entered
    # exactly twice, and no replacement cache was ever built.
    assert h.decoded == [FULL_1]
    assert len(h.on_deltas) == 2
    assert len(h.caches) == 1


def test_a_warm_failure_before_any_delta_still_retries_cold():
    from sous.engine.base import Delta

    seen: list[Delta] = []
    h = FakeHooks(trimmable=True)
    pc = PrefixCache(h)
    pc.generate(STABLE_1, FULL_1, 16)
    h.fail_once = True  # raises before emitting anything (stream_before_fail stays False)
    with pytest.warns(UserWarning, match="retrying cold"):
        assert pc.generate(STABLE_2, FULL_2, 16, seen.append) == "text"
    assert pc.stats()["cold_retries"] == 1
    assert seen == [Delta("text", 1, "stop")]  # only the retry's delta got out
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_engine_base.py tests/test_promptcache.py -q -k "delta or chunked or streamed"`
Expected: FAIL — `ImportError: cannot import name 'Delta'` / `TypeError: ... unexpected keyword argument 'on_delta'`.

- [ ] **Step 3: Add `Delta`/`OnDelta` and thread the callback through `base.py`**

In `src/sous/engine/base.py`, add after the imports (`dataclass` needs importing from `dataclasses`):

```python
@dataclass(frozen=True)
class Delta:
    """One streamed piece of a generation, delivered as the engine produces it.

    `output_tokens` counts everything generated so far, this piece included.
    `finish_reason` is None until the final piece, then "stop" (the model ended
    its turn) or "length" (max_tokens was reached). The final piece may carry
    empty text — the detokenizer's flush — so an empty delta is not "nothing
    happened".
    """

    text: str
    output_tokens: int
    finish_reason: str | None = None


# Called on the generating thread, inside the decode loop: it must return
# quickly and must never raise — an exception here fails the generation.
OnDelta = Callable[[Delta], None]
```

Change the `Engine` protocol method:

```python
    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str: ...
```

`ManagedEngine.generate`:

```python
    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        with self._gen_lock:
            return self._inner.generate(messages, tools, max_tokens, on_delta)
```

`GenerationSession.generate` — add the parameter and put it in the request tuple (`_loop` already calls `self._managed._inner.generate(*req)`, so it needs no change):

```python
    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        timeout: float,
        on_delta: OnDelta | None = None,
    ) -> str:
        assert not self._closed and not self._abandoned.is_set(), (
            "session reused after close() or a stall"
        )
        self._requests.put_nowait((messages, tools, max_tokens, on_delta))
```

Append to the `GenerationSession` class docstring:

```
    on_delta, when given, fires on this thread from inside the engine's decode
    loop — mid-generation, under _gen_lock. A stalled-and-abandoned generation
    keeps firing it until it ends, so a consumer must tolerate deltas that
    arrive after generate() has already raised GenerationStalled.
```

- [ ] **Step 4: Thread `on_delta` through `promptcache.py`**

In `src/sous/engine/promptcache.py`, import the types (this module stays mlx-free; `base` has no module-level mlx import either):

```python
from sous.engine.base import Delta, OnDelta
```

`CacheHooks.decode`:

```python
    def decode(
        self, cache: list, token_ids: list[int], max_tokens: int, on_delta: OnDelta | None
    ) -> str: ...
```

`PrefixCache.generate` — new signature and body changes (everything not shown is unchanged):

```python
    def generate(
        self,
        stable_ids: list[int],
        full_ids: list[int],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        hooks = self._hooks
        if not self.enabled:
            return hooks.decode(hooks.new_cache(), list(full_ids), max_tokens, on_delta)
        ...
        if reuse_length(stable_ids, full_ids) == 0:
            ...
            return hooks.decode(hooks.new_cache(), list(full_ids), max_tokens, on_delta)
        ...
        # A warm attempt that already streamed text cannot be retried: the
        # consumer has forwarded those deltas, and a cold re-run would deliver
        # the turn a second time. Counting is all the decision below needs.
        emitted = 0

        def _counting(sink: OnDelta) -> OnDelta:
            def relay(delta: Delta) -> None:
                nonlocal emitted
                emitted += 1
                sink(delta)

            return relay

        relay = _counting(on_delta) if on_delta is not None else None

        text = ""
        retry_reason: str | None = None
        try:
            text = self._run(stats, warm, stable_ids, full_ids, reuse, max_tokens, relay)
        except Exception as e:
            if reuse == 0:
                raise
            retry_reason = str(e)

        if retry_reason is not None:
            if emitted:
                raise RuntimeError(
                    f"warm generation failed after streaming {emitted} delta(s) "
                    f"({retry_reason}); not retrying cold, which would replay the turn"
                )
            stats.cold_retries += 1
            warnings.warn(...)  # unchanged
            warm = hooks.new_cache()
            text = self._run(stats, warm, stable_ids, full_ids, 0, max_tokens, relay)
```

`_run` gains a trailing `on_delta: OnDelta | None` parameter and passes it to both `hooks.decode(...)` calls.

- [ ] **Step 5: Emit deltas from both engines**

`src/sous/engine/lm.py` — import `Delta, OnDelta` from `sous.engine.base`; `decode` and `generate`:

```python
    def decode(
        self, cache: list, token_ids: list[int], max_tokens: int, on_delta: OnDelta | None = None
    ) -> str:
        from mlx_lm import stream_generate

        model, tokenizer = self._loaded()
        chunks: list[str] = []
        for r in stream_generate(
            model,
            tokenizer,
            token_ids,
            max_tokens=max_tokens,
            sampler=self._sampler,
            prompt_cache=cache,
        ):
            chunks.append(r.text)
            if on_delta is not None:
                on_delta(Delta(r.text, r.generation_tokens, r.finish_reason))
        return "".join(chunks)

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        full_ids = self._ids("full", messages, tools)
        stable_ids = self._ids("stable", messages, tools) if self._cache.enabled else []
        return self._cache.generate(stable_ids, full_ids, max_tokens, on_delta)
```

`src/sous/engine/vlm.py` — same `generate` change; `decode` moves from `generate()` to `stream_generate()`, which is exactly what `generate()` wraps (it iterates `stream_generate(..., **kwargs)`, skips `is_draft` rows, concatenates `.text`, and resets the shared stopping criteria first):

```python
    def decode(
        self, cache: list, token_ids: list[int], max_tokens: int, on_delta: OnDelta | None = None
    ) -> str:
        import mlx.core as mx
        from mlx_vlm import stream_generate

        model, processor = self._loaded()
        # Speculative decoding rides on the decode call only: prefill has no
        # tokens to draft, and generate_step captures the hidden states the
        # drafter needs during its own prefill of these input_ids. block size
        # 0 means None — let the drafter's own policy pick the depth.
        draft_kwargs = (
            {
                "draft_model": self._draft,
                "draft_kind": self._draft_kind,
                "draft_block_size": self._draft_block_size or None,
            }
            if self._draft is not None
            else {}
        )
        # generate() resets the tokenizer's shared stopping criteria before
        # every call and stream_generate does not; mirror it so a criteria
        # left mutated by another caller cannot change where this turn stops.
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)
        chunks: list[str] = []
        # prompt_cache plus input_ids, not prompt_cache_state. mlx-vlm primes
        # Qwen mRoPE state before feeding a suffix, and that priming turns out
        # to be bit-identical to no priming for text-only prompts — so sous
        # owns the cache outright rather than driving mlx-vlm's reuse path.
        for r in stream_generate(
            model,
            processor,
            "",
            max_tokens=max_tokens,
            sampler=self._sampler,
            verbose=False,
            prompt_cache=cache,
            input_ids=mx.array(token_ids)[None],
            **draft_kwargs,
        ):
            # Draft rows are the speculator's proposals, not accepted output;
            # generate() skips them the same way.
            if r.is_draft:
                continue
            chunks.append(r.text)
            if on_delta is not None:
                on_delta(Delta(r.text, r.generation_tokens, r.finish_reason))
        return "".join(chunks)
```

(`prefill` keeps calling `generate(..., max_tokens=0)`: that prefill-only path is a property of the shared step loop and is unchanged.)

- [ ] **Step 6: Update the fakes**

`tests/fake_engine.py` becomes:

```python
"""Scripted Engine implementations for model-free tests."""

import threading
import time

from sous.engine.base import Delta, OnDelta


class FakeEngine:
    model_id = "fake/model"

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.tools_seen: list[list[dict]] = []
        self.max_tokens_seen: list[int] = []
        self.unloaded = False
        self.resets = 0
        self.stats: dict = {}
        # Which thread ran each call: the per-task-thread design (issue #34)
        # is pinned on these. Thread objects, not idents — idents recycle.
        self.generate_threads: list[threading.Thread] = []
        self.reset_idents: list[int] = []

    def _take(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        self.generate_threads.append(threading.current_thread())
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append(list(tools))
        self.max_tokens_seen.append(max_tokens)
        if not self.script:
            raise AssertionError("FakeEngine script exhausted")
        return self.script.pop(0)

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        text = self._take(messages, tools, max_tokens)
        if on_delta is not None:
            # One delta per generation is the minimum streaming contract; tests
            # that need to watch pieces arrive use ChunkedFakeEngine.
            on_delta(Delta(text, max(1, len(text.split())), "stop"))
        return text

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return sum(len(str(m)) for m in messages) // 4

    def reset_prompt_cache(self) -> None:
        self.resets += 1
        self.reset_idents.append(threading.get_ident())

    def prompt_cache_stats(self) -> dict:
        return dict(self.stats)

    def unload(self) -> None:
        self.unloaded = True


class ChunkedFakeEngine(FakeEngine):
    """Streams each scripted reply in pieces split on `|`, sleeping `delay`
    seconds before each piece, so gateway tests can watch deltas and
    keepalives arrive while a generation is still running. `finished` is set
    when a generation completes — the drain-on-disconnect tests wait on it."""

    finish_reason = "stop"

    def __init__(self, script: list[str], delay: float = 0.0):
        super().__init__(script)
        self.delay = delay
        self.finished = threading.Event()

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        pieces = self._take(messages, tools, max_tokens).split("|")
        for n, piece in enumerate(pieces, start=1):
            if self.delay:
                time.sleep(self.delay)
            if on_delta is not None:
                last = n == len(pieces)
                on_delta(Delta(piece, n, self.finish_reason if last else None))
        self.finished.set()
        return "".join(pieces)
```

Every test-local `FakeEngine` subclass that overrides `generate` must accept the new positional argument, because `GenerationSession._loop` now calls `generate(*req)` with four values. Change each of these signatures from `def generate(self, messages, tools, max_tokens):` to `def generate(self, messages, tools, max_tokens, on_delta=None):` (bodies unchanged; their `super().generate(messages, tools, max_tokens)` calls stay valid):

- `tests/test_engine_base.py`: `_BlockingEngine.generate`, and the local classes `BlockingEngine` (in `test_reset_prompt_cache_does_not_wait_for_the_generation_lock`), `Flaky`, `Gated` (two of them), `Wedged`.
- `tests/test_worker.py`: `CancelAfterGen`, `DyingEngine`, `StallingEngine`, `ExplodingEngine`.

In `tests/test_promptcache.py`, `FakeHooks`: add two attributes and record the callback —

```python
    def __init__(
        self,
        trimmable: bool,
        layers: int = 2,
        fail_once: bool = False,
        decode_impl=None,
    ):
        ...
        self.fail_once = fail_once
        # When a failure is scripted, whether one delta escapes first — the
        # streaming no-retry rule is decided on exactly that difference.
        self.stream_before_fail = False
        self.on_deltas: list = []
        ...

    def decode(self, cache, token_ids, max_tokens, on_delta=None):
        self.on_deltas.append(on_delta)
        if self.fail_once:
            self.fail_once = False
            if self.stream_before_fail and on_delta is not None:
                on_delta(Delta("partial", 1, None))
            raise RuntimeError("boom")
        if self.decode_impl is not None:
            return self.decode_impl(self, cache, token_ids, max_tokens)
        self.decoded.append(list(token_ids))
        self._advance(cache, len(token_ids) + len(self.generated))
        if on_delta is not None:
            on_delta(Delta("text", 1, "stop"))
        return "text"
```

with `from sous.engine.base import Delta` added to the test module's imports.

- [ ] **Step 7: Run the suite**

Run: `uv run pytest -m "not model" -q`
Expected: all PASS, output pristine (no `TypeError` from a fake with the old arity — if one appears, the traceback names the class to fix).

- [ ] **Step 8: Add the model-marked streaming tests (local only)**

Append to `tests/test_engine_lm.py`:

```python
def test_lm_engine_streams_deltas_that_reassemble_the_reply():
    from sous.engine.base import Delta
    from sous.engine.lm import LMEngine
    from sous.protocol import WORKER_TOOLS

    e = LMEngine(TINY)
    seen: list[Delta] = []
    msgs = [{"role": "user", "content": "Count from one to five."}]
    out = e.generate(msgs, WORKER_TOOLS, max_tokens=32, on_delta=seen.append)
    assert "".join(d.text for d in seen) == out
    assert [d.output_tokens for d in seen] == sorted(d.output_tokens for d in seen)
    assert seen[-1].finish_reason in ("stop", "length")
    assert all(d.finish_reason is None for d in seen[:-1])
    e.unload()
```

Append to `tests/test_engine_vlm.py` (uses the file's existing `TINY_VLM`):

```python
def test_vlm_engine_streams_deltas_that_reassemble_the_reply():
    from sous.engine.base import Delta
    from sous.engine.vlm import VLMEngine
    from sous.protocol import WORKER_TOOLS

    e = VLMEngine(TINY_VLM)
    seen: list[Delta] = []
    msgs = [{"role": "user", "content": "Count from one to five."}]
    out = e.generate(msgs, WORKER_TOOLS, max_tokens=32, on_delta=seen.append)
    assert "".join(d.text for d in seen) == out
    assert seen[-1].finish_reason in ("stop", "length")
    assert all(d.finish_reason is None for d in seen[:-1])
    assert [d.output_tokens for d in seen] == sorted(d.output_tokens for d in seen)
    # mlx-vlm yields every token and then a flush result: on an EOS stop the
    # loop breaks before yielding, so the flush carries len(seen); on a
    # max_tokens finish it repeats the last count, len(seen) - 1.
    assert seen[-1].output_tokens in (len(seen), len(seen) - 1)
    e.unload()
```

Run: `uv run pytest tests/test_engine_lm.py tests/test_engine_vlm.py -q -m model -k streams`
Expected: PASS (downloads ~350 MB and ~1 GB once). Also run the existing drafter test so the `stream_generate` switch is proven on the speculative path: `uv run pytest tests/test_engine_vlm.py -q -m model -k drafter` — Expected: PASS.

- [ ] **Step 9: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/engine tests/fake_engine.py tests/test_engine_base.py tests/test_worker.py tests/test_promptcache.py tests/test_engine_lm.py tests/test_engine_vlm.py
git commit -m "feat(engine): stream generation deltas through the engine stack

The gateway needs text as it is produced plus the final token count and
stop reason; the worker needs nothing new. An optional on_delta callback
rides in GenerationSession's request tuple and fires on the session thread
from inside the decode loop, leaving the one-slot request/reply protocol
and its stall semantics untouched. VLMEngine.decode moves from generate()
to the stream_generate() it wraps, replicating its stopping-criteria reset
and its draft-row skipping. PrefixCache refuses a cold retry once any delta
has been streamed: a retry would replay the turn to the consumer.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 4: `gateway/convert.py` — Anthropic request → chat messages and tools

**Files:**
- Create: `src/sous/gateway/__init__.py` (empty)
- Create: `src/sous/gateway/convert.py`
- Test: `tests/test_gateway_convert.py`

**Interfaces:**
- Produces: `RequestError(status: int, error_type: str, message: str)` with `.status`, `.error_type`, `.message`, `.body() -> dict` (Anthropic shape `{"type": "error", "error": {"type", "message"}}`); `ChatRequest` dataclass (`model: str`, `messages: list[dict]`, `tools: list[dict]`, `max_tokens: int`, `stream: bool`, `dropped_server_tools: list[str]`); `parse_messages_request(body: object) -> ChatRequest` (raises `RequestError` 400 on shape problems); `parse_count_tokens_request(body: object) -> ChatRequest`; `strip_volatile(text: str) -> str`; `chat_messages(system, messages) -> list[dict]`; `chat_tools(tools) -> tuple[list[dict], list[str]]`. The `tools` list is OpenAI-style `{"type": "function", "function": {"name", "description", "parameters"}}` — what `ToolSet.from_tools` (Task 2) and the chat template consume.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gateway_convert.py`:

```python
"""Anthropic request → chat-template inputs. Pure; every Claude Code
accommodation from the gateway spec's checklist is pinned here."""

import pytest

from sous.gateway.convert import (
    ChatRequest,
    RequestError,
    chat_messages,
    chat_tools,
    parse_count_tokens_request,
    parse_messages_request,
    strip_volatile,
)

READ_TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
}


def _body(**overrides) -> dict:
    body = {
        "model": "sous-local",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


# --- system prompt: canonical field, inline system messages, volatile markers --


def test_string_system_becomes_the_leading_system_message():
    out = chat_messages("Be terse.", [{"role": "user", "content": "hi"}])
    assert out == [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}]


def test_system_blocks_are_joined_and_the_billing_header_block_is_dropped():
    """Checklist item 4: the x-anthropic-billing-header block carries
    per-request random values that would defeat the prefix cache."""
    system = [
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cc_is_subagent=true"},
        {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Rules follow."},
    ]
    out = chat_messages(system, [{"role": "user", "content": "hi"}])
    assert out[0] == {"role": "system", "content": "You are Claude Code.\n\nRules follow."}


def test_inline_system_messages_fold_into_the_leading_system_message():
    """Checklist item 1: Claude Code >= 2.1.154 puts system content in
    messages[] (gate 2 saw a 7.8 KB string there); Qwen's template accepts a
    system message only at index 0, canonical field first."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "system", "content": "Inline instructions."},
        {"role": "system", "content": [{"type": "text", "text": "More."}]},
    ]
    out = chat_messages([{"type": "text", "text": "Canonical."}], messages)
    assert out == [
        {"role": "system", "content": "Canonical.\n\nInline instructions.\n\nMore."},
        {"role": "user", "content": "task"},
    ]


def test_inline_system_without_a_canonical_field_still_leads():
    out = chat_messages(None, [{"role": "user", "content": "q"}, {"role": "system", "content": "S"}])
    assert out[0] == {"role": "system", "content": "S"}


def test_total_tokens_markers_are_stripped_from_system_and_user_text():
    """Checklist item 3: a freshly decremented copy is appended every request
    and the stale ones kept — without stripping, no prefix past it is reusable."""
    marker = "\n\n<total_tokens>15000000 tokens left</total_tokens>"
    assert strip_volatile("Rules." + marker) == "Rules."
    assert strip_volatile("a" + marker + marker) == "a"
    assert strip_volatile("no marker here") == "no marker here"
    wrapped = "<system-reminder>\n<total_tokens>5 tokens left</total_tokens>\n</system-reminder>"
    assert strip_volatile("Hi.\n\n" + wrapped) == "Hi."
    out = chat_messages("Sys." + marker, [{"role": "user", "content": "Hi." + marker}])
    assert out == [{"role": "system", "content": "Sys."}, {"role": "user", "content": "Hi."}]


def test_marker_only_text_after_tool_results_adds_no_user_turn():
    """Claude Code emits the marker as an attachment after every tool-result
    batch (bare, or wrapped in a system-reminder). Stripped, nothing remains,
    and an empty user turn after the tool responses would tell the model the
    user said nothing."""
    marker = "<total_tokens>123 tokens left</total_tokens>"
    for trailer in (marker, f"<system-reminder>\n{marker}\n</system-reminder>"):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "A"}, {"type": "text", "text": trailer}]},
            {"role": "user", "content": trailer},
        ]
        assert chat_messages(None, messages)[2:] == [{"role": "tool", "content": "A"}]


def test_tool_result_content_is_not_marker_stripped():
    """A file the model read may legitimately contain the marker text; only
    the volatile places Claude Code writes it are stripped."""
    text = "<total_tokens>5 tokens left</total_tokens>"
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": text}]},
    ]
    assert chat_messages(None, messages)[-1] == {"role": "tool", "content": text}


# --- user turns -------------------------------------------------------------------


def test_user_text_blocks_join_with_newlines():
    out = chat_messages(None, [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert out == [{"role": "user", "content": "a\nb"}]


def test_tool_results_become_tool_messages_in_order_with_text_flushed_first():
    """Qwen's template matches results to calls by ORDER (it never renders the
    id) and groups consecutive tool messages into one user turn; text that
    preceded a result must be its own turn to keep that order."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a"}},
                {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "b"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "note before"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "A"},
                {"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "B1"}, {"type": "text", "text": "B2"}], "is_error": True},
                {"type": "text", "text": "note after"},
            ],
        },
    ]
    out = chat_messages(None, messages)
    assert out[1] == {
        "role": "assistant",
        "content": "Reading.",
        "tool_calls": [
            {"type": "function", "function": {"name": "Read", "arguments": {"file_path": "a"}}},
            {"type": "function", "function": {"name": "Read", "arguments": {"file_path": "b"}}},
        ],
    }
    assert out[2:] == [
        {"role": "user", "content": "note before"},
        {"role": "tool", "content": "A"},
        {"role": "tool", "content": "B1\nB2"},
        {"role": "user", "content": "note after"},
    ]


def test_images_and_documents_become_placeholders_and_unknown_blocks_are_skipped():
    """Checklist item 8: tolerate, never 4xx. sous serves text only."""
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "text", "text": "what is this?"},
        {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "x"}},
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
        {"type": "something_new_20270101", "payload": 1},
    ]
    out = chat_messages(None, [{"role": "user", "content": content}])
    assert out == [
        {
            "role": "user",
            "content": (
                "[image omitted: sous serves text only]\nwhat is this?\n"
                "[document omitted: sous serves text only]"
            ),
        }
    ]


def test_empty_user_content_yields_an_empty_user_turn():
    assert chat_messages(None, [{"role": "user", "content": []}]) == [{"role": "user", "content": ""}]


def test_tool_result_without_content_is_an_empty_tool_message():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]},
    ]
    assert chat_messages(None, messages)[-1] == {"role": "tool", "content": ""}


# --- assistant turns --------------------------------------------------------------


def test_assistant_thinking_blocks_are_dropped_and_string_input_becomes_empty_args():
    content = [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "redacted_thinking", "data": "..."},
        {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": "not-a-dict"},
    ]
    out = chat_messages(None, [{"role": "user", "content": "q"}, {"role": "assistant", "content": content}])
    assert out[1] == {
        "role": "assistant",
        "content": "ok",
        "tool_calls": [{"type": "function", "function": {"name": "Bash", "arguments": {}}}],
    }


def test_assistant_string_content_passes_through():
    out = chat_messages(None, [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A"}])
    assert out[1] == {"role": "assistant", "content": "A"}


def test_unknown_role_is_a_400():
    with pytest.raises(RequestError) as exc:
        chat_messages(None, [{"role": "tool", "content": "x"}])
    assert exc.value.status == 400 and exc.value.error_type == "invalid_request_error"


def test_non_list_non_string_content_is_a_400():
    with pytest.raises(RequestError):
        chat_messages(None, [{"role": "user", "content": {"type": "text", "text": "x"}}])


# --- tools ------------------------------------------------------------------------


def test_client_tools_convert_to_function_schemas():
    tools, dropped = chat_tools([READ_TOOL, {"name": "NoDesc", "input_schema": {"type": "object"}}])
    assert tools == [
        {"type": "function", "function": {"name": "Read", "description": "Read a file", "parameters": READ_TOOL["input_schema"]}},
        {"type": "function", "function": {"name": "NoDesc", "description": "", "parameters": {"type": "object"}}},
    ]
    assert dropped == []


def test_server_side_tools_are_dropped_and_named():
    """Checklist item 7: anything with a non-custom `type` executes inside
    Anthropic's API; toolset entries carry no `name` at all."""
    tools, dropped = chat_tools([
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
        {"type": "browser_toolset_20260801", "configs": {}},
        {"type": "custom", **READ_TOOL},
        READ_TOOL,
    ])
    assert [t["function"]["name"] for t in tools] == ["Read", "Read"]
    assert dropped == ["web_search_20250305:web_search", "browser_toolset_20260801:"]


def test_client_tool_without_schema_or_name_is_a_400():
    with pytest.raises(RequestError, match="input_schema"):
        chat_tools([{"name": "Broken"}])
    with pytest.raises(RequestError, match="name"):
        chat_tools([{"input_schema": {"type": "object"}}])


def test_no_tools_is_fine():
    assert chat_tools(None) == ([], [])


# --- whole requests ---------------------------------------------------------------


def test_parse_messages_request_converts_and_ignores_unknown_fields():
    """Decision 9: thinking, output_config, context_management, metadata and
    the sampling knobs are accepted and ignored; gate 1 saw all of them."""
    body = _body(
        stream=True,
        system=[{"type": "text", "text": "S"}],
        tools=[READ_TOOL],
        thinking={"type": "adaptive", "display": "omitted"},
        output_config={"effort": "high"},
        context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        metadata={"user_id": "abc"},
        temperature=1,
        tool_choice={"type": "auto"},
        stop_sequences=["x"],
        totally_new_field=True,
    )
    chat = parse_messages_request(body)
    assert isinstance(chat, ChatRequest)
    assert chat.model == "sous-local" and chat.stream is True and chat.max_tokens == 32000
    assert chat.messages == [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    assert chat.tools[0]["function"]["name"] == "Read"
    assert chat.dropped_server_tools == []


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ([], "JSON object"),
        (_body(model=""), "model"),
        (_body(messages=[]), "messages"),
        (_body(messages=[{"role": "robot", "content": "x"}]), "role"),
        (_body(messages=["not a dict"]), "role"),
        (_body(max_tokens=0), "max_tokens"),
        (_body(max_tokens=True), "max_tokens"),
        ({k: v for k, v in _body().items() if k != "max_tokens"}, "max_tokens"),
        (_body(stream="yes"), "stream"),
        (_body(tools="nope"), "tools"),
    ],
)
def test_parse_messages_request_rejects_bad_shapes_with_400(body, fragment):
    with pytest.raises(RequestError) as exc:
        parse_messages_request(body)
    assert exc.value.status == 400
    assert exc.value.error_type == "invalid_request_error"
    assert fragment in exc.value.message
    assert exc.value.body() == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": exc.value.message},
    }


def test_parse_count_tokens_request_needs_no_max_tokens_or_stream():
    body = {"model": "sous-local", "messages": [{"role": "user", "content": "hi"}], "tools": [READ_TOOL]}
    chat = parse_count_tokens_request(body)
    assert chat.stream is False and chat.tools[0]["function"]["name"] == "Read"
    with pytest.raises(RequestError):
        parse_count_tokens_request("not an object")


def test_gate_capture_shape_converts_cleanly():
    """The structure gate 1 captured (comment 5440572099 / dump inspection):
    billing header first in a 3-block system list, cache_control on some
    blocks, a two-block user turn, then a plain-string inline system message."""
    body = _body(
        system=[
            {
                "type": "text",
                "text": (
                    "x-anthropic-billing-header: cc_version=2.1.238; cc_entrypoint=cli; "
                    "cc_is_subagent=true"
                ),
            },
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Long system prompt.", "cache_control": {"type": "ephemeral"}},
        ],
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Agent prompt."},
                {"type": "text", "text": "Environment.", "cache_control": {"type": "ephemeral"}},
            ]},
            {"role": "system", "content": "Inline subagent system text."},
        ],
        tools=[READ_TOOL],
        thinking={"type": "adaptive", "display": "omitted"},
    )
    chat = parse_messages_request(body)
    assert chat.messages == [
        {
            "role": "system",
            "content": (
                "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
                "Long system prompt.\n\nInline subagent system text."
            ),
        },
        {"role": "user", "content": "Agent prompt.\nEnvironment."},
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_convert.py -q`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'sous.gateway'`.

- [ ] **Step 3: Create the package and `convert.py`**

Create empty `src/sous/gateway/__init__.py`. Create `src/sous/gateway/convert.py`:

```python
"""Anthropic Messages requests → the chat messages and tools the engine renders.

Pure: no engine, no mlx, no I/O. Everything Claude Code sends that the local
model cannot use is dropped here, deliberately and visibly (the spec's
accommodation checklist): server-side tools, thinking blocks, images, and the
per-request volatile markers that would otherwise defeat the prefix cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A client tool's `type` is absent, null, or "custom"; every other value names
# a tool that runs inside Anthropic's API (web_search_*, web_fetch_*,
# code_execution_*, bash_*, text_editor_*, computer_*, memory_*,
# browser_toolset_*, tool_search_tool_*), which no local endpoint can run.
_CLIENT_TOOL_TYPES = (None, "custom")

# The first system block of a Claude Code request: per-request random values
# (checklist item 4), so it can never be part of a reusable prefix.
_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"

# Claude Code appends a freshly decremented copy of this marker on every
# request and keeps the stale ones (oMLX measured full ~60k-token re-prefills
# per turn until it was stripped). Nothing downstream reads it. The regex is
# oMLX's, verbatim: it eats up to two preceding newlines with the marker.
_TOTAL_TOKENS_RE = re.compile(r"\n{0,2}<total_tokens>\d+ tokens left</total_tokens>")
# One of Claude Code's assembly paths wraps that marker in a system-reminder;
# stripping the marker leaves the hollow shell, which goes too.
_EMPTY_REMINDER_RE = re.compile(r"\n{0,2}<system-reminder>\s*</system-reminder>")

_ROLES = ("user", "assistant", "system")
_OMITTED = "[{kind} omitted: sous serves text only]"


class RequestError(Exception):
    """A request the gateway rejects, in Anthropic's error vocabulary."""

    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message

    def body(self) -> dict:
        return {"type": "error", "error": {"type": self.error_type, "message": self.message}}


def _invalid(message: str) -> RequestError:
    return RequestError(400, "invalid_request_error", message)


@dataclass
class ChatRequest:
    """What the engine needs from a /v1/messages body, plus what the response
    has to echo (`model`) and the log has to mention (dropped tools)."""

    model: str
    messages: list[dict]
    tools: list[dict]
    max_tokens: int
    stream: bool
    dropped_server_tools: list[str] = field(default_factory=list)


def strip_volatile(text: str) -> str:
    if "<total_tokens>" not in text:
        return text
    return _EMPTY_REMINDER_RE.sub("", _TOTAL_TOKENS_RE.sub("", text))


def _text_parts(content: object, *, drop_billing: bool) -> list[str]:
    """The text of a string-or-blocks content value; non-text blocks skipped."""
    if isinstance(content, str):
        return [content] if content else []
    if not isinstance(content, list):
        raise _invalid("content must be a string or a list of content blocks")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text", "")
        if not isinstance(text, str) or not text:
            continue
        if drop_billing and text.startswith(_BILLING_HEADER_PREFIX):
            continue
        parts.append(text)
    return parts


def _system_text(system: object, messages: list[dict]) -> str:
    """One system prompt from the canonical field plus every inline system
    message, canonical first. Qwen's template accepts a system message only
    at index 0 (it raises otherwise), so this is the only place they can go."""
    parts = _text_parts(system, drop_billing=True) if system is not None else []
    for msg in messages:
        if msg.get("role") == "system":
            parts.extend(_text_parts(msg.get("content"), drop_billing=True))
    return strip_volatile("\n\n".join(parts))


def _tool_result_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") in ("image", "document"):
                parts.append(_OMITTED.format(kind=item["type"]))
        return "\n".join(parts)
    return str(content)


def _user_turns(content: object) -> list[dict]:
    if isinstance(content, str):
        text = strip_volatile(content)
        # A message that was only the marker contributes nothing; a genuinely
        # empty message keeps its (empty) turn.
        return [{"role": "user", "content": text}] if text or not content else []
    if not isinstance(content, list):
        raise _invalid("content must be a string or a list of content blocks")
    out: list[dict] = []
    texts: list[str] = []
    recognized = False

    def flush() -> None:
        # Claude Code appends the <total_tokens> marker as its own text block
        # after every tool-result batch. Once stripped there is nothing left,
        # and an empty user turn right after the tool responses would read to
        # the model as "the user said nothing".
        text = strip_volatile("\n".join(texts))
        texts.clear()
        if text:
            out.append({"role": "user", "content": text})

    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            recognized = True
            texts.append(str(block.get("text", "")))
        elif kind == "tool_result":
            # The template groups consecutive tool messages into one user turn
            # and matches results to calls by order only, so text that came
            # before a result must be its own turn to keep the sequence.
            recognized = True
            flush()
            out.append({"role": "tool", "content": _tool_result_text(block.get("content"))})
        elif kind in ("image", "document"):
            recognized = True
            texts.append(_OMITTED.format(kind=kind))
        # thinking, redacted_thinking, server_tool_use, *_tool_result and
        # anything newer: dropped (checklist item 8 — tolerate, never 4xx).
    flush()
    if not out and not recognized:
        out.append({"role": "user", "content": ""})
    return out


def _assistant_turn(content: object) -> dict:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        raise _invalid("content must be a string or a list of content blocks")
    texts: list[str] = []
    calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            texts.append(str(block.get("text", "")))
        elif kind == "tool_use":
            arguments = block.get("input")
            # The template iterates arguments as a mapping (a JSON string
            # raises inside Jinja), so anything else becomes an empty call.
            calls.append({
                "type": "function",
                "function": {
                    "name": str(block.get("name", "")),
                    "arguments": arguments if isinstance(arguments, dict) else {},
                },
            })
        # thinking / redacted_thinking: the local model never produced them
        # and cannot verify them; dropped.
    turn: dict = {"role": "assistant", "content": "\n".join(texts)}
    if calls:
        turn["tool_calls"] = calls
    return turn


def chat_messages(system: object, messages: list[dict]) -> list[dict]:
    """The chat-template message list for a request. Raises RequestError."""
    out: list[dict] = []
    prompt = _system_text(system, messages)
    if prompt:
        out.append({"role": "system", "content": prompt})
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue  # folded into the system prompt above
        if role == "user":
            out.extend(_user_turns(msg.get("content")))
        elif role == "assistant":
            out.append(_assistant_turn(msg.get("content")))
        else:
            raise _invalid(f"messages: role must be one of {_ROLES}, got {role!r}")
    return out


def chat_tools(tools: object) -> tuple[list[dict], list[str]]:
    """OpenAI-style function schemas for the template, plus the server-side
    tools dropped on the way ("type:name", for the log)."""
    if tools is None:
        return [], []
    if not isinstance(tools, list):
        raise _invalid("tools must be a list")
    out: list[dict] = []
    dropped: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise _invalid("tools: each entry must be an object")
        if tool.get("type") not in _CLIENT_TOOL_TYPES:
            dropped.append(f"{tool.get('type')}:{tool.get('name', '')}")
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not name:
            raise _invalid("tools: each client tool needs a name")
        if not isinstance(schema, dict):
            raise _invalid(f"tools: {name} needs an input_schema object")
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description") or ""),
                "parameters": schema,
            },
        })
    return out, dropped


def parse_messages_request(body: object) -> ChatRequest:
    """Validate a /v1/messages body and convert it. Unknown top-level fields
    (thinking, output_config, context_management, metadata, temperature,
    top_p, top_k, stop_sequences, tool_choice, ...) are ignored: sampling is
    the daemon's [model] configuration and thinking is off."""
    if not isinstance(body, dict):
        raise _invalid("request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise _invalid("model: field required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _invalid("messages: at least one message is required")
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") not in _ROLES:
            raise _invalid(f"messages: each message needs a role in {_ROLES}")
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise _invalid("max_tokens: must be a positive integer")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise _invalid("stream: must be true or false")
    tools, dropped = chat_tools(body.get("tools"))
    return ChatRequest(
        model, chat_messages(body.get("system"), messages), tools, max_tokens, stream, dropped
    )


def parse_count_tokens_request(body: object) -> ChatRequest:
    """count_tokens has neither max_tokens nor stream; otherwise identical."""
    if not isinstance(body, dict):
        raise _invalid("request body must be a JSON object")
    return parse_messages_request({**body, "max_tokens": 1, "stream": False})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gateway_convert.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/gateway/__init__.py src/sous/gateway/convert.py tests/test_gateway_convert.py
git commit -m "feat(gateway): convert Anthropic requests to chat-template inputs

Pure conversion with every Claude Code accommodation the spec lists:
inline system messages fold into the leading system message (Qwen's
template rejects any other position), the billing-header block and the
<total_tokens> markers are stripped so the prefix cache can hold, server-
side tools are dropped by type, tool results become role:tool turns in
call order, and unknown block types are skipped rather than rejected.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 5: `gateway/response.py` — deltas → Anthropic events and message

**Files:**
- Create: `src/sous/gateway/response.py`
- Test: `tests/test_gateway_response.py`

**Interfaces:**
- Consumes: `sous.engine.base.Delta` (Task 3); `sous.protocol.ToolSet`, `parse_tool_calls`, `ParseError`, `ToolCall` (Task 2).
- Produces: `TextSplitter` (`feed(text) -> str`, `finish() -> str`); `TurnAssembler(message_id: str, model: str, toolset: ToolSet)` with `start(input_tokens) -> list[dict]`, `feed(delta) -> list[dict]`, `finish(text, output_tokens, finish_reason) -> list[dict]`, `message() -> dict`, attributes `input_tokens`, `output_tokens`, `stop_reason`; `new_message_id() -> str` (`msg_` + 24 hex), `new_tool_use_id() -> str` (`toolu_` + 24 hex); `stop_reason(finish_reason, has_calls) -> str`. Events are plain dicts whose `"type"` is the SSE event name (Task 7 frames them).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gateway_response.py`:

```python
"""Turn output → Anthropic content blocks and stream events. Pure."""

import json
import re

import pytest

from sous.engine.base import Delta
from sous.gateway.response import (
    TextSplitter,
    TurnAssembler,
    new_message_id,
    new_tool_use_id,
    stop_reason,
)
from sous.protocol import ToolSet

TOOLS = ToolSet.from_tools(
    [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
            },
        }
    ],
    strict=False,
)
XML_CALL = (
    "<tool_call>\n<function=Read>\n<parameter=file_path>\na.py\n</parameter>\n"
    "</function>\n</tool_call>"
)


def _types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def _stream(text_pieces: list[str], toolset: ToolSet = TOOLS, finish: str = "stop") -> tuple[TurnAssembler, list[dict]]:
    """Feed pieces as deltas, then finish with the joined text."""
    a = TurnAssembler("msg_x", "sous-local", toolset)
    events = a.start(100)
    for n, piece in enumerate(text_pieces, start=1):
        events += a.feed(Delta(piece, n, None))
    events += a.finish("".join(text_pieces), len(text_pieces), finish)
    return a, events


# --- TextSplitter -------------------------------------------------------------------


def test_splitter_emits_plain_text_but_holds_trailing_whitespace_and_tag_prefixes():
    s = TextSplitter()
    assert s.feed("Hello") == "Hello"
    assert s.feed(" world\n\n") == " world"  # trailing newlines held
    assert s.feed("<tool") == ""  # could be <tool_call>
    assert s.feed("s are fun") == "\n\n<tools are fun"  # it wasn't: released, with the held newlines
    assert s.finish() == ""


def test_splitter_drops_leading_whitespace():
    s = TextSplitter()
    assert s.feed("\n\n") == ""
    assert s.feed("  Hi") == "Hi"


def test_splitter_stops_at_tool_call_and_finish_returns_the_tail():
    s = TextSplitter()
    assert s.feed("Reading now.\n\n<tool_") == "Reading now."
    assert s.feed("call>\n<function=Read>") == ""
    assert s.feed("\n</function>\n</tool_call>") == ""
    # The blank lines before the tag ride with the tail, so an unparseable
    # call can be returned verbatim.
    assert s.finish() == "\n\n<tool_call>\n<function=Read>\n</function>\n</tool_call>"


def test_splitter_finish_flushes_held_text_when_no_call_came():
    s = TextSplitter()
    assert s.feed("Done <") == "Done"
    assert s.finish() == " <"


# --- TurnAssembler: streaming ---------------------------------------------------------


def test_text_only_turn_streams_one_text_block():
    a, events = _stream(["Hel", "lo", "!"])
    assert _types(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0]["message"] == {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "sous-local",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    assert events[1] == {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    assert [e["delta"]["text"] for e in events[2:5]] == ["Hel", "lo", "!"]
    assert events[5] == {"type": "content_block_stop", "index": 0}
    assert events[6] == {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 3}}
    assert a.message()["content"] == [{"type": "text", "text": "Hello!"}]
    assert a.message()["stop_reason"] == "end_turn"


def test_tool_call_after_prose_streams_text_then_one_buffered_tool_use_block():
    """Decision 4: prose streams; the call is parsed whole and emitted as
    start(input: {}) → one input_json_delta → stop, indices contiguous."""
    pieces = ["I will read", " a.py.\n\n", "<tool_", "call>\n<function=Read>\n<parameter=file_path>\na.py\n</parameter>\n</function>\n</tool_call>"]
    a, events = _stream(pieces)
    assert _types(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert [e["delta"]["text"] for e in events[2:4]] == ["I will read", " a.py."]
    start = events[5]
    assert start["index"] == 1
    assert start["content_block"]["type"] == "tool_use"
    assert start["content_block"]["name"] == "Read"
    assert start["content_block"]["input"] == {}
    assert re.fullmatch(r"toolu_[0-9a-f]{24}", start["content_block"]["id"])
    assert events[6] == {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": json.dumps({"file_path": "a.py"})}}
    assert events[8]["delta"]["stop_reason"] == "tool_use"
    assert a.message()["content"] == [
        {"type": "text", "text": "I will read a.py."},
        {"type": "tool_use", "id": start["content_block"]["id"], "name": "Read", "input": {"file_path": "a.py"}},
    ]


def test_tool_call_alone_produces_no_phantom_text_block():
    """Checklist item 6: leading newlines before the call never open a text
    block; the tool_use block takes index 0."""
    a, events = _stream(["\n", XML_CALL])
    assert _types(events) == ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert events[1]["index"] == 0 and events[1]["content_block"]["type"] == "tool_use"
    assert a.message()["content"][0]["type"] == "tool_use"


def test_two_tool_calls_get_consecutive_indices():
    a, events = _stream([XML_CALL + "\n" + XML_CALL])
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [s["index"] for s in starts] == [0, 1]
    assert len({s["content_block"]["id"] for s in starts}) == 2


def test_trailing_prose_after_a_tool_call_is_dropped():
    a, _ = _stream([XML_CALL + "\nDone!"])
    assert [b["type"] for b in a.message()["content"]] == ["tool_use"]


def test_unparseable_tool_call_comes_back_as_text_with_end_turn():
    """The model's malformed turn is shown, not hidden."""
    raw = "Sure.\n\n<tool_call>\n<function=Read>\n<parameter=file_path>\nnever closed"
    a, events = _stream(["Sure.\n\n", "<tool_call>\n<function=Read>\n<parameter=file_path>\nnever closed"])
    assert a.message()["content"] == [{"type": "text", "text": raw}]
    assert a.stop_reason == "end_turn"
    assert [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"] == ["Sure.", "\n\n<tool_call>\n<function=Read>\n<parameter=file_path>\nnever closed"]


def test_unknown_tool_name_passes_through_as_tool_use():
    text = (
        "<tool_call>\n<function=Imaginary>\n<parameter=x>\n1\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    a, _ = _stream([text])
    assert a.message()["content"] == [{"type": "tool_use", "id": a.message()["content"][0]["id"], "name": "Imaginary", "input": {"x": "1"}}]


def test_empty_reply_yields_one_empty_text_block():
    a, events = _stream(["", ""])
    assert _types(events) == ["message_start", "content_block_start", "content_block_stop", "message_delta", "message_stop"]
    assert a.message()["content"] == [{"type": "text", "text": ""}]


def test_length_finish_maps_to_max_tokens_unless_a_call_parsed():
    a, _ = _stream(["cut off mid"], finish="length")
    assert a.stop_reason == "max_tokens"
    b, _ = _stream([XML_CALL], finish="length")
    assert b.stop_reason == "tool_use"
    assert stop_reason(None, False) == "end_turn"


def test_final_empty_delta_still_updates_output_tokens():
    """Both mlx libraries flush the detokenizer with a possibly empty final
    piece that carries the real total and the finish reason."""
    a = TurnAssembler("msg_x", "sous-local", TOOLS)
    a.start(1)
    a.feed(Delta("hi", 1, None))
    a.feed(Delta("", 2, "stop"))
    a.finish("hi", 2, "stop")
    assert a.output_tokens == 2
    assert a.message()["usage"] == {"input_tokens": 1, "output_tokens": 2}


# --- non-streaming path is the same code path ----------------------------------------


@pytest.mark.parametrize("split", [1, 3, 7, 12])
def test_non_streaming_message_equals_the_streamed_one(split):
    text = "Let me look.\n\n" + XML_CALL
    a = TurnAssembler("msg_x", "sous-local", TOOLS)
    a.start(5)
    pieces = [text[i : i + split] for i in range(0, len(text), split)]
    for n, piece in enumerate(pieces, start=1):
        a.feed(Delta(piece, n, None))
    a.finish(text, len(pieces), "stop")
    b = TurnAssembler("msg_x", "sous-local", TOOLS)
    b.start(5)
    b.finish(text, len(pieces), "stop")
    def strip_ids(m: dict) -> dict:
        return {**m, "content": [{k: v for k, v in b.items() if k != "id"} for b in m["content"]]}

    assert strip_ids(a.message()) == strip_ids(b.message())
    assert b.message()["content"][0] == {"type": "text", "text": "Let me look."}


def test_message_shape():
    a, _ = _stream(["ok"])
    m = a.message()
    assert set(m) == {"id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "usage"}
    assert m["type"] == "message" and m["role"] == "assistant" and m["stop_sequence"] is None


def test_ids_have_anthropic_shapes():
    assert re.fullmatch(r"msg_[0-9a-f]{24}", new_message_id())
    assert re.fullmatch(r"toolu_[0-9a-f]{24}", new_tool_use_id())
    assert new_message_id() != new_message_id()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_response.py -q`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'sous.gateway.response'`.

- [ ] **Step 3: Create `response.py`**

```python
"""Turn output → Anthropic content blocks and stream events.

Pure. `TurnAssembler` is fed the engine's deltas as they arrive and emits
Anthropic stream events (as dicts; SSE framing is the route's job); the same
object then yields the non-streaming Message, so both response shapes come
from one code path — Claude Code retries a failed stream non-streaming with a
byte-identical body, and the two must agree.
"""

from __future__ import annotations

import json
import secrets

from sous.engine.base import Delta
from sous.protocol import ParseError, ToolCall, ToolSet, parse_tool_calls

_OPEN = "<tool_call>"


def new_message_id() -> str:
    return "msg_" + secrets.token_hex(12)


def new_tool_use_id() -> str:
    return "toolu_" + secrets.token_hex(12)


def stop_reason(finish_reason: str | None, has_calls: bool) -> str:
    if has_calls:
        return "tool_use"
    return "max_tokens" if finish_reason == "length" else "end_turn"


def _tag_prefix_len(text: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of
    `<tool_call>` — text that must wait for the next delta to be classified."""
    for n in range(min(len(_OPEN) - 1, len(text)), 0, -1):
        if text.endswith(_OPEN[:n]):
            return n
    return 0


class TextSplitter:
    """Decides, delta by delta, which text is safe to send as `text_delta`.

    Three things wait until more text settles them: a suffix that could be
    the start of `<tool_call>` (the tag arrives split across deltas), trailing
    whitespace (the turn's text is `.strip()`ped, and the model puts newlines
    before its tool call), and everything from the first `<tool_call>` on
    (tool calls are parsed whole, at the end). Leading whitespace is dropped
    so a text block never opens with a blank — the phantom block of checklist
    item 6.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._started = False
        self._tail = ""  # from the first <tool_call> on; feed() never emits it

    def feed(self, text: str) -> str:
        if self._tail:
            self._tail += text
            return ""
        self._pending += text
        if not self._started:
            self._pending = self._pending.lstrip()
        idx = self._pending.find(_OPEN)
        if idx != -1:
            # The whitespace between the prose and the tag goes with the tail:
            # if the call turns out unparseable, the raw text is returned
            # verbatim, blank lines included.
            out = self._pending[:idx].rstrip()
            self._tail = self._pending[len(out) :]
            self._pending = ""
        else:
            tag = _tag_prefix_len(self._pending)
            body = self._pending[: len(self._pending) - tag]
            hold = tag + (len(body) - len(body.rstrip()))
            cut = len(self._pending) - hold
            out, self._pending = self._pending[:cut], self._pending[cut:]
        self._started = self._started or bool(out)
        return out

    def finish(self) -> str:
        """Everything not yet emitted: held text plus the raw tool-call tail."""
        out = self._pending + self._tail
        self._pending = self._tail = ""
        return out if self._started else out.lstrip()


class TurnAssembler:
    """Builds one assistant message from deltas (streaming) or from the final
    text alone (non-streaming), emitting Anthropic stream events either way.

    Block order: a text block if there is non-blank prose, then one tool_use
    block per parsed call, indices contiguous from 0. An empty reply gets one
    empty text block (what the API itself returns). Prose after a tool call is
    dropped — the template forbids it. A tool call the parser cannot read is
    returned as text with end_turn: the model's malformed turn is shown, not
    hidden.
    """

    def __init__(self, message_id: str, model: str, toolset: ToolSet):
        self.message_id = message_id
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason = "end_turn"
        self._toolset = toolset
        self._splitter = TextSplitter()
        self._fed = ""
        self._blocks: list[dict] = []
        self._text_index: int | None = None

    def start(self, input_tokens: int) -> list[dict]:
        self.input_tokens = input_tokens
        return [
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }
        ]

    def feed(self, delta: Delta) -> list[dict]:
        self._fed += delta.text
        self.output_tokens = max(self.output_tokens, delta.output_tokens)
        return self._emit_text(self._splitter.feed(delta.text))

    def finish(self, text: str, output_tokens: int, finish_reason: str | None) -> list[dict]:
        """Close the turn. `text` is the engine's whole reply; whatever the
        deltas did not carry (all of it, on the non-streaming path) is fed
        first so both paths see identical text."""
        events: list[dict] = []
        if len(text) > len(self._fed):
            events += self._emit_text(self._splitter.feed(text[len(self._fed) :]))
            self._fed = text
        self.output_tokens = max(self.output_tokens, output_tokens)
        remainder = self._splitter.finish()
        calls: list[ToolCall] | None
        try:
            calls = parse_tool_calls(text, self._toolset)
        except ParseError:
            calls = None
        if calls:
            events += self._close_text()
            for call in calls:
                index = len(self._blocks)
                block = {
                    "type": "tool_use",
                    "id": new_tool_use_id(),
                    "name": call.name,
                    "input": call.arguments,
                }
                self._blocks.append(block)
                events.append(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {**block, "input": {}},
                    }
                )
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(call.arguments),
                        },
                    }
                )
                events.append({"type": "content_block_stop", "index": index})
        else:
            events += self._emit_text(remainder.rstrip())
            events += self._close_text()
            if not self._blocks:
                self._blocks.append({"type": "text", "text": ""})
                events.append(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
                events.append({"type": "content_block_stop", "index": 0})
        self.stop_reason = stop_reason(finish_reason, bool(calls))
        events.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": self.output_tokens},
            }
        )
        events.append({"type": "message_stop"})
        return events

    def message(self) -> dict:
        return {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [dict(block) for block in self._blocks],
            "stop_reason": self.stop_reason,
            "stop_sequence": None,
            # No cache accounting fields (the SDKs read absent ones as null):
            # a truthful split needs the per-turn reuse count, which arrives
            # with keyed cache slots (Phase 3a). message_start says the same.
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
        }

    def _emit_text(self, text: str) -> list[dict]:
        if not text:
            return []
        events: list[dict] = []
        if self._text_index is None:
            self._text_index = len(self._blocks)
            self._blocks.append({"type": "text", "text": ""})
            events.append(
                {
                    "type": "content_block_start",
                    "index": self._text_index,
                    "content_block": {"type": "text", "text": ""},
                }
            )
        self._blocks[self._text_index]["text"] += text
        events.append(
            {
                "type": "content_block_delta",
                "index": self._text_index,
                "delta": {"type": "text_delta", "text": text},
            }
        )
        return events

    def _close_text(self) -> list[dict]:
        if self._text_index is None:
            return []
        index, self._text_index = self._text_index, None
        return [{"type": "content_block_stop", "index": index}]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gateway_response.py -q`
Expected: all PASS. If `test_splitter_emits_plain_text_but_holds_trailing_whitespace_and_tag_prefixes` disagrees on the exact held/released string, fix the implementation, not the test: the invariant is "emitted text never ends in whitespace or a `<tool_call>` prefix, and everything is eventually released".

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/gateway/response.py tests/test_gateway_response.py
git commit -m "feat(gateway): assemble Anthropic content blocks and stream events

One assembler serves both response shapes so a stream and its non-streaming
retry agree byte for byte in content. Prose streams as text_delta with a
hold-back for split tags and trailing whitespace; tool calls are parsed
whole at the end and emitted as one input_json_delta each, the shape Claude
Code has years of mileage on through oMLX. A leading newline before a call
never opens a phantom text block, and indices stay contiguous.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 6: `gateway/turn.py` — one serialized turn on the shared engine

**Files:**
- Create: `src/sous/gateway/turn.py`
- Test: `tests/test_gateway_turn.py`

**Interfaces:**
- Consumes: `EngineManager`, `ManagedEngine`, `GenerationSession`, `GenerationStalled`, `Delta`, `release_mlx_thread_state` (Task 3); `SousConfig.gateway_max_context_tokens`, `gateway_generation_timeout_minutes` (Task 1); `tests.fake_engine.ChunkedFakeEngine` (Task 3).
- Produces: `Sink` protocol (`started(input_tokens: int) -> None`, `delta(delta: Delta) -> None`; both called off the event loop, must not block or raise); `TurnResult` (frozen: `text`, `input_tokens`, `output_tokens`, `finish_reason: str | None`, `cache_hit: bool`, `reused_tokens: int`, `seconds: float`); `PromptTooLong(tokens, window)` (message `prompt is too long: N tokens > M maximum`, attributes `.tokens`, `.window`); `GatewayBusy`; `TurnAbandoned`; `TurnRunner(engines, config)` with `run(messages, tools, max_tokens, sink, abandoned: threading.Event | None = None) -> TurnResult` (raises `TurnAbandoned` without generating when `abandoned` is already set once the lock is acquired), `count_tokens(messages, tools) -> int`, `close()`. All synchronous; the route calls them on pool threads. A stall (`GenerationStalled`) drops the session **and** resets the engine's prompt cache.
- Also modify: the last sentence of `GenerationSession`'s class docstring in `src/sous/engine/base.py` — "Every reset belongs to the worker thread." becomes "Every reset belongs to the thread that owns the session: the worker thread for tasks, the gateway's turn thread after a stall (`sous.gateway.turn`)."

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gateway_turn.py`:

```python
"""One gateway turn on the shared engine: serialized, drained, session reuse."""

import threading
import time
from pathlib import Path

import pytest

from sous.config import SousConfig
from sous.engine.base import Delta, EngineManager, GenerationStalled
from sous.gateway.turn import (
    GatewayBusy,
    PromptTooLong,
    TurnAbandoned,
    TurnResult,
    TurnRunner,
)
from tests.fake_engine import ChunkedFakeEngine, FakeEngine


class RecordingSink:
    def __init__(self):
        self.started_with: list[int] = []
        self.deltas: list[Delta] = []
        self.threads: set[threading.Thread] = set()

    def started(self, input_tokens: int) -> None:
        self.started_with.append(input_tokens)
        self.threads.add(threading.current_thread())

    def delta(self, delta: Delta) -> None:
        self.deltas.append(delta)
        self.threads.add(threading.current_thread())


def _cfg(tmp_path: Path, **overrides) -> SousConfig:
    overrides.setdefault("gateway_enabled", True)
    return SousConfig(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
        **overrides,
    )


def _runner(tmp_path: Path, inner, **overrides) -> tuple[TurnRunner, EngineManager]:
    engines = EngineManager(_cfg(tmp_path, **overrides), engine_factory=lambda mid: inner)
    return TurnRunner(engines, _cfg(tmp_path, **overrides)), engines


MSGS = [{"role": "user", "content": "hello"}]


def test_run_streams_deltas_and_reports_real_counts(tmp_path: Path):
    inner = ChunkedFakeEngine(["Hel|lo"])
    runner, _ = _runner(tmp_path, inner)
    sink = RecordingSink()
    result = runner.run(MSGS, [], 4096, sink)
    assert isinstance(result, TurnResult)
    assert result.text == "Hello"
    assert result.input_tokens == inner.count_tokens(MSGS, [])
    assert result.output_tokens == 2 and result.finish_reason == "stop"
    assert sink.started_with == [result.input_tokens]
    assert [d.text for d in sink.deltas] == ["Hel", "lo"]
    assert result.seconds >= 0 and result.cache_hit is False and result.reused_tokens == 0


def test_max_tokens_is_clamped_to_the_room_left_in_the_window(tmp_path: Path):
    inner = FakeEngine(["ok"])
    runner, _ = _runner(tmp_path, inner)
    # 100_000 > the 65536 window: with a request that already fits, the test
    # would be vacuous (Claude Code's 32000 fits comfortably).
    runner.run(MSGS, [], 100_000, RecordingSink())
    room = runner._window - inner.count_tokens(MSGS, [])
    assert inner.max_tokens_seen == [room]
    inner.script.append("ok")
    runner.run(MSGS, [], 10, RecordingSink())
    assert inner.max_tokens_seen[-1] == 10


def test_prompt_that_fills_the_window_is_rejected_before_generating(tmp_path: Path):
    inner = FakeEngine(["never"])
    runner, _ = _runner(tmp_path, inner)  # window 65536; FakeEngine counts len/4
    sink = RecordingSink()
    with pytest.raises(PromptTooLong) as exc:
        runner.run([{"role": "user", "content": "x" * 300_000}], [], 100, sink)
    assert exc.value.window == 65536 and exc.value.tokens >= 65536
    assert "prompt is too long" in str(exc.value)
    assert sink.started_with == [] and inner.calls == []


def test_all_turns_share_one_session_thread_so_the_prompt_cache_can_survive(tmp_path: Path):
    inner = FakeEngine(["a", "b"])
    runner, _ = _runner(tmp_path, inner)
    runner.run(MSGS, [], 100, RecordingSink())
    runner.run(MSGS, [], 100, RecordingSink())
    assert inner.generate_threads[0] is inner.generate_threads[1]
    assert inner.resets == 0  # unlike run_task, a turn never resets the cache
    runner.close()


def test_a_reloaded_engine_gets_a_fresh_session(tmp_path: Path):
    """After an idle unload the next get() builds a new ManagedEngine; the old
    session's thread would call into weights that are gone."""
    made: list[FakeEngine] = []

    def factory(mid):
        e = FakeEngine(["a", "b"])
        made.append(e)
        return e

    cfg = _cfg(tmp_path, idle_unload_minutes=0)
    engines = EngineManager(cfg, engine_factory=factory)
    runner = TurnRunner(engines, cfg)
    runner.run(MSGS, [], 100, RecordingSink())
    first_session = runner._session
    assert first_session is not None
    time.sleep(0.01)
    assert engines.unload_if_idle() is True
    runner.run(MSGS, [], 100, RecordingSink())
    assert len(made) == 2 and made[1].calls  # second turn ran on the new engine
    assert runner._session is not first_session
    first_session._thread.join(5)
    assert not first_session._thread.is_alive()


def test_turns_are_serialized(tmp_path: Path):
    inner = ChunkedFakeEngine(["one|two", "three"], delay=0.2)
    runner, _ = _runner(tmp_path, inner)
    order: list[str] = []

    def go(label):
        runner.run(MSGS, [], 100, RecordingSink())
        order.append(label)

    a = threading.Thread(target=go, args=("a",))
    b = threading.Thread(target=go, args=("b",))
    a.start()
    time.sleep(0.05)
    b.start()
    a.join(5)
    b.join(5)
    assert order == ["a", "b"]
    assert len(inner.generate_threads) == 2


def test_busy_gateway_gives_up_after_the_timeout(tmp_path: Path):
    entered = threading.Event()

    class Announcing(ChunkedFakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            entered.set()  # t is past session.generate(timeout=...) by now
            return super().generate(messages, tools, max_tokens, on_delta)

    inner = Announcing(["slow|slow|slow"], delay=0.3)
    runner, _ = _runner(tmp_path, inner)
    t = threading.Thread(target=runner.run, args=(MSGS, [], 100, RecordingSink()))
    t.start()
    assert entered.wait(5)  # t holds the gateway lock and is generating
    # Config is minutes-granular; the waiter needs a sub-second bound. Set it
    # only now, after t captured the long timeout for its own generation.
    runner._timeout = 0.2
    with pytest.raises(GatewayBusy):
        runner.run(MSGS, [], 100, RecordingSink())
    t.join(5)
    assert inner.finished.wait(5)


def test_a_stall_drops_the_session_and_the_next_turn_gets_a_new_one(tmp_path: Path):
    gate = threading.Event()

    class Gated(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            gate.wait(10)
            return super().generate(messages, tools, max_tokens, on_delta)

    inner = Gated(["late", "fresh"])
    runner, _ = _runner(tmp_path, inner)
    runner._timeout = 0.1
    with pytest.raises(GenerationStalled):
        runner.run(MSGS, [], 100, RecordingSink())
    assert runner._session is None
    assert inner.resets == 1  # the abandoned thread's cache must never be adopted
    gate.set()  # the stalled thread finishes and releases the engine lock
    time.sleep(0.2)
    runner._timeout = 5
    assert runner.run(MSGS, [], 100, RecordingSink()).text == "fresh"
    assert inner.generate_threads[0] is not inner.generate_threads[1]


def test_a_turn_abandoned_while_queued_never_generates(tmp_path: Path):
    """Drain-to-completion covers a generation that started. One whose client
    left while it was still waiting for the lock must not start."""
    inner = FakeEngine(["never"])
    runner, _ = _runner(tmp_path, inner)
    gone = threading.Event()
    gone.set()
    with pytest.raises(TurnAbandoned):
        runner.run(MSGS, [], 100, RecordingSink(), abandoned=gone)
    assert inner.calls == []
    assert not runner._lock.locked()


def test_run_releases_mlx_thread_state_and_touches_the_engine(tmp_path: Path, monkeypatch):
    import sous.gateway.turn as turn

    released: list[bool] = []
    monkeypatch.setattr(turn, "release_mlx_thread_state", lambda: released.append(True))
    inner = FakeEngine(["ok"])
    runner, engines = _runner(tmp_path, inner)
    runner.run(MSGS, [], 100, RecordingSink())
    assert released == [True]
    idle = engines.status()["idle_seconds"]
    assert idle is not None and idle < 1.0


def test_count_tokens_uses_the_engine_and_releases(tmp_path: Path, monkeypatch):
    import sous.gateway.turn as turn

    released: list[bool] = []
    monkeypatch.setattr(turn, "release_mlx_thread_state", lambda: released.append(True))
    inner = FakeEngine([])
    runner, _ = _runner(tmp_path, inner)
    assert runner.count_tokens(MSGS, []) == inner.count_tokens(MSGS, [])
    assert released == [True]


def test_cache_hit_is_reported_from_the_engines_counters(tmp_path: Path):
    inner = FakeEngine(["a", "b"])
    inner.stats = {"hits": 0, "reused_tokens": 0}
    runner, _ = _runner(tmp_path, inner)

    # Simulate the engine's stats moving during the second turn.
    original = inner.generate

    def generate(messages, tools, max_tokens, on_delta=None):
        out = original(messages, tools, max_tokens, on_delta)
        if len(inner.calls) == 2:
            inner.stats = {"hits": 1, "reused_tokens": 900}
        return out

    inner.generate = generate  # ty: ignore[invalid-assignment]
    first = runner.run(MSGS, [], 100, RecordingSink())
    second = runner.run(MSGS, [], 100, RecordingSink())
    assert (first.cache_hit, first.reused_tokens) == (False, 0)
    assert (second.cache_hit, second.reused_tokens) == (True, 900)
```

(Keep the `# ty: ignore[invalid-assignment]` on the bound-method assignment: ty flags assigning over a method, and this is the one place a test needs to.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_turn.py -q`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'sous.gateway.turn'`.

- [ ] **Step 3: Create `turn.py`**

```python
"""One gateway turn on the shared engine: serialized, thread-bridged, drained.

The daemon has one engine and one generation lock; the worker and the gateway
share both. A turn takes the gateway's own lock first (so gateway turns queue
in order and never find the one-slot GenerationSession busy), then the
engine's lock through the session, exactly as run_task does. Everything here
is synchronous and runs on whatever pool thread the route hands it; progress
crosses back to the event loop through the Sink.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol

from sous.config import SousConfig
from sous.engine.base import (
    Delta,
    EngineManager,
    GenerationSession,
    GenerationStalled,
    ManagedEngine,
    release_mlx_thread_state,
)


class Sink(Protocol):
    """Where a turn reports progress. Neither method runs on the event loop —
    `started` on the turn's thread, `delta` on the engine's session thread
    mid-decode — and neither may block or raise."""

    def started(self, input_tokens: int) -> None: ...
    def delta(self, delta: Delta) -> None: ...


class PromptTooLong(Exception):
    def __init__(self, tokens: int, window: int):
        super().__init__(f"prompt is too long: {tokens} tokens > {window} maximum")
        self.tokens = tokens
        self.window = window


class GatewayBusy(Exception):
    """The gateway lock was not acquired within the turn timeout."""


class TurnAbandoned(Exception):
    """The client left while the turn was still queued for the gateway lock."""


@dataclass(frozen=True)
class TurnResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    cache_hit: bool
    reused_tokens: int
    seconds: float


class TurnRunner:
    def __init__(self, engines: EngineManager, config: SousConfig):
        self._engines = engines
        self._window = config.gateway_max_context_tokens
        self._timeout = float(config.gateway_generation_timeout_minutes * 60)
        self._lock = threading.Lock()
        # One long-lived session for every gateway turn: the prompt cache
        # lives on the session thread's mlx streams (#34), so a per-request
        # session would throw the cache away between a subagent's turns — and
        # the cache is what turns gate 2's ~200s cold prefill into seconds.
        self._session: GenerationSession | None = None
        self._session_engine: ManagedEngine | None = None

    def run(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        sink: Sink,
        abandoned: threading.Event | None = None,
    ) -> TurnResult:
        if not self._lock.acquire(timeout=self._timeout):
            raise GatewayBusy(f"no generation slot within {self._timeout:.0f}s")
        started = time.monotonic()
        try:
            if abandoned is not None and abandoned.is_set():
                # Queued behind another turn, and the client gave up meanwhile:
                # a generation nobody reads would only delay the live requests
                # behind it. (A turn that has started still drains — the GPU
                # cannot be interrupted and the lock discipline depends on it.)
                raise TurnAbandoned
            engine = self._engines.get()
            session = self._session_for(engine)
            input_tokens = engine.count_tokens(messages, tools)
            room = self._window - input_tokens
            if room <= 0:
                raise PromptTooLong(input_tokens, self._window)
            # Hit/miss is for the log only: the counters are global, and a
            # worker task resetting the cache mid-turn zeroes them, so a hit
            # can read as a miss. Exact per-turn reuse comes with keyed slots.
            before = engine.prompt_cache_stats()
            sink.started(input_tokens)
            final: Delta | None = None

            def on_delta(delta: Delta) -> None:
                nonlocal final
                final = delta
                sink.delta(delta)

            try:
                text = session.generate(
                    messages, tools, min(max_tokens, room), timeout=self._timeout, on_delta=on_delta
                )
            except GenerationStalled:
                # The session is unusable after a stall (its thread may still
                # be generating, holding the engine lock); the next turn gets a
                # fresh one and waits on the lock like the worker would. Reset
                # the cache too: when the abandoned thread finishes it publishes
                # the KV cache it built on ITS streams, and a cache is usable
                # only from the thread that built it (#34). The reset's epoch
                # bump makes that late publish drop itself — the same guard
                # run_task's finally relies on.
                self._drop_session()
                engine.reset_prompt_cache()
                raise
            after = engine.prompt_cache_stats()
            return TurnResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=final.output_tokens if final else 0,
                finish_reason=final.finish_reason if final else "stop",
                cache_hit=after.get("hits", 0) > before.get("hits", 0),
                reused_tokens=max(0, after.get("reused_tokens", 0) - before.get("reused_tokens", 0)),
                seconds=time.monotonic() - started,
            )
        finally:
            self._engines.touch()
            self._lock.release()
            # engines.get() may have loaded the model on this thread. Pool
            # threads outlive the call, but the invariant is per thread that
            # touched mlx (ml-explore/mlx#4327), and keeping it unconditional
            # is what makes it checkable.
            release_mlx_thread_state()

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        try:
            count = self._engines.get().count_tokens(messages, tools)
            self._engines.touch()
            return count
        finally:
            release_mlx_thread_state()

    def _session_for(self, engine: ManagedEngine) -> GenerationSession:
        if self._session is None or self._session_engine is not engine:
            # A different ManagedEngine means the model was idle-unloaded and
            # reloaded; the old session's thread would call into an engine
            # whose weights are gone.
            self._drop_session()
            self._session = engine.session()
            self._session_engine = engine
        return self._session

    def _drop_session(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._session_engine = None

    def close(self) -> None:
        with self._lock:
            self._drop_session()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gateway_turn.py -q`
Expected: all PASS. `test_turns_are_serialized` and `test_busy_gateway_gives_up_after_the_timeout` involve real sleeps (~1 s total). If either is flaky, run it in a loop (`for i in $(seq 20); do uv run pytest tests/test_gateway_turn.py -q -k "serialized or busy" || break; done`) and fix the timing margins in the test, never the semantics.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add src/sous/gateway/turn.py tests/test_gateway_turn.py
git commit -m "feat(gateway): run one serialized turn on the shared engine

Gateway turns queue on their own lock and then take the engine lock through
one long-lived GenerationSession, so the strict-prefix cache survives
between a subagent's turns (the difference between ~200s and seconds per
turn, per gate 2). The session is replaced only after a stall or an idle
unload/reload. Token counts come from the engine's own tokenization, never
from library prompt_tokens, and a prompt that fills the window is refused
before anything is generated.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 7: `gateway/routes.py`, mounting on the daemon, explicit dependencies

**Files:**
- Create: `src/sous/gateway/routes.py`
- Modify: `src/sous/server.py` (`create_server`, `main`)
- Modify: `pyproject.toml`, `uv.lock`
- Test: `tests/test_gateway_routes.py`

**Interfaces:**
- Consumes: `parse_messages_request`, `parse_count_tokens_request`, `RequestError`, `ChatRequest` (Task 4); `TurnAssembler`, `new_message_id` (Task 5); `TurnRunner`, `TurnResult`, `PromptTooLong`, `GatewayBusy` (Task 6); `ToolSet` (Task 2); `Delta`, `GenerationStalled` (Task 3); `SousConfig.gateway_*` (Task 1).
- Produces: `mount_gateway(mcp: MCPServer, engines: EngineManager, config: SousConfig) -> Gateway`, registering `POST /v1/messages`, `POST /v1/messages/count_tokens`, `GET|HEAD /api/hello`; module constants `MAX_REQUEST_BYTES = 32 * 1024 * 1024`, `PING_INTERVAL_SECONDS = 10` (tests monkeypatch both). `create_server` mounts the gateway when `config.gateway_enabled`.

- [ ] **Step 1: Declare the dependencies the gateway imports directly**

In `pyproject.toml`, `dependencies` gains (keep alphabetical order within the list):

```toml
    "psutil>=7",
    # The gateway (src/sous/gateway) and server.serve import these directly.
    # mcp already pulls all three in, but a direct import must be a direct
    # dependency: mcp is free to drop or swap them in a minor release and
    # lockfile drift would hide it.
    "sse-starlette>=3.4",
    "starlette>=1.6",
    "tomlkit>=0.13",
    "uvicorn>=0.31",
```

and the `dev` group gains `"httpx>=0.28",` (the gateway tests drive the ASGI app with `httpx.ASGITransport`; today httpx is present only as huggingface-hub's transitive dependency).

Run: `uv lock && uv sync && uv lock --check`
Expected: lockfile updated (no version changes — both are already installed at these versions), check passes.

- [ ] **Step 2: Write the failing route tests**

Create `tests/test_gateway_routes.py`:

```python
"""The gateway's HTTP surface, in-process through the ASGI app create_server
builds. httpx's ASGITransport buffers a response, so streaming tests read the
whole SSE body after the fact; timing-sensitive behaviour lives in
test_gateway_http.py against a real server."""

import asyncio
import json
from pathlib import Path

import httpx

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.server import create_server
from sous.tasks import TaskStore
from tests.fake_engine import ChunkedFakeEngine, FakeEngine

READ_TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
}
XML_CALL = (
    "<tool_call>\n<function=Read>\n<parameter=file_path>\na.py\n</parameter>\n"
    "</function>\n</tool_call>"
)


def _app(tmp_path: Path, engine, **overrides):
    overrides.setdefault("gateway_enabled", True)
    cfg = SousConfig(
        data_dir=tmp_path / "data",
        config_path=tmp_path / "config.toml",
        **overrides,
    )
    store = TaskStore(tmp_path / "tasks.db")
    engines = EngineManager(cfg, engine_factory=lambda mid: engine)
    return create_server(store, engines, cfg).streamable_http_app()


def _request(app, method: str, path: str, body=None, headers=None) -> httpx.Response:
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8383") as client:
            content = body if isinstance(body, bytes | str | type(None)) else json.dumps(body)
            return await client.request(
                method,
                path,
                content=content,
                headers={"content-type": "application/json", **(headers or {})},
            )

    return asyncio.run(go())


def _post(app, body, path="/v1/messages", headers=None) -> httpx.Response:
    return _request(app, "POST", path, body, headers)


def _body(**overrides) -> dict:
    body = {"model": "sous-local", "max_tokens": 4096, "messages": [{"role": "user", "content": "hi"}]}
    body.update(overrides)
    return body


def _events(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for frame in text.split("\n\n"):
        event: str | None = None
        data: dict = {}
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event is not None:
            out.append((event, data))
    return out


# --- probes and routing -------------------------------------------------------------


def test_hello_answers_head_and_get(tmp_path: Path):
    app = _app(tmp_path, FakeEngine([]))
    assert _request(app, "HEAD", "/api/hello").status_code == 200
    assert _request(app, "GET", "/api/hello").status_code == 200


def test_routes_are_absent_when_the_gateway_is_disabled(tmp_path: Path):
    app = _app(tmp_path, FakeEngine([]), gateway_enabled=False)
    assert _request(app, "HEAD", "/api/hello").status_code == 404
    assert _post(app, _body()).status_code == 404


def test_unknown_model_is_an_anthropic_shaped_404(tmp_path: Path):
    """Phase 2 forwards these upstream; until then the client hears exactly
    what the real API says for an unknown model, and gate 1 showed the main
    loop recovers from that."""
    app = _app(tmp_path, FakeEngine([]))
    r = _post(app, _body(model="claude-opus-5"))
    assert r.status_code == 404
    assert r.json() == {"type": "error", "error": {"type": "not_found_error", "message": "model: claude-opus-5"}}


def test_configured_local_models_are_all_served(tmp_path: Path):
    app = _app(tmp_path, FakeEngine(["ok"]), gateway_local_models=("sous-local", "sous-fast"))
    assert _post(app, _body(model="sous-fast")).status_code == 200


def test_host_header_must_be_loopback(tmp_path: Path):
    """Custom routes skip the /mcp transport's Host check; a page whose hostname
    re-resolves to 127.0.0.1 must not get to drive the local model."""
    import sous.gateway.routes as routes

    app = _app(tmp_path, FakeEngine([]))
    for host in ("evil.example:8383", "evil.example"):
        r = _post(app, _body(), headers={"host": host})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"
        assert _request(app, "HEAD", "/api/hello", headers={"host": host}).status_code == 403
    for host in ("127.0.0.1:8383", "localhost", "[::1]:8383", "[::1]"):
        assert _request(app, "HEAD", "/api/hello", headers={"host": host}).status_code == 200
    assert set(routes._ALLOWED_HOSTS) == {"127.0.0.1", "localhost", "[::1]"}


# --- request validation ---------------------------------------------------------------


def test_invalid_json_and_bad_shapes_are_400s(tmp_path: Path):
    app = _app(tmp_path, FakeEngine([]))
    r = _post(app, b"{not json")
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    r = _post(app, _body(max_tokens=0))
    assert r.status_code == 400 and "max_tokens" in r.json()["error"]["message"]
    r = _post(app, [1, 2, 3])
    assert r.status_code == 400


def test_oversized_bodies_are_413_by_header_and_by_actual_size(tmp_path: Path, monkeypatch):
    import sous.gateway.routes as routes

    app = _app(tmp_path, FakeEngine([]))
    r = _post(app, _body(), headers={"content-length": str(routes.MAX_REQUEST_BYTES + 1)})
    assert r.status_code == 413 and r.json()["error"]["type"] == "request_too_large"
    monkeypatch.setattr(routes, "MAX_REQUEST_BYTES", 64)
    r = _post(app, _body(messages=[{"role": "user", "content": "x" * 200}]))
    assert r.status_code == 413


# --- streaming --------------------------------------------------------------------------


def test_streamed_text_turn_has_the_anthropic_event_sequence(tmp_path: Path):
    inner = FakeEngine(["Hello there"])
    app = _app(tmp_path, inner)
    r = _post(app, _body(stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    assert [e for e, _ in events] == [
        "ping",
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[1][1]["message"]
    assert start["model"] == "sous-local" and start["usage"]["input_tokens"] == inner.count_tokens(
        [{"role": "user", "content": "hi"}], []
    )
    assert events[3][1]["delta"] == {"type": "text_delta", "text": "Hello there"}
    assert events[5][1] == {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 2}}


def test_streamed_tool_call_turn(tmp_path: Path):
    inner = ChunkedFakeEngine(
        [
            "I will read a.py.\n\n|<tool_call>\n<function=Read>\n|<parameter=file_path>\na.py\n"
            "</parameter>\n</function>\n</tool_call>"
        ]
    )
    app = _app(tmp_path, inner)
    r = _post(app, _body(stream=True, tools=[READ_TOOL]))
    events = _events(r.text)
    types = [e for e, _ in events]
    assert types == [
        "ping",
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    tool_start = events[5][1]
    assert tool_start["index"] == 1
    assert tool_start["content_block"]["type"] == "tool_use"
    assert tool_start["content_block"]["name"] == "Read"
    assert tool_start["content_block"]["id"].startswith("toolu_")
    assert json.loads(events[6][1]["delta"]["partial_json"]) == {"file_path": "a.py"}
    assert events[8][1]["delta"]["stop_reason"] == "tool_use"
    # The template got the converted tool, not the Anthropic shape.
    assert inner.tools_seen[0][0] == {
        "type": "function",
        "function": {"name": "Read", "description": "Read a file", "parameters": READ_TOOL["input_schema"]},
    }


def test_non_streaming_turn_matches_the_streamed_content(tmp_path: Path):
    script = "Reading.\n\n" + XML_CALL
    streamed = _post(_app(tmp_path / "a", ChunkedFakeEngine([script])), _body(stream=True, tools=[READ_TOOL]))
    plain = _post(_app(tmp_path / "b", FakeEngine([script])), _body(tools=[READ_TOOL]))
    assert plain.status_code == 200
    message = plain.json()
    assert set(message) == {"id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "usage"}
    assert message["id"].startswith("msg_") and message["stop_reason"] == "tool_use"
    assert message["content"][0] == {"type": "text", "text": "Reading."}
    assert message["content"][1]["name"] == "Read" and message["content"][1]["input"] == {"file_path": "a.py"}
    assert message["usage"]["output_tokens"] > 0  # FakeEngine counts whitespace-separated words
    streamed_types = [b["content_block"]["type"] for e, b in _events(streamed.text) if e == "content_block_start"]
    assert streamed_types == ["text", "tool_use"]


def test_prompt_conversion_reaches_the_engine(tmp_path: Path):
    """Inline system, billing header and the volatile marker: the engine sees
    one stable system message, so the prefix cache can hold across turns."""
    inner = FakeEngine(["ok"])
    app = _app(tmp_path, inner)
    body = _body(
        system=[
            {"type": "text", "text": "x-anthropic-billing-header: cc_is_subagent=true"},
            {"type": "text", "text": "Canonical.\n\n<total_tokens>99 tokens left</total_tokens>"},
        ],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {"role": "system", "content": "Inline."},
        ],
        tools=[READ_TOOL, {"type": "web_search_20250305", "name": "web_search"}],
    )
    assert _post(app, body).status_code == 200
    assert inner.calls[0] == [
        {"role": "system", "content": "Canonical.\n\nInline."},
        {"role": "user", "content": "task"},
    ]
    assert [t["function"]["name"] for t in inner.tools_seen[0]] == ["Read"]


def test_max_tokens_is_clamped_to_the_window(tmp_path: Path):
    inner = FakeEngine(["ok", "ok"])
    app = _app(tmp_path, inner)
    _post(app, _body(max_tokens=100_000))  # above the 65536 window, so the clamp bites
    assert inner.max_tokens_seen == [65536 - inner.count_tokens([{"role": "user", "content": "hi"}], [])]
    _post(app, _body(max_tokens=10))
    assert inner.max_tokens_seen[-1] == 10


def test_prompt_too_long_is_400_plain_and_an_error_event_streamed(tmp_path: Path):
    inner = FakeEngine(["never", "never"])
    app = _app(tmp_path, inner)
    huge = _body(messages=[{"role": "user", "content": "x" * 300_000}])
    r = _post(app, huge)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "prompt is too long" in r.json()["error"]["message"]
    r = _post(app, {**huge, "stream": True})
    assert r.status_code == 200  # headers are already out; the error is in-band
    events = _events(r.text)
    assert events[0][0] == "ping"
    event, payload = events[-1]
    assert event == "error"
    tokens = inner.count_tokens(huge["messages"], [])
    assert payload == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"prompt is too long: {tokens} tokens > 65536 maximum",
        },
    }
    assert inner.calls == []


def test_engine_failure_is_500_plain_and_an_api_error_event_streamed(tmp_path: Path):
    class Exploding(FakeEngine):
        def generate(self, messages, tools, max_tokens, on_delta=None):
            raise ValueError("boom")

    app = _app(tmp_path, Exploding([]))
    r = _post(app, _body())
    assert r.status_code == 500 and r.json()["error"]["type"] == "api_error"
    r = _post(app, _body(stream=True))
    events = _events(r.text)
    assert events[-1][0] == "error" and events[-1][1]["error"]["type"] == "api_error"
    assert "boom" in events[-1][1]["error"]["message"]


def test_malformed_tool_call_comes_back_as_text(tmp_path: Path):
    app = _app(tmp_path, FakeEngine(["<tool_call>\n<function=Read>\n<parameter=file_path>\nunterminated"]))
    message = _post(app, _body(tools=[READ_TOOL])).json()
    assert message["content"] == [{"type": "text", "text": "<tool_call>\n<function=Read>\n<parameter=file_path>\nunterminated"}]
    assert message["stop_reason"] == "end_turn"


def test_count_tokens(tmp_path: Path):
    inner = FakeEngine([])
    app = _app(tmp_path, inner)
    body = {"model": "sous-local", "messages": [{"role": "user", "content": "hello world"}], "tools": [READ_TOOL]}
    r = _post(app, body, path="/v1/messages/count_tokens")
    assert r.status_code == 200
    assert r.json() == {"input_tokens": inner.count_tokens([{"role": "user", "content": "hello world"}], [])}
    assert _post(app, {**body, "model": "claude-x"}, path="/v1/messages/count_tokens").status_code == 404


# --- logging discipline --------------------------------------------------------------------


def test_log_lines_carry_metadata_only(tmp_path: Path, capsys):
    """Spec security posture: bodies and header values never reach a log."""
    secret_text = "SECRET-PROMPT-TEXT-7f3a"
    secret_token = "sk-ant-oat01-SECRET-TOKEN-9c1d"
    app = _app(tmp_path, FakeEngine(["a reply"]))
    r = _post(
        app,
        _body(messages=[{"role": "user", "content": secret_text}], stream=True),
        headers={"authorization": f"Bearer {secret_token}", "anthropic-beta": "oauth-2025-04-20"},
    )
    assert r.status_code == 200
    err = capsys.readouterr().err
    assert "sous gateway: POST /v1/messages" in err
    assert "model=sous-local" in err and "stream=1" in err and "input_tokens=" in err
    assert secret_text not in err
    assert secret_token not in err and "oauth-2025-04-20" not in err
    assert "a reply" not in err
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_routes.py -q`
Expected: FAIL — the enabled-gateway tests get 404s (no routes mounted), `test_oversized_bodies_are_413_by_header_and_by_actual_size` and `test_host_header_must_be_loopback` fail at their `import sous.gateway.routes` with `ModuleNotFoundError`, and `test_log_lines_carry_metadata_only` finds no log line. `test_routes_are_absent_when_the_gateway_is_disabled` passes already: today's app answers 404 on those paths, which is exactly what it pins.

- [ ] **Step 4: Create `routes.py`**

```python
"""The gateway's HTTP surface: Anthropic-shaped routes on the daemon's app.

Never logs a request body or a header value. Never executes a tool: tool_use
blocks go back to Claude Code, whose permission system runs them (toolexec.py
is not in this path).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import threading
from collections.abc import AsyncIterator

from mcp.server import MCPServer
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response

from sous.config import SousConfig
from sous.engine.base import Delta, EngineManager, GenerationStalled
from sous.gateway.convert import (
    ChatRequest,
    RequestError,
    parse_count_tokens_request,
    parse_messages_request,
)
from sous.gateway.response import TurnAssembler, new_message_id
from sous.gateway.turn import (
    GatewayBusy,
    PromptTooLong,
    TurnAbandoned,
    TurnResult,
    TurnRunner,
)
from sous.protocol import ToolSet

# Anthropic's own request cap. The MCP transport's 4 MiB limit wraps only the
# /mcp handler; custom routes get nothing unless they enforce it themselves.
MAX_REQUEST_BYTES = 32 * 1024 * 1024
# Claude Code disconnects when nothing arrives for a while (oMLX saw it on
# 90k-token prefills). Both official SDKs drop `ping` events in the SSE
# iterator before their accumulator sees anything, so pings are safe anywhere
# in the stream — including before message_start, which is where the lock
# wait, the model load and the prefill all happen.
PING_INTERVAL_SECONDS = 10

# A ServerSentEvent carries its own separator (the response-level `sep` only
# applies to dicts and strings), and its default is "\r\n"; every frame here
# says "\n" so the whole stream is one canonical shape.
_SEP = "\n"
_PING = ServerSentEvent(event="ping", data='{"type": "ping"}', sep=_SEP)

# Custom routes get none of the Host validation the /mcp transport applies to
# loopback binds; without it a web page whose hostname re-resolves to
# 127.0.0.1 could drive the local model. Same allow-list as the SDK's.
_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")

# sse-starlette logs every frame it sends at DEBUG — the model's reply,
# verbatim. The daemon runs at INFO, but the no-bodies-in-logs rule must not
# depend on that: pin the library's logger above DEBUG where the frames are made.
logging.getLogger("sse_starlette").setLevel(logging.INFO)


def _log(message: str) -> None:
    print(f"sous gateway: {message}", file=sys.stderr, flush=True)


def _error_response(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": error_type, "message": message}}, status_code=status
    )


def _classify(exc: Exception) -> tuple[int, str, str]:
    """(status, error type, message) for a failure while turning."""
    if isinstance(exc, PromptTooLong):
        return 400, "invalid_request_error", str(exc)
    if isinstance(exc, GatewayBusy):
        return 529, "overloaded_error", str(exc)
    if isinstance(exc, GenerationStalled):
        return 500, "api_error", str(exc)
    return 500, "api_error", f"generation failed: {exc}"


async def _read_json(request: Request) -> object:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
        raise RequestError(413, "request_too_large", f"request body exceeds {MAX_REQUEST_BYTES} bytes")
    chunks: list[bytes] = []
    size = 0
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                raise RequestError(
                    413, "request_too_large", f"request body exceeds {MAX_REQUEST_BYTES} bytes"
                )
            chunks.append(chunk)
    except ClientDisconnect:
        # Nobody will read this response; it exists so the disconnect is not
        # a 500 with a traceback in the daemon log.
        raise RequestError(400, "invalid_request_error", "client disconnected mid-body") from None
    try:
        return json.loads(b"".join(chunks))
    except ValueError:
        raise RequestError(400, "invalid_request_error", "request body is not valid JSON") from None


def _check_host(request: Request) -> None:
    host = request.headers.get("host", "")
    # Strip the port; an IPv6 literal keeps its brackets.
    if host.startswith("[") and "]" in host:
        host = host[: host.index("]") + 1]
    else:
        host = host.split(":", 1)[0]
    if host not in _ALLOWED_HOSTS:
        raise RequestError(403, "permission_error", "the gateway serves loopback hosts only")


def _frame(event: dict) -> ServerSentEvent:
    return ServerSentEvent(
        event=event["type"], data=json.dumps(event, separators=(",", ":")), sep=_SEP
    )


class _QueueSink:
    """Relays a turn's progress from its threads onto the event loop's queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self._loop = loop
        self._queue = queue

    def put(self, item: tuple) -> None:
        # A closed loop means the daemon is shutting down: nobody is listening.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def started(self, input_tokens: int) -> None:
        self.put(("started", input_tokens))

    def delta(self, delta: Delta) -> None:
        self.put(("delta", delta))


class _NullSink:
    def started(self, input_tokens: int) -> None: ...

    def delta(self, delta: Delta) -> None: ...


class Gateway:
    def __init__(self, engines: EngineManager, config: SousConfig):
        self._config = config
        self._runner = TurnRunner(engines, config)

    async def hello(self, request: Request) -> Response:
        # Claude Code probes HEAD /api/hello at startup (gate 1, O3).
        try:
            _check_host(request)
        except RequestError as e:
            return JSONResponse(e.body(), status_code=e.status)
        return Response(status_code=200)

    async def count_tokens(self, request: Request) -> Response:
        try:
            _check_host(request)
            chat = parse_count_tokens_request(await _read_json(request))
            self._check_model(chat.model)
        except RequestError as e:
            return JSONResponse(e.body(), status_code=e.status)
        try:
            count = await asyncio.to_thread(self._runner.count_tokens, chat.messages, chat.tools)
        except Exception as e:  # noqa: BLE001 — every failure becomes an Anthropic error body
            return _error_response(*_classify(e))
        return JSONResponse({"input_tokens": count})

    async def messages(self, request: Request) -> Response:
        try:
            _check_host(request)
            chat = parse_messages_request(await _read_json(request))
            self._check_model(chat.model)
        except RequestError as e:
            _log(f"POST /v1/messages status={e.status} error={e.error_type}")
            return JSONResponse(e.body(), status_code=e.status)
        if chat.dropped_server_tools:
            _log(
                f"dropped {len(chat.dropped_server_tools)} server-side tool(s) not executable "
                f"locally: {', '.join(chat.dropped_server_tools)}"
            )
        assembler = TurnAssembler(
            new_message_id(), chat.model, ToolSet.from_tools(chat.tools, strict=False)
        )
        if chat.stream:
            return EventSourceResponse(
                self._stream(chat, assembler),
                ping=PING_INTERVAL_SECONDS,
                ping_message_factory=lambda: _PING,
                sep=_SEP,
            )
        try:
            result = await asyncio.to_thread(
                self._runner.run, chat.messages, chat.tools, chat.max_tokens, _NullSink()
            )
        except Exception as e:  # noqa: BLE001 — every failure becomes an Anthropic error body
            status, error_type, message = _classify(e)
            _log(f"POST /v1/messages model={chat.model} stream=0 status={status} error={error_type}")
            return _error_response(status, error_type, message)
        assembler.start(result.input_tokens)
        assembler.finish(result.text, result.output_tokens, result.finish_reason)
        self._log_turn(chat, result, assembler, stream=False)
        return JSONResponse(assembler.message())

    def _check_model(self, model: str) -> None:
        # Phase 2 forwards other models upstream; until then they are simply
        # not served here, in the vocabulary the client already understands.
        if model not in self._config.gateway_local_models:
            raise RequestError(404, "not_found_error", f"model: {model}")

    async def _stream(
        self, chat: ChatRequest, assembler: TurnAssembler
    ) -> AsyncIterator[ServerSentEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sink = _QueueSink(loop, queue)
        abandoned = threading.Event()

        def turn() -> None:
            # Completion and failure both travel through the queue, so this
            # generator has exactly one thing to wait on — and a client that
            # disconnects cancels only the waiting, never a started turn: the
            # thread drains the generation to completion regardless.
            try:
                sink.put(
                    ("done", self._runner.run(chat.messages, chat.tools, chat.max_tokens, sink, abandoned))
                )
            except TurnAbandoned:
                _log(f"POST /v1/messages model={chat.model} stream=1 abandoned while queued")
            except Exception as e:  # noqa: BLE001 — relayed as an in-band error event
                sink.put(("error", e))

        loop.run_in_executor(None, turn)
        try:
            yield _PING
            while True:
                kind, value = await queue.get()
                if kind == "started":
                    for event in assembler.start(value):
                        yield _frame(event)
                elif kind == "delta":
                    for event in assembler.feed(value):
                        yield _frame(event)
                elif kind == "done":
                    for event in assembler.finish(
                        value.text, value.output_tokens, value.finish_reason
                    ):
                        yield _frame(event)
                    self._log_turn(chat, value, assembler, stream=True)
                    return
                else:
                    status, error_type, message = _classify(value)
                    _log(
                        f"POST /v1/messages model={chat.model} stream=1 status=200 "
                        f"error={error_type}"
                    )
                    yield _frame({"type": "error", "error": {"type": error_type, "message": message}})
                    return
        finally:
            # Reached on a normal finish, on the client disconnecting (the
            # CancelledError lands at queue.get) and on generator close: a turn
            # still waiting for the lock sees this and never starts.
            abandoned.set()

    def _log_turn(
        self, chat: ChatRequest, result: TurnResult, assembler: TurnAssembler, *, stream: bool
    ) -> None:
        _log(
            f"POST /v1/messages model={chat.model} stream={int(stream)} status=200 "
            f"input_tokens={result.input_tokens} output_tokens={result.output_tokens} "
            f"stop={assembler.stop_reason} cache={'hit' if result.cache_hit else 'miss'} "
            f"reused_tokens={result.reused_tokens} seconds={result.seconds:.1f}"
        )


def mount_gateway(mcp: MCPServer, engines: EngineManager, config: SousConfig) -> Gateway:
    """Register the Anthropic-compatible routes on the daemon's Starlette app.
    custom_route adds bare routes: no auth (loopback only, like /mcp), no
    body limit (enforced above), no DNS-rebinding check (the /mcp transport's
    settings do not reach here)."""
    gateway = Gateway(engines, config)
    mcp.custom_route("/v1/messages", methods=["POST"])(gateway.messages)
    mcp.custom_route("/v1/messages/count_tokens", methods=["POST"])(gateway.count_tokens)
    mcp.custom_route("/api/hello", methods=["GET", "HEAD"])(gateway.hello)
    return gateway
```

- [ ] **Step 5: Mount it from `create_server` and announce it at startup**

In `src/sous/server.py`, import `from sous.gateway.routes import mount_gateway` and, at the end of `create_server` before `return mcp`:

```python
    if config.gateway_enabled:
        mount_gateway(mcp, engines, config)
```

In `main()`, after `mcp = create_server(store, engines, config)`:

```python
    if config.gateway_enabled:
        print(
            f"sous: gateway (experimental) serving {', '.join(config.gateway_local_models)} "
            f"at http://127.0.0.1:{config.server_port}/v1/messages"
        )
```

Then bound uvicorn's graceful shutdown. Add `import anyio` and `import uvicorn` to `server.py`'s imports (both are runtime dependencies now — anyio via mcp, as `proxy.py` already relies on), and add, above `main()`:

```python
# uvicorn owns SIGTERM/SIGINT while it serves: capture_signals swaps sous's
# handler out and re-raises the signal only after its own shutdown returns,
# and by default that shutdown waits for every open connection with no bound.
# A non-streaming gateway turn (Claude Code's retry shape) can hold one for
# the whole generation timeout, which would defer the command-group kill in
# _install_shutdown_handler by the same amount. Bound it: streams already
# cancel themselves on the exit signal (sse-starlette), and a cancelled
# non-streaming handler leaves its turn draining on the executor thread.
GRACEFUL_SHUTDOWN_SECONDS = 5


def uvicorn_config(app, host: str, port: int, log_level: str = "info") -> uvicorn.Config:
    return uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )


def serve(mcp: MCPServer, host: str, port: int) -> None:
    """What `mcp.run(transport="streamable-http")` does, minus the unbounded
    shutdown wait: the SDK builds this same app and uvicorn config but exposes
    no graceful-shutdown timeout."""
    app = mcp.streamable_http_app(host=host)
    anyio.run(uvicorn.Server(uvicorn_config(app, host, port)).serve)
```

and replace `mcp.run(transport="streamable-http", host="127.0.0.1", port=config.server_port)` in `main()` with `serve(mcp, "127.0.0.1", config.server_port)`. In `tests/daemon_stub.py`, replace its `mcp.run(...)` line with `serve(mcp, "127.0.0.1", port)` (import `serve` alongside `_acquire_singleton_lock, create_server`) so the stub keeps mirroring `main()`. The three existing `main()` tests that make the port bind fail still see `SystemExit`: uvicorn's `startup()` calls `sys.exit(1)` on a bind error exactly as it did under `mcp.run`.

Append to `tests/test_server.py`:

```python
def test_uvicorn_config_bounds_graceful_shutdown():
    """uvicorn holds SIGTERM while serving and, unbounded, waits for open
    connections — a long non-streaming gateway turn would defer sous's own
    shutdown handler by minutes. The bound is what keeps `sous stop` prompt."""
    from sous.server import GRACEFUL_SHUTDOWN_SECONDS, uvicorn_config

    cfg = uvicorn_config(object(), "127.0.0.1", 0)
    assert cfg.timeout_graceful_shutdown == GRACEFUL_SHUTDOWN_SECONDS == 5
    assert cfg.host == "127.0.0.1"
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_gateway_routes.py tests/test_server.py -q`
Expected: all PASS, output pristine. If sse-starlette emits a deprecation or "unclosed" warning under `ASGITransport`, fix the cause (e.g. ensure the generator returns after `message_stop`), do not filter it.

Then the whole suite: `uv run pytest -m "not model" -q` — Expected: PASS.

- [ ] **Step 7: Lint, type-check, lockfile, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check && uv lock --check
git add pyproject.toml uv.lock src/sous/gateway/routes.py src/sous/server.py tests/daemon_stub.py tests/test_server.py tests/test_gateway_routes.py
git commit -m "feat(gateway): mount /v1/messages, count_tokens and /api/hello on the daemon

Off unless [gateway].enabled. Streams through sse-starlette: pings flow from
the first byte because the lock wait, model load and prefill all sit between
the request and message_start, and a stream cancels itself on the daemon's
exit signal. uvicorn owns SIGTERM while serving and by default waits for
every open connection, so the daemon now drives uvicorn itself with a
bounded graceful shutdown — a non-streaming turn could otherwise defer the
shutdown handler by a whole generation. Everything that needs the engine
reports in-band once a stream is open; shape errors, unknown models,
non-loopback hosts and oversized bodies get real statuses first. Logs carry
metadata only — never a body or a header value. starlette, sse-starlette
and uvicorn become direct dependencies because sous now imports them.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 8: Real-server behaviour — keepalives, disconnect drain, real model

**Files:**
- Create: `tests/test_gateway_http.py` (`slow`-marked: real uvicorn on a loopback port, real sockets, real timing)
- Create: `tests/test_gateway_model.py` (`model`-marked: the tiny real model through the real route)

**Interfaces:**
- Consumes: everything from Tasks 3–7; `tests.fake_engine.ChunkedFakeEngine`; `sous.gateway.routes.PING_INTERVAL_SECONDS` (monkeypatched to 1 — it is typed `int`, sse-starlette's `ping` is `int`); `sous.server.uvicorn_config` and `GRACEFUL_SHUTDOWN_SECONDS` (Task 7) so the real server under test is configured exactly like the daemon.

- [ ] **Step 1: Write the slow HTTP tests**

Create `tests/test_gateway_http.py`:

```python
"""Gateway behaviour only a real server shows: keepalive pings while the
model is silent, and a client that hangs up mid-stream. httpx's in-process
transport buffers whole responses, so these run uvicorn on a loopback port in
a thread — same stack as the daemon, no subprocess."""

import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.server import GRACEFUL_SHUTDOWN_SECONDS, create_server, uvicorn_config
from sous.tasks import TaskStore
from tests.fake_engine import ChunkedFakeEngine

pytestmark = pytest.mark.slow


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _app(tmp_path: Path, engine):
    cfg = SousConfig(
        data_dir=tmp_path / "data", config_path=tmp_path / "config.toml", gateway_enabled=True
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: engine)
    return create_server(TaskStore(tmp_path / "tasks.db"), engines, cfg).streamable_http_app()


@contextlib.contextmanager
def _serve(app) -> Iterator[tuple[str, uvicorn.Server, threading.Thread]]:
    """Run the ASGI app under uvicorn in a thread, with the daemon's own
    uvicorn configuration so the graceful-shutdown bound under test is the
    real one. Yields the base URL, the server (off the main thread its
    should_exit flag is the only stop switch) and the serving thread."""
    port = _free_port()
    server = uvicorn.Server(uvicorn_config(app, "127.0.0.1", port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn never started"
    try:
        yield f"http://127.0.0.1:{port}", server, thread
    finally:
        server.should_exit = True
        thread.join(GRACEFUL_SHUTDOWN_SECONDS + 10)


def _wait_for_generation(inner: ChunkedFakeEngine) -> None:
    deadline = time.monotonic() + 5
    while not inner.generate_threads and time.monotonic() < deadline:
        time.sleep(0.05)
    assert inner.generate_threads, "the turn never started"


def _body(**overrides) -> dict:
    body = {
        "model": "sous-local",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


def _frames(lines: Iterator[str]) -> Iterator[tuple[str, dict | None]]:
    event = data = None
    for line in lines:
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            data = json.loads(line[len("data: ") :])
        elif line == "" and event is not None:
            yield event, data
            event = data = None


def test_pings_keep_flowing_while_the_model_is_silent(tmp_path: Path, monkeypatch):
    """Checklist item 2: Claude Code disconnects on a silent stream. The first
    ping is the first byte; sse-starlette repeats it on the interval while the
    fake engine sleeps between its two pieces."""
    import sous.gateway.routes as routes

    monkeypatch.setattr(routes, "PING_INTERVAL_SECONDS", 1)
    inner = ChunkedFakeEngine(["slow|reply"], delay=1.3)
    with (
        _serve(_app(tmp_path, inner)) as (base, _server, _thread),
        httpx.Client(timeout=30) as client,
        client.stream("POST", f"{base}/v1/messages", json=_body()) as r,
    ):
        assert r.status_code == 200
        events = list(_frames(r.iter_lines()))
    kinds = [e for e, _ in events]
    assert kinds[0] == "ping"
    assert kinds.count("ping") >= 2, kinds
    # A ping may land between message_stop and the stream closing, so the last
    # non-ping frame is what pins the sequence.
    assert [k for k in kinds if k != "ping"][-1] == "message_stop"
    text = "".join(d["delta"]["text"] for e, d in events if e == "content_block_delta" and d)
    assert text == "slowreply"


def test_client_disconnect_drains_the_turn_and_never_wedges_the_next(tmp_path: Path):
    """Spec Phase 1 requirement: an undrained producer holding the engine lock
    would block every later generation. The thread finishes the turn after the
    client is gone, and the next request runs on the same session."""
    inner = ChunkedFakeEngine(["a|b|c|d|e", "second"], delay=0.3)
    with _serve(_app(tmp_path, inner)) as (base, _server, _thread), httpx.Client(timeout=30) as client:
        with client.stream("POST", f"{base}/v1/messages", json=_body()) as r:
            for line in r.iter_lines():
                if "text_delta" in line:
                    break  # hang up after the first piece
        assert inner.finished.wait(10), "the abandoned turn never completed"
        t0 = time.monotonic()
        second = client.post(f"{base}/v1/messages", json=_body(stream=False))
        assert second.status_code == 200
        assert second.json()["content"] == [{"type": "text", "text": "second"}]
        assert time.monotonic() - t0 < 5
    assert inner.generate_threads[0] is inner.generate_threads[1]


def test_non_streaming_over_a_real_socket(tmp_path: Path):
    inner = ChunkedFakeEngine(["hello| world"])
    with _serve(_app(tmp_path, inner)) as (base, _server, _thread), httpx.Client(timeout=30) as client:
        r = client.post(f"{base}/v1/messages", json=_body(stream=False))
        assert r.status_code == 200
        assert r.json()["content"] == [{"type": "text", "text": "hello world"}]
        assert r.json()["usage"]["output_tokens"] == 2
        assert client.head(f"{base}/api/hello").status_code == 200


def test_a_request_abandoned_while_queued_never_generates(tmp_path: Path):
    """Drain-to-completion covers a generation that started. One still waiting
    for the lock when its client leaves must not start: it would only delay the
    live requests queued behind it by a whole turn."""
    inner = ChunkedFakeEngine(["a|b|c|d|e|f", "third"], delay=0.4)
    with _serve(_app(tmp_path, inner)) as (base, _server, _thread), httpx.Client(timeout=30) as client:

        def first_turn() -> None:
            with contextlib.suppress(Exception):
                client.post(f"{base}/v1/messages", json=_body(stream=False))

        first = threading.Thread(target=first_turn, daemon=True)
        first.start()
        _wait_for_generation(inner)  # the first turn holds the gateway lock
        with client.stream("POST", f"{base}/v1/messages", json=_body()) as r:
            assert r.status_code == 200  # headers and the first ping arrive while queued
        # Leaving the block closed the second request while it was still queued.
        first.join(10)
        time.sleep(1.0)  # every chance for an abandoned turn to (wrongly) start
        assert len(inner.calls) == 1
        third = client.post(f"{base}/v1/messages", json=_body(stream=False))
        assert third.json()["content"] == [{"type": "text", "text": "third"}]


def test_shutdown_is_bounded_while_a_non_streaming_turn_runs(tmp_path: Path):
    """uvicorn owns SIGTERM while it serves and, unbounded, waits for open
    connections before sous's own handler runs; a non-streaming gateway turn
    could hold one for the whole generation timeout. The daemon's uvicorn
    config bounds that wait, so `sous stop` and launchd restarts stay prompt
    while the turn itself still drains on its thread."""
    inner = ChunkedFakeEngine(["slow|slow|slow|slow|slow|slow"], delay=1.0)
    client = httpx.Client(timeout=30)
    with _serve(_app(tmp_path, inner)) as (base, server, thread):

        def request() -> None:
            with contextlib.suppress(Exception):  # the connection is cut by the shutdown
                client.post(f"{base}/v1/messages", json=_body(stream=False))

        threading.Thread(target=request, daemon=True).start()
        _wait_for_generation(inner)
        t0 = time.monotonic()
        server.should_exit = True
        thread.join(GRACEFUL_SHUTDOWN_SECONDS + 5)
        assert not thread.is_alive(), "uvicorn waited for the turn instead of bounding the shutdown"
        assert time.monotonic() - t0 < GRACEFUL_SHUTDOWN_SECONDS + 4
    assert inner.finished.wait(10)  # the turn still drained to completion
    client.close()
```

- [ ] **Step 2: Run them (for real — never mocked)**

Run: `uv run pytest tests/test_gateway_http.py -q`
Expected: 5 PASS in roughly 20 s total. Run the disconnect, abandoned and shutdown tests five times in a loop to check for flakiness before moving on: `for i in 1 2 3 4 5; do uv run pytest tests/test_gateway_http.py -q -k "disconnect or abandoned or shutdown" || break; done`.

- [ ] **Step 3: Write the model-marked test**

Create `tests/test_gateway_model.py`:

```python
"""The gateway route in front of a real (tiny) model: the event stream is
well-formed end to end, counts are real, and a second, longer request reuses
the prompt cache through the gateway-owned session."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.server import create_server
from sous.tasks import TaskStore

pytestmark = pytest.mark.model

TINY = "mlx-community/Qwen3-0.6B-4bit"  # ~350 MB; text-only, so the LM backend


def _post(app, body: dict) -> httpx.Response:
    async def go():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8383", timeout=600
        ) as client:
            return await client.post("/v1/messages", json=body)

    return asyncio.run(go())


def _events(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for frame in text.split("\n\n"):
        event: str | None = None
        data: dict = {}
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event is not None:
            out.append((event, data))
    return out


def test_real_model_streams_a_well_formed_turn_and_reuses_the_cache(tmp_path: Path):
    from sous.engine.lm import LMEngine

    cfg = SousConfig(
        data_dir=tmp_path / "data", config_path=tmp_path / "config.toml", gateway_enabled=True
    )
    engines = EngineManager(cfg, engine_factory=lambda mid: LMEngine(TINY, prompt_cache=True))
    app = create_server(TaskStore(tmp_path / "tasks.db"), engines, cfg).streamable_http_app()
    tool = {
        "name": "echo",
        "description": "Echo a word back",
        "input_schema": {"type": "object", "properties": {"word": {"type": "string"}}},
    }
    first = {
        "model": "sous-local",
        "max_tokens": 48,
        "stream": True,
        "tools": [tool],
        "messages": [{"role": "user", "content": "Say the word banana and nothing else."}],
    }
    r = _post(app, first)
    assert r.status_code == 200
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "ping" and kinds[1] == "message_start"
    assert [k for k in kinds if k != "ping"][-1] == "message_stop"  # a late ping may trail it
    start = events[1][1]["message"]
    assert start["usage"]["input_tokens"] > 0
    delta = next(d for e, d in events if e == "message_delta")
    assert delta["usage"]["output_tokens"] > 0
    assert delta["delta"]["stop_reason"] in ("end_turn", "max_tokens", "tool_use")
    indices = [d["index"] for e, d in events if e == "content_block_start"]
    assert indices == list(range(len(indices)))

    # Turn 2 extends turn 1's conversation: the gateway's long-lived session
    # keeps the KV cache, so this must be a prefix-cache hit.
    reply_text = "".join(
        d["delta"]["text"] for e, d in events if e == "content_block_delta" and d["delta"]["type"] == "text_delta"
    )
    second = {
        **first,
        "stream": False,
        "messages": first["messages"]
        + [
            {"role": "assistant", "content": reply_text or "banana"},
            {"role": "user", "content": "Now say kiwi and nothing else."},
        ],
    }
    r2 = _post(app, second)
    assert r2.status_code == 200
    assert r2.json()["usage"]["input_tokens"] > start["usage"]["input_tokens"]
    stats = engines.get().prompt_cache_stats()
    assert stats["hits"] >= 1, stats
    engines.get().unload()
```

Run: `uv run pytest tests/test_gateway_model.py -q -m model`
Expected: PASS (first run downloads the 0.6B weights). The `hits >= 1` assertion is the proof that the gateway-owned session preserves the cache across HTTP requests — the property the whole Phase 1 latency story rests on.

- [ ] **Step 4: Lint, type-check, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
git add tests/test_gateway_http.py tests/test_gateway_model.py
git commit -m "test(gateway): real-server keepalive, disconnect drain, real-model turn

The in-process ASGI transport buffers responses, so the two behaviours the
spec names as Phase 1 requirements — pings while the model is silent, and
a disconnected client never wedging the next turn — are pinned against a
real uvicorn on a loopback port. The model-marked test proves the gateway-
owned session keeps the prompt cache warm between HTTP requests.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---
### Task 9: Documentation — README, CLAUDE.md

**Files:**
- Modify: `README.md` (Configuration block; new "Gateway mode (experimental)" section; Security model; Limitations)
- Modify: `CONTRIBUTING.md` (new "Verifying the gateway endpoint" section — the whole-session-local recipe lives here, not in the README)
- Modify: `CLAUDE.md` (Gotchas, Security boundary)

**Interfaces:** none (docs). Everything stated here must match the code from Tasks 1–8 exactly (key names, defaults, env vars, error behaviour).

- [ ] **Step 1: Configuration block**

In `README.md`, inside the `toml` block under `## Configuration — ~/.sous/config.toml`, after the `[tasks]` table:

```toml
[gateway]
# EXPERIMENTAL — see "Gateway mode" below. Serve Claude Code from the local
# model over an Anthropic-compatible /v1/messages on the same port.
enabled = false
local_models = ["sous-local"]   # model ids served locally; anything else is a 404 for now.
                                # Never claude-*: Claude Code ignores its context-window
                                # env vars for those ids (the config rejects them).
max_context_tokens = 65536      # window advertised to Claude Code; clamped up to 49152
generation_timeout_minutes = 30
```

- [ ] **Step 2: The "Gateway mode" section**

Insert a new section between `## What Claude gets` and `## Configuration — ~/.sous/config.toml`:

```markdown
## Gateway mode (experimental)

sous can also stand in for `api.anthropic.com`: with `[gateway].enabled =
true` the daemon serves `POST /v1/messages` (streaming and not),
`POST /v1/messages/count_tokens` and `HEAD /api/hello` on the same
`127.0.0.1:8383`, answering requests for the model id `sous-local` with the
local model. Claude Code executes the tool calls itself, with its usual
permission prompts; the model only decides what to call.

This is the first half of the hybrid design in issue #41. The intended mode
is a frontier main loop with local subagents — `CLAUDE_CODE_SUBAGENT_MODEL=sous-local`,
tier variables left alone, so a frontier model always reviews the local
model's work — and it needs the routing half and the `sous claude` launcher,
which come next. Until then a Claude Code session pointed at the gateway runs
*entirely* on the local model: exactly the trade "Why" and "How sous
compares" above argue against, supported only as a way to exercise the
endpoint, not as a mode. The recipe for that lives in
[CONTRIBUTING.md](CONTRIBUTING.md#verifying-the-gateway-endpoint).

What a locally served turn gives up, stated plainly:

- **No sandbox.** The gateway returns `tool_use` blocks and never runs a
  tool; `toolexec.py` (path confinement, allowlist, audit) is not in this
  path. Claude Code's own permission system is the boundary, and a 27B model
  inherits whatever permissiveness you configured for frontier subagents.
- **No server-side tools.** `WebSearch`, `WebFetch`-as-server-tool, code
  execution and the other tools that run inside Anthropic's API are dropped
  from the request (logged as `dropped N server-side tool(s)`) — they cannot
  execute on a local endpoint.
- **No thinking, no request-level sampling.** `thinking`, `temperature`,
  `top_p`, `top_k`, `stop_sequences` and `tool_choice` are accepted and
  ignored; the daemon's `[model]` sampler applies. Images and documents in
  messages become a one-line `[image omitted]` placeholder.
- **One turn at a time, one cache slot.** Turns are serialized behind the
  same lock as delegated tasks, and the prompt cache has one slot: a
  subagent's consecutive turns reuse it (turn 2 of a 57k-token prompt
  prefills only the new tokens), but while a delegated task and a gateway
  conversation overlap — or two conversations interleave — every turn on
  both sides is a cold prefill, because each evicts the other. Do not run
  both at once until keyed cache slots land.
- **A client that disconnects does not stop the model.** The turn runs to
  completion (so the next request never waits on a wedged lock); aborting
  mid-generation comes with batching, later.

The gateway logs one line per request to the daemon's stderr — method,
model, status, token counts, stop reason, cache hit/miss, seconds — and never
a request body or a header value. Errors are Anthropic-shaped
(`{"type": "error", "error": {"type": ..., "message": ...}}`); an unknown
model id is a `404 not_found_error`, an oversized body (over 32 MiB) a
`413 request_too_large`, and a prompt that fills the window an
`invalid_request_error` saying `prompt is too long`.
```

- [ ] **Step 3: Security model and Limitations**

Append to the `## Security model` list:

```markdown
- **Gateway mode bypasses the sandbox by design.** A locally served Claude
  Code turn never touches `toolexec.py`: the gateway hands `tool_use` blocks
  back and Claude Code executes them under its own permission rules. The
  gateway binds to `127.0.0.1` only, stores no credential, and never logs
  request bodies or header values — including the `Authorization` header
  Claude Code sends with every request. It is off by default.
```

Append to `## Limitations`:

```markdown
- Gateway mode (experimental) serves one turn at a time on a single-slot
  prompt cache, drops Anthropic server-side tools, ignores request-level
  sampling and thinking, and finishes a turn even after the client hangs up.
  It serves *every* request that reaches it locally; routing frontier models
  upstream (the hybrid in issue #41) is not built yet.
```

- [ ] **Step 4: CONTRIBUTING.md — the verification recipe**

Append to `CONTRIBUTING.md` (the block below contains its own fenced command, hence the four-backtick fence):

````markdown
## Verifying the gateway endpoint

Until the routing half of issue #41 lands, every request that reaches the
gateway is served locally, so the only way to drive it from Claude Code is a
*whole-session-local* run. That is a verification setup, not a supported
mode (see README, "Gateway mode"). With `[gateway].enabled = true` and the
daemon restarted:

```bash
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
  ANTHROPIC_BASE_URL=http://127.0.0.1:8383 \
  ANTHROPIC_DEFAULT_OPUS_MODEL=sous-local \
  ANTHROPIC_DEFAULT_SONNET_MODEL=sous-local \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=sous-local \
  CLAUDE_CODE_SUBAGENT_MODEL=sous-local \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 \
  CLAUDE_CODE_AUTO_COMPACT_WINDOW=65536 \
  API_TIMEOUT_MS=3000000 \
  claude --disallowedTools LSP
```

Do **not** set `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`: either one
switches Claude Code from your subscription login to API-credit billing the
moment traffic goes upstream again. The two context variables must match
`[gateway].max_context_tokens` (Claude Code honours them only for model ids
that are not `claude-*`, which is why the served id is honest).
`API_TIMEOUT_MS` covers model load plus a long prefill; `--disallowedTools
LSP` keeps a language server from appending its schema mid-session and
re-prefilling the whole conversation. Watch `~/.sous/daemon.log` for the
`sous gateway:` lines — the first turn is a cold prefill, later turns should
report `cache=hit`.
````

(The plan's Task 10 uses this recipe verbatim plus headless flags; keep the two identical.)

- [ ] **Step 5: CLAUDE.md**

Add to `## Gotchas`:

```markdown
- Engine `on_delta` callbacks (`engine/base.py:Delta`) fire on the generation
  thread from inside the decode loop: never block or raise in one, and expect
  late deltas from a stalled-and-abandoned session.
- `PrefixCache` refuses its cold retry once any delta has streamed (a retry
  would replay the turn to the client). That is deliberate, not a missing
  retry.
```

Add to `## Security boundary`, after the toolexec paragraph:

```markdown
`src/sous/gateway/` is deliberately outside that boundary: it never executes a
tool (Claude Code does, under its own permissions) and never logs a request
body or header value. A change that makes it do either needs the spec
(`docs/superpowers/specs/2026-08-26-hybrid-gateway-design.md`) changed first.
```

- [ ] **Step 6: Verify and commit**

Run: `uv run ruff check . && uv run ruff format --check .` (docs are ruff-excluded; this confirms nothing else moved) and read the README and CONTRIBUTING sections once more against `src/sous/config.py` defaults and `src/sous/gateway/routes.py` behaviour — every number and env var above has one source of truth in code.

```bash
git add README.md CONTRIBUTING.md CLAUDE.md
git commit -m "docs: describe gateway mode and what a local turn gives up

The gateway is off by default and experimental; the docs say plainly that
it bypasses the toolexec sandbox by design, drops server-side tools,
serializes turns on one cache slot, and never logs bodies or headers — what
the spec's security posture requires now that the endpoint exists. Whole-
session-local is documented as a verification setup only, in CONTRIBUTING;
the shipped positioning stays subagent-only and lands with Phase 2 routing.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Phase 1 exit — one full-local Claude Code session (manual)

**Files:** none in the repo. Output: a comment on issue #41 with observed data.

This task is the spec's Phase 1 exit criterion: "one full-local Claude Code session (no routing yet) completes a real bounded task against sous". It is manual because it needs the 27B, a logged-in Claude Code CLI (the CLI login is separate from Claude Desktop's), and a human judging the diff. Run it after Tasks 1–9 are merged to `main` and the daemon is restarted from that build.

- [ ] **Step 1: Enable and restart**

Add to `~/.sous/config.toml`:

```toml
[gateway]
enabled = true
```

Restart the daemon (launchd-managed: `launchctl kickstart -k gui/$(id -u)/com.sous.daemon`; otherwise `sous stop && sous serve` in a terminal). Confirm the startup line `sous: gateway (experimental) serving sous-local at http://127.0.0.1:8383/v1/messages` in `~/.sous/daemon.log` (or the terminal), and:

```bash
curl -sS -I http://127.0.0.1:8383/api/hello | head -1
```

Expected: `HTTP/1.1 200 OK`.

- [ ] **Step 2: Smoke the endpoint by hand (loads the model on first call)**

```bash
curl -sS -N http://127.0.0.1:8383/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"sous-local","max_tokens":64,"stream":true,
       "messages":[{"role":"user","content":"Reply with the single word: ready"}]}'
```

Expected: `event: ping` first (repeated every 10 s while the model loads), then `message_start` … `content_block_delta` frames containing `ready` … `message_delta` with `"stop_reason":"end_turn"` and a non-zero `output_tokens`, then `message_stop`. Then a count and a tool call:

```bash
curl -sS http://127.0.0.1:8383/v1/messages/count_tokens \
  -H 'content-type: application/json' \
  -d '{"model":"sous-local","messages":[{"role":"user","content":"hello"}]}'
```

Expected: `{"input_tokens": N}` with N in the low tens.

```bash
curl -sS http://127.0.0.1:8383/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"sous-local","max_tokens":256,
       "tools":[{"name":"Read","description":"Read a file","input_schema":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}],
       "messages":[{"role":"user","content":"Use the Read tool to read /etc/hosts. Call the tool; do not answer in prose."}]}'
```

Expected: JSON with `"stop_reason":"tool_use"` and a `content` entry `{"type":"tool_use","id":"toolu_…","name":"Read","input":{"file_path":"/etc/hosts"}}`. Record the daemon log line for each request (`input_tokens=… output_tokens=… stop=… cache=… seconds=…`).

- [ ] **Step 3: The full-local session**

Make a scratch repo and run Claude Code headless against the gateway. The environment is the CONTRIBUTING.md "Verifying the gateway endpoint" recipe verbatim (no credential env vars — the CLI's own login is used for nothing here, but the rule is the rule); the only additions are the headless flags:

```bash
mkdir -p /tmp/gw-demo && cd /tmp/gw-demo && git init -q && echo "# demo" > README.md && git add . && git -c user.email=t@t -c user.name=t commit -qm init
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
  ANTHROPIC_BASE_URL=http://127.0.0.1:8383 \
  ANTHROPIC_DEFAULT_OPUS_MODEL=sous-local \
  ANTHROPIC_DEFAULT_SONNET_MODEL=sous-local \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=sous-local \
  CLAUDE_CODE_SUBAGENT_MODEL=sous-local \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 \
  CLAUDE_CODE_AUTO_COMPACT_WINDOW=65536 \
  API_TIMEOUT_MS=3000000 \
  claude --disallowedTools LSP --allowedTools "Read,Write,Edit,Glob,Grep,Bash(python3 *)" \
  -p "Create hello.py that prints 'hello from sous', run it with python3 hello.py, and report the exact output."
```

Watch `~/.sous/daemon.log` in another terminal. Expected: several `sous gateway: POST /v1/messages model=sous-local stream=1 status=200 …` lines; the first turn reports `cache=miss` with a large `input_tokens` (Claude Code's system prompt plus its tool schemas — tens of thousands of tokens) and tens of seconds to minutes; subsequent turns report `cache=hit reused_tokens=…` close to the previous `input_tokens` and complete in seconds. The session should end with `hello.py` on disk, executed, and the output reported. If Claude Code prints `unrecognized_model`, the tier env vars did not take — check the shell exported them; if it reports a timeout, check that `API_TIMEOUT_MS` reached it and that pings appear in the log-adjacent `curl -N` test.

Judge the task like a PR: did the model pick the right tools, did the edit land, did it stop when done. Note anything the model did that the gate-2 caveat predicted (retrying simpler `Bash` rather than pivoting tools) and anything Claude Code did that the spec's checklist did not anticipate (new event, new block type, a 4xx it did not recover from). Also record which shape the post-tool `<total_tokens>` marker arrived in — a bare text block after the tool results, or one wrapped in `<system-reminder>` — by temporarily adding a `_log(...)` of the last user message's block types in `Gateway.messages` for this run (remove it afterwards; it must never log text). Both shapes must yield no extra user turn (Task 4 pins both).

- [ ] **Step 4: Record the exit on issue #41**

Post a comment on `krcm0209/sous#41` titled "Phase 1 exit: full-local session" with: the sous commit, Claude Code version (`claude --version`), model id, per-turn table from the log (`input_tokens`, `output_tokens`, `stop`, `cache`, `reused_tokens`, `seconds`), whether the task completed and the diff was correct, and the observations from Step 3. State plainly whether the exit criterion is met. Then run `gh issue view 44 --repo krcm0209/sous` and, if the session ran the main loop locally without incident, note on #44 that its gate 1 ("can a full-local session work at all") was answered here.

- [ ] **Step 5: Turn it off again**

Set `[gateway] enabled = false` (or remove the section) and restart the daemon, unless you intend to keep using it — the routing half is not built, so a live gateway serves every request it gets locally.

---

## Self-review notes (for the executor and the reviewer)

**Spec coverage → task:** opt-in `[gateway]` config (T1); `/v1/messages` + `count_tokens` mounted on the daemon (T7); streaming threaded through `GenerationSession` (T3, via `on_delta` — decision 2); request-scoped tool validation replacing `protocol.py:173` (T2); disconnect never wedges the session — drain-to-completion (T6 design, T8 test); client tool schemas via the chat template and back as `tool_use` (T4, T5); checklist item 1 inline system (T4), item 2 pings (T7, T8), items 3–4 marker/billing stripping (T4), item 5 thinking signature (N/A while thinking is off — recorded in decision 9), item 6 phantom text block (T5), item 7 server-side tools (T4 + README T9), item 8 tolerate unknown fields/blocks with Anthropic-shaped errors (T4, T7), item 9 launcher env (the verification recipe in CONTRIBUTING.md via T9; the launcher itself is Phase 2); prompt stabilization from day one (T4); 48K floor (T1); security posture — loopback only (inherited binding), no body/header logging (T7 test), no credential storage (nothing stores one), sandbox bypass stated in docs (T9); exit criterion (T10).

**Not in this phase, on purpose:** upstream forwarding and the `sous claude` launcher (Phase 2); keyed cache slots (3a); batching and true mid-generation abort (3b); `SECURITY.md` gateway section (spec: Phase 2); a real-HTTP-status pre-flight for "prompt is too long" on streams (in-band for now — decision 7; revisit if Claude Code's auto-compact does not trigger on the in-band error); stripping `$schema`/`additionalProperties` from tool schemas to save prefix tokens (measure first); honouring `stop_sequences`/`tool_choice: none`.

**Type consistency check:** `Delta(text, output_tokens, finish_reason)` is constructed identically in `lm.py`, `vlm.py`, `fake_engine.py` and the tests; `OnDelta | None` is the parameter type everywhere it appears; `ToolSet.from_tools(tools, *, strict)` is keyword-only for `strict` in every call; `TurnAssembler(message_id, model, toolset)` positional in `routes.py` and the tests; `TurnRunner.run(messages, tools, max_tokens, sink)` matches its four call sites (routes ×2, tests); `RequestError.body()` is the only error-body builder for validation errors, `_error_response`/`_classify` for turn failures.
