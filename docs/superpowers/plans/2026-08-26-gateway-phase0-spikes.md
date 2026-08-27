# Gateway Phase 0 (Spike Gates) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the two kill-criteria spikes for the hybrid gateway — gate 1 (does subscription billing survive a transparent localhost proxy, and does `CLAUDE_CODE_SUBAGENT_MODEL` reach the wire verbatim) and gate 2 (can the default 27B act as a Claude Code subagent) — and document both outcomes on issue #41 with observed wire data.

**Architecture:** Two standalone throwaway scripts under `scripts/spikes/` on a branch that is never merged to `main`. Gate 1 is a Starlette/httpx transparent reverse proxy in front of `api.anthropic.com` that logs routing metadata (never bodies, never header values). Gate 2 replays a captured subagent request body against the local model via `mlx_lm`, with a permissive tool-call parser and read-only tool execution through sous's existing `ToolExecutor`.

**Tech Stack:** Python 3.14, uv, starlette + uvicorn + httpx (all already in the venv as transitive deps of `mcp`), mlx_lm (already a dependency), `sous.toolexec.ToolExecutor` (read-only ops only).

**Spec:** `docs/superpowers/specs/2026-08-26-hybrid-gateway-design.md` (Phase 0 section; the credential rule is in the "Client configuration" core decision).

## Global Constraints

- Python >= 3.14; everything runs via `uv run`. Never pip.
- **Never set `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` when launching Claude Code for these spikes** — either one switches Claude Code off the OAuth path gate 1 exists to observe (spec, "Client configuration").
- **Never log request bodies or header values.** Log only derived metadata: presence booleans, token *kind* inferred from prefix, sizes, the model field, and the `anthropic-beta` feature list. The one exception is the explicit `--dump-model` capture for gate 2, which writes to a local directory outside the repo and must never be committed (dumps contain session content).
- Spike code is throwaway: it lives on branch `spike/gateway-gates`, is committed there for reproducibility, and is **not** merged to `main`. Findings land on issue #41, not in the codebase.
- Any thread that touches mlx must call `sous.engine.base.release_mlx_thread_state()` before it exits (CLAUDE.md gotcha; the gate 2 script runs mlx on the main thread but calls it in a `finally` anyway, matching house practice).
- Gate 2 downloads/loads multi-GB weights — local/manual only, same as `model`-marked tests. Nothing in this plan runs in CI.
- Commit messages: Conventional Commits, imperative lowercase subject, why in the body.

---

### Task 1: Spike branch and the gate 1 transparent proxy

**Files:**
- Create: `scripts/spikes/README.md`
- Create: `scripts/spikes/gate1_proxy.py`

**Interfaces:**
- Produces: a runnable proxy (`uv run python scripts/spikes/gate1_proxy.py --port 8399`) whose stdout is one JSON object per request with keys `ts, method, path, model, stream, body_bytes, status, dur_ms, auth, x_api_key, anthropic_beta`; and a `--dump-model NAME --dump-dir DIR` mode Task 3 and Task 5 rely on, which writes each matching request body to `DIR/<unix-ts>-<n>.json`.

- [ ] **Step 1: Create the branch**

```bash
git checkout main && git pull && git checkout -b spike/gateway-gates
```

- [ ] **Step 2: Write `scripts/spikes/README.md`**

```markdown
# Gateway spike gates (issue #41 / Phase 0)

Throwaway probes for the hybrid gateway's two kill criteria. This branch is
never merged; results are recorded on issue #41. See
docs/superpowers/specs/2026-08-26-hybrid-gateway-design.md, Phase 0.

- `gate1_proxy.py` — transparent passthrough proxy to api.anthropic.com.
  Logs routing metadata only (no bodies, no header values). `--dump-model`
  writes matching request bodies to a LOCAL directory for gate 2; dumps
  contain session content and must never be committed.
- `gate2_replay.py` — replays a captured subagent request against the local
  model and drives a bounded tool loop.

Neither script is imported by sous and neither runs in CI.
```

- [ ] **Step 3: Write `scripts/spikes/gate1_proxy.py`**

```python
"""Gate 1 spike: transparent Anthropic passthrough proxy. THROWAWAY.

Sits between Claude Code and api.anthropic.com to answer, by observation:
  1. Does subscription (OAuth) billing survive a localhost proxy?
  2. What does the `model` field contain per request — and does
     CLAUDE_CODE_SUBAGENT_MODEL's value arrive verbatim?
  3. Does any traffic we did not expect match the subagent model id?

Logging discipline (spec: "never bodies or headers"): stdout gets one JSON
line per request with derived metadata only. Header VALUES are never
printed; the Authorization header is reduced to a token-kind guess from its
prefix. The one exception is --dump-model, an explicit opt-in that writes
matching request bodies to a local directory for gate 2's replay — those
dumps contain session content and stay out of git.
"""

import argparse
import json
import time
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

UPSTREAM = "https://api.anthropic.com"

# RFC 9110 §7.6.1 connection-scoped headers, plus host/content-length which
# httpx regenerates for the new origin.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
}

_dump_model: str | None = None
_dump_dir: Path | None = None
_dump_count = 0


def _token_kind(request: Request) -> str:
    """Classify the Authorization header without exposing it."""
    auth = request.headers.get("authorization", "")
    if not auth:
        return "absent"
    if not auth.lower().startswith("bearer "):
        return "non-bearer"
    token = auth[7:]
    # Observed Claude Code OAuth access tokens start "sk-ant-oat"; API keys
    # start "sk-ant-api". Anything else is reported as "other" — update this
    # classifier from observation rather than trusting it blindly.
    if token.startswith("sk-ant-oat"):
        return "bearer:oauth"
    if token.startswith("sk-ant-api"):
        return "bearer:api-key"
    return "bearer:other"


def _maybe_dump(body: bytes, model: str) -> None:
    global _dump_count
    if _dump_dir is None or model != _dump_model:
        return
    _dump_count += 1
    path = _dump_dir / f"{int(time.time())}-{_dump_count}.json"
    path.write_bytes(body)
    print(f"# dumped {len(body)} bytes for model={model!r} -> {path}", flush=True)


async def proxy(request: Request) -> Response:
    started = time.monotonic()
    body = await request.body()
    model, stream = "", False
    try:
        parsed = json.loads(body) if body else {}
        model = parsed.get("model", "")
        stream = bool(parsed.get("stream", False))
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        pass
    _maybe_dump(body, model)

    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    url = UPSTREAM + request.url.path
    if request.url.query:
        url += "?" + request.url.query

    client: httpx.AsyncClient = request.app.state.client
    upstream = client.build_request(
        request.method, url, headers=fwd_headers, content=body
    )
    resp = await client.send(upstream, stream=True)

    record = {
        "ts": round(time.time(), 3),
        "method": request.method,
        "path": request.url.path,
        "model": model,
        "stream": stream,
        "body_bytes": len(body),
        "status": resp.status_code,
        "dur_ms": round((time.monotonic() - started) * 1000),
        "auth": _token_kind(request),
        "x_api_key": "x-api-key" in request.headers,
        # The beta header is a feature-flag list, not a credential; #41 says
        # OAuth capability is forwarded through it, so its value is the data.
        "anthropic_beta": request.headers.get("anthropic-beta", ""),
    }
    print(json.dumps(record), flush=True)

    resp_headers = {
        k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=resp_headers,
        background=None,
    )


def main() -> None:
    global _dump_model, _dump_dir
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--dump-model", help="dump request bodies whose model matches")
    ap.add_argument("--dump-dir", help="where dumps go (created; keep out of git)")
    args = ap.parse_args()
    if bool(args.dump_model) != bool(args.dump_dir):
        ap.error("--dump-model and --dump-dir must be used together")
    if args.dump_model:
        _dump_model = args.dump_model
        _dump_dir = Path(args.dump_dir).expanduser()
        _dump_dir.mkdir(parents=True, exist_ok=True)

    app = Starlette(routes=[
        Route("/{rest:path}", proxy,
              methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
    ])
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke-test the proxy without Claude Code**

```bash
uv run python scripts/spikes/gate1_proxy.py --port 8399 &
sleep 2
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8399/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"probe","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'
kill %1
```

Expected: curl prints `401` (upstream rejects the unauthenticated probe — proving the request reached api.anthropic.com), and the proxy's stdout shows one JSON line with `"model": "probe"`, `"auth": "absent"`, `"status": 401`. A connection error or a 5xx from the proxy itself means the forwarding is broken — fix before proceeding.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check scripts/spikes/ && uv run ruff format scripts/spikes/
git add scripts/spikes/
git commit -m "spike: gate 1 transparent passthrough proxy (#41)

Throwaway probe for Phase 0 gate 1 of the hybrid gateway spec. Logs
routing metadata only; --dump-model captures bodies locally for gate 2."
```

---

### Task 2: Experiment 1a — subscription passthrough

This task is manual observation, driven by whoever executes the plan, with Kyle at the keyboard for the Claude Code side (it needs his logged-in session).

**Files:**
- Create: none (observations go in a scratch note, then issue #41 in Task 4)

**Interfaces:**
- Consumes: `gate1_proxy.py --port 8399` from Task 1.
- Produces: written answers to observations O1–O5 below.

- [ ] **Step 1: Start the proxy with logging to a file**

```bash
uv run python scripts/spikes/gate1_proxy.py --port 8399 | tee /tmp/gate1a.jsonl
```

- [ ] **Step 2: Launch Claude Code through it — credentials deliberately absent**

In a second terminal (note the explicit unsets — the user's shell may export these):

```bash
env -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_API_KEY \
  ANTHROPIC_BASE_URL=http://127.0.0.1:8399 \
  claude
```

Expected: Claude Code starts normally against the existing claude.ai login. If it demands an API key or shows an auth error at startup, record that verbatim — it is a gate-1 kill signal.

- [ ] **Step 3: Run one small prompt and one subagent, then record observations**

In the Claude Code session: ask one trivial question (e.g. "what is 2+2"), then trigger a subagent (e.g. "use the Explore agent to list what's in this directory"). Then answer, from `/tmp/gate1a.jsonl` and the session:

- **O1 (billing kill criterion):** Did the session work end-to-end with no credential env vars set? Does `claude` report the subscription account (run `/status` in the session), with no API-credit warnings?
- **O2:** Every request line's `auth` value. Expected `bearer:oauth` throughout; `bearer:api-key` or `absent` on inference requests means OAuth did not pass through.
- **O3:** The set of distinct `path` values (what surface must the real gateway forward — count_tokens? anything unexpected?).
- **O4:** The set of distinct `model` values, and which ones the subagent used with no model overrides set.
- **O5:** The `anthropic_beta` values — is there an OAuth-related capability token in the list (per #41, the gateway must forward this header for subscription auth to work)?

```bash
jq -s 'map(.path) | unique' /tmp/gate1a.jsonl
jq -s 'map(.model) | unique' /tmp/gate1a.jsonl
jq -s 'map(.auth) | unique' /tmp/gate1a.jsonl
jq -s 'map(.anthropic_beta) | unique' /tmp/gate1a.jsonl
```

- [ ] **Step 4: Decision point**

If O1 or O2 fails (billing does not survive, or OAuth is not on the wire), **stop the plan here** and go directly to Task 4 to document the kill. Per the spec: "If billing falls back to API credits, the hybrid is dead — stop."

---

### Task 3: Experiment 1b — the subagent discriminator, plus the gate 2 capture

**Files:**
- Create: none (dumps land in `~/.sous/spikes/dumps/`, outside the repo)

**Interfaces:**
- Consumes: `gate1_proxy.py` from Task 1.
- Produces: at least one dumped subagent request body at `~/.sous/spikes/dumps/*.json` (Task 5's input), and answers to observations O6–O8.

- [ ] **Step 1: Restart the proxy with dump mode on**

```bash
uv run python scripts/spikes/gate1_proxy.py --port 8399 \
  --dump-model sous-local --dump-dir ~/.sous/spikes/dumps \
  | tee /tmp/gate1b.jsonl
```

- [ ] **Step 2: Launch Claude Code with the subagent model pinned**

```bash
env -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_API_KEY \
  ANTHROPIC_BASE_URL=http://127.0.0.1:8399 \
  CLAUDE_CODE_SUBAGENT_MODEL=sous-local \
  claude
```

- [ ] **Step 3: Trigger one Explore subagent and let it fail**

Ask: "use the Explore agent to find where the config is loaded in this repo." The subagent request carries `model: sous-local`, which api.anthropic.com will 404 — that failure is expected and harmless; the request body is captured on its way through. Confirm the main conversation keeps working after the subagent errors.

- [ ] **Step 4: Record observations**

- **O6 (discriminator):** Does a request with `"model": "sous-local"` appear in `/tmp/gate1b.jsonl` — the env var's value verbatim, no alias expansion? (`jq -s 'map(.model) | unique' /tmp/gate1b.jsonl`)
- **O7 (predicate purity):** Does anything *other* than the deliberate subagent request carry `sous-local`? Expected: no. Any hit here (titles, compaction, background traffic) must be listed.
- **O8 (capture):** At least one dump file exists and parses: `python3 -c "import json,glob; d=json.load(open(sorted(glob.glob('$HOME/.sous/spikes/dumps/*.json'))[-1])); print(d['model'], len(d.get('tools',[])), len(json.dumps(d.get('system',''))))"` — prints `sous-local`, a non-zero tool count, and a large system size.

---

### Task 4: Document gate 1 on issue #41

**Files:**
- Create: none (a GitHub comment)

**Interfaces:**
- Consumes: observations O1–O8 from Tasks 2–3.
- Produces: the gate 1 verdict comment on #41; unblocks (or kills) everything after.

- [ ] **Step 1: Write the comment**

Post with `gh issue comment 41 --repo krcm0209/sous --body-file <scratch file>`. Structure: **Gate 1 verdict: PASS/KILL**, then O1–O8 each with its observed data (paths, model sets, auth kinds, beta values — the wire data the spec's exit criterion demands), then "Method: `scripts/spikes/gate1_proxy.py` on branch `spike/gateway-gates` at <commit sha>". If the verdict is KILL, state which observation failed and close out: the remaining tasks do not run, and the spec's Phase 1+ is not built.

- [ ] **Step 2: Commit nothing** — this task changes no files.

---

### Task 5: The gate 2 replay harness

**Files:**
- Create: `scripts/spikes/gate2_replay.py`

**Interfaces:**
- Consumes: a dump file from Task 3 (`~/.sous/spikes/dumps/*.json`); `sous.toolexec.ToolExecutor` (existing: `read_file(path, offset, limit)`, `glob(pattern)`, `grep(pattern, glob_pattern)`); `sous.engine.base.release_mlx_thread_state` (existing).
- Produces: a runnable replay (`uv run python scripts/spikes/gate2_replay.py DUMP.json --project-root DIR --turns 5`) printing a per-turn verdict table Task 6 records.

- [ ] **Step 1: Write `scripts/spikes/gate2_replay.py`**

```python
"""Gate 2 spike: replay a captured Claude Code subagent request locally. THROWAWAY.

Feeds a gate-1b dump (real subagent system prompt + real tool schemas) to the
configured sous model and drives a bounded tool loop. Read-only tools (Read,
Glob, Grep) execute for real through sous's ToolExecutor; anything else asks
the operator to paste a result. The operator judges quality; this script
measures parseability and timing.

Success rubric (spec, Phase 0 gate 2): across the requested turns, every
model response contains at least one parseable tool call naming a tool from
the request's schema with its required arguments present. Prose-only turns,
unknown tool names, and unparseable markup are failures worth recording.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# sous internals, reused read-only. Run via `uv run` from the repo root.
from sous.config import load_config
from sous.engine.base import release_mlx_thread_state
from sous.toolexec import ToolExecutor

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def anthropic_to_chat(dump: dict) -> tuple[list[dict], list[dict], int]:
    """Convert the captured Anthropic request to chat-template inputs.

    Returns (messages, tools, dropped_server_side). First subagent requests
    are system + one user turn of text blocks; anything fancier is out of
    scope for the spike and fails loudly.
    """
    messages: list[dict] = []
    system = dump.get("system", "")
    if isinstance(system, list):
        system = "\n\n".join(b["text"] for b in system if b.get("type") == "text")
    if system:
        messages.append({"role": "system", "content": system})
    for msg in dump["messages"]:
        content = msg["content"]
        if isinstance(content, list):
            parts = []
            for block in content:
                if block.get("type") != "text":
                    raise SystemExit(f"unsupported block in capture: {block.get('type')}")
                parts.append(block["text"])
            content = "\n".join(parts)
        messages.append({"role": msg["role"], "content": content})

    tools, dropped = [], 0
    for t in dump.get("tools", []):
        if "input_schema" not in t:  # server-side tool (web_search_*, ...)
            dropped += 1
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        })
    return messages, tools, dropped


def parse_calls(text: str) -> list[tuple[str, dict | None, str]]:
    """Permissive tool-call extraction: (name, args-or-None, error)."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        payload = m.group(1).strip()
        if payload.startswith("{"):
            try:
                obj = json.loads(payload)
                calls.append((str(obj.get("name")), obj.get("arguments"), ""))
            except json.JSONDecodeError as e:
                calls.append(("", None, f"bad json: {e}"))
        elif payload.startswith("<function="):
            name_m = re.match(r"<function=([^>]+)>", payload)
            args = dict(re.findall(
                r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", payload, re.DOTALL))
            calls.append((name_m.group(1) if name_m else "", args, ""))
        else:
            calls.append(("", None, f"unrecognized payload head: {payload[:40]!r}"))
    return calls


def execute(name: str, args: dict, tx: ToolExecutor) -> str:
    """Read-only execution for the three safe Claude tools; operator otherwise."""
    try:
        if name == "Read":
            return tx.read_file(args["file_path"],
                                offset=int(args.get("offset", 0)),
                                limit=int(args.get("limit", 2000)))
        if name == "Glob":
            return tx.glob(args["pattern"])
        if name == "Grep":
            return tx.grep(args["pattern"], args.get("glob", "**/*"))
    except Exception as e:  # spike: any executor error is a result string
        return f"error: {e}"
    print(f"\n== operator input needed for {name}({json.dumps(args)[:200]})")
    print("== paste a plausible tool result, end with a lone '.' line:")
    lines = []
    for line in sys.stdin:
        if line.rstrip("\n") == ".":
            break
        lines.append(line)
    return "".join(lines) or "error: tool unavailable in spike"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--project-root", type=Path, required=True,
                    help="repo the read-only tools run against")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    dump = json.loads(args.dump.read_text())
    messages, tools, dropped = anthropic_to_chat(dump)
    schema_names = {t["function"]["name"] for t in tools}
    print(f"# capture: {len(messages)} messages, {len(tools)} client tools, "
          f"{dropped} server-side tools dropped")

    cfg = load_config()
    tx = ToolExecutor(args.project_root.resolve(),
                      config_path=cfg.config_path, data_dir=cfg.data_dir)

    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    print(f"# loading {cfg.model_id} ...")
    model, tokenizer = load(cfg.model_id)
    sampler = make_sampler(temp=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)

    results = []
    try:
        for turn in range(1, args.turns + 1):
            prompt = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True,
                tokenize=False, enable_thinking=False)
            ids = tokenizer.encode(prompt)
            t0 = time.monotonic()
            text = "".join(
                r.text for r in stream_generate(
                    model, tokenizer, ids,
                    max_tokens=args.max_tokens, sampler=sampler))
            dt = time.monotonic() - t0
            calls = parse_calls(text)
            ok = [c for c in calls if c[0] in schema_names and isinstance(c[1], dict)]
            results.append((turn, len(ids), round(dt, 1), len(calls), len(ok)))
            print(f"\n=== turn {turn}: {len(ids)} prompt tokens, {dt:.1f}s, "
                  f"{len(calls)} calls parsed, {len(ok)} schema-valid")
            print(text[:1500])
            if not ok:
                print("!! no valid tool call this turn — operator judges: continue? [y/N]")
                if input().strip().lower() != "y":
                    break
            messages.append({"role": "assistant", "content": text})
            # Mirror sous's worker convention: results return as user turns.
            blocks = []
            for name, cargs, _err in ok:
                result = execute(name, cargs or {}, tx)
                blocks.append(f'<tool_result name="{name}">\n{result[:8000]}\n</tool_result>')
            if blocks:
                messages.append({"role": "user", "content": "\n".join(blocks)})
    finally:
        release_mlx_thread_state()

    print("\n# turn | prompt_tokens | seconds | calls | schema-valid")
    for row in results:
        print("# " + " | ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the converter and parser without a model** (fast, no weights)

```bash
uv run python - <<'EOF'
import json, sys
sys.path.insert(0, "scripts/spikes")
from gate2_replay import anthropic_to_chat, parse_calls

dump = {"model": "sous-local",
        "system": [{"type": "text", "text": "You are an agent."}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "find config"}]}],
        "tools": [{"name": "Read", "description": "read", "input_schema": {"type": "object"}},
                  {"name": "web_search", "type": "web_search_20250305"}]}
msgs, tools, dropped = anthropic_to_chat(dump)
assert [m["role"] for m in msgs] == ["system", "user"] and dropped == 1, (msgs, dropped)
assert tools[0]["function"]["name"] == "Read"

calls = parse_calls('x<tool_call>{"name":"Read","arguments":{"file_path":"a.py"}}</tool_call>')
assert calls == [("Read", {"file_path": "a.py"}, "")], calls
calls = parse_calls("<tool_call><function=Grep><parameter=pattern>\nfoo\n</parameter></function></tool_call>")
assert calls[0][0] == "Grep" and calls[0][1] == {"pattern": "foo"}, calls
print("converter+parser OK")
EOF
```

Expected: `converter+parser OK`. An AssertionError means the conversion or parsing logic is wrong — fix before burning model time.

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff check scripts/spikes/ && uv run ruff format scripts/spikes/
git add scripts/spikes/gate2_replay.py
git commit -m "spike: gate 2 subagent replay harness (#41)

Replays a captured subagent request against the local model with
read-only tool execution through ToolExecutor; the operator judges
quality, the script measures parseability and timing."
```

---

### Task 6: Run gate 2 against the real model

Manual, on Kyle's machine (multi-GB weights, ~30 GiB unified memory).

**Files:**
- Create: none

**Interfaces:**
- Consumes: the freshest dump from Task 3, `gate2_replay.py` from Task 5.
- Produces: the per-turn results table plus the operator's quality judgment, for Task 7.

- [ ] **Step 1: Run the replay**

```bash
uv run python scripts/spikes/gate2_replay.py \
  "$(ls -t ~/.sous/spikes/dumps/*.json | head -1)" \
  --project-root ~/code/personal/sous --turns 5
```

- [ ] **Step 2: Record, per turn:** prompt tokens, seconds (the first turn's number is the cold-prefill datapoint for a harness-sized prompt — flag it separately), calls parsed vs schema-valid, and the operator's judgment: did the model pursue the actual task, did it use real tool names with sensible arguments, did it derail into prose or hallucinated tools? The spec's bar: "well-formed tool_use across a multi-turn loop." A single malformed turn is data, not automatically a kill — three consecutive is (mirroring the worker's `MAX_CONSECUTIVE_MALFORMED = 3`).

---

### Task 7: Document gate 2 and close out Phase 0

**Files:**
- Create: none (a GitHub comment; branch push)

**Interfaces:**
- Consumes: Task 6's results.
- Produces: the Phase 0 verdict on #41; the pushed `spike/gateway-gates` branch as the reproducibility record.

- [ ] **Step 1: Post the gate 2 comment on #41**

Same shape as Task 4: **Gate 2 verdict: PASS/KILL**, the per-turn table, the cold-prefill wall-clock, the operator judgment, method pointer to the branch + sha. End the comment with the Phase 0 conclusion: either "both gates pass — Phase 1 (spec `2026-08-26-hybrid-gateway-design.md`) is unblocked" or the specific kill rationale.

- [ ] **Step 2: Push the spike branch as a record, without a PR**

```bash
git push -u origin spike/gateway-gates
```

The branch is the reproducibility record referenced from #41. It is deliberately not merged: `scripts/spikes/` is throwaway per the spec, and the plan's Global Constraints say so.

---

## Self-review notes

- Spec coverage: gate 1's three questions map to O1–O5 (Task 2) and O6–O7 (Task 3); the capture requirement to O8; gate 2's replay + rubric to Tasks 5–6; the exit criterion ("documented on #41 with observed wire data") to Tasks 4 and 7; the credential rule is enforced in every launch command via `env -u`.
- The retired gate 3 (internal-traffic contamination) is covered by O7 rather than a separate task, per the spec.
- Interface check done against current source: `ToolExecutor(project_root, config_path, data_dir=None)` (`toolexec.py:288`), `read_file(path, offset=0, limit=2000)`, `glob(pattern)`, `grep(pattern, glob_pattern="**/*")`, `load_config()` returning `config_path`/`data_dir`, and `release_mlx_thread_state` (`engine/base.py:30`) — the plan's code matches these signatures.
