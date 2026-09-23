"""sous CLI: serve / status / top / statusline / stop / claude / install- and
uninstall-launchd."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import plistlib
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from sous.config import load_config

LABEL = "com.sous.daemon"

# What `sous claude` sets, and — as important — what it never sets. Either
# credential variable switches Claude Code from the subscription login to
# API-credit billing (the load-bearing #41 fact); the tier variables would pull
# the main loop onto the local model. oMLX's launcher sets all of them because
# it never forwards anything; sous forwards the main loop, so it must not.
_CREDENTIAL_VARS = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")
# Not set here, but inherited from the shell if the user exported one — the
# whole-session recipe in CONTRIBUTING does exactly that — and then Claude Code
# runs that tier of the MAIN loop on the local model instead of upstream, which
# is the hybrid this launcher exists to set up. Warned about, never stripped:
# the user's environment is the user's, same ruling as _CREDENTIAL_VARS.
_TIER_VARS = (
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)
_DISALLOWED_FLAGS = ("--disallowedTools", "--disallowed-tools")
_LSP_OFF = ["--disallowedTools", "LSP"]
# Invocations that exit at once: holding the model for them would start a
# load nobody waits for and, once the process is gone, restart the idle
# clock under weights nobody is using. Only these four are knowable here,
# and Claude Code honours them after other options too (`--model opus
# --version` prints the version), so they count anywhere before a `--`.
_NO_HOLD_FLAGS = ("--help", "-h", "--version", "-v")
# Model load plus a long prefill: minutes, not the SDK's default.
_API_TIMEOUT_MS = "3000000"
# Nothing about /sous/status or /sous/hold is slow: one reads the engine's
# counters and the turn registry (a millisecond), the other registers a
# pid; neither waits for the model.
_STATUS_TIMEOUT_SECONDS = 15.0
# The status document is tens of KB with its ring of recent turns and a hold
# reply is three fields; a reply past this is not the daemon's.
_REPLY_LIMIT = 1 << 20
_PREDATES_MESSAGE = (
    "sous claude: the running daemon predates this CLI; restart it (sous stop; sous serve)"
)
# Claude Code runs the status-line command about once a second and shows
# whatever it prints; a slow answer is a frozen line, so the whole call —
# the stdin drain and the fetch, on a thread joined for this long — is
# bounded well inside that second.
_STATUSLINE_TIMEOUT_SECONDS = 0.5
_STATUSLINE_DOWN = "sous: daemon down"


def claude_argv(user_args: list[str]) -> list[str]:
    """The user's arguments plus `--disallowedTools LSP` (a language server
    connecting mid-session appends its schema to every request and re-prefills
    the conversation) unless they chose their own. Appended, never prepended:
    the option is variadic, so ahead of a positional prompt it would swallow
    the prompt as a tool name. Ahead of a `--`, which ends Claude Code's
    option parsing, if there is one — and only the arguments before that `--`
    can be the user's own choice, since anything after it is literal text
    (`sous claude -- --disallowedTools` asks for that prompt, not that flag)."""
    end = user_args.index("--") if "--" in user_args else len(user_args)
    for arg in user_args[:end]:
        if arg in _DISALLOWED_FLAGS or arg.startswith(tuple(f"{f}=" for f in _DISALLOWED_FLAGS)):
            return list(user_args)
    return [*user_args[:end], *_LSP_OFF, *user_args[end:]]


def claude_env(
    port: int,
    local_models: Sequence[str],
    max_context_tokens: int,
    base: Mapping[str, str],
) -> dict[str, str]:
    """The inherited environment plus the five variables Claude Code needs.

    They are the RUNNING daemon's values (see _daemon_status), not the config
    file's: pinning subagents to an id the daemon does not serve would send
    every one of them upstream.

    CLAUDE_CODE_MAX_CONTEXT_TOKENS is honoured only for non-claude-* ids, so
    it sizes the local subagent's window and leaves the main loop alone.
    CLAUDE_CODE_AUTO_COMPACT_WINDOW is deliberately NOT set: it is global
    ("the minimum of this setting and your model's maximum context window",
    per Claude Code's own /config copy), so pinning it to the local window
    would make the frontier main loop compact far too early. The subagent's
    threshold is already bounded by its window.

    CLAUDE_CODE_SUBAGENT_MODEL_FORCE: as of Claude Code 2.1.26x,
    CLAUDE_CODE_SUBAGENT_MODEL only sets the *default* subagent model — a
    built-in agent's own `model:` (Explore, for one) or an explicit per-spawn
    model now overrides it, which sent Explore upstream to claude-opus-5 in
    practice. The FORCE variable makes it apply to every subagent regardless.
    """
    env = dict(base)
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    env["CLAUDE_CODE_SUBAGENT_MODEL"] = local_models[0]
    env["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] = "1"
    env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(max_context_tokens)
    env["API_TIMEOUT_MS"] = _API_TIMEOUT_MS
    return env


def _sous_request(port: int, method: str, path: str, json: dict | None = None) -> tuple[int, bytes]:
    """One HTTP call to the daemon on `port`: the status code and the whole
    reply, bounded in wall-clock time and in size. httpx's timeout is per
    phase and its read timeout re-arms on every chunk, so the deadline is
    kept here: what holds the port may not be the daemon, and must not be
    able to park the launcher by dribbling."""
    import httpx

    deadline = time.monotonic() + _STATUS_TIMEOUT_SECONDS
    # 127.0.0.1 is loopback: a proxy variable must never route or see this
    # call — same rule as api/upstream.py's trust_env=False.
    with (
        httpx.Client(timeout=_STATUS_TIMEOUT_SECONDS, trust_env=False) as client,
        client.stream(method, f"http://127.0.0.1:{port}{path}", json=json) as reply,
    ):
        chunks: list[bytes] = []
        size = 0
        for chunk in reply.iter_bytes():
            size += len(chunk)
            if size > _REPLY_LIMIT:
                raise httpx.ReadError(f"reply exceeds {_REPLY_LIMIT} bytes")
            if time.monotonic() > deadline:
                raise httpx.ReadTimeout(f"no complete reply within {_STATUS_TIMEOUT_SECONDS:.0f}s")
            chunks.append(chunk)
        return reply.status_code, b"".join(chunks)


def _json_object(raw: bytes) -> dict | None:
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _no_status_answer(port: int, why: str) -> None:
    print(f"sous claude: 127.0.0.1:{port} did not answer /sous/status ({why})", file=sys.stderr)


def _daemon_status(port: int, data_dir: Path) -> dict | None:
    """What the daemon on `port` reports about itself (GET /sous/status), or
    None if nothing there answers as one. Exits 1 on a 404: from the process
    holding `data_dir`'s daemon lock that is a daemon from before the route
    existed, and the fix is a restart — the CLI and the daemon ship in one
    package, so there is no older dialect to fall back to; from anything
    else it is not sous on the port at all, and a restart would not help.

    The config FILE is not the truth: the daemon loads it once at startup and
    holds that snapshot for its whole life, so an edit since then — a changed
    `local_models`, a wider `max_context_tokens` — is a setting the running
    daemon does not have. /sous/status is where it reports the configuration
    it is actually serving.
    """
    import httpx

    if not _port_open(port):
        return None
    try:
        status, raw = _sous_request(port, "GET", "/sous/status")
    except httpx.HTTPError as exc:
        # The type only: an httpx message can carry the URL it was building.
        _no_status_answer(port, type(exc).__name__)
        return None
    body = _json_object(raw) if status == 200 else None
    if body is not None and not isinstance(body.get("config"), dict):
        # Someone else's JSON object: read as the document it would be a
        # traceback at the first field.
        body = None
    if body is not None:
        return body
    # Whatever the answer was, the lock says whether a daemon is there at
    # all: nothing holding it means the port is another service's, and a
    # restart would only collide with it.
    if not _lock_is_held(data_dir):
        print(
            f"sous claude: 127.0.0.1:{port} is not a sous daemon (nothing holds "
            f"{data_dir / 'daemon.lock'}); stop what listens there, or change [server].port",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if status == 404:
        print(_PREDATES_MESSAGE, file=sys.stderr)
        raise SystemExit(1)
    _no_status_answer(port, f"status {status}" if status != 200 else "not the status document")
    return None


def _hold_warning(why: str) -> None:
    print(
        f"sous claude: warning: could not hold the model ({why}); launching anyway",
        file=sys.stderr,
    )


def _hold(port: int) -> dict | None:
    """Ask the daemon to keep the model loaded while this process lives.
    execve keeps the pid, so the process that leaves is the Claude Code
    session, and its start time tells the daemon a reused pid from a new
    one. Any failure is a warning: the session works without the hold, it
    just pays a reload after an idle unload."""
    import httpx
    import psutil

    try:
        body = {"pid": os.getpid(), "create_time": psutil.Process().create_time()}
        status, raw = _sous_request(port, "POST", "/sous/hold", json=body)
    except (httpx.HTTPError, psutil.Error) as exc:
        # The type only: an httpx message can carry the URL it was building.
        _hold_warning(type(exc).__name__)
        return None
    if status != 200:
        _hold_warning(f"status {status}")
        return None
    answer = _json_object(raw)
    if answer is None:
        _hold_warning("unexpected answer")
        return None
    return answer


def _cmd_claude(user_args: list[str]) -> None:
    """Replace this process with Claude Code pointed at the daemon.

    Every served value Claude Code is given comes from the running daemon;
    the config file supplies the port to reach it on, the path to name in an
    error message, and the two values compared against the daemon's for the
    drift note — never the values themselves.
    """
    config = load_config()
    exe = shutil.which("claude")
    if exe is None:
        print("sous claude: `claude` is not on PATH; install Claude Code first", file=sys.stderr)
        raise SystemExit(1)
    status = _daemon_status(config.server_port, config.data_dir)
    if status is None:
        if _port_open(config.server_port):
            # `sous serve` would only fail on the daemon lock: something is
            # there, it just did not answer as a daemon should.
            print(
                f"sous claude: 127.0.0.1:{config.server_port} is listening but did not answer "
                "as a sous daemon; if it is one, restart it (sous stop; sous serve)",
                file=sys.stderr,
            )
        else:
            print(
                f"sous claude: no daemon on 127.0.0.1:{config.server_port}; start it with: "
                "sous serve   (or: sous install-launchd)",
                file=sys.stderr,
            )
        raise SystemExit(1)
    local_models = status.get("config", {}).get("local_models")
    max_context_tokens = status.get("config", {}).get("max_context_tokens")
    if (
        not isinstance(local_models, list)
        or not local_models
        or type(max_context_tokens) is not int
    ):
        # A 200 with the wrong document: sous reports both keys or neither
        # route, so whatever answered is not the daemon this CLI shipped with.
        print(
            f"sous claude: 127.0.0.1:{config.server_port} answered /sous/status with an "
            "unexpected document; restart the daemon from this install (sous stop; sous serve)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    # The daemon's values win, so a file edited since it started is a note,
    # not a silent difference between what is advertised and what is served.
    drifted = [
        key
        for key, running, on_disk in (
            ("local_models", list(local_models), list(config.local_models)),
            ("max_context_tokens", max_context_tokens, config.max_context_tokens),
        )
        if running != on_disk
    ]
    if drifted:
        print(
            f"sous claude: note: config.toml differs from the running daemon "
            f"({', '.join(drifted)}); using the daemon's values — restart it to apply the file",
            file=sys.stderr,
        )
    for var in _CREDENTIAL_VARS:
        if os.environ.get(var):
            print(
                f"sous claude: warning: {var} is set, so Claude Code will bill the main loop "
                "to API credits instead of your subscription; unset it to use the login",
                file=sys.stderr,
            )
    for var in _TIER_VARS:
        # The value is a model id, not a secret, and it is the whole point of
        # the warning: the user has to recognise their own leftover export.
        if value := os.environ.get(var):
            print(
                f"sous claude: warning: {var}={value} is set in your shell, so that tier of "
                "the main loop will not run upstream; unset it for a hybrid session",
                file=sys.stderr,
            )
    env = claude_env(config.server_port, local_models, max_context_tokens, os.environ)
    end = user_args.index("--") if "--" in user_args else len(user_args)
    exits_at_once = any(arg in _NO_HOLD_FLAGS for arg in user_args[:end])
    held = None if exits_at_once else _hold(config.server_port)
    if held is not None:
        model_id = status.get("engine", {}).get("model_id", "the model")
        if held.get("loaded"):
            what = f"{model_id} already loaded"
        elif held.get("loading"):
            what = f"preloading {model_id}"
        else:
            # Only when the daemon could not start its preload thread — but
            # this is another process's JSON, so never an assumption.
            what = f"{model_id} not loaded yet"
        print(f"sous claude: {what}; held while this session runs", file=sys.stderr)
    print(
        f"sous claude: ANTHROPIC_BASE_URL={env['ANTHROPIC_BASE_URL']} "
        f"CLAUDE_CODE_SUBAGENT_MODEL={env['CLAUDE_CODE_SUBAGENT_MODEL']} "
        f"CLAUDE_CODE_SUBAGENT_MODEL_FORCE={env['CLAUDE_CODE_SUBAGENT_MODEL_FORCE']} "
        f"CLAUDE_CODE_MAX_CONTEXT_TOKENS={env['CLAUDE_CODE_MAX_CONTEXT_TOKENS']} "
        f"API_TIMEOUT_MS={env['API_TIMEOUT_MS']}",
        file=sys.stderr,
    )
    # exec, not a subprocess: the TTY, the signals and the exit code are
    # Claude Code's own from here on.
    os.execve(exe, [exe, *claude_argv(user_args)], env)


def launchd_plist(sous_executable: str, log_dir: Path) -> str:
    # plistlib handles XML escaping — a path containing & or < must still
    # produce a plist launchctl can parse (string formatting silently
    # produced invalid XML while install-launchd reported success).
    # No EnvironmentVariables block on purpose: the daemon runs no commands
    # of its own, so launchd's bare system PATH is all it needs, and it reads
    # no credential or proxy from its environment (api/upstream.py) — a block
    # here would only snapshot the installing shell into the plist.
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": [sous_executable, "serve"],
            "RunAtLoad": True,
            "KeepAlive": True,
            # Both streams to one file, in write order. The stderr file was
            # named daemon.err.log and held everything the daemon said — Python
            # logging, the daemon's own lines, warnings, every library's
            # handler all write to stderr — while daemon.log held the banner. One file,
            # and the level on every line (sous.logs) says what is an error.
            "StandardOutPath": f"{log_dir}/daemon.log",
            "StandardErrorPath": f"{log_dir}/daemon.log",
        },
        sort_keys=False,
    ).decode()


def status_lines(document: dict, now: float) -> list[str]:
    """`sous status` from the status document: the engine, the turns in
    flight and the last five served, newest first. `now` is only for a
    turn's elapsed time, like statusline_text."""
    engine = document.get("engine")
    engine = engine if isinstance(engine, dict) else {}
    config = document.get("config")
    config = config if isinstance(config, dict) else {}
    if engine.get("loading"):
        state = "loading"
    elif engine.get("unloading"):
        state = "unloading"
    elif engine.get("loaded"):
        state = "loaded"
    else:
        state = "unloaded"
    engine_parts = [
        f"engine: {engine.get('model_id', '?')} {state}",
        f"holders {engine.get('holders') or 0}",
    ]
    idle = engine.get("idle_seconds")
    if state == "loaded" and isinstance(idle, int | float):
        engine_parts.append(f"idle {_span(idle)}")
    lines = [
        f"sous daemon: listening on 127.0.0.1:{config.get('port', '?')}",
        "  " + " · ".join(engine_parts),
    ]
    disk = (engine.get("prompt_cache") or {}).get("disk")
    if isinstance(disk, dict) and disk.get("state") == "active":
        gb = (disk.get("bytes") or 0) / (1 << 30)
        lines.append(f"  forks on disk: {disk.get('forks', 0)} · {gb:.1f} GB")
    elif isinstance(disk, dict) and disk.get("state") == "unavailable":
        lines.append(f"  forks on disk: unavailable ({disk.get('reason') or '?'})")
    # Shape-checked entry by entry: whatever answered on the port is read
    # here, and a wrong element must not become a traceback.
    inflight = [t for t in _entries(document.get("inflight")) if isinstance(t, dict)]
    if inflight:
        for turn in inflight:
            since = turn.get("started_at")
            age = f" {_clock(now - since)}" if isinstance(since, int | float) else ""
            lines.append(
                f"  turn {turn.get('id', '?')} {turn.get('model', '?')} "
                f"{turn.get('phase', 'queued')}{age}"
            )
    else:
        lines.append("  turns in flight: none")
    # The ring is newest first already.
    recent = [s for s in _entries(document.get("recent_turns")) if isinstance(s, dict)][:5]
    if recent:
        lines.append("  recent turns:")
        lines.extend("    " + _recent_turn_text(s) for s in recent)
    return lines


def _entries(value: object) -> list:
    return value if isinstance(value, list) else []


def _recent_turn_text(s: dict) -> str:
    seconds = s.get("seconds")
    took = f"{seconds:.1f}s" if isinstance(seconds, int | float) else "-"
    head = f"{s.get('id', '?')} {s.get('model', '?')} status={s.get('status', '?')}"
    if s.get("error"):
        # A refused, abandoned or failed turn: its summary carries the error
        # and none of the served fields.
        return f"{head} error={s['error']} {took}"
    return (
        f"{head} stop={s.get('stop_reason') or '-'} cache={s.get('cache') or '-'} "
        f"in={s.get('input_tokens', '-')} out={s.get('output_tokens', '-')} {took}"
    )


def _cmd_status() -> None:
    config = load_config()
    if not _port_open(config.server_port):
        print(f"sous daemon: not running (port {config.server_port})")
        print("start it with: sous serve   (or: sous install-launchd)")
        return
    import httpx

    try:
        status, raw = _sous_request(config.server_port, "GET", "/sous/status")
    except httpx.HTTPError as e:
        print(
            f"sous daemon: port {config.server_port} did not answer /sous/status "
            f"({type(e).__name__}); is it sous?"
        )
        raise SystemExit(1) from None
    document = _json_object(raw) if status == 200 else None
    if document is not None and not (
        isinstance(document.get("engine"), dict) and isinstance(document.get("config"), dict)
    ):
        # A JSON object without the engine block is some other service on the
        # port; read as a document it would print a confident report about a
        # daemon that isn't there.
        document = None
    if document is None:
        # Whatever the answer was — a 404, a 503, HTML, someone else's JSON —
        # the lock says whether there is a daemon to restart: nothing holding
        # it means the port is another service's, and a restart would only
        # collide with it.
        if not _lock_is_held(config.data_dir):
            print(
                f"sous daemon: 127.0.0.1:{config.server_port} is not a sous daemon (nothing "
                f"holds {config.data_dir / 'daemon.lock'}); stop what listens there, or "
                "change [server].port"
            )
            raise SystemExit(1)
        answer = (
            "/sous/status with something that is not the status document"
            if status == 200
            else f"{status} to /sous/status"
        )
        print(
            f"sous daemon: port {config.server_port} answered {answer}; "
            f"restart it ({restart_hint(managed=_launchd_loaded(LABEL))})"
        )
        raise SystemExit(1)
    for line in status_lines(document, now=time.time()):
        print(line)


def _span(seconds: float) -> str:
    """`48s`, `4m 12s`, `1h 03m`: the idle span, in the shape `sous top` gives
    it (span_text lives in tui.py, which this module must not import)."""
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def statusline_text(document: dict, now: float) -> str:
    """One line for Claude Code's status bar from the status document: the
    turn in flight if there is one, else the engine. Every number is the
    daemon's; `now` is only for the elapsed time of a turn."""
    engine = document.get("engine") or {}
    inflight = document.get("inflight") or []
    if not inflight:
        if not engine.get("loaded"):
            return "sous: loading" if engine.get("loading") else "sous: idle · model unloaded"
        parts = ["sous: idle"]
        slots = (engine.get("prompt_cache") or {}).get("slots")
        if slots is not None:
            parts.append(f"{slots} slots")
        if engine.get("holders"):
            parts.append("held")
        return " · ".join(parts)
    turn = inflight[0]
    phase = turn.get("phase", "queued")
    # A turn has a size — and an ETA worth printing — once it decodes, or
    # prefills a render the cache has measured.
    sized = phase == "decode" or (phase == "prefill" and bool(turn.get("to_prefill")))
    head = f"sous: {phase}"
    parts: list[str] = []
    if phase == "decode":
        head = f"{head} {turn.get('generated_tokens', 0):,} tok"
        if turn.get("decode_tps"):
            parts.append(f"{turn['decode_tps']:.1f} tok/s")
    elif sized:
        head = f"{head} {turn['to_prefill']:,} tok"
    eta = turn.get("eta_seconds")
    if eta is not None and sized:
        parts.append(f"eta {int(eta)}s")
    elif turn.get("started_at") is not None:
        parts.append(_clock(now - turn["started_at"]))
    if len(inflight) > 1:
        parts.append(f"+{len(inflight) - 1} queued")
    return " · ".join([head, *parts])


def _statusline_fetch(port: int) -> str | None:
    """The line for the status bar, or None when nothing on `port` answered
    as the daemon. Runs on a thread of its own: nothing here carries a
    deadline of its own — a stdin held open blocks the drain, and urllib's
    timeout re-arms on every byte, so a listener that dribbles never
    returns — and the thread is what the budget is applied to."""
    import urllib.request

    # Claude Code writes its own JSON (model, cwd, cost) to stdin; none of
    # it is needed here, but a pipe nobody reads can block the writer. A
    # stdin that cannot be read (a closed handle, pytest's capture) is as
    # harmless as an empty one.
    with contextlib.suppress(OSError, ValueError):
        if not sys.stdin.isatty():
            sys.stdin.read()
    # No proxy, whatever the environment says: 127.0.0.1 is loopback, the
    # same rule as the launcher's httpx client and the daemon's forwarder.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{port}/sous/status"
    try:
        with opener.open(url, timeout=_STATUSLINE_TIMEOUT_SECONDS) as reply:
            document = _json_object(reply.read(_REPLY_LIMIT))
    except Exception:  # noqa: BLE001 — whatever holds the port is not the daemon
        # urllib's URLError and HTTPError are OSErrors, but http.client's own
        # BadStatusLine and IncompleteRead — a listener that is not HTTP, a
        # body cut short — are not, and anything escaping this thread would
        # print a traceback beside the line.
        return None
    if document is None or "engine" not in document:
        # A JSON object without the engine block is some other service on the
        # port, and reading it as a document would print a confident
        # `sous: idle` about a daemon that isn't there.
        return None
    return statusline_text(document, time.time())


def _cmd_statusline() -> None:
    config = load_config()
    answer: list[str | None] = []
    worker = threading.Thread(
        target=lambda: answer.append(_statusline_fetch(config.server_port)), daemon=True
    )
    worker.start()
    # The wall-clock bound. A worker still parked past it is left to the
    # process exit — a daemon thread — and the bar says the daemon is down.
    worker.join(_STATUSLINE_TIMEOUT_SECONDS)
    line = answer[0] if answer else None
    print(_STATUSLINE_DOWN if line is None else line)


def _has_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _cmd_top() -> None:
    """The terminal. Textual is imported here and nowhere else in sous, so
    the daemon and every other subcommand run without it."""
    config = load_config()
    if not _has_terminal():
        # Textual would draw the pass into the pipe and wait for a key that
        # never comes: an agent's shell, or a redirect, gets told instead.
        print("sous top: needs a terminal (try sous status, or sous statusline)", file=sys.stderr)
        raise SystemExit(2)
    from sous.tui import run_top

    raise SystemExit(run_top(config.server_port))


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


_UNLOAD_GRACE_SECONDS = 5.0


def _await_port_closed(port: int, seconds: float) -> bool:
    """Wait up to `seconds` for the port to stop accepting."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _port_open(port):
            return True
        time.sleep(0.2)
    return not _port_open(port)


# How long install-launchd waits for the daemon it just unloaded to let go of
# the lock. Past the port closing, that daemon still runs uvicorn's graceful
# bound (server.GRACEFUL_SHUTDOWN_SECONDS), the turn runner's close and the
# teardown of a resident model — well past _UNLOAD_GRACE_SECONDS.
_DAEMON_EXIT_SECONDS = 30.0


def _await_lock_free(data_dir: Path, seconds: float) -> bool:
    """Wait up to `seconds` for the daemon lock to be free.

    The port closes as soon as launchd's listener stops accepting, but a
    daemon unloading a resident model is still running lifespan shutdown —
    and still holding the flock — for seconds after that. Poll the lock
    itself rather than the port, since the lock is what the caller actually
    needs free.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _lock_is_held(data_dir):
            return True
        time.sleep(0.2)
    return not _lock_is_held(data_dir)


def _launchd_loaded(label: str) -> bool:
    """Whether launchd is currently managing the daemon.

    Absent or unusable launchctl reads as "not managed": the caller then falls
    back to signalling directly, which is the right answer on a machine where
    launchd is not in the picture.
    """
    try:
        listed = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=10)
    except OSError, subprocess.SubprocessError:
        return False
    return any(line.split("\t")[-1] == label for line in listed.stdout.splitlines())


# launchctl's exit code for a service target that is not loaded. Every other
# nonzero code is a real failure and must not be mistaken for "already gone".
_BOOTOUT_NOT_LOADED = 3


def _bootout(label: str) -> int:
    """Unload the job by service target, returning launchctl's exit code.

    By label rather than by plist path: the plist may have been deleted by hand
    while the label is still bootstrapped, and that is precisely the state
    where uninstalling has to do something.
    """
    try:
        done = subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError, subprocess.SubprocessError:
        return _BOOTOUT_NOT_LOADED  # no launchctl here: nothing to unload
    return done.returncode


def _daemon_pid(data_dir: Path) -> int | None:
    """The pid the running daemon recorded when it took its lock."""
    try:
        return int((data_dir / "daemon.lock").read_text().strip())
    except OSError, ValueError:
        return None


def _lock_is_held(data_dir: Path) -> bool:
    """Whether a live daemon currently holds the lock.

    daemon.lock outlives the daemon that wrote it, so the pid inside proves
    nothing on its own — the OS may have recycled it onto an unrelated process,
    and a port probe only shows that *something* is listening. The flock is the
    authoritative signal: if we can take it, nobody is holding it.
    """
    try:
        handle = (data_dir / "daemon.lock").open("r+b")
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        handle.close()  # releases anything we just took
    return False


def restart_hint(*, managed: bool) -> str:
    """The command that restarts the daemon — launchd's kickstart when it
    manages the daemon, `sous stop` then `sous serve` otherwise. One spelling
    for `sous stop`, `sous tune` and the docs: the label and the launchd
    domain live here, not in every message that names them."""
    if managed:
        return f"launchctl kickstart -k gui/{os.getuid()}/{LABEL}"
    return "sous stop, then sous serve"


def _cmd_stop() -> None:
    config = load_config()
    # launchd first, deliberately. A KeepAlive job can be loaded while its
    # daemon is mid-restart; checking the port first would report "not running"
    # and exit 0 in exactly that window, while launchd brings it straight back.
    if _launchd_loaded(LABEL):
        print("sous daemon: managed by launchd, which would restart it immediately")
        print("  remove it:  sous uninstall-launchd")
        print(f"  restart it: {restart_hint(managed=True)}")
        raise SystemExit(1)
    if not _port_open(config.server_port):
        print(f"sous daemon: not running (port {config.server_port})")
        return
    pid = _daemon_pid(config.data_dir)
    if pid is None:
        print(f"sous daemon: listening on {config.server_port}, but no pid in daemon.lock")
        print("  stop it by hand, or upgrade the running daemon to one that records it")
        raise SystemExit(1)
    if not _lock_is_held(config.data_dir):
        print(f"sous: daemon.lock names pid {pid} but nothing holds the lock")
        print(f"  something else is on port {config.server_port}; not signalling a stale pid")
        raise SystemExit(1)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"sous: pid {pid} is already gone")
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _port_open(config.server_port):
        time.sleep(0.2)
    if _port_open(config.server_port):
        print(f"sous daemon: sent SIGTERM to {pid} but port {config.server_port} is still open")
        raise SystemExit(1)
    print(f"sous daemon: stopped (pid {pid})")


def _cmd_uninstall_launchd() -> None:
    plist_path = _plist_path()
    # Unload first and unconditionally: a plist deleted by hand leaves the label
    # bootstrapped and the daemon running, which is exactly the state where
    # returning early would claim to have uninstalled something and not have.
    code = _bootout(LABEL)
    if code not in (0, _BOOTOUT_NOT_LOADED):
        # Anything else (permissions, launchctl error) leaves the KeepAlive job
        # alive; deleting the plist here would report success over a live daemon.
        print(f"sous: launchctl bootout failed (exit {code}); leaving {plist_path} in place")
        raise SystemExit(1)

    had_plist = plist_path.exists()
    plist_path.unlink(missing_ok=True)
    if had_plist:
        print(f"removed {plist_path}")
    if code == 0:
        print("unloaded the launchd agent; it will no longer start at login")
    elif not had_plist:
        print(f"sous: launchd agent not installed ({plist_path})")
        return

    config = load_config()
    if code == 0:
        # bootout terminates the job it just unloaded, so checking the port
        # straight away races that teardown: the daemon is still accepting for
        # a moment, and advising `sous stop` sends the user after a process
        # that is already exiting — they run it and are told "not running",
        # which reads as bad advice rather than early advice. Only a daemon
        # that outlives the unload was never launchd's to begin with.
        _await_port_closed(config.server_port, _UNLOAD_GRACE_SECONDS)
    if _port_open(config.server_port):
        print("the running daemon is now unmanaged; stop it with: sous stop")


def _fold_legacy_stderr_log(data_dir: Path) -> bool:
    """Append a pre-one-file `daemon.err.log` onto `daemon.log` and remove it.

    The history is worth keeping; the name is not — it was launchd's stderr
    path, not an error log, and it will never be written again. Byte for
    byte, so nothing in it is reformatted. Only ever called with no daemon
    holding the lock: an open descriptor would keep writing into the unlinked
    inode and those lines would be lost."""
    legacy = data_dir / "daemon.err.log"
    if not legacy.exists():
        return False
    target = data_dir / "daemon.log"
    if target.exists() and os.path.samefile(legacy, target):
        # One file under both names, linked by hand: appending it to itself
        # would read back its own writes until the disk filled. Drop the old
        # name only where that cannot take the data with it — not when
        # daemon.log is the link and daemon.err.log the only real copy.
        if target.is_symlink() or not (legacy.is_symlink() or legacy.stat().st_nlink > 1):
            print(f"{legacy.name} and {target.name} are already one file; left as they are")
            return False
        try:
            legacy.unlink()
        except OSError as exc:
            print(f"sous: could not remove {legacy.name}, a link to {target.name} ({exc})")
            raise SystemExit(1) from None
        print(f"removed {legacy.name}, a link to {target.name}")
        return True
    try:
        with legacy.open("rb") as src, target.open("ab") as dst:
            shutil.copyfileobj(src, dst)
            # The appended bytes are only in the page cache at this point,
            # while the unlink below is journaled metadata; without forcing
            # them out first, a panic between the two loses the folded
            # history even though the source is already gone.
            dst.flush()
            os.fsync(dst.fileno())
    except OSError as exc:
        # No rollback attempt: a partial append may already be in target, so a
        # retry after the underlying problem (full disk, an unwritable target)
        # is fixed will duplicate that prefix — an accepted cost of not losing
        # anything, since the legacy file is left in place either way.
        print(
            f"sous: could not fold {legacy.name} into {target.name} ({exc}); "
            f"{legacy.name} left in place"
        )
        raise SystemExit(1) from None
    try:
        legacy.unlink()
    except OSError as exc:
        # The append above already landed and was fsynced, so — same stance
        # as the copy failure — there is nothing to roll back. Unlike that
        # case, the fold itself succeeded here; say so, and name what a
        # human needs to do before the next run appends this file's bytes
        # a second time, rather than let the exception escape as a traceback.
        print(
            f"sous: folded {legacy.name} into {target.name}, but could not remove "
            f"{legacy.name} ({exc}); delete it by hand before re-running or its "
            f"contents will be appended twice"
        )
        raise SystemExit(1) from None
    print(f"folded {legacy.name} into {target.name}")
    return True


def _cmd_install_launchd() -> None:
    config = load_config()
    exe = shutil.which("sous") or sys.argv[0]
    plist_path = _plist_path()
    # Out first, unconditionally: `bootstrap` fails on a label that is already
    # loaded, so a re-run (a new plist — this release moved both log streams
    # to daemon.log — or a moved executable) never applied. Same rule and
    # same helper as uninstall: anything but "unloaded" or "was not loaded"
    # leaves a live KeepAlive job, and writing a plist over it would report
    # success for a daemon still running the old one.
    code = _bootout(LABEL)
    # launchctl answers a job it cannot find with codes other than
    # _BOOTOUT_NOT_LOADED too — a domain error over SSH with no GUI login,
    # for one — so only a job launchd still lists makes a failure fatal.
    if code not in (0, _BOOTOUT_NOT_LOADED) and _launchd_loaded(LABEL):
        print(f"sous: launchctl bootout failed (exit {code}); not touching {plist_path}")
        raise SystemExit(1)
    # Nothing below reloads the job a successful bootout just unloaded, so
    # every failure from here on has to say so: otherwise the daemon, and
    # every `sous claude` session's upstream with it, is simply gone.
    unloaded = "; the launchd job is unloaded until install-launchd succeeds" if code == 0 else ""
    if code == 0 and not _await_lock_free(config.data_dir, _DAEMON_EXIT_SECONDS):
        print(
            f"sous: the unloaded daemon still holds the lock after {_DAEMON_EXIT_SECONDS:.0f}s"
            f"{unloaded}; retry once it exits"
        )
        raise SystemExit(1)
    if _lock_is_held(config.data_dir):
        # A daemon started by hand (`sous serve`): folding a log out from
        # under it would lose lines.
        print(f"sous: a daemon still holds the lock{unloaded}; stop it first: sous stop")
        raise SystemExit(1)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        _fold_legacy_stderr_log(config.data_dir)
    except SystemExit:
        if unloaded:
            print(unloaded.removeprefix("; "))
        raise
    try:
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(launchd_plist(exe, config.data_dir))
    except OSError as exc:
        print(f"sous: could not write {plist_path} ({exc}){unloaded}")
        raise SystemExit(1) from None
    print(f"wrote {plist_path}")
    cmd = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)]
    try:
        subprocess.run(cmd, check=True)
        print("daemon loaded; it will start at login and stay alive")
    except subprocess.CalledProcessError, FileNotFoundError:
        # Bootout above is unconditional now, so a failed load here is worse
        # than the pre-existing state: exiting 0 would report success with no
        # daemon running at all, not the previously-loaded job left in place.
        print("could not load automatically; run manually:")
        print("  " + " ".join(cmd))
        raise SystemExit(1) from None


def _positive_int(text: str) -> int:
    """0 or fewer suite runs per task would measure nothing while still
    reporting a clean exit: reject it as a usage error before the suite
    starts rather than silently produce an empty comparison."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv: list[str] | None = None) -> None:
    raw = sys.argv[1:] if argv is None else argv
    if raw[:1] == ["claude"]:
        # Before argparse: every argument after `claude` is Claude Code's,
        # including a leading `-p` or `--help`, which argparse.REMAINDER would
        # reject as an unknown option of ours.
        _cmd_claude(raw[1:])
        return
    parser = argparse.ArgumentParser(prog="sous", description="local MLX sous-chef for Claude")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the daemon (the endpoint on 127.0.0.1)")
    status = sub.add_parser("status", help="the daemon, its engine and the recent turns")
    status.add_argument("--watch", action="store_true", help="the live terminal (sous top)")
    sub.add_parser(
        "statusline",
        help="one line for Claude Code's statusLine setting (reads and ignores its stdin JSON)",
    )
    sub.add_parser("top", help="watch the pass live: the order on it, the line, the recent orders")
    sub.add_parser("stop", help="stop the daemon (unmanaged daemons only)")
    sub.add_parser("install-launchd", help="install start-at-login LaunchAgent")
    sub.add_parser("uninstall-launchd", help="remove the start-at-login LaunchAgent")
    tune = sub.add_parser(
        "tune",
        help="measure this machine, grade the candidates and propose [model] settings "
        "(--quick: throughput only — drafter, block size, window)",
    )
    tune.add_argument(
        "--quick", action="store_true", help="the throughput stage only; never changes the model"
    )
    tune.add_argument(
        "--models", nargs="+", metavar="ID", help="candidate ids instead of the table"
    )
    tune.add_argument(
        "--runs", type=_positive_int, default=2, help="suite runs per task (full run)"
    )
    tune.add_argument("--repeat", type=int, default=2, help="attempts per measurement (best wins)")
    tune.add_argument("--resume", metavar="RUN_ID", help="continue a run under ~/.sous/tune")
    tune.add_argument("--yes", action="store_true", help="approve every download and apply")
    tune.add_argument("--apply", action="store_true", help="apply the diff without asking")
    # Registered for `sous --help` only: the verb is dispatched at the top of
    # main(), before argparse ever sees it, so there is no `claude` branch below.
    sub.add_parser(
        "claude",
        help="launch Claude Code with its subagents served locally and the main loop "
        "upstream (every following argument passes through to claude)",
    )
    args = parser.parse_args(raw)
    if args.command == "serve":
        from sous.server import main as serve_main

        serve_main()
    elif args.command == "status":
        _cmd_top() if args.watch else _cmd_status()
    elif args.command == "top":
        _cmd_top()
    elif args.command == "statusline":
        _cmd_statusline()
    elif args.command == "stop":
        _cmd_stop()
    elif args.command == "install-launchd":
        _cmd_install_launchd()
    elif args.command == "uninstall-launchd":
        _cmd_uninstall_launchd()
    elif args.command == "tune":
        from sous.tune import main as tune_main

        raise SystemExit(tune_main(args))
