import plistlib
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sous.cli import LABEL, launchd_plist


def _free_cli_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_plist_is_valid_and_correct():
    xml = launchd_plist("/usr/local/bin/sous", Path("/Users/x/.sous"))
    data = plistlib.loads(xml.encode())
    assert data["Label"] == LABEL
    assert data["ProgramArguments"] == ["/usr/local/bin/sous", "serve"]
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert data["StandardOutPath"] == "/Users/x/.sous/daemon.log"
    assert data["StandardErrorPath"] == "/Users/x/.sous/daemon.err.log"


def test_plist_sets_no_environment_variables():
    """PATH is deliberately NOT baked into the plist: the daemon adopts the
    user's login-shell PATH at startup (server._login_shell_path), which stays
    current and works however the daemon was launched. An install-time PATH
    snapshot here would be a second, staler mechanism that shadows it."""
    xml = launchd_plist("/Users/x/.local/bin/sous", Path("/Users/x/.sous"))
    data = plistlib.loads(xml.encode())
    assert "EnvironmentVariables" not in data


def test_plist_escapes_xml_special_characters():
    """C5: a valid macOS path containing & or < must yield a plist that
    launchctl can actually parse — not silently-invalid XML."""
    xml = launchd_plist("/opt/a&b/sous", Path("/Users/x/My & <Special> Docs/.sous"))
    data = plistlib.loads(xml.encode())
    assert data["ProgramArguments"] == ["/opt/a&b/sous", "serve"]
    assert data["StandardOutPath"] == "/Users/x/My & <Special> Docs/.sous/daemon.log"
    assert data["StandardErrorPath"] == "/Users/x/My & <Special> Docs/.sous/daemon.err.log"


def test_status_reports_not_running(tmp_path, capsys, monkeypatch):
    from sous import cli
    from sous.config import SousConfig

    cfg = SousConfig(
        server_port=1,
        data_dir=tmp_path,  # port 1: never listening
        config_path=tmp_path / "c.toml",
    )
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "not running" in out


def test_mcp_subcommand_dispatches_to_the_proxy(monkeypatch):
    """`sous mcp` must reach the proxy and propagate its exit code, so a failed
    cold start surfaces to the launching client instead of exiting 0."""
    import sous.proxy
    from sous.cli import main

    called = {}

    def fake_run():
        called["ran"] = True
        return 1

    monkeypatch.setattr(sous.proxy, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        main(["mcp"])
    assert called.get("ran") is True
    assert exc.value.code == 1


# --- stop / uninstall-launchd -------------------------------------------------


def _cfg(tmp_path: Path, port: int):
    from sous.config import SousConfig

    return SousConfig(server_port=port, data_dir=tmp_path, config_path=tmp_path / "c.toml")


def _fake_daemon(tmp_path: Path, port: int, hold_lock: bool = True):
    """A stand-in that holds the port, the flock, and its pid, like the real one.

    Holding the port matters: `stop` waits for it to close, so a victim that
    does not own it would let the test pass against code that never signalled.
    Holding the flock matters too — that is how `stop` tells a live daemon from
    a leftover pid file. `hold_lock=False` fakes the stale-pid case.
    """
    # It must ACCEPT, not just listen: `stop` probes the port repeatedly while
    # waiting for it to close, and an unaccepted backlog fills up and starts
    # refusing — which reads as "daemon not running" and hides the real result.
    lock_line = (
        f"import fcntl;lk=open({str(tmp_path / 'daemon.lock')!r},'r+b');"
        f"fcntl.flock(lk.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        if hold_lock
        else ""
    )
    if hold_lock:
        (tmp_path / "daemon.lock").touch()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            lock_line + f"import socket;s=socket.socket();"
            f"s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
            f"s.bind(('127.0.0.1',{port}));s.listen(64)\n"
            f"while True:\n    c,_=s.accept()\n    c.close()",
        ]
    )
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                break
        except OSError:
            time.sleep(0.1)
    else:  # never connected — fail here, not later with a confusing symptom
        proc.kill()
        raise AssertionError(f"fake daemon never listened on {port}")
    (tmp_path / "daemon.lock").write_text(f"{proc.pid}\n")
    return proc


def test_stop_reports_when_not_running(tmp_path, capsys, monkeypatch):
    from sous import cli

    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))  # port 1: never up
    # Never consult the host's real launchd: this machine may well have the
    # agent loaded, which would make the result depend on the developer's box.
    monkeypatch.setattr(cli, "_launchd_loaded", lambda label: False)
    cli.main(["stop"])
    assert "not running" in capsys.readouterr().out


def test_stop_refuses_when_launchd_manages_the_daemon(tmp_path, capsys, monkeypatch):
    """Under KeepAlive a signal is undone in a second, so refuse and say why."""
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)
    try:
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_launchd_loaded", lambda label: True)
        with pytest.raises(SystemExit) as exc:
            cli.main(["stop"])
        assert exc.value.code != 0
        out = capsys.readouterr().out + capsys.readouterr().err
        assert "uninstall-launchd" in out
        assert daemon.poll() is None, "signalled a launchd-managed daemon anyway"
    finally:
        daemon.kill()


def test_stop_signals_the_recorded_pid(tmp_path, capsys, monkeypatch):
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)
    try:
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_launchd_loaded", lambda label: False)
        cli.main(["stop"])
        daemon.wait(timeout=10)
        assert daemon.returncode is not None
    finally:
        daemon.kill()


def test_uninstall_launchd_boots_out_and_removes_the_plist(tmp_path, capsys, monkeypatch):
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"
    plist.write_text("<plist/>")
    booted = []
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli, "_bootout", lambda label: (booted.append(label), 0)[1])
    cli.main(["uninstall-launchd"])
    assert booted == [cli.LABEL]
    assert not plist.exists()


def test_uninstall_launchd_is_idempotent(tmp_path, capsys, monkeypatch):
    """Running it twice, or with nothing installed, must not blow up."""
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"  # never created
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli, "_bootout", lambda label: cli._BOOTOUT_NOT_LOADED)
    cli.main(["uninstall-launchd"])
    assert "not installed" in capsys.readouterr().out.lower()


def test_stop_checks_launchd_before_the_port(tmp_path, capsys, monkeypatch):
    """A KeepAlive job can be loaded while its daemon is mid-restart.

    Checking the port first reports "not running" and exits 0 in that window,
    even though launchd is about to bring it back — which breaks the
    unmanaged-only contract exactly when it matters.
    """
    from sous import cli

    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))  # port down
    monkeypatch.setattr(cli, "_launchd_loaded", lambda label: True)
    with pytest.raises(SystemExit) as exc:
        cli.main(["stop"])
    assert exc.value.code != 0
    assert "launchd" in capsys.readouterr().out


def test_stop_refuses_a_stale_pid_file(tmp_path, capsys, monkeypatch):
    """daemon.lock outlives the daemon, so the pid in it may be recycled.

    Something is listening and a pid is recorded, but nobody holds the lock —
    signalling that pid could hit an unrelated process.
    """
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port, hold_lock=False)
    try:
        (tmp_path / "daemon.lock").write_text("999999\n")  # not the listener
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_launchd_loaded", lambda label: False)
        with pytest.raises(SystemExit) as exc:
            cli.main(["stop"])
        assert exc.value.code != 0
        assert daemon.poll() is None, "signalled despite a stale lock"
    finally:
        daemon.kill()


def test_stop_warns_about_all_active_tasks(tmp_path, capsys, monkeypatch):
    """The warning must cover every state recover_interrupted() will fail.

    list_recent() caps at 20 rows, so a long-running task drops off the list
    once 20 newer ones are queued, and AWAITING_APPROVAL was missed entirely.
    """
    from sous import cli
    from sous.tasks import TaskStore

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)
    try:
        store = TaskStore(tmp_path / "tasks.db")
        old = store.enqueue("long-runner", "x", "/tmp", [], [])
        store.claim_next()
        approving = store.enqueue("needs-approval", "x", "/tmp", [], [])
        store.claim_next()
        store.request_approval(approving.id, "pytest")
        for i in range(25):  # push the running task off list_recent()
            store.enqueue(f"filler{i}", "x", "/tmp", [], [])
        still_running = store.get(old.id)
        assert still_running is not None and still_running.state == "running"

        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_launchd_loaded", lambda label: False)
        cli.main(["stop"])
        out = capsys.readouterr().out
        assert "running" in out and "awaiting_approval" in out
    finally:
        daemon.kill()


def test_uninstall_keeps_the_plist_when_bootout_really_fails(tmp_path, capsys, monkeypatch):
    """Only "not loaded" is benign. A permissions error must not be reported as
    success with the plist deleted and the KeepAlive job still running."""
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"
    plist.write_text("<plist/>")
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli, "_bootout", lambda label: 5)  # I/O error, not "absent"
    with pytest.raises(SystemExit) as exc:
        cli.main(["uninstall-launchd"])
    assert exc.value.code != 0
    assert plist.exists(), "deleted the plist despite a failed bootout"


def test_uninstall_boots_out_even_when_the_plist_is_gone(tmp_path, capsys, monkeypatch):
    """A hand-deleted plist does not unload the job; the label can still be
    bootstrapped, so uninstall must target the label regardless."""
    from sous import cli

    plist = tmp_path / "com.sous.daemon.plist"  # never created
    booted = []
    monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, 1))
    monkeypatch.setattr(cli, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli, "_bootout", lambda label: (booted.append(label), 0)[1])
    cli.main(["uninstall-launchd"])
    assert booted == [cli.LABEL], "did not attempt bootout without a plist"


def test_uninstall_does_not_advise_stop_for_a_daemon_it_just_unloaded(
    tmp_path, capsys, monkeypatch
):
    """bootout terminates the job, so an immediate port check races its exit.

    Advising `sous stop` then names a daemon that is already going away, and
    running it reports "not running" — which reads as though the advice was
    wrong rather than merely early.
    """
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)
    plist = tmp_path / "com.sous.daemon.plist"
    plist.write_text("<plist/>")

    def bootout_then_it_dies(label):
        threading.Timer(0.8, daemon.kill).start()  # launchd tears it down shortly after
        return 0

    try:
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_plist_path", lambda: plist)
        monkeypatch.setattr(cli, "_bootout", bootout_then_it_dies)
        cli.main(["uninstall-launchd"])
        out = capsys.readouterr().out
        assert "unloaded the launchd agent" in out
        assert "sous stop" not in out, "advised stopping a daemon that was already exiting"
    finally:
        daemon.kill()


def test_uninstall_still_advises_stop_when_a_daemon_outlives_the_unload(
    tmp_path, capsys, monkeypatch
):
    """The wait must not swallow the case the advice exists for.

    Goes through the successful-unload branch on purpose: that is the one the
    wait was added to, so a regression suppressing the advice whenever bootout
    succeeds has to fail here. The grace is shortened rather than waited out,
    and the elapsed time is asserted so the timeout path is genuinely taken
    instead of the daemon merely looking absent.
    """
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)  # nothing kills this one
    plist = tmp_path / "com.sous.daemon.plist"
    plist.write_text("<plist/>")

    try:
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_plist_path", lambda: plist)
        monkeypatch.setattr(cli, "_bootout", lambda label: 0)  # unload succeeded
        monkeypatch.setattr(cli, "_UNLOAD_GRACE_SECONDS", 0.5)
        started = time.monotonic()
        cli.main(["uninstall-launchd"])
        elapsed = time.monotonic() - started

        out = capsys.readouterr().out
        assert "unloaded the launchd agent" in out
        assert "sous stop" in out, "suppressed the advice for a daemon that survived"
        assert elapsed >= 0.5, f"returned in {elapsed:.2f}s — the wait never ran"
    finally:
        daemon.kill()


def test_uninstall_advises_stop_when_there_was_nothing_to_unload(tmp_path, capsys, monkeypatch):
    """No unload happened, so nothing was going to terminate the daemon and the
    advice applies immediately — the branch the wait must not reach."""
    from sous import cli

    port = _free_cli_port()
    daemon = _fake_daemon(tmp_path, port)
    plist = tmp_path / "com.sous.daemon.plist"
    plist.write_text("<plist/>")

    try:
        monkeypatch.setattr(cli, "load_config", lambda: _cfg(tmp_path, port))
        monkeypatch.setattr(cli, "_plist_path", lambda: plist)
        monkeypatch.setattr(cli, "_bootout", lambda label: cli._BOOTOUT_NOT_LOADED)
        cli.main(["uninstall-launchd"])
        assert "sous stop" in capsys.readouterr().out
    finally:
        daemon.kill()


# --- wait ----------------------------------------------------------------------


def _wait_store(tmp_path, monkeypatch):
    from sous import cli
    from sous.config import SousConfig
    from sous.tasks import TaskStore

    cfg = SousConfig(data_dir=tmp_path, config_path=tmp_path / "c.toml")
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    return TaskStore(tmp_path / "tasks.db")


def _enqueue(store):
    return store.enqueue(
        title="t", instructions="x", project_root="/", context_files=[], verify_commands=[]
    )


def test_wait_returns_immediately_for_a_finished_task(tmp_path, capsys, monkeypatch):
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    t = _enqueue(store)
    store.claim_next()
    store.finish(t.id, "completed", {"summary": "s"})
    cli.main(["wait", t.id])
    out = capsys.readouterr().out
    assert "done" in out and "completed" in out


def test_wait_blocks_until_approval_is_requested(tmp_path, capsys, monkeypatch):
    """The point of `wait`: agents park it in a background shell instead of
    tight-polling task_status or reading tasks.db by hand — so it must wake on
    awaiting_approval (a human is needed NOW), not only on terminal states."""
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    t = _enqueue(store)
    store.claim_next()

    def approve_later():
        time.sleep(0.3)
        store.request_approval(t.id, "git diff")

    flipper = threading.Thread(target=approve_later)
    started = time.monotonic()
    flipper.start()
    try:
        cli.main(["wait", t.id, "--interval", "0.05"])
    finally:
        flipper.join()
    elapsed = time.monotonic() - started
    out = capsys.readouterr().out
    assert "awaiting_approval" in out
    assert "git diff" in out, "the pending command is the thing the human must see"
    assert elapsed >= 0.3, f"returned in {elapsed:.2f}s — never actually waited"


def test_wait_unknown_task_exits_2(tmp_path, capsys, monkeypatch):
    from sous import cli

    _wait_store(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        cli.main(["wait", "nope"])
    assert exc.value.code == 2


def test_wait_timeout_exits_1(tmp_path, capsys, monkeypatch):
    """A queued task that never advances must not hang the caller forever when
    a timeout was asked for — and the timeout must be exit 1, distinct from
    unknown-task (2), so scripts can tell 'still running' from 'gone'."""
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    t = _enqueue(store)  # stays queued: nothing ever claims it
    with pytest.raises(SystemExit) as exc:
        cli.main(["wait", t.id, "--timeout", "0.3", "--interval", "0.05"])
    assert exc.value.code == 1
    assert "queued" in capsys.readouterr().out


def test_wait_timeout_is_not_quantized_by_interval(tmp_path, capsys, monkeypatch):
    """A 0.3s timeout must expire near 0.3s even with the default 2s interval —
    each sleep has to be capped to the remaining budget, or the deadline check
    only runs on interval boundaries (and a task finishing inside the overrun
    would be reported as success AFTER the caller's deadline)."""
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    t = _enqueue(store)  # stays queued
    started = time.monotonic()
    with pytest.raises(SystemExit) as exc:
        cli.main(["wait", t.id, "--timeout", "0.3"])  # interval left at default
    elapsed = time.monotonic() - started
    assert exc.value.code == 1
    assert elapsed < 1.0, f"timed out after {elapsed:.2f}s — quantized to the interval"


def test_wait_rejects_degenerate_intervals_and_timeouts(tmp_path, capsys, monkeypatch):
    """--interval 0 recreates the tight-polling this command exists to prevent,
    a negative interval raises out of time.sleep, and a NaN timeout never
    expires — all three must be argparse usage errors (exit 2), not runtime
    misbehavior."""
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    t = _enqueue(store)
    store.claim_next()
    store.finish(t.id, "completed", {"summary": "s"})  # even a done task: reject first
    for argv in (
        ["wait", t.id, "--interval", "0", "--timeout", "0.2"],
        ["wait", t.id, "--interval", "-1", "--timeout", "0.2"],
        ["wait", t.id, "--timeout", "nan"],
        ["wait", t.id, "--interval", "nan", "--timeout", "0.2"],
        ["wait", t.id, "--timeout", "-5"],
    ):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2, argv


def test_wait_timeout_zero_is_an_immediate_probe(tmp_path, capsys, monkeypatch):
    """--timeout 0 is the defined non-blocking form: one state check, then
    report — exit 0 if the task already needs attention, exit 1 otherwise."""
    from sous import cli

    store = _wait_store(tmp_path, monkeypatch)
    pending = _enqueue(store)
    with pytest.raises(SystemExit) as exc:
        cli.main(["wait", pending.id, "--timeout", "0"])
    assert exc.value.code == 1
    done = _enqueue(store)
    store.claim_next()  # claims `pending`... order: claim_next takes oldest queued
    store.claim_next()
    store.finish(done.id, "completed", {"summary": "s"})
    cli.main(["wait", done.id, "--timeout", "0"])
    assert "done" in capsys.readouterr().out


# --- sous claude ---------------------------------------------------------------------


def _claude_setup(tmp_path, monkeypatch, *, probe=(200, True), **overrides):
    """A gateway-enabled config, a `claude` on PATH, a daemon that answers the
    probe, and an execve that records instead of replacing the process."""
    import os

    from sous import cli
    from sous.config import SousConfig

    # Popped rather than passed alongside **overrides: a test overriding
    # gateway_enabled would otherwise collide with a literal keyword of the
    # same name below and raise "got multiple values for keyword argument".
    gateway_enabled = overrides.pop("gateway_enabled", True)
    cfg = SousConfig(
        server_port=8383,
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        gateway_enabled=gateway_enabled,
        **overrides,
    )
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: "/opt/bin/claude" if name == "claude" else None
    )
    monkeypatch.setattr(cli, "_probe_gateway", lambda port: probe)
    for var in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "API_TIMEOUT_MS"):
        monkeypatch.delenv(var, raising=False)
    calls: list[tuple] = []
    monkeypatch.setattr(os, "execve", lambda exe, argv, env: calls.append((exe, argv, env)))
    return cfg, calls


def test_claude_argv_appends_the_lsp_opt_out_unless_the_user_chose_their_own():
    """Claude Code's --disallowedTools is variadic, so it goes last (prepended,
    it would swallow a positional prompt) — and before a `--` if there is one."""
    from sous.cli import claude_argv

    assert claude_argv([]) == ["--disallowedTools", "LSP"]
    assert claude_argv(["-p", "hi"]) == ["-p", "hi", "--disallowedTools", "LSP"]
    assert claude_argv(["--", "fix it"]) == ["--disallowedTools", "LSP", "--", "fix it"]
    for own in (
        ["--disallowedTools", "Foo"],
        ["--disallowedTools=Foo"],
        ["--disallowed-tools", "Foo", "Bar"],
        ["--disallowed-tools=Foo"],
    ):
        assert claude_argv(["-p", "x", *own]) == ["-p", "x", *own]
    # After a `--` the same text is a prompt, not an option Claude Code will
    # parse, so it cannot be the user choosing their own opt-out.
    assert claude_argv(["--", "--disallowedTools"]) == [
        "--disallowedTools",
        "LSP",
        "--",
        "--disallowedTools",
    ]
    assert claude_argv(["--disallowedTools", "X", "--", "y"]) == [
        "--disallowedTools",
        "X",
        "--",
        "y",
    ]


def test_claude_env_sets_the_gateway_variables_and_nothing_credential_shaped(tmp_path):
    from sous.cli import claude_env
    from sous.config import SousConfig

    cfg = SousConfig(
        server_port=9999,
        data_dir=tmp_path,
        config_path=tmp_path / "c.toml",
        gateway_enabled=True,
        gateway_local_models=("sous-fast", "sous-local"),
        gateway_max_context_tokens=131072,
    )
    env = claude_env(cfg, {"PATH": "/usr/bin", "API_TIMEOUT_MS": "5", "HOME": "/Users/x"})
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sous-fast"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    assert env["API_TIMEOUT_MS"] == "3000000"
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/Users/x"
    # Global, not per-model: pinning it would make the frontier main loop
    # compact at the local window (decision 10).
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env
    for forbidden in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ):
        assert forbidden not in env


def test_claude_execs_claude_with_the_gateway_environment(tmp_path, capsys, monkeypatch):
    from sous import cli

    cfg, calls = _claude_setup(tmp_path, monkeypatch)
    cli.main(["claude", "-p", "hi"])
    [(exe, argv, env)] = calls
    assert exe == "/opt/bin/claude"
    assert argv == ["/opt/bin/claude", "-p", "hi", "--disallowedTools", "LSP"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8383"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sous-local"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    assert env["API_TIMEOUT_MS"] == "3000000"
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
    err = capsys.readouterr().err
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8383" in err
    assert "CLAUDE_CODE_SUBAGENT_MODEL=sous-local" in err
    assert "warning" not in err


def test_claude_passes_every_argument_through_including_help(tmp_path, monkeypatch):
    """Dispatch happens before argparse: `sous claude --help` is Claude Code's
    help, and a leading option is not sous's to reject."""
    from sous import cli

    _, calls = _claude_setup(tmp_path, monkeypatch)
    cli.main(["claude", "--help"])
    cli.main(["claude", "--model", "opus", "--verbose"])
    assert calls[0][1] == ["/opt/bin/claude", "--help", "--disallowedTools", "LSP"]
    assert calls[1][1] == [
        "/opt/bin/claude",
        "--model",
        "opus",
        "--verbose",
        "--disallowedTools",
        "LSP",
    ]


def test_claude_warns_about_a_credential_variable_but_still_launches(tmp_path, capsys, monkeypatch):
    """Spec: warn, don't refuse — either variable moves the main loop from the
    subscription to API-credit billing, and that is the user's call to make."""
    from sous import cli

    _, calls = _claude_setup(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-canary")
    cli.main(["claude"])
    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "warning" in err and "ANTHROPIC_API_KEY" in err and "API credits" in err
    assert "sk-ant-api03-canary" not in err
    assert calls[0][2]["ANTHROPIC_API_KEY"] == "sk-ant-api03-canary"  # untouched, not unset


def test_claude_refuses_when_the_gateway_is_disabled(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls = _claude_setup(tmp_path, monkeypatch, gateway_enabled=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude"])
    assert exc.value.code == 1 and calls == []
    assert "[gateway]" in capsys.readouterr().err


def test_claude_refuses_when_no_daemon_listens(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls = _claude_setup(tmp_path, monkeypatch, probe=None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude"])
    assert exc.value.code == 1 and calls == []
    assert "sous serve" in capsys.readouterr().err


def test_claude_refuses_a_daemon_running_without_the_gateway(tmp_path, capsys, monkeypatch):
    """A 404 with no Via is the daemon's own "no such route": the config was
    edited but the daemon was not restarted."""
    from sous import cli

    _, calls = _claude_setup(tmp_path, monkeypatch, probe=(404, False))
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude"])
    assert exc.value.code == 1 and calls == []
    assert "restart" in capsys.readouterr().err


def test_claude_launches_with_a_warning_when_the_upstream_is_down(tmp_path, capsys, monkeypatch):
    """A 5xx carrying Via came from the gateway's forwarder: the gateway is up,
    the network is not. Claude Code's own error is the truth, so launch."""
    from sous import cli

    _, calls = _claude_setup(tmp_path, monkeypatch, probe=(502, True))
    cli.main(["claude"])
    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "warning" in err and "upstream" in err


def test_claude_launches_against_a_phase_1_daemon(tmp_path, monkeypatch):
    """A local 200 without Via is a daemon that still answers hello itself."""
    from sous import cli

    _, calls = _claude_setup(tmp_path, monkeypatch, probe=(200, False))
    cli.main(["claude"])
    assert len(calls) == 1


def test_claude_refuses_without_the_claude_binary(tmp_path, capsys, monkeypatch):
    from sous import cli

    _, calls = _claude_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["claude"])
    assert exc.value.code == 1 and calls == []
    assert "PATH" in capsys.readouterr().err


def test_probe_gateway_reads_status_and_via_over_a_real_socket():
    import http.server

    from sous.cli import _probe_gateway

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            status, via = self.server.answer  # ty: ignore[unresolved-attribute]
            self.send_response(status)
            if via:
                self.send_header("Via", "1.1 sous")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):  # ty: ignore[invalid-method-override]
            pass

    port = _free_cli_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        server.answer = (502, True)  # ty: ignore[unresolved-attribute]
        assert _probe_gateway(port) == (502, True)
        server.answer = (404, False)  # ty: ignore[unresolved-attribute]
        assert _probe_gateway(port) == (404, False)
    finally:
        server.shutdown()
        server.server_close()
    assert _probe_gateway(port) is None
