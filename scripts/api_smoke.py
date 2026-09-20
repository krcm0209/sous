"""One served turn against a tiny real model: the check to run after a
dependency change, since CI cannot load a model.

Starts `sous serve` on a free port with a config and data dir under a temp
HOME, posts one streaming /v1/messages request for `sous-local` that carries
a tool, and passes when a tool_use block streams back. Judged by the block,
not by the model finishing its turn: a 0.6B model cannot reliably close a
turn, and the plumbing is what this checks.

    uv run python scripts/api_smoke.py
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TINY = "mlx-community/Qwen3-0.6B-4bit"
START_TIMEOUT = 120.0
TURN_TIMEOUT = 600.0


def build_request(model: str = "sous-local") -> dict:
    return {
        "model": model,
        "max_tokens": 200,
        "stream": True,
        "system": "You are a coding assistant. Use the write_file tool to write files.",
        "tools": [
            {
                "name": "write_file",
                "description": "Write a file at path with content.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Write hello.txt containing exactly: hello from sous"}
        ],
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise SystemExit("the daemon did not open its port")


def main() -> int:
    import httpx

    port = _free_port()
    home = Path(tempfile.mkdtemp(prefix="sous-smoke-"))
    sous_dir = home / ".sous"
    sous_dir.mkdir()
    # An empty drafter id disables speculative decoding, as tune's drafterless
    # arms do; the floor window keeps the KV reserve small for a 0.6B model.
    (sous_dir / "config.toml").write_text(
        f'[server]\nport = {port}\n\n[model]\nid = "{TINY}"\nmax_context_tokens = 49152\n'
        'speculative_draft_id = ""\n'
    )
    # HOME points the daemon at the temp config and data dir; HF_HOME keeps
    # the real model cache so the tiny model is not downloaded again.
    env = {
        **os.environ,
        "HOME": str(home),
        "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
    }
    daemon = subprocess.Popen(
        [sys.executable, "-c", "from sous.cli import main; main()", "serve"],
        env=env,
        stdout=sys.stderr,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_port(port, time.monotonic() + START_TIMEOUT)
        saw_tool_use = False
        with (
            httpx.Client(timeout=TURN_TIMEOUT, trust_env=False) as client,
            client.stream(
                "POST", f"http://127.0.0.1:{port}/v1/messages", json=build_request()
            ) as reply,
        ):
            print(f"status {reply.status_code}", file=sys.stderr)
            for line in reply.iter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                block = event.get("content_block") or {}
                if event.get("type") == "content_block_start" and block.get("type") == "tool_use":
                    saw_tool_use = True
                    print(f"tool_use: {json.dumps(block)}", file=sys.stderr)
                if event.get("type") == "message_delta":
                    stop = (event.get("delta") or {}).get("stop_reason")
                    print(f"stop_reason: {stop}", file=sys.stderr)
        verdict = "PASS: a tool_use block came back" if saw_tool_use else "FAIL: no tool_use block"
        print(verdict, file=sys.stderr)
        return 0 if saw_tool_use else 1
    finally:
        daemon.send_signal(signal.SIGTERM)
        try:
            daemon.wait(30)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait()
        # Only once it is gone: the temp HOME holds the daemon's data dir.
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
