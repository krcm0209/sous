"""End-to-end smoke test with a tiny real model (no MCP, no Claude needed).

Run:  uv run python scripts/e2e_smoke.py
Downloads ~350 MB on first run. Expected: task reaches state=done and
hello.txt exists with the right content.
"""

import tempfile
import threading
import time
from pathlib import Path

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.tasks import FINISHED_STATES, TaskStore
from sous.worker import run_worker_loop

TINY = "mlx-community/Qwen3-0.6B-4bit"


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        proj = base / "proj"
        proj.mkdir()
        cfg = SousConfig(
            model_id=TINY, data_dir=base / "data",
            config_path=base / "config.toml", max_turns=8, max_minutes=5,
        )
        store = TaskStore(base / "tasks.db")
        task = store.enqueue(
            title="smoke",
            instructions=("Create a file named hello.txt containing exactly "
                          "this one line: hello sous"),
            project_root=str(proj), context_files=[], verify_commands=[],
        )
        stop = threading.Event()
        threading.Thread(
            target=run_worker_loop,
            args=(store, EngineManager(cfg), cfg, stop), daemon=True,
        ).start()
        while store.get(task.id).state not in FINISHED_STATES:
            t = store.get(task.id)
            print(f"  state={t.state} turns={t.turns_used} {t.last_activity}")
            time.sleep(2)
        stop.set()
        final = store.get(task.id)
        print(f"\nstate={final.state} outcome={final.outcome}")
        print(f"report: {final.report}")
        hello = proj / "hello.txt"
        print(f"hello.txt exists: {hello.exists()}")
        if hello.exists():
            print(f"content: {hello.read_text()!r}")


if __name__ == "__main__":
    main()
