"""End-to-end smoke test with a tiny real model (no MCP, no Claude needed).

Run:  uv run python scripts/e2e_smoke.py
      SOUS_PROMPT_CACHE=0 uv run python scripts/e2e_smoke.py

Downloads ~350 MB on first run. The task is deliberately multi-turn (read
a file, then write one) so the prompt_cache block is non-trivial. Run it
twice and compare budget.seconds to see the before-and-after that issue #27
asks for. The 0.6B model is text-only and therefore fully trimmable, so it
exercises the one-call trim path, not the snapshot path. Exercises the real
multi-turn agent loop (generate -> parse -> execute -> append) against the
smallest available Qwen3 model. hello.txt is usually written correctly on
the first or second turn (check "content:" below) — but this 0.6B model is
unreliable at then emitting a well-formed `finish` tool call, so the task
can still end `failed` (model-confused) or `done`/`budget-exhausted` even
when the file is right. Judge success from the printed report and hello.txt
content, not just the final state; the transcript_path in the report has
full turn-by-turn detail if something looks wrong. Sous's default model
(mlx-community/Qwen3.8-27B-mxfp8) is far larger and far more reliable at
closing out the loop than this tiny one.
"""

import os
import tempfile
import threading
import time
from pathlib import Path

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.tasks import FINISHED_STATES, Task, TaskStore
from sous.worker import run_worker_loop

TINY = "mlx-community/Qwen3-0.6B-4bit"


def _require(store: TaskStore, task_id: str) -> Task:
    """The smoke task, which must exist — this script enqueued it itself."""
    task = store.get(task_id)
    if task is None:
        raise RuntimeError(f"task {task_id} vanished from the store")
    return task


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        proj = base / "proj"
        proj.mkdir()
        proj_file = proj / "notes.md"
        proj_file.write_text("# Notes\n\nalpha\nbeta\ngamma\n")
        cfg = SousConfig(
            model_id=TINY,
            data_dir=base / "data",
            config_path=base / "config.toml",
            max_turns=8,
            max_minutes=5,
            prompt_cache=os.environ.get("SOUS_PROMPT_CACHE", "1") != "0",
        )
        store = TaskStore(base / "tasks.db")
        task = store.enqueue(
            title="smoke",
            instructions=(
                "Read notes.md, then create hello.txt containing exactly this one line: hello sous"
            ),
            project_root=str(proj),
            context_files=[],
            verify_commands=[],
        )
        stop = threading.Event()
        threading.Thread(
            target=run_worker_loop,
            args=(store, EngineManager(cfg), cfg, stop),
            daemon=True,
        ).start()
        # One read per iteration; the task that ends the loop is the final one.
        while (current := _require(store, task.id)).state not in FINISHED_STATES:
            print(f"  state={current.state} turns={current.turns_used} {current.last_activity}")
            time.sleep(2)
        stop.set()
        print(f"\nstate={current.state} outcome={current.outcome}")
        print(f"report: {current.report}")
        cache = (current.report or {}).get("prompt_cache")
        print(f"prompt_cache={cfg.prompt_cache} stats: {cache}")
        hello = proj / "hello.txt"
        print(f"hello.txt exists: {hello.exists()}")
        if hello.exists():
            print(f"content: {hello.read_text()!r}")


if __name__ == "__main__":
    main()
