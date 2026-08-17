"""Run a real sous daemon against a tmp data dir. Test support, never shipped.

Mirrors server.main()'s startup — including the singleton flock — so proxy
tests exercise the real guard rather than a stand-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sous.config import SousConfig
from sous.engine.base import EngineManager
from sous.server import _acquire_singleton_lock, create_server
from sous.tasks import TaskStore
from tests.fake_engine import FakeEngine


def main() -> None:
    data = Path(sys.argv[1])
    port = int(sys.argv[2])
    data.mkdir(parents=True, exist_ok=True)
    cfg = SousConfig(data_dir=data, config_path=data / "config.toml", server_port=port)
    if not cfg.config_path.exists():
        cfg.config_path.write_text("")
    _lock = _acquire_singleton_lock(data)
    store = TaskStore(data / "tasks.db")
    engines = EngineManager(cfg, engine_factory=lambda model_id: FakeEngine([]))
    mcp = create_server(store, engines, cfg)
    mcp.run(transport="streamable-http", host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
