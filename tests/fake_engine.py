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
        self.reset_owners: list[threading.Thread | None] = []
        self.stats_owners: list[threading.Thread | None] = []
        # Which thread ran each call: the per-task-thread design (issue #34)
        # is pinned on these. Thread objects, not idents — idents recycle.
        self.generate_threads: list[threading.Thread] = []
        self.reset_idents: list[int] = []
        # The on_delta object itself (not called) — pins whether a caller
        # wrapped it in ReplaySafe (sous.gateway.turn's replay_safe contract).
        self.on_deltas_seen: list[OnDelta | None] = []

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
        self.on_deltas_seen.append(on_delta)
        text = self._take(messages, tools, max_tokens)
        if on_delta is not None:
            # One delta per generation is the minimum streaming contract; tests
            # that need to watch pieces arrive use ChunkedFakeEngine.
            on_delta(Delta(text, max(1, len(text.split())), "stop"))
        return text

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return sum(len(str(m)) for m in messages) // 4

    def reset_prompt_cache(self, owner: threading.Thread | None = None) -> None:
        self.resets += 1
        self.reset_idents.append(threading.get_ident())
        self.reset_owners.append(owner)

    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict:
        self.stats_owners.append(owner)
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
        self.on_deltas_seen.append(on_delta)
        pieces = self._take(messages, tools, max_tokens).split("|")
        for n, piece in enumerate(pieces, start=1):
            if self.delay:
                time.sleep(self.delay)
            if on_delta is not None:
                last = n == len(pieces)
                on_delta(Delta(piece, n, self.finish_reason if last else None))
        self.finished.set()
        return "".join(pieces)
