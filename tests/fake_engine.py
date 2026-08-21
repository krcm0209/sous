"""Scripted Engine implementation for model-free tests."""

import threading


class FakeEngine:
    model_id = "fake/model"

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.max_tokens_seen: list[int] = []
        self.unloaded = False
        self.resets = 0
        self.stats: dict = {}
        # Which thread ran each call: the per-task-thread design (issue #34)
        # is pinned on these. Thread objects, not idents — idents recycle.
        self.generate_threads: list[threading.Thread] = []
        self.reset_idents: list[int] = []

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        self.generate_threads.append(threading.current_thread())
        self.calls.append([dict(m) for m in messages])
        self.max_tokens_seen.append(max_tokens)
        if not self.script:
            raise AssertionError("FakeEngine script exhausted")
        return self.script.pop(0)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return sum(len(str(m)) for m in messages) // 4

    def reset_prompt_cache(self) -> None:
        self.resets += 1
        self.reset_idents.append(threading.get_ident())

    def prompt_cache_stats(self) -> dict:
        return dict(self.stats)

    def unload(self) -> None:
        self.unloaded = True
