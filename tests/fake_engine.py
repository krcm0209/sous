"""Scripted Engine implementation for model-free tests."""


class FakeEngine:
    model_id = "fake/model"

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.max_tokens_seen: list[int] = []
        self.unloaded = False
        self.resets = 0
        self.stats: dict = {}

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        self.calls.append([dict(m) for m in messages])
        self.max_tokens_seen.append(max_tokens)
        if not self.script:
            raise AssertionError("FakeEngine script exhausted")
        return self.script.pop(0)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return sum(len(str(m)) for m in messages) // 4

    def reset_prompt_cache(self) -> None:
        self.resets += 1

    def prompt_cache_stats(self) -> dict:
        return dict(self.stats)

    def unload(self) -> None:
        self.unloaded = True
