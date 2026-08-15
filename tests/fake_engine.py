"""Scripted Engine implementation for model-free tests."""


class FakeEngine:
    model_id = "fake/model"

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.unloaded = False

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            raise AssertionError("FakeEngine script exhausted")
        return self.script.pop(0)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return sum(len(str(m)) for m in messages) // 4

    def unload(self) -> None:
        self.unloaded = True
