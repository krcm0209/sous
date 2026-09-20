"""An in-memory count of named items."""


class InventoryStore:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def add(self, name: str, quantity: int = 1) -> None:
        self._counts[name] = self._counts.get(name, 0) + quantity

    def count(self, name: str) -> int:
        return self._counts.get(name, 0)

    def names(self) -> list[str]:
        return sorted(self._counts)
