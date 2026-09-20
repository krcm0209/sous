from store import InventoryStore


def summarize(store: InventoryStore) -> str:
    parts = [f"{name} x{store.count(name)}" for name in store.names()]
    return f"{len(parts)} items: {', '.join(parts)}"
