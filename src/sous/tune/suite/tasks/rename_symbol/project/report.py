from store import ItemStore


def summarize(store: ItemStore) -> str:
    parts = [f"{name} x{store.count(name)}" for name in store.names()]
    return f"{len(parts)} items: {', '.join(parts)}"
