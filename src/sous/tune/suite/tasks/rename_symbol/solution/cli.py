"""python cli.py NAME[:QUANTITY] ... prints a summary of the items given."""

import sys

from report import summarize
from store import InventoryStore


def main(argv: list[str]) -> int:
    store = InventoryStore()
    for arg in argv:
        name, _, quantity = arg.partition(":")
        store.add(name, int(quantity) if quantity else 1)
    print(summarize(store))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
