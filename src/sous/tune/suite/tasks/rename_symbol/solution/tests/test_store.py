import unittest

from report import summarize
from store import InventoryStore


class StoreTests(unittest.TestCase):
    def test_counts_add_up(self):
        store = InventoryStore()
        store.add("bolt", 3)
        store.add("bolt")
        self.assertEqual(store.count("bolt"), 4)

    def test_summary_lists_items_in_name_order(self):
        store = InventoryStore()
        store.add("nut")
        store.add("bolt", 3)
        self.assertEqual(summarize(store), "2 items: bolt x3, nut x1")
