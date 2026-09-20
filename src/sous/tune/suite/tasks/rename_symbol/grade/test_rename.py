import unittest
from pathlib import Path


class RenameTests(unittest.TestCase):
    def test_the_new_name_is_the_store(self):
        from store import InventoryStore

        store = InventoryStore()
        store.add("bolt", 3)
        self.assertEqual(store.count("bolt"), 3)

    def test_the_report_takes_the_new_name(self):
        from report import summarize
        from store import InventoryStore

        store = InventoryStore()
        store.add("nut")
        store.add("bolt", 3)
        self.assertEqual(summarize(store), "2 items: bolt x3, nut x1")

    def test_the_old_name_is_gone_from_every_file(self):
        import store

        self.assertFalse(hasattr(store, "ItemStore"))
        for path in sorted(Path.cwd().rglob("*.py")):
            self.assertNotIn("ItemStore", path.read_text(), path.name)

    def test_the_cli_and_the_tests_use_the_new_name(self):
        for name in ("cli.py", "tests/test_store.py"):
            self.assertIn("InventoryStore", Path(name).read_text(), name)
