import json
import unittest
from pathlib import Path

EXPECTED = {
    "alpha": {"server": {"host": "alpha.local", "port": "8080"}, "limits": {"retries": "3"}},
    "beta": {"server": {"host": "beta.local", "port": "8081"}, "limits": {"retries": "5"}},
    "gamma": {"server": {"host": "gamma.local", "port": "9000"}, "limits": {"retries": "1"}},
}
CONFIGS = Path("configs")


class MigrationTests(unittest.TestCase):
    def test_every_ini_became_a_json_file(self):
        for name in EXPECTED:
            self.assertTrue((CONFIGS / f"{name}.json").is_file(), name)
            self.assertFalse((CONFIGS / f"{name}.ini").exists(), name)

    def test_the_json_holds_the_sections_with_string_values(self):
        for name, expected in EXPECTED.items():
            with (CONFIGS / f"{name}.json").open() as f:
                self.assertEqual(json.load(f), expected, name)

    def test_load_reads_the_json(self):
        from settings import load

        self.assertFalse((CONFIGS / "beta.ini").exists())
        self.assertEqual(load("beta"), EXPECTED["beta"])
        self.assertEqual(load("gamma"), EXPECTED["gamma"])
