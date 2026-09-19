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
    def test_every_ini_has_a_json_twin(self):
        for name in EXPECTED:
            self.assertTrue((CONFIGS / f"{name}.json").is_file(), name)

    def test_the_json_holds_the_sections_with_string_values(self):
        for name, expected in EXPECTED.items():
            with (CONFIGS / f"{name}.json").open() as f:
                self.assertEqual(json.load(f), expected, name)

    def test_load_reads_the_json_not_the_ini(self):
        self.assertNotIn("configparser", Path("settings.py").read_text())
        from settings import load

        self.assertEqual(load("beta"), EXPECTED["beta"])
        # Proof that the JSON file is what load reads: change it and watch
        # load follow, then put it back.
        path = CONFIGS / "gamma.json"
        original = path.read_text()
        try:
            new_content = json.dumps(
                {"server": {"host": "sentinel", "port": "1"}, "limits": {"retries": "9"}}
            )
            path.write_text(new_content)
            self.assertEqual(load("gamma")["server"]["host"], "sentinel")
        finally:
            path.write_text(original)
