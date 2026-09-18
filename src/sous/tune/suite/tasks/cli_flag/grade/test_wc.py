import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _wc(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "wc.py", *args], capture_output=True, text=True, timeout=30, check=False
    )


class LinesFlagTests(unittest.TestCase):
    def test_lines_counts_lines_and_words_stay_the_default(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "f.txt"
            f.write_text("a b\nc\n")
            self.assertEqual(_wc("--lines", str(f)).stdout.strip(), f"2 {f}")
            self.assertEqual(_wc(str(f)).stdout.strip(), f"3 {f}")

    def test_two_files_print_a_line_total(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a.txt", Path(td) / "b.txt"
            a.write_text("a\nb\nc\n")
            b.write_text("x\n")
            out = _wc("--lines", str(a), str(b))
            self.assertEqual(out.stdout.strip().splitlines(), [f"3 {a}", f"1 {b}", "4 total"])

    def test_help_describes_the_flag(self):
        out = _wc("--help")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--lines", out.stdout)
