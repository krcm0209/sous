"""`python -m sous.tune.suite.unittests DIR`: run the unittest modules under
DIR and print the counts as one JSON line. The grader runs this in a
subprocess with the candidate's project as the working directory — `python -m`
puts that directory first on sys.path, so the hidden tests import the
candidate's modules by name — and a project that hangs or crashes takes the
subprocess with it, never the tune."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


def run(tests_dir: Path) -> dict:
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern="test_*.py", top_level_dir=str(tests_dir))
    result = unittest.TestResult()
    suite.run(result)
    failed = [str(case) for case, _ in [*result.failures, *result.errors]]
    total = result.testsRun
    return {"passed": total - len(failed), "total": total, "failed": failed}


def main(argv: list[str]) -> int:
    print(json.dumps(run(Path(argv[1]).resolve())), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
