"""python wc.py FILE [FILE ...] prints the word count of each file as
"<count> <file>", and a "<count> total" line when more than one file is
given. With --lines it counts lines instead."""

import argparse
import sys


def count_words(text: str) -> int:
    return len(text.split())


def count_lines(text: str) -> int:
    return len(text.splitlines())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="count words in files")
    parser.add_argument("files", nargs="+", help="files to count")
    parser.add_argument("--lines", action="store_true", help="count lines instead of words")
    args = parser.parse_args(argv)
    total = 0
    for name in args.files:
        with open(name, encoding="utf-8") as f:
            text = f.read()
        count = count_lines(text) if args.lines else count_words(text)
        total += count
        print(f"{count} {name}")
    if len(args.files) > 1:
        print(f"{total} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
