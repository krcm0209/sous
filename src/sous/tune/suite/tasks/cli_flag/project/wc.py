"""python wc.py FILE [FILE ...] prints the word count of each file as
"<count> <file>", and a "<count> total" line when more than one file is
given."""

import argparse
import sys


def count_words(text: str) -> int:
    return len(text.split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="count words in files")
    parser.add_argument("files", nargs="+", help="files to count")
    args = parser.parse_args(argv)
    total = 0
    for name in args.files:
        with open(name, encoding="utf-8") as f:
            count = count_words(f.read())
        total += count
        print(f"{count} {name}")
    if len(args.files) > 1:
        print(f"{total} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
