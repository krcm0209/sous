"""slugify(text, max_length=40) turns a title into a URL slug:

- letters are lower-cased;
- every run of characters that are not ASCII letters or digits becomes one
  "-";
- leading and trailing "-" are removed;
- the result is cut to at most max_length characters, and a "-" the cut
  leaves at the end is removed too.
"""

import re

_NON_WORD = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_length: int = 40) -> str:
    slug = _NON_WORD.sub("-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-")
