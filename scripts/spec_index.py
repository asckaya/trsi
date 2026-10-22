"""Report a specification the architecture map does not list, or lists wrongly.

Run with `--check` to exit non-zero, which is how pre-commit uses it.
"""

import re
from pathlib import Path

SPECS = Path(__file__).parent.parent / "docs" / "spec"
MAP = SPECS / "architecture.md"
LINK = re.compile(r"\]\(\./([a-z0-9-]+\.md)\)")


def main() -> int:
    """Print the mismatches between the map and the installed specifications."""
    pages = {path.name for path in SPECS.glob("*.md")}
    named = set(LINK.findall(MAP.read_text(encoding="utf-8")))
    unlisted = sorted(pages - named)
    dangling = sorted(named - pages)
    for name in unlisted:
        print(f"docs/spec/{name}: not reachable from {MAP.name}")
    for name in dangling:
        print(f"docs/spec/{name}: named by {MAP.name} and absent")
    if unlisted or dangling:
        return 1
    print(f"{len(pages)} specifications, all reachable from {MAP.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
