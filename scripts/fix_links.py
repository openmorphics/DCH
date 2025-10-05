#!/usr/bin/env python3
"""
Fix relative links inside docs/ and docs/sections/ so they resolve correctly.

Problem:
- Many markdown files under docs/ and docs/sections/ use link targets prefixed with "docs/...".
- When a file inside docs/ links to "docs/...", the resolved path becomes docs/docs/... (invalid).
- When a file inside docs/sections/ links to "docs/..." or "docs/sections/...", the resolved path becomes docs/sections/docs/... (invalid).

Strategy:
- For files directly under docs/ (excluding nested subdirs):
    - Replace "](docs/" with "](./"
- For any file under docs/sections/ (or deeper):
    - Replace "](docs/" with "](../"

Notes:
- This is a conservative textual rewrite limited to markdown link syntax substrings.
- External links (http/https/mailto) are not affected (we only target the "](docs/" prefix).
- A report summary is printed to stdout and written to docs/LinksFixReport.md.

Usage:
  python3 scripts/fix_links.py
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPORT = DOCS / "LinksFixReport.md"

def rewrite_content(text: str, mode: str) -> tuple[str, int]:
    """
    mode: 'docs' or 'sections'
    - docs:      '](docs/' -> '](./'
    - sections:  '](docs/' -> '](../'
    Returns: (new_text, replacements_count)
    """
    before = text
    if mode == "docs":
        new = before.replace("](docs/", "](./")
    else:
        new = before.replace("](docs/", "](../")
    count = 0
    # Count occurrences replaced by comparing
    # quick-and-dirty: count target in old minus in new (not exact if nested; acceptable here)
    count = before.count("](docs/")
    return new, count

def classify(md_path: Path) -> str:
    """
    Return 'sections' if file path is under docs/sections, otherwise 'docs'
    """
    try:
        rel = md_path.resolve().relative_to(DOCS)
    except Exception:
        # Not under docs; skip
        return "skip"
    parts = rel.parts
    if len(parts) >= 1 and parts[0] == "sections":
        return "sections"
    else:
        return "docs"

def main() -> int:
    if not DOCS.exists():
        print(f"[fix-links] docs directory not found: {DOCS}", file=sys.stderr)
        return 1

    md_files = list(DOCS.rglob("*.md"))
    total_files = 0
    total_repl = 0
    per_file = []

    for md in md_files:
        mode = classify(md)
        if mode == "skip":
            continue
        text = md.read_text(encoding="utf-8")
        new_text, n = rewrite_content(text, mode)
        if n > 0:
            md.write_text(new_text, encoding="utf-8")
            per_file.append((md, n, mode))
            total_repl += n
            total_files += 1

    # Write report
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as f:
        f.write("# Links Fix Report\n\n")
        f.write(f"- Files modified: {total_files}\n")
        f.write(f"- Total replacements: {total_repl}\n\n")
        if per_file:
            f.write("## Changes by file\n\n")
            for md, n, mode in sorted(per_file, key=lambda t: str(t[0])):
                rel = md.resolve().relative_to(ROOT)
                f.write(f"- {rel} ({mode}): {n} replacements\n")

    print(f"[fix-links] files_modified={total_files} replacements={total_repl}")
    print(f"[fix-links] report: {REPORT.resolve().relative_to(ROOT)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())