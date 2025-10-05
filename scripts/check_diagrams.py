#!/usr/bin/env python3
"""
Check Mermaid diagrams across docs for basic validity and generate a report.

- Scans Markdown files under:
    docs/
    docs/sections/
- Finds code fences starting with ```mermaid
- Performs basic validation:
    * Block contains one of: 'flowchart', 'sequenceDiagram', 'graph'
    * Block is non-empty
- Outputs a summary to stdout and writes a report to docs/DiagramsReport.md

Exit codes:
  0 = no issues found
  1 = issues found
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIRS = [os.path.join(ROOT, "docs"), os.path.join(ROOT, "docs", "sections")]
REPORT_PATH = os.path.join(ROOT, "docs", "DiagramsReport.md")

REQUIRED_TOKENS = ("flowchart", "sequenceDiagram", "graph")


def find_markdown_files(dirs: List[str]) -> List[str]:
    files: List[str] = []
    for base in dirs:
        if not os.path.isdir(base):
            continue
        for root, _, filenames in os.walk(base):
            for fn in filenames:
                if fn.lower().endswith(".md"):
                    files.append(os.path.join(root, fn))
    return sorted(files)


def extract_mermaid_blocks(path: str) -> List[Tuple[int, List[str]]]:
    """
    Return a list of (start_line_number, lines) for each mermaid code block.
    """
    blocks: List[Tuple[int, List[str]]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return blocks

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip().startswith("```mermaid"):
            start = i + 1  # 1-based content line
            i += 1
            block_lines: List[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                block_lines.append(lines[i].rstrip("\n"))
                i += 1
            blocks.append((start + 1, block_lines))  # +1 for human-friendly display
        i += 1
    return blocks


def validate_block(block_lines: List[str]) -> List[str]:
    issues: List[str] = []
    text = "\n".join(block_lines).strip()
    if not text:
        issues.append("Empty mermaid block")
        return issues
    if not any(tok in text for tok in REQUIRED_TOKENS):
        issues.append("No known mermaid diagram type token found (expected one of: flowchart, sequenceDiagram, graph)")
    return issues


def main() -> int:
    md_files = find_markdown_files(DOCS_DIRS)
    total_blocks = 0
    total_bad = 0
    per_file_issues: List[str] = []

    for path in md_files:
        blocks = extract_mermaid_blocks(path)
        if not blocks:
            continue
        file_bad = 0
        for start_line, block_lines in blocks:
            total_blocks += 1
            issues = validate_block(block_lines)
            if issues:
                file_bad += 1
                total_bad += 1
                per_file_issues.append(f"- {os.path.relpath(path, ROOT)}:{start_line} :: " + "; ".join(issues))
        # Optional: note files with blocks but no issues
        if blocks and file_bad == 0:
            per_file_issues.append(f"- {os.path.relpath(path, ROOT)} :: OK ({len(blocks)} mermaid block(s))")

    # Write report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as rep:
        rep.write("# Diagrams Report\n\n")
        rep.write(f"Scanned {len(md_files)} markdown file(s).\n\n")
        rep.write(f"- Mermaid blocks found: {total_blocks}\n")
        rep.write(f"- Blocks with issues: {total_bad}\n\n")
        if per_file_issues:
            rep.write("## Details\n")
            rep.write("\n".join(per_file_issues))
            rep.write("\n")

        rep.write("\n## Guidance\n")
        rep.write("- Ensure each mermaid block begins with one of: 'flowchart', 'sequenceDiagram', or 'graph'.\n")
        rep.write("- Avoid unbalanced code fences or stray backticks.\n")
        rep.write("- See docs/DiagramsIndex.md for the authoritative list and locations.\n")

    # Print summary
    print(f"[diagrams] files={len(md_files)} blocks={total_blocks} issues={total_bad}")
    print(f"[diagrams] report: {os.path.relpath(REPORT_PATH, ROOT)}")

    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())