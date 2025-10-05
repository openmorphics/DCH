#!/usr/bin/env python3
import re
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

"""
Link and anchor checker for Markdown docs.
- Scans repository recursively for .md files (default: docs/ and root).
- Extracts heading anchors (ATX and Setext). Supports Pandoc explicit IDs {#id}.
- Validates internal links:
    * Local anchors: (#anchor) must exist in same file.
    * Cross-file anchors: (path#anchor) file must exist and anchor present.
    * File links: (path) file must exist.
    * Image links: basic existence check.
- Emits summary to stdout and writes a detailed report to docs/LinksReport.md
Exit code: 0 always; report includes issues count.
"""

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "docs" / "LinksReport.md"

LINK_RE = re.compile(r'!\[[^\]]*\]\(([^)]+)\)|\[[^\]]*\]\(([^)]+)\)')
ATX_RE = re.compile(r'^\s{0,3}(#{1,6})\s+(.+?)\s*(?:#+\s*)?$')
SETEXT_RE = re.compile(r'^\s*([=-]{3,})\s*$')
EXPLICIT_ID_RE = re.compile(r'\s*\{#([A-Za-z0-9][A-Za-z0-9\-_\.]*)\}\s*$')

SKIP_DIRS = {".git", ".github", "node_modules", ".venv", "venv", "__pycache__"}


def slugify(text: str) -> str:
    # Remove explicit ID if present
    m = EXPLICIT_ID_RE.search(text)
    if m:
        return m.group(1)
    t = text.strip()
    # Strip surrounding backticks and trailing colons
    t = re.sub(r'[`*_~]', '', t)
    t = t.lower()
    # Remove anything that's not alnum, space, hyphen or underscore or dot
    t = re.sub(r'[^a-z0-9 _\-.]+', '', t)
    # Replace spaces with hyphens
    t = re.sub(r'\s+', '-', t)
    # Collapse multiple hyphens
    t = re.sub(r'-{2,}', '-', t)
    # Trim hyphens/dots
    t = t.strip('-. ')
    return t


def extract_anchors(md_path: Path) -> Set[str]:
    anchors: Set[str] = set()
    try:
        lines = md_path.read_text(encoding='utf-8').splitlines()
    except Exception:
        return anchors
    prev_line = ""
    for i, line in enumerate(lines):
        m = ATX_RE.match(line)
        if m:
            title = m.group(2)
            # Extract explicit ID if present
            m_id = EXPLICIT_ID_RE.search(title)
            if m_id:
                anchors.add(m_id.group(1))
                # Remove the {#id} from title before slugging
                title = EXPLICIT_ID_RE.sub('', title)
            anchors.add(slugify(title))
        else:
            m2 = SETEXT_RE.match(line)
            if m2 and prev_line.strip():
                # Setext heading uses previous line as title
                title = prev_line.strip()
                m_id2 = EXPLICIT_ID_RE.search(title)
                if m_id2:
                    anchors.add(m_id2.group(1))
                    title = EXPLICIT_ID_RE.sub('', title)
                anchors.add(slugify(title))
        prev_line = line
    return anchors


def gather_md_files(root: Path) -> List[Path]:
    md_files: List[Path] = []
    for p in root.rglob('*.md'):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        md_files.append(p)
    return md_files


def norm_rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def is_external(target: str) -> bool:
    return target.startswith('http://') or target.startswith('https://') or target.startswith('mailto:') or target.startswith('tel:')


def split_target(target: str) -> Tuple[str, str]:
    # Returns (path_part, anchor_part). Any may be ''.
    if target.startswith('#'):
        return ('', target[1:])
    if '#' in target:
        p, a = target.split('#', 1)
        return (p, a)
    return (target, '')


def check_links(md_files: List[Path], anchors_map: Dict[str, Set[str]]) -> Tuple[int, List[Dict]]:
    issues: List[Dict] = []
    links_total = 0
    for md in md_files:
        try:
            content = md.read_text(encoding='utf-8')
        except Exception as e:
            issues.append({'file': norm_rel(md), 'line': 0, 'link': '', 'error': f'cannot read file: {e}'})
            continue
        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            for m in LINK_RE.finditer(line):
                target = m.group(1) or m.group(2) or ''
                if not target or target.startswith('<'):
                    continue
                links_total += 1
                if is_external(target):
                    continue
                path_part, anchor_part = split_target(target)
                if not path_part:
                    # Same-file anchor
                    if anchor_part and anchor_part not in anchors_map.get(norm_rel(md), set()):
                        issues.append({'file': norm_rel(md), 'line': idx, 'link': target, 'error': 'missing local anchor'})
                    continue
                # Normalize and resolve relative to current file
                ref_path = (md.parent / path_part).resolve()
                # Strip URL fragments like ? or query?
                # For safety, ignore query/params
                try:
                    ref_rel = norm_rel(ref_path)
                except Exception:
                    # Path escapes repo (e.g., absolute path). Flag it.
                    issues.append({'file': norm_rel(md), 'line': idx, 'link': target, 'error': 'path escapes repository or invalid'})
                    continue
                if not ref_path.exists():
                    issues.append({'file': norm_rel(md), 'line': idx, 'link': target, 'error': 'referenced path does not exist'})
                    continue
                if anchor_part:
                    anchors = anchors_map.get(ref_rel, set())
                    if anchor_part not in anchors:
                        issues.append({'file': norm_rel(md), 'line': idx, 'link': target, 'error': 'missing target anchor'})
    return (links_total, issues)


def write_report(links_total: int, issues: List[Dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open('w', encoding='utf-8') as f:
        f.write('# Links Report\n\n')
        f.write(f'- Files scanned: docs and repository markdown\n')
        f.write(f'- Links found: {links_total}\n')
        f.write(f'- Issues: {len(issues)}\n\n')
        if issues:
            f.write('## Issues\n\n')
            for it in issues:
                f.write(f"- {it['file']}:{it['line']}: {it['error']} -> {it['link']}\n")
        else:
            f.write('No issues found.\n')
    print(f"[links] links={links_total} issues={len(issues)}")
    print(f"[links] report: {norm_rel(REPORT_PATH)}")


def main() -> int:
    root = ROOT
    md_files = gather_md_files(root)
    # Precompute anchors map for all md files
    anchors_map: Dict[str, Set[str]] = {}
    for md in md_files:
        anchors_map[norm_rel(md)] = extract_anchors(md)
    links_total, issues = check_links(md_files, anchors_map)
    write_report(links_total, issues)
    # Do not fail CI here; just report
    return 0


if __name__ == '__main__':
    raise SystemExit(main())