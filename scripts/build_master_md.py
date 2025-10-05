#!/usr/bin/env python3
"""
Assemble a single Markdown file from the master spec and all section files.

Output:
  docs/DCH_TechSpec_v0.1_assembled.md

Rationale:
- Keeps links valid by placing the assembled file under docs/.
- While concatenating files from docs/sections/, adjust relative links:
    '](../'  --> '](./'
  so references that originally pointed from docs/sections/* back to docs/*
  remain correct when viewed from docs/ (the assembled file's location).

Notes:
- This does not alter source files; the rewrite is applied only to the assembled output.
- The order matches 'Export and assembly notes' in the master spec.
"""

from pathlib import Path
import datetime

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SECTIONS = DOCS / "sections"
OUT = DOCS / "DCH_TechSpec_v0.1_assembled.md"

# Master preface (overview, TOC, diagrams)
MASTER = DOCS / "DCH_TechSpec_v0.1.md"

# Section files in canonical order
SECTION_FILES = [
    SECTIONS / "DCH_Section1_FormalFoundations.md",
    SECTIONS / "DCH_Section2_DHG_TCkNN.md",
    SECTIONS / "DCH_Section3_Plasticity.md",
    SECTIONS / "DCH_Section4_HyperpathEmbedding.md",
    SECTIONS / "DCH_Section5_CreditAssignment.md",
    SECTIONS / "DCH_Section6_FSM.md",
    SECTIONS / "DCH_Section7_HierarchicalAbstraction.md",
    SECTIONS / "DCH_Section8_TaskAwareScaffolding.md",
    SECTIONS / "DCH_Section9_Interfaces.md",
    SECTIONS / "DCH_Section10_ComplexityResource.md",
    SECTIONS / "DCH_Section11_SoftwareBlueprint.md",
    SECTIONS / "DCH_Section12_Evaluation.md",
    SECTIONS / "DCH_Section13_ParamsTuning.md",
    SECTIONS / "DCH_Section14_RiskMitigations.md",
    SECTIONS / "DCH_Section15_CausaChip.md",
]

# Tail matter
REFERENCES = DOCS / "References.md"
DIAGRAMS = DOCS / "DiagramsIndex.md"


def read_text(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return p.read_text(encoding="utf-8")


def adjust_for_docs_base(text: str) -> str:
    """
    Rewrites links that originate from docs/sections/* to be correct
    when viewed from docs/* (the assembled file location).
    - '](../' --> '](./'
    """
    return text.replace("](../", "](./")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    parts = []
    parts.append(f"<!-- Assembled {now} by scripts/build_master_md.py -->\n")

    # 0. Master preface
    parts.append(read_text(MASTER).rstrip() + "\n")

    # 1..15 Sections (with link adjustment)
    for sec in SECTION_FILES:
        body = read_text(sec)
        body = adjust_for_docs_base(body)
        # Separator
        parts.append("\n\n---\n\n")
        parts.append(body.rstrip() + "\n")

    # References
    parts.append("\n\n---\n\n")
    parts.append(read_text(REFERENCES).rstrip() + "\n")

    # Diagrams Index
    parts.append("\n\n---\n\n")
    parts.append(read_text(DIAGRAMS).rstrip() + "\n")

    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"[assemble-md] wrote {OUT.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())