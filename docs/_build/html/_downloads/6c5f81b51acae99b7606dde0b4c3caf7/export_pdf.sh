#!/usr/bin/env bash
# Export DCH master spec to PDF with Mermaid rendering when possible.
# Targets:
# - Input:  docs/DCH_TechSpec_v0.1.md
# - Output: docs/export/DCH_TechSpec_v0.1.pdf
#
# Attempts (in order):
# 1) pandoc with pandoc-mermaid filter
# 2) pandoc with mermaid-filter
# 3) pandoc without Mermaid filter (diagrams may not render)
#
# If none succeed, prints setup instructions.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/docs/DCH_TechSpec_v0.1.md"
OUT_DIR="${ROOT_DIR}/docs/export"
OUT="${OUT_DIR}/DCH_TechSpec_v0.1.pdf"

mkdir -p "${OUT_DIR}"

have() { command -v "$1" >/dev/null 2>&1; }

echo "[export] Source: ${SRC}"
echo "[export] Target: ${OUT}"

if ! [ -f "${SRC}" ]; then
  echo "[export] ERROR: cannot find source markdown at ${SRC}" >&2
  exit 1
fi

# Try pandoc with pandoc-mermaid (common filter name)
if have pandoc && have mermaid-filter; then
  echo "[export] Using pandoc + mermaid-filter"
  pandoc --from=gfm --to=pdf \
    --filter mermaid-filter \
    -o "${OUT}" "${SRC}"
  echo "[export] Wrote ${OUT}"
  exit 0
fi

# Try pandoc with pandoc-mermaid (alternative filter name)
if have pandoc; then
  if python3 -c "import importlib; importlib.import_module('pandoc_mermaid')" >/dev/null 2>&1; then
    echo "[export] Using pandoc + pandoc-mermaid"
    pandoc --from=gfm --to=pdf \
      --filter pandoc-mermaid \
      -o "${OUT}" "${SRC}"
    echo "[export] Wrote ${OUT}"
    exit 0
  fi
fi

# Fallback: pandoc without Mermaid rendering (diagrams may not render)
if have pandoc; then
  echo "[export] Fallback: pandoc without Mermaid filter (diagrams may not render)"
  pandoc --from=gfm --to=pdf -o "${OUT}" "${SRC}" || true
  if [ -f "${OUT}" ]; then
    echo "[export] Wrote ${OUT} (without Mermaid rendering)"
    exit 0
  fi
fi

cat <<EOF 1>&2
[export] Unable to export with current toolchain.
Please install one of the following and re-run:

Option A: pandoc + mermaid filter
  - brew install pandoc
  - pip3 install pandoc-mermaid  # or: pip3 install mermaid-filter
  - bash scripts/export_pdf.sh

Option B: Use a VS Code extension that supports Mermaid-in-PDF export and export from UI.

The master spec is at:
  ${SRC}
EOF
exit 2