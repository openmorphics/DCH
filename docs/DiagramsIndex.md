# Dynamic Causal Hypergraph DCH — Diagrams Index and Render Guide

Purpose  
This index catalogs all Mermaid diagrams across the DCH spec to support consistency checks and PDF export. Each entry links to the source section. Use the Render Guide below to validate diagrams.

Render Guide
- VS Code preview Install the Mermaid Markdown extension; open each linked file and ensure diagrams render.  
- CLI export Option A mermaid-cli
  - npm install -g @mermaid-js/mermaid-cli
  - mmdc -i input.md -o output.pdf with a markdown-to-pdf pipeline that preserves Mermaid.  
- CLI export Option B Pandoc with Mermaid filter
  - pandoc --from gfm --to pdf --filter pandoc-mermaid -o docs/export/DCH_TechSpec_v0.1.pdf docs/DCH_TechSpec_v0.1.md
- Verify anchors After export, click internal links in the PDF to confirm they resolve.

Master overview
- Neuro-symbolic learning loop
  - [docs/DCH_TechSpec_v0.1.md](./DCH_TechSpec_v0.1.md)

Core algorithms
- TC-kNN DHG flow enhanced
  - [docs/sections/DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)
- Constrained backward traversal with AND frontier
  - [docs/sections/DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md)
- FSM pipeline normalization to rule promotion
  - [docs/sections/DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md)
- Hierarchical abstraction creation and usage HOEs
  - [docs/sections/DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md)

Learning control and interfaces
- Task-aware scaffolding REUSE ISOLATE HYBRID control loop
  - [docs/sections/DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md)
- Module interaction map data flow and control
  - [docs/sections/DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)
- Performance pipeline with latency budgets
  - [docs/sections/DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md)

Software blueprint and evaluation
- Software dataflow orchestration lanes
  - [docs/sections/DCH_Section11_SoftwareBlueprint.md](./sections/DCH_Section11_SoftwareBlueprint.md)
- Evaluation workflows data-to-metrics
  - [docs/sections/DCH_Section12_Evaluation.md](./sections/DCH_Section12_Evaluation.md)

Risk and governance
- Risk-to-action control loop monitors and auto knobs
  - [docs/sections/DCH_Section14_RiskMitigations.md](./sections/DCH_Section14_RiskMitigations.md)

Hardware co-design
- Chip-level dataflow units and control
  - [docs/sections/DCH_Section15_CausaChip.md](./sections/DCH_Section15_CausaChip.md)
- GMF update pipeline with PIM atomic ops
  - [docs/sections/DCH_Section15_CausaChip.md](./sections/DCH_Section15_CausaChip.md)
- PTA PE array topology and reduction path
  - [docs/sections/DCH_Section15_CausaChip.md](./sections/DCH_Section15_CausaChip.md)
- FSM engine pipeline in hardware
  - [docs/sections/DCH_Section15_CausaChip.md](./sections/DCH_Section15_CausaChip.md)

Outline quick diagrams duplicate anchors
- Neuro-symbolic learning loop quick view
  - [docs/DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- DHG construction around a post spike quick view
  - [docs/DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- Constrained backward traversal quick view
  - [docs/DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- Causa-Chip dataflow quick view
  - [docs/DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)

Checklist for reviewers
- Diagrams render without syntax errors in all listed files.  
- Terminology nodes and labels match the text e.g., DHG, PTA, FSM, HOE, WL, SAGE.  
- Arrows reflect correct data and control flow according to the section narratives.  
- Duplicate diagrams quick views in the Outline are consistent with the authoritative versions in Sections 2, 5, 6, 7, 8, 9, 10, 11, 12, 14, and 15.

Known formatting caveats
- Avoid double quotes and parentheses inside Mermaid node labels to prevent parse errors; the spec conforms to this requirement.  
- Long labels may wrap differently across renderers ensure readability in both VS Code preview and CLI export.

End of Diagrams Index