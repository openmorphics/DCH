#!/usr/bin/env python3
"""
Export docs/DCH_TechSpec_v0.1.md to a single-file HTML artifact:
- Embeds the markdown source directly in the HTML (no external conversion step)
- Uses Marked.js (CDN) to render Markdown client-side
- Converts ```mermaid fenced blocks into <div class="mermaid"> for on-load rendering
- Includes Mermaid.js (CDN) and minimal CSS/print styles
- Generates a simple in-page TOC (H1..H3) for navigation

Usage:
  python3 scripts/export_html.py
Outputs:
  docs/export/DCH_TechSpec_v0.1.html
"""
from pathlib import Path
import json
import datetime

ROOT = Path(__file__).resolve().parent.parent
IN_MD = ROOT / "docs" / "DCH_TechSpec_v0.1.md"
OUT_HTML = ROOT / "docs" / "export" / "DCH_TechSpec_v0.1.html"

def main():
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    if not IN_MD.exists():
        raise SystemExit(f"Input markdown not found: {IN_MD}")

    md_text = IN_MD.read_text(encoding="utf-8")
    # Safely embed MD as a JS string literal via JSON
    md_js = json.dumps(md_text)

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Dynamic Causal Hypergraph (DCH) — v0.1 Spec</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="generator" content="export_html.py" />
  <meta name="exported-at" content="{now}" />
  <style>
    :root {{
      --bg: #0b0d10;
      --panel: #101419;
      --text: #e6edf3;
      --muted: #9aa7b2;
      --accent: #6cb6ff;
      --code-bg: #0d1117;
      --border: #222a33;
      --link: #91cbff;
      --toc-bg: #0f1318;
      --toc-border: #1b212a;
      --mermaid-bg: #0d1117;
    }}
    html, body {{
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif;
      line-height: 1.55;
    }}
    a {{ color: var(--link); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .container {{
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 24px;
    }}
    .sidebar {{
      position: sticky;
      top: 0;
      max-height: 100vh;
      overflow: auto;
      padding: 20px 16px;
      background: var(--toc-bg);
      border-right: 1px solid var(--toc-border);
    }}
    .content {{
      padding: 32px 32px 80px 0;
      max-width: 1200px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }}
    pre, code {{
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    pre {{
      padding: 14px;
      overflow: auto;
    }}
    code {{
      padding: 2px 6px;
    }}
    h1, h2, h3, h4 {{
      margin-top: 28px;
      color: #eaf2f9;
    }}
    h1 {{
      margin-top: 0;
      font-size: 1.9rem;
    }}
    hr {{
      border: none;
      border-top: 1px solid var(--border);
      margin: 28px 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 12px;
    }}
    .toc-title {{
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .toc ul {{
      list-style: none;
      padding-left: 8px;
      margin: 0;
    }}
    .toc li {{
      margin: 4px 0;
    }}
    .toc a {{
      color: var(--muted);
    }}
    .toc a:hover {{
      color: var(--text);
    }}
    .toc .lvl-1 {{ margin-left: 0; }}
    .toc .lvl-2 {{ margin-left: 12px; }}
    .toc .lvl-3 {{ margin-left: 24px; }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      margin-right: 6px;
      font-size: 12px;
    }}
    .mermaid {{
      background: var(--mermaid-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      margin: 16px 0;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: linear-gradient(180deg, rgba(11,13,16,0.92), rgba(11,13,16,0.85) 70%, rgba(11,13,16,0));
      backdrop-filter: blur(6px);
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
    }}
    .topbar .title {{
      font-weight: 600;
    }}
    .btns {{
      float: right;
    }}
    .btn {{
      display: inline-block;
      padding: 6px 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      margin-left: 8px;
      cursor: pointer;
      background: #0f141a;
    }}
    .btn:hover {{
      background: #131922;
    }}
    @media print {{
      .sidebar, .topbar, .btns {{ display: none !important; }}
      .container {{
        display: block;
      }}
      .content {{
        padding: 0;
        max-width: none;
      }}
      a[href]::after {{ content: ""; }}
      body {{ background: white; color: black; }}
    }}
  </style>
  <!-- Marked.js for Markdown rendering -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js" defer></script>
  <!-- Mermaid.js for diagrams -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js" defer></script>
</head>
<body>
  <div class="topbar">
    <span class="title">Dynamic Causal Hypergraph (DCH) — v0.1</span>
    <span class="btns">
      <button class="btn" onclick="window.print()">Print to PDF</button>
      <a class="btn" href="./DCH_TechSpec_v0.1.html" download>Download HTML</a>
    </span>
  </div>
  <div class="container">
    <aside class="sidebar">
      <div class="panel">
        <div class="meta">Artifact exported: {now}</div>
        <div>
          <span class="badge">DCH</span>
          <span class="badge">Spec v0.1</span>
          <span class="badge">Mermaid</span>
        </div>
      </div>
      <div class="panel" style="margin-top: 12px;">
        <div class="toc-title">Contents</div>
        <nav class="toc" id="toc"></nav>
      </div>
    </aside>
    <main class="content">
      <article id="content" class="panel">
        <div class="meta">Source: docs/DCH_TechSpec_v0.1.md</div>
        <div id="md-target">Rendering...</div>
      </article>
    </main>
  </div>

  <script>
    // Configuration for Marked
    window.addEventListener('DOMContentLoaded', () => {{
      // Configure marked
      marked.setOptions({{
        gfm: true,
        breaks: false,
        mangle: false,
        headerIds: true
      }});

      const md = {md_js};
      const target = document.getElementById('md-target');

      // Render markdown
      const html = marked.parse(md);

      // Inject rendered HTML
      target.innerHTML = html;

      // Transform fenced mermaid blocks into <div class="mermaid">
      document.querySelectorAll('pre code.language-mermaid').forEach((code) => {{
        const pre = code.parentElement;
        const wrapper = document.createElement('div');
        wrapper.className = 'mermaid';
        // Preserve raw mermaid source
        wrapper.textContent = code.textContent;
        pre.replaceWith(wrapper);
      }});

      // Build a simple TOC (H1..H3)
      const tocRoot = document.getElementById('toc');
      const heads = document.querySelectorAll('#md-target h1, #md-target h2, #md-target h3');
      const ul = document.createElement('ul');
      heads.forEach((h) => {{
        // Ensure ID exists
        if (!h.id) {{
          // derive slug
          const slug = (h.textContent || '')
            .toLowerCase()
            .trim()
            .replace(/[`*_~]/g, '')
            .replace(/[^a-z0-9 \\-\\.]+/g, '')
            .replace(/\\s+/g, '-')
            .replace(/-{{2,}}/g, '-')
            .replace(/^[\\-.]+|[\\-.]+$/g, '');
          h.id = slug;
        }}
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent || '';
        const lvl = h.tagName === 'H1' ? 'lvl-1' : (h.tagName === 'H2' ? 'lvl-2' : 'lvl-3');
        li.className = lvl;
        li.appendChild(a);
        ul.appendChild(li);
      }});
      tocRoot.appendChild(ul);

      // Initialize mermaid after content is in place
      if (window.mermaid) {{
        try {{
          mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
          // In case startOnLoad misses (defer), explicitly run init
          setTimeout(() => {{
            mermaid.init();
          }}, 50);
        }} catch (e) {{
          console.error('Mermaid init failed', e);
        }}
      }}
    }});
  </script>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[export-html] wrote {OUT_HTML.relative_to(ROOT)}")

if __name__ == "__main__":
    main()