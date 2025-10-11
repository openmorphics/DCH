# Configuration file for the Sphinx documentation builder.
# Optional docsite for the Dynamic Causal Hypergraph (DCH) project.
#
# Build guidance (optional):
#   python -m pip install -U sphinx myst-parser furo
#   make -C docs html
#   # or:
#   python -m sphinx -b html docs docs/_build/html

from __future__ import annotations
import os
from pathlib import Path

# -- Project information -----------------------------------------------------
project = "Dynamic Causal Hypergraph (DCH)"
author = "DCH Maintainers"
# Keep the version a lightweight placeholder; avoid importing project code.
release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST (Markdown) config
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
primary_domain = "py"
pygments_style = "sphinx"
highlight_language = "python"

# -- Options for HTML output -------------------------------------------------
try:
    import importlib.util as _ilu  # noqa: F401
    html_theme = "furo" if _ilu.find_spec("furo") else "alabaster"
except Exception:
    html_theme = "alabaster"

html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
}

# -- Paths -------------------------------------------------------------------
# Do not modify sys.path to import project code; keep docs build import-safe.
repo_root = Path(__file__).resolve().parents[1]

# -- nitpicky options --------------------------------------------------------
# Avoid build failures if intersphinx or external refs are absent.
nitpicky = False

# -- End of configuration ----------------------------------------------------