# Minimal Makefile for Sphinx documentation
# Optional build dependencies (not required to use DCH):
#   python -m pip install -U sphinx myst-parser furo
# Build HTML:
#   make html
# Clean build artifacts:
#   make clean

SPHINXBUILD = sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help
help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR)

.PHONY: html
html:
	@$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR)

.PHONY: clean
clean:
	@rm -rf $(BUILDDIR)