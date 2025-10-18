# DCH 0.9.0 — Release Draft

Artifacts:
- Wheel: dist/dch-0.9.0-py3-none-any.whl
- sdist: dist/dch-0.9.0.tar.gz

Gate review:
- Gate JSON: artifacts/gate_report.json/gate_report.json
- passed: true

SHA256 checksums:
bd654dfde0456235af58fe78ad7631257e2f2752ca08c2083d23fef3b2d1ce88  dist/dch-0.9.0-py3-none-any.whl
117b321f0b75d736e96e8123da68d4a68b49af27a9542d43d75c16e9bf5ce436  dist/dch-0.9.0.tar.gz

Install notes:
- Requires Python >= 3.10 (per [pyproject.toml](pyproject.toml:1))
- Local wheel install: python -m pip install dist/dch-0.9.0-py3-none-any.whl

Post-approval steps:
1) Upload to TestPyPI (credentials required):
   twine upload --repository testpypi dist/*
2) Validate install in clean venv (Python 3.10+):
   python -m pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple 'dch==0.9.0'
3) Create signed tag and publish GitHub release with attached dists and checksums.
