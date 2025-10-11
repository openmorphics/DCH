# DCH Usage Guide

This guide covers installation options (torch-free base and optional extras), dataset workflow, CLI usage, configuration files, reproducibility, and logging/artifacts.

Note: DCH is torch-optional at runtime. The DCH pipeline and tests run without torch/norse/tonic. SNN features only activate when explicitly enabled.

---

## 1) Installation

Recommended: Python 3.10–3.12 on Linux/macOS.

Torch-free minimal environment (no install of the package, run from source):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

# Minimal runtime deps (no torch)
pip install "hydra-core>=1.3" "numpy>=1.24,<2.1" "scipy>=1.10" "scikit-learn>=1.3,<2" \
            "pandas>=2.0,<3" "networkx>=3,<4" "pyyaml>=6" "statsmodels>=0.14" \
            "tqdm>=4.65" "rich>=13" "typing_extensions>=4.7"
```

Optional: install the package to enable the CLI entrypoint `dch-run`:
```bash
# Editable install; may bring additional dependencies defined by the project
pip install -e .
```

Docsite build requirements (optional):
```bash
python -m pip install -U sphinx myst-parser furo
```

---

## 2) Optional extras (SNN and datasets)

Install only if you plan to use SNNs or dataset loaders:
- pip:
  ```bash
  pip install "torch>=2.2" "norse>=0.0.9" "tonic>=1.4.0"
  ```
- conda:
  ```bash
  conda install -c conda-forge pytorch norse tonic
  ```

If these are not installed, the repository still imports cleanly; SNN paths are gated behind `snn.enabled` and dataset code falls back to synthetic examples.

---

## 3) Dataset workflow

Download datasets (requires `tonic`):
```bash
python scripts/download_datasets.py --dataset nmnist --root ./data/nmnist --split train
python scripts/download_datasets.py --dataset dvs_gesture --root ./data/dvs_gesture --split train
```

Notes:
- Pass `--root` to place files under a local directory (e.g., `./data/...`).
- If `tonic` is not installed, loaders fall back to synthetic small sequences at runtime with an explanatory message.
- For scripted splits, see `scripts/make_splits.py` (if applicable in your workflow).

---

## 4) Running experiments via CLI

If installed (`pip install -e .`), use the entrypoint:
```bash
# Torch-free path
dch-run experiment=dvs_gesture snn.enabled=false
```

Enable SNN path (requires `torch` and `norse`):
```bash
dch-run experiment=nmnist snn.enabled=true model=norse_lif
```

If `torch`/`norse` are missing when `snn.enabled=true`, the runner exits with:
```
SNN is enabled but required optional dependencies are missing: torch, norse
- Try: pip install 'torch>=2.2' 'norse>=0.0.9'
- Or:  conda install -c conda-forge pytorch norse
Alternatively run with 'snn.enabled=false'.
```

Run without installation (direct script invocation):
```bash
# Equivalent to dch-run, from repo root:
python scripts/run_experiment.py experiment=dvs_gesture snn.enabled=false
python scripts/run_experiment.py experiment=nmnist snn.enabled=true model=norse_lif
```

---

## 5) Configuration overview

Key configuration files:
- Pipeline defaults: [configs/pipeline.yaml](configs/pipeline.yaml)
- Experiments: [configs/experiments/dvs_gesture.yaml](configs/experiments/dvs_gesture.yaml), [configs/experiments/nmnist.yaml](configs/experiments/nmnist.yaml)
- DCH core: [configs/dch.yaml](configs/dch.yaml)
- FSM: [configs/fsm.yaml](configs/fsm.yaml)
- Scaffolding: [configs/scaffolding.yaml](configs/scaffolding.yaml)
- SNN model (optional): [configs/model/norse_lif.yaml](configs/model/norse_lif.yaml)

Override any config via dotlist on the CLI, for example:
```bash
dch-run experiment=dvs_gesture dch.k=4 traversal.horizon=8 fsm.theta=0.8
```

---

## 6) Reproducibility knobs

- Seeds and environment capture: [dch_pipeline/seeding.py](dch_pipeline/seeding.py)
- Set seeds via config or CLI dotlist, e.g.:
  ```bash
  dch-run experiment=dvs_gesture experiment.seeds=[123]
  ```
- Deterministic flags for torch (enabled internally when available) are set in the runner; on CPU-only runs this is a no-op.

For formal guidance, see:
- [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- [docs/EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)

---

## 7) Logging and artifacts

The runner writes artifacts using [dch_pipeline/logging_utils.py](dch_pipeline/logging_utils.py):
- CSV metrics: `./artifacts/<exp>*/metrics.csv`
- JSONL logs: `./artifacts/<exp>*/metrics.jsonl`
- Merged config: `./artifacts/<exp>*/config.merged.json`
- TensorBoard (if TB installed): `./artifacts/<exp>*/tb` (logging gracefully no-ops if missing)

File locations can be adjusted by config (e.g., `experiment.artifacts_dir`).

---

## 8) Benchmarks

Deterministic, single-line JSON outputs:
```bash
python benchmarks/benchmark_traversal.py --num-neurons 512 --num-edges 10000
python benchmarks/benchmark_pipeline.py --num-neurons 256 --event-rate 200 --steps 500
```
See source: [benchmarks/benchmark_traversal.py](../benchmarks/benchmark_traversal.py), [benchmarks/benchmark_pipeline.py](../benchmarks/benchmark_pipeline.py).

---

## 9) Optional docsite build

Building docs is optional and not required to use DCH:
```bash
python -m pip install -U sphinx myst-parser furo
make -C docs html
# or:
python -m sphinx -b html docs docs/_build/html
```
The root page is [docs/index.rst](index.rst).