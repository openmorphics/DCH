# Troubleshooting and FAQ

This page lists common issues and their fixes. DCH is torch-optional: the DCH-only path runs without torch/norse/tonic. SNN features are activated only when explicitly requested.

---

## 1) Import errors: optional dependencies

Symptoms often show up when requesting features that need optional packages. The runner provides actionable messages.

- SNN path requested but torch/norse missing
  - Message (from [scripts/run_experiment.py](scripts/run_experiment.py)):
    ```
    SNN is enabled but required optional dependencies are missing: torch, norse
    - Try: pip install 'torch>=2.2' 'norse>=0.0.9'
    - Or:  conda install -c conda-forge pytorch norse
    Alternatively run with 'snn.enabled=false'.
    ```
  - Fix:
    - pip:
      ```bash
      pip install "torch>=2.2" "norse>=0.0.9"
      ```
    - conda:
      ```bash
      conda install -c conda-forge pytorch norse
      ```
    - Or disable SNN:
      ```bash
      dch-run experiment=dvs_gesture snn.enabled=false
      ```

- Datasets requested but tonic missing
  - Typical behavior: falls back to a small synthetic sequence with a note explaining the fallback.
  - Fix (to use real datasets):
    - pip:
      ```bash
      pip install "tonic>=1.4.0"
      ```
    - conda:
      ```bash
      conda install -c conda-forge tonic
      ```
    - Then download:
      ```bash
      python scripts/download_datasets.py --dataset nmnist --root ./data/nmnist --split train
      python scripts/download_datasets.py --dataset dvs_gesture --root ./data/dvs_gesture --split train
      ```

- Numpy missing (required for metrics)
  - Message (from [scripts/run_experiment.py](scripts/run_experiment.py)):
    ```
    Optional dependency 'numpy' is required for metrics.
    - Try: pip install numpy
    - Or:  conda install -c conda-forge numpy
    ```
  - Fix:
    ```bash
    pip install "numpy>=1.24,<2.1"
    # or:
    conda install -c conda-forge numpy
    ```

Notes:
- You can always force the torch-free path with `snn.enabled=false`.
- The repository imports cleanly without torch/norse/tonic; optional features are lazily gated.

---

## 2) CLI and exit codes

The experiment runner [scripts/run_experiment.py](scripts/run_experiment.py) exits with code 2 for actionable errors (missing optional deps, missing configs). Read the stderr message; it always includes concrete pip/conda commands or a torch-free alternative.

Examples:
```bash
# Torch-free path
dch-run experiment=dvs_gesture snn.enabled=false

# SNN with explicit model (requires torch+norse)
dch-run experiment=nmnist snn.enabled=true model=norse_lif
```

---

## 3) Sphinx docsite build issues (optional)

Building the docs is optional and not required to use DCH. If you do build them:

- Install doc tooling:
  ```bash
  python -m pip install -U sphinx myst-parser furo
  ```
- Build:
  ```bash
  make -C docs html
  # or:
  python -m sphinx -b html docs docs/_build/html
  ```
- Theme errors (e.g., `ImportError: No module named 'furo'`):
  - The config in [docs/conf.py](conf.py) falls back to "alabaster" if `furo` is not available.
  - Installing `furo` yields a nicer theme but is not required.

---

## 4) Paths, logging, and artifacts

The runner writes artifacts using [dch_pipeline/logging_utils.py](../dch_pipeline/logging_utils.py):
- CSV: `./artifacts/<exp>*/metrics.csv`
- JSONL: `./artifacts/<exp>*/metrics.jsonl`
- Merged config: `./artifacts/<exp>*/config.merged.json`
- TensorBoard (if installed): `./artifacts/<exp>*/tb` (no-op if TB missing)

Adjust via configs or CLI dotlist, e.g.:
```bash
dch-run experiment=dvs_gesture experiment.artifacts_dir=./runs
```

---

## 5) Where to find configuration and protocol docs

- Pipeline and core configs:
  - [configs/pipeline.yaml](../configs/pipeline.yaml)
  - [configs/experiments/dvs_gesture.yaml](../configs/experiments/dvs_gesture.yaml)
  - [configs/experiments/nmnist.yaml](../configs/experiments/nmnist.yaml)
  - [configs/dch.yaml](../configs/dch.yaml), [configs/fsm.yaml](../configs/fsm.yaml), [configs/scaffolding.yaml](../configs/scaffolding.yaml)
- Evaluation and reproducibility:
  - [docs/EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)
  - [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- Usage guide:
  - [docs/USAGE.md](USAGE.md)

If an issue persists after following the steps above, re-run with `-v` or capture stderr and open an issue including the full command and environment details (Python version, OS).