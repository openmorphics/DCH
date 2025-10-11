# DCH Baselines

Two optional, fast-running baselines are provided. They are import-safe by default (no heavy deps imported at module import time), deterministic given the same seed, and print a single-line JSON summary when executed.

Baselines:
- Norse surrogate-gradient training baseline
- BindsNET unsupervised STDP baseline

Optional dependencies
- pip:
  - Norse SG: `pip install "torch>=2.2" "norse>=0.0.9" "tonic>=1.4.0"`
  - BindsNET STDP: `pip install "torch>=2.2" "bindsnet>=0.3" "tonic>=1.4.0"`
- conda (example):
  - `conda install pytorch -c pytorch -c nvidia` (choose per your hardware)
  - `pip install norse tonic bindsnet`

Runtime expectations
- Defaults target tiny, deterministic runs on CPU with synthetic data.
- Typical runtime: a few seconds.
- Datasets (NMNIST/DVS Gesture) are gated; actionable ImportError is raised if required deps are missing. Minimal dataset loaders are intentionally omitted to keep scope small.

Console scripts
- Norse SG: `dch-baseline-norse`
- BindsNET STDP: `dch-baseline-bindsnet`

Example invocations
- Norse SG (synthetic):
  - `dch-baseline-norse --config configs/baselines/norse_sg.yaml`
- BindsNET STDP (synthetic):
  - `dch-baseline-bindsnet --config configs/baselines/bindsnet_stdp.yaml`

Config files
- [configs/baselines/norse_sg.yaml](configs/baselines/norse_sg.yaml)
- [configs/baselines/bindsnet_stdp.yaml](configs/baselines/bindsnet_stdp.yaml)

Files
- [baselines/norse_sg.py](baselines/norse_sg.py)
- [baselines/bindsnet_stdp.py](baselines/bindsnet_stdp.py)

Notes
- On missing deps, scripts exit with code 2 and print an actionable message.
- Output is a single-line JSON with keys: baseline, dataset, accuracy|macro_f1|loss (if computed), elapsed_s, seed.
