# Dynamic Causal Hypergraph (DCH) — Multi-stage Dockerfile
# Targets:
#   - dch_cpu  : CPU-only environment (Python 3.11)
#   - dch_cuda : CUDA 12.1 + PyTorch runtime (from official PyTorch image)

# -------- CPU image --------
FROM python:3.11-slim AS dch_cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential graphviz ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# Install project with dev extras for linting/tests/docs
RUN python -m pip install -U pip wheel \
 && python -m pip install -e ".[dev]"

ENV PYTHONPATH=/workspace
ENTRYPOINT ["bash"]

# -------- CUDA image --------
# Uses PyTorch official runtime with CUDA 12.1 and cuDNN8
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS dch_cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential graphviz ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# Ensure pip is recent; install project with dev extras
RUN python -m pip install -U pip wheel \
 && python -m pip install -e ".[dev]"

# Sanity check: CUDA must be available
RUN python - <<'PY'\nimport torch\nassert torch.cuda.is_available(), 'CUDA not available in image'\nprint('CUDA device:', torch.cuda.get_device_name(0))\nPY

ENV PYTHONPATH=/workspace
ENTRYPOINT ["bash"]