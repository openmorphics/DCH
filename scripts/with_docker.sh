#!/usr/bin/env bash
# Dynamic Causal Hypergraph (DCH)
# Helper to build and run Docker images for CPU and CUDA targets.
# See also: docs/FrameworkDecision.md and Dockerfile

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_CPU="dch:cpu"
IMAGE_CUDA="dch:cuda"
TARGET_CPU="dch_cpu"
TARGET_CUDA="dch_cuda"

usage() {
  cat <<'USAGE'
Usage:
  scripts/with_docker.sh build [cpu|cuda]
  scripts/with_docker.sh run   [cpu|cuda] [-- ARGS...]

Examples:
  # Build images
  scripts/with_docker.sh build cpu
  scripts/with_docker.sh build cuda

  # Run CPU image and open shell
  scripts/with_docker.sh run cpu -- bash

  # Run tests inside CPU image
  scripts/with_docker.sh run cpu -- pytest -q

  # Run CUDA image and verify GPU
  scripts/with_docker.sh run cuda -- python -c "import torch;print(torch.cuda.is_available())"
USAGE
}

build_image() {
  local which="${1:-cpu}"
  case "$which" in
    cpu)
      docker build -t "${IMAGE_CPU}" -f "${REPO_ROOT}/Dockerfile" --target "${TARGET_CPU}" "${REPO_ROOT}"
      ;;
    cuda)
      docker build -t "${IMAGE_CUDA}" -f "${REPO_ROOT}/Dockerfile" --target "${TARGET_CUDA}" "${REPO_ROOT}"
      ;;
    *)
      echo "Unknown build target: ${which}" 1>&2
      exit 2
      ;;
  esac
}

run_container() {
  local which="${1:-cpu}"
  shift || true
  local args=( "$@" )

  local uid gid user_flag
  uid="$(id -u)"; gid="$(id -g)"
  user_flag=( "--user" "${uid}:${gid}" )

  local vol_flag=( "-v" "${REPO_ROOT}:/workspace" )
  local workdir=( "-w" "/workspace" )

  case "$which" in
    cpu)
      docker run --rm -it "${user_flag[@]}" "${vol_flag[@]}" "${workdir[@]}" \
        -e PYTHONUNBUFFERED=1 -e PIP_NO_CACHE_DIR=1 \
        "${IMAGE_CPU}" \
        "${args[@]}"
      ;;
    cuda)
      # Requires nvidia-container-toolkit; will use all GPUs by default
      docker run --rm -it --gpus all "${user_flag[@]}" "${vol_flag[@]}" "${workdir[@]}" \
        -e PYTHONUNBUFFERED=1 -e PIP_NO_CACHE_DIR=1 \
        "${IMAGE_CUDA}" \
        "${args[@]}"
      ;;
    *)
      echo "Unknown run target: ${which}" 1>&2
      exit 3
      ;;
  esac
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local cmd="${1:-}"; shift || true
  case "$cmd" in
    help|-h|--help)
      usage
      ;;
    build)
      if [[ $# -lt 1 ]]; then
        echo "Specify target: cpu|cuda" 1>&2
        exit 1
      fi
      build_image "$1"
      ;;
    run)
      if [[ $# -lt 1 ]]; then
        echo "Specify target: cpu|cuda" 1>&2
        exit 1
      fi
      local target="$1"; shift || true
      if [[ "${1:-}" == "--" ]]; then shift; fi
      run_container "$target" "$@"
      ;;
    *)
      echo "Unknown command: ${cmd}" 1>&2
      usage
      exit 1
      ;;
  esac
}

main "$@"