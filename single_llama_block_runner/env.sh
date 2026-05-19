#!/usr/bin/env bash
# Resolve Python/torchrun from repo-root .venv, conda vllm env, or PATH.

VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-vllm-py312}"

resolve_conda_env_bin() {
  local exe="$1"
  if [[ -n "${CONDA_PREFIX:-}" && "$(basename "$CONDA_PREFIX")" == "$VLLM_CONDA_ENV" && -x "${CONDA_PREFIX}/bin/${exe}" ]]; then
    echo "${CONDA_PREFIX}/bin/${exe}"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -x "${conda_base}/envs/${VLLM_CONDA_ENV}/bin/${exe}" ]]; then
      echo "${conda_base}/envs/${VLLM_CONDA_ENV}/bin/${exe}"
      return 0
    fi
  fi
  return 1
}

if [[ -x "../.venv/bin/python" ]]; then
  PYTHON="../.venv/bin/python"
  TORCHRUN="../.venv/bin/torchrun"
elif conda_python="$(resolve_conda_env_bin python)"; then
  PYTHON="$conda_python"
  TORCHRUN="$(resolve_conda_env_bin torchrun || echo torchrun)"
else
  PYTHON="${PYTHON:-python}"
  TORCHRUN="${TORCHRUN:-torchrun}"
fi

results_json_path() {
  local tp_size="$1"
  mkdir -p results
  echo "results/tp${tp_size}_$(date -u +%Y%m%dT%H%M%SZ).json"
}
