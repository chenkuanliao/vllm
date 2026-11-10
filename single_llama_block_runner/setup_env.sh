#!/usr/bin/env bash
# Create or update this repository's one environment for the Llama block
# runner, `vllm-experiments`, for the GPU type of this machine.
#
#   ./setup_env.sh v100
#   ./setup_env.sh a100
#
# The run_*.sh scripts find the env through env.sh; no activation needed.
#
# torch is 2.11.0 on both GPUs: the version this vLLM source pins
# (requirements/cuda.txt), which its compiled kernels need.
#   - V100: the default attention backend is PyTorch SDPA, so vLLM itself is
#     not installed. torch comes from the CUDA 12.6 build, which ships V100
#     (sm_70) kernels and runs on the V100 box's CUDA 12.4 driver.
#   - A100: vLLM is installed from this repository with its precompiled
#     kernels (FlashAttention-2 varlen, triton-prefill), using the CUDA 12.9
#     builds of torch and of vLLM's kernels (the recipe previously documented
#     in README.md for the A100 box).
set -euo pipefail
cd "$(dirname "$0")"

GPU="${1:-}"
ENV_NAME="${VLLM_CONDA_ENV:-vllm-experiments}"
PYTHON_VERSION="3.12"
TORCH_VERSION="2.11.0"
REPO_ROOT="$(cd .. && pwd)"

case "$GPU" in
  v100|V100|a100|A100) GPU="$(echo "$GPU" | tr '[:upper:]' '[:lower:]')" ;;
  *) echo "usage: ./setup_env.sh v100|a100" >&2; exit 2 ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
  want="$(echo "$GPU" | tr '[:lower:]' '[:upper:]')"
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "$want"; then
    echo "warning: requested $want but this machine has:" \
      "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | tr '\n' ' ')" >&2
  fi
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env $ENV_NAME (python $PYTHON_VERSION)"
  conda create -y -q -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "Updating conda env $ENV_NAME"
fi
conda activate "$ENV_NAME"

python -m pip install -q --upgrade pip
if [[ "$GPU" == "v100" ]]; then
  python -m pip install -q "torch==$TORCH_VERSION" numpy \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple
else
  python -m pip install -q uv
  py="$(command -v python)"
  uv pip install --python "$py" --upgrade setuptools wheel
  uv pip install --python "$py" "torch==$TORCH_VERSION" torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu129
  (cd "$REPO_ROOT" && VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 VLLM_USE_PRECOMPILED=1 \
    uv pip install --python "$py" -e . --torch-backend=auto)
fi
python -m pip check

python - "$GPU" "$TORCH_VERSION" <<'EOF'
import sys, torch
gpu, want = sys.argv[1], sys.argv[2]
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), arch {torch.cuda.get_arch_list()}")
if torch.__version__.split("+")[0] != want:
    sys.exit(f"expected torch {want}")
if gpu == "a100":
    from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
    if not is_flash_attn_varlen_func_available():
        sys.exit("vLLM FlashAttention-2 varlen is not available")
    print("vLLM FlashAttention-2 varlen: available")
EOF
echo "Ready: the run_*.sh scripts use conda env $ENV_NAME"
