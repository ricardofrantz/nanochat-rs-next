#!/usr/bin/env bash
set -euo pipefail

# Colab helper for GPU benchmark runs against nanoGPT / nanochat.
#
# Usage (from repo root):
#   bash scripts/colab_gpu_benchmark.sh
#   BASELINE=nanogpt bash scripts/colab_gpu_benchmark.sh
#   BASELINE=both bash scripts/colab_gpu_benchmark.sh
#   PROFILE=full BASELINE=both bash scripts/colab_gpu_benchmark.sh
#   BASELINE=nanochat EXTRA_ARGS="--nanochat-num-iterations 30" bash scripts/colab_gpu_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASELINE="${BASELINE:-nanogpt}"
PROFILE="${PROFILE:-auto}" # auto|quick|full
EXTRA_ARGS="${EXTRA_ARGS:-}"
TORCH_PIP_INDEX_URL="${TORCH_PIP_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_PIP_FALLBACK_INDEX_URL="${TORCH_PIP_FALLBACK_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
OURS_CARGO_FEATURES="${OURS_CARGO_FEATURES:-tch-backend}"
REQUIRE_GPU="${REQUIRE_GPU:-1}" # 1|0
RUN_DOCTOR="${RUN_DOCTOR:-1}" # 1|0
NANOGPT_REF="${NANOGPT_REF:-auto}"
NANOCHAT_REF="${NANOCHAT_REF:-auto}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU runtime required. In Colab: Runtime -> Change runtime type -> GPU."
  exit 1
fi

if ! command -v rustup >/dev/null 2>&1; then
  curl -LsSf https://sh.rustup.rs | sh -s -- -y
fi

source "${HOME}/.cargo/env"
rustup toolchain install stable --profile minimal
rustup default stable

python3 --version
cargo --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1

if [[ "${RUN_DOCTOR}" == "1" ]]; then
  REQUIRE_GPU="${REQUIRE_GPU}" bash scripts/nvidia_doctor.sh
fi

if [[ "${PROFILE}" == "auto" ]]; then
  GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
  if [[ "${GPU_MEM_MB}" =~ ^[0-9]+$ ]] && (( GPU_MEM_MB >= 30000 )); then
    PROFILE="full"
  else
    PROFILE="quick"
  fi
fi

echo "Using PROFILE=${PROFILE} BASELINE=${BASELINE} NANOGPT_REF=${NANOGPT_REF} NANOCHAT_REF=${NANOCHAT_REF}"

ARGS=(
  --baseline "${BASELINE}"
  --install-deps
  --ours-cargo-features "${OURS_CARGO_FEATURES}"
  --nanogpt-ref "${NANOGPT_REF}"
  --nanochat-ref "${NANOCHAT_REF}"
  --torch-pip-index-url "${TORCH_PIP_INDEX_URL}"
  --torch-pip-fallback-index-url "${TORCH_PIP_FALLBACK_INDEX_URL}"
)

if [[ "${REQUIRE_GPU}" == "1" ]]; then
  ARGS+=(--require-gpu)
else
  ARGS+=(--no-require-gpu)
fi

if [[ "${PROFILE}" == "quick" ]]; then
  ARGS+=(
    --ours-steps 300
    --nanogpt-max-iters 80
    --nanogpt-eval-interval 20
    --nanogpt-eval-iters 20
    --nanogpt-log-interval 10
    --nanochat-num-shards 1
    --nanochat-max-chars 10000000
    --nanochat-depth 4
    --nanochat-device-batch-size 2
    --nanochat-total-batch-size 4096
    --nanochat-eval-every 20
    --nanochat-eval-tokens 32768
    --nanochat-num-iterations 30
  )
elif [[ "${PROFILE}" != "full" ]]; then
  echo "Unknown PROFILE='${PROFILE}'. Use auto, quick, or full."
  exit 1
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  ARGS+=("${EXTRA_ARR[@]}")
fi

python3 scripts/benchmark_karpathy.py "${ARGS[@]}"
