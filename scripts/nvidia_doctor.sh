#!/usr/bin/env bash
set -euo pipefail

# Diagnose whether this environment is ready for the NVIDIA/tch backend.
#
# Usage:
#   bash scripts/nvidia_doctor.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

REQUIRE_GPU="${REQUIRE_GPU:-1}" # 1 => fail if NVIDIA CUDA is unavailable

echo "== System =="
uname -a
echo

echo "== Rust Toolchain =="
if command -v cargo >/dev/null 2>&1; then
  cargo --version
else
  echo "cargo not found"
fi
echo

echo "== NVIDIA Driver / GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader
else
  echo "nvidia-smi not found"
  if [[ "${REQUIRE_GPU}" == "1" ]]; then
    echo "nvidia_doctor: FAIL (NVIDIA GPU required)"
    exit 1
  fi
fi
echo

echo "== Python Torch CUDA =="
python3 - <<'PY'
import json
try:
    import torch
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
    raise SystemExit(0)

data = {
    "ok": True,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
}
print(json.dumps(data, indent=2))
PY
echo

CUDA_OK="$(python3 - <<'PY'
try:
    import torch
    print("1" if torch.cuda.is_available() else "0")
except Exception:
    print("0")
PY
)"
if [[ "${REQUIRE_GPU}" == "1" && "${CUDA_OK}" != "1" ]]; then
  echo "nvidia_doctor: FAIL (torch CUDA unavailable)"
  exit 1
fi

echo "== libtorch path from python torch =="
TORCH_LIB="$(python3 - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
echo "${TORCH_LIB}"
echo

echo "== cargo check (tch-backend) =="
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
if [[ "$(uname -s)" == "Darwin" ]]; then
  export DYLD_LIBRARY_PATH="${TORCH_LIB}"
else
  export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi
cargo check --features tch-backend
echo

echo "== tensor runtime probe =="
PROBE_OUTPUT="$(cargo run --quiet --features tch-backend -- train --mode tensor --style classic --steps 1 --data input.txt --seed 1337 2>&1 || true)"
echo "${PROBE_OUTPUT}"
if [[ "${PROBE_OUTPUT}" == *"tensor train failed"* ]]; then
  echo "nvidia_doctor: FAIL (tensor runtime probe failed)"
  exit 1
fi
if [[ "${REQUIRE_GPU}" == "1" && "${PROBE_OUTPUT}" != *"using_gpu=true"* ]]; then
  echo "nvidia_doctor: FAIL (tensor runtime did not report using_gpu=true)"
  exit 1
fi

echo

echo "nvidia_doctor: OK"
