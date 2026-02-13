#!/usr/bin/env bash
set -euo pipefail

# GPU throughput profiling wrapper for this repo vs nanochat baseline.
#
# Usage:
#   bash scripts/profile_gpu.sh
#   REQUIRE_GPU=0 OURS_CARGO_FEATURES="" NANOCHAT_DEVICE_TYPE=cpu bash scripts/profile_gpu.sh
#   EXTRA_BENCH_ARGS="--nanochat-num-iterations 30 --nanochat-eval-every 10" bash scripts/profile_gpu.sh
#   BASELINE_MODE=skip bash scripts/profile_gpu.sh
#   NANOCHAT_TRAIN_TIMEOUT_SEC=0 bash scripts/profile_gpu.sh
#   DRY_RUN=1 BASELINE_MODE=skip bash scripts/profile_gpu.sh
#
# Artifacts:
#   results/gpu_profile_<timestamp>/run.log
#   results/gpu_profile_<timestamp>/summary.txt
#   results/gpu_profile_<timestamp>/nvidia_smi_{pre,post}.txt (if nvidia-smi is available)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

REQUIRE_GPU="${REQUIRE_GPU:-1}"                 # 1|0
INSTALL_DEPS="${INSTALL_DEPS:-0}"               # 1|0
OURS_CARGO_FEATURES="${OURS_CARGO_FEATURES-tch-backend}"
OURS_STEPS="${OURS_STEPS:-1000}"
OURS_DATA="${OURS_DATA:-input.txt}"
SEED="${SEED:-1337}"
BASELINE="${BASELINE:-nanochat}"
NANOCHAT_DEVICE_TYPE="${NANOCHAT_DEVICE_TYPE:-cuda}"
NANOCHAT_NUM_ITERATIONS="${NANOCHAT_NUM_ITERATIONS:-60}"
NANOCHAT_TRAIN_TIMEOUT_SEC="${NANOCHAT_TRAIN_TIMEOUT_SEC:-900}"
NANOCHAT_DISABLE_COMPILE="${NANOCHAT_DISABLE_COMPILE:-0}" # 1|0
BASELINE_MODE="${BASELINE_MODE:-run}" # run|skip
DRY_RUN="${DRY_RUN:-0}" # 1|0
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"

if [[ "${BASELINE}" != "nanochat" ]]; then
  echo "profile_gpu: only BASELINE=nanochat is supported."
  exit 1
fi

if [[ "${BASELINE_MODE}" != "run" && "${BASELINE_MODE}" != "skip" ]]; then
  echo "profile_gpu: BASELINE_MODE must be 'run' or 'skip'."
  exit 1
fi

if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "profile_gpu: DRY_RUN must be '0' or '1'."
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]] && [[ "${REQUIRE_GPU}" == "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "profile_gpu: nvidia-smi not found and REQUIRE_GPU=1."
  exit 1
fi

BENCH_CMD=(
  python3 scripts/benchmark_karpathy.py
  --baseline "${BASELINE}"
  --ours-cargo-features "${OURS_CARGO_FEATURES}"
  --ours-steps "${OURS_STEPS}"
  --ours-data "${OURS_DATA}"
  --seed "${SEED}"
  --nanochat-device-type "${NANOCHAT_DEVICE_TYPE}"
  --nanochat-num-iterations "${NANOCHAT_NUM_ITERATIONS}"
  --nanochat-train-timeout-sec "${NANOCHAT_TRAIN_TIMEOUT_SEC}"
  --baseline-mode "${BASELINE_MODE}"
)

if [[ "${REQUIRE_GPU}" == "1" ]]; then
  BENCH_CMD+=(--require-gpu)
else
  BENCH_CMD+=(--no-require-gpu)
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  BENCH_CMD+=(--install-deps)
fi

if [[ "${NANOCHAT_DISABLE_COMPILE}" == "1" ]]; then
  BENCH_CMD+=(--nanochat-disable-compile)
fi

if [[ -n "${EXTRA_BENCH_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_BENCH_ARGS})
  BENCH_CMD+=("${EXTRA_ARR[@]}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="results/gpu_profile_${STAMP}"
  RUN_LOG="${RUN_DIR}/run.log"
  SUMMARY_TXT="${RUN_DIR}/summary.txt"
  NVIDIA_PRE="${RUN_DIR}/nvidia_smi_pre.txt"
  NVIDIA_POST="${RUN_DIR}/nvidia_smi_post.txt"
  echo "profile_gpu: dry-run command preview (DRY_RUN=1)"
  echo "profile_gpu: would write artifacts under ${RUN_DIR}"
  echo "profile_gpu: running benchmark command:"
  printf '  %q' "${BENCH_CMD[@]}"
  echo
  exit 0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="results/gpu_profile_${STAMP}"
mkdir -p "${RUN_DIR}"

RUN_LOG="${RUN_DIR}/run.log"
SUMMARY_TXT="${RUN_DIR}/summary.txt"
NVIDIA_PRE="${RUN_DIR}/nvidia_smi_pre.txt"
NVIDIA_POST="${RUN_DIR}/nvidia_smi_post.txt"

capture_gpu_snapshot() {
  local output_file="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader >"${output_file}" || true
  fi
}

capture_gpu_snapshot "${NVIDIA_PRE}"

echo "profile_gpu: writing artifacts under ${RUN_DIR}"
echo "profile_gpu: running benchmark command:"
printf '  %q' "${BENCH_CMD[@]}"
echo

"${BENCH_CMD[@]}" | tee "${RUN_LOG}"

capture_gpu_snapshot "${NVIDIA_POST}"

BENCH_JSON="$(sed -n 's/^benchmark complete: //p' "${RUN_LOG}" | tail -n1)"
if [[ -z "${BENCH_JSON}" ]]; then
  echo "profile_gpu: could not locate benchmark json path in ${RUN_LOG}"
  exit 1
fi
if [[ ! -f "${BENCH_JSON}" ]]; then
  echo "profile_gpu: benchmark json path does not exist: ${BENCH_JSON}"
  exit 1
fi

python3 - "${BENCH_JSON}" >"${SUMMARY_TXT}" <<'PY'
import json
import pathlib
import sys

json_path = pathlib.Path(sys.argv[1])
data = json.loads(json_path.read_text(encoding="utf-8"))

ours = data.get("runs", {}).get("ours_tensor", {})
ours_metrics = ours.get("metrics", {})
nano = data.get("runs", {}).get("nanochat", {})
nano_steps = nano.get("steps_tail", []) or []

ours_tps = ours_metrics.get("tokens_per_sec")
nano_tps = None
if nano_steps:
    nano_tps = nano_steps[-1].get("tok_per_sec")

ratio = None
if isinstance(ours_tps, (int, float)) and isinstance(nano_tps, (int, float)) and nano_tps > 0:
    ratio = float(ours_tps) / float(nano_tps)

print(f"benchmark_json={json_path}")
print(f"benchmark_md={data.get('artifact_md')}")
print(f"logs_dir={data.get('artifact_logs_dir')}")
print("")
print(f"ours.status={ours.get('status')}")
print(f"ours.elapsed_sec={ours.get('elapsed_sec')}")
print(f"ours.backend={ours_metrics.get('backend')}")
print(f"ours.device={ours_metrics.get('device')}")
print(f"ours.using_gpu={ours_metrics.get('using_gpu')}")
print(f"ours.steps_per_sec={ours_metrics.get('steps_per_sec')}")
print(f"ours.tokens_per_sec={ours_tps}")
print(f"ours.final_loss={ours_metrics.get('final_loss')}")
print(f"ours.val_loss={ours_metrics.get('val_loss')}")
print("")
print(f"nanochat.status={nano.get('status')}")
print(f"nanochat.elapsed_sec={nano.get('elapsed_sec')}")
print(f"nanochat.last_step={nano.get('last_step')}")
print(f"nanochat.last_eval={nano.get('last_eval')}")
print(f"nanochat.min_val_bpb={nano.get('min_val_bpb')}")
print(f"nanochat.last_tok_per_sec={nano_tps}")
if ratio is None:
    print("throughput_ratio.ours_over_nanochat=<unavailable>")
else:
    print(f"throughput_ratio.ours_over_nanochat={ratio:.4f}")
PY

cat "${SUMMARY_TXT}"
