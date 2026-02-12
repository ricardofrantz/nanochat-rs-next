#!/usr/bin/env python3
"""
GPU-first benchmark harness for comparing this repo against Karpathy baselines.

Outputs:
- results/benchmark_karpathy_<timestamp>.json
- results/benchmark_karpathy_<timestamp>.md
- results/benchmark_karpathy_<timestamp>_logs/

Usage:
  python3 scripts/benchmark_karpathy.py --baseline nanochat --install-deps
  python3 scripts/benchmark_karpathy.py --baseline nanochat --install-deps --run-ours-scalar-fallback
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
EXTERNAL_DIR = REPO_ROOT / "external"
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
DEFAULT_TORCH_FALLBACK_INDEX = "https://download.pytorch.org/whl/cu126"
DEFAULT_NANOCHAT_PINNED_REF = "2f096867244e3d00a50284d1be05fa3f5dcfb84b"
GPU_BACKEND_HINTS = ("tch", "candle", "wgpu", "cuda", "metal", "vulkan", "opencl", "burn")


def run_bash(
    command: str,
    *,
    cwd: pathlib.Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
        elapsed = time.perf_counter() - started
        result = {
            "command": command,
            "cwd": str(cwd) if cwd else str(REPO_ROOT),
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "output": proc.stdout,
            "timed_out": False,
            "timeout_sec": timeout_sec,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        output = ""
        if exc.stdout is not None:
            output = exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", errors="replace")
        result = {
            "command": command,
            "cwd": str(cwd) if cwd else str(REPO_ROOT),
            "returncode": 124,
            "elapsed_sec": elapsed,
            "output": output,
            "timed_out": True,
            "timeout_sec": timeout_sec,
        }
        if check:
            tail = output[-8000:]
            raise RuntimeError(
                f"command timed out ({timeout_sec}s): {command}\n"
                f"--- output tail ---\n{tail}\n--- end tail ---"
            )
    if check and result["returncode"] != 0:
        tail = result["output"][-8000:]
        raise RuntimeError(
            f"command failed ({result['returncode']}): {command}\n"
            f"--- output tail ---\n{tail}\n--- end tail ---"
        )
    return result


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_command_log(log_dir: pathlib.Path, name: str, result: dict[str, Any]) -> None:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    log_path = log_dir / f"{safe_name}.log"
    lines = [
        f"$ {result['command']}",
        f"cwd: {result['cwd']}",
        f"returncode: {result['returncode']}",
        f"elapsed_sec: {result['elapsed_sec']:.3f}",
        f"timed_out: {result.get('timed_out', False)}",
        f"timeout_sec: {result.get('timeout_sec')}",
        "",
        result["output"],
    ]
    write_text(log_path, "\n".join(lines))


def resolve_origin_default_ref(repo_path: pathlib.Path, log_dir: pathlib.Path, name: str) -> str:
    symbolic = run_bash(
        "git symbolic-ref --quiet --short refs/remotes/origin/HEAD", cwd=repo_path, check=False
    )
    write_command_log(log_dir, f"{name}_origin_head_symbolic_ref", symbolic)
    if symbolic["returncode"] == 0 and symbolic["output"].strip():
        return symbolic["output"].strip()

    remote_show = run_bash(
        "git remote show origin | sed -n '/HEAD branch/s/.*: //p'",
        cwd=repo_path,
        check=False,
    )
    write_command_log(log_dir, f"{name}_origin_head_remote_show", remote_show)
    head_branch = remote_show["output"].strip()
    if remote_show["returncode"] == 0 and head_branch:
        return f"origin/{head_branch}"

    return "origin/main"


def checkout_ref(repo_path: pathlib.Path, name: str, ref: str, log_dir: pathlib.Path) -> str:
    target = ref
    if ref == "auto":
        target = resolve_origin_default_ref(repo_path, log_dir, name)

    checkout = run_bash(
        f"git checkout --detach {shlex.quote(target)}", cwd=repo_path, check=False
    )
    write_command_log(log_dir, f"{name}_checkout_{target.replace('/', '_')}", checkout)
    if checkout["returncode"] == 0:
        return target

    if "/" not in target:
        prefixed = f"origin/{target}"
        checkout_prefixed = run_bash(
            f"git checkout --detach {shlex.quote(prefixed)}",
            cwd=repo_path,
            check=False,
        )
        write_command_log(
            log_dir,
            f"{name}_checkout_{prefixed.replace('/', '_')}",
            checkout_prefixed,
        )
        if checkout_prefixed["returncode"] == 0:
            return prefixed

    raise RuntimeError(
        f"failed to checkout ref '{ref}' for {name}; resolved target='{target}'"
    )


def ensure_repo(url: str, name: str, ref: str, log_dir: pathlib.Path) -> tuple[pathlib.Path, str]:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    repo_path = EXTERNAL_DIR / name
    if not repo_path.exists():
        result = run_bash(
            f"git clone {shlex.quote(url)} {shlex.quote(str(repo_path))}"
        )
        write_command_log(log_dir, f"clone_{name}", result)

    fetch = run_bash("git fetch --tags --prune origin", cwd=repo_path)
    write_command_log(log_dir, f"fetch_{name}", fetch)
    resolved_ref = checkout_ref(repo_path, name, ref, log_dir)
    return repo_path, resolved_ref


def git_commit(repo_path: pathlib.Path) -> str:
    result = run_bash("git rev-parse HEAD", cwd=repo_path)
    return result["output"].strip()


def parse_key_values(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)", line):
        out[key] = value
    return out


def maybe_float(value: str) -> Any:
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
        return float(value)
    if value in {"true", "false"}:
        return value == "true"
    return value


def query_gpu_info(require_gpu: bool) -> dict[str, Any]:
    probe = run_bash("command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", check=False)
    if probe["returncode"] != 0 or not probe["output"].strip():
        if require_gpu:
            raise RuntimeError("GPU required but nvidia-smi was not available.")
        return {"available": False, "devices": []}

    devices = []
    for line in probe["output"].strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        devices.append(
            {
                "name": parts[0],
                "memory_total": parts[1],
                "driver_version": parts[2],
            }
        )
    return {"available": bool(devices), "devices": devices}


def parse_ours_features(ours_cargo_features: str) -> set[str]:
    return {
        feature.strip().lower()
        for feature in ours_cargo_features.split(",")
        if feature.strip()
    }


def detect_requested_gpu_backend(ours_cargo_features: str) -> dict[str, Any]:
    ours_features = parse_ours_features(ours_cargo_features)
    matched = [hint for hint in GPU_BACKEND_HINTS if any(hint in feature for feature in ours_features)]
    return {
        "gpu_backend_detected": bool(matched),
        "backend_hints": matched,
        "requested_features": sorted(ours_features),
    }


def detect_repo_gpu_backend() -> dict[str, Any]:
    cargo_toml = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8").lower()
    matched = [hint for hint in GPU_BACKEND_HINTS if hint in cargo_toml]
    return {
        "gpu_backend_detected": bool(matched),
        "backend_hints": matched,
    }


def install_python_torch_cuda(
    *,
    install_deps: bool,
    log_dir: pathlib.Path,
    index_url: str,
    fallback_index_url: str,
) -> None:
    if not install_deps:
        return
    install = run_bash(
        "python3 -m pip install --upgrade pip && "
        f"python3 -m pip install torch --index-url {shlex.quote(index_url)}",
        check=False,
    )
    write_command_log(log_dir, "install_python_torch_primary", install)
    if install["returncode"] == 0:
        return
    fallback = run_bash(
        "python3 -m pip install --upgrade pip && "
        f"python3 -m pip install torch --index-url {shlex.quote(fallback_index_url)}",
        check=False,
    )
    write_command_log(log_dir, "install_python_torch_fallback", fallback)
    if fallback["returncode"] != 0:
        raise RuntimeError(
            "failed to install python torch from both primary and fallback indexes"
        )


def probe_python_torch_cuda(log_dir: pathlib.Path) -> dict[str, Any]:
    probe = run_bash(
        "python3 - <<'PY'\n"
        "import json\n"
        "try:\n"
        "    import torch\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'error': str(e)}))\n"
        "    raise SystemExit(0)\n"
        "data = {\n"
        "    'ok': True,\n"
        "    'torch_version': torch.__version__,\n"
        "    'cuda_available': bool(torch.cuda.is_available()),\n"
        "    'cuda_version': torch.version.cuda,\n"
        "    'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,\n"
        "    'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],\n"
        "}\n"
        "print(json.dumps(data))\n"
        "PY",
        check=False,
    )
    write_command_log(log_dir, "probe_python_torch_cuda", probe)
    try:
        return json.loads(probe["output"].strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": "failed to parse python torch probe output"}


def ensure_ours_tch_deps(
    install_deps: bool,
    log_dir: pathlib.Path,
    index_url: str,
    fallback_index_url: str,
) -> dict[str, Any]:
    if not install_deps:
        return probe_python_torch_cuda(log_dir)
    install_python_torch_cuda(
        install_deps=True,
        log_dir=log_dir,
        index_url=index_url,
        fallback_index_url=fallback_index_url,
    )
    return probe_python_torch_cuda(log_dir)


def run_ours_tensor(args: argparse.Namespace, log_dir: pathlib.Path, dataset_path: pathlib.Path) -> dict[str, Any]:
    feature_flag = ""
    if args.ours_cargo_features.strip():
        feature_flag = f"--features {shlex.quote(args.ours_cargo_features.strip())}"
    cmd = (
        f"cargo run --release {feature_flag} -- train --mode tensor --style classic "
        f"--steps {args.ours_steps} --data {shlex.quote(str(dataset_path))} --seed {args.seed}"
    )
    env = os.environ.copy()
    # Use Python-installed torch as libtorch provider in Colab and similar environments.
    env["LIBTORCH_USE_PYTORCH"] = "1"
    env.setdefault("LIBTORCH_BYPASS_VERSION_CHECK", "1")
    torch_lib = run_bash(
        "python3 - <<'PY'\nimport os, torch\nprint(os.path.join(os.path.dirname(torch.__file__), 'lib'))\nPY",
        check=False,
    )
    if torch_lib["returncode"] == 0:
        torch_lib_path = torch_lib["output"].strip()
        if sys.platform == "darwin":
            existing = env.get("DYLD_LIBRARY_PATH", "")
            env["DYLD_LIBRARY_PATH"] = (
                f"{torch_lib_path}:{existing}" if existing else torch_lib_path
            )
        else:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{torch_lib_path}:{existing}" if existing else torch_lib_path
            )
    result = run_bash(cmd, cwd=REPO_ROOT, env=env, check=False)
    write_command_log(log_dir, "ours_tensor", result)

    output = result["output"]
    last_line = ""
    for line in reversed(output.splitlines()):
        if line.strip():
            last_line = line.strip()
            break

    unsupported = "not implemented yet" in output.lower()
    metrics = parse_key_values(last_line)
    parsed_metrics = {k: maybe_float(v) for k, v in metrics.items()}
    return {
        "status": "unsupported" if unsupported else ("ok" if result["returncode"] == 0 else "failed"),
        "returncode": result["returncode"],
        "elapsed_sec": result["elapsed_sec"],
        "message": last_line,
        "metrics": parsed_metrics,
    }


def ensure_ours_used_cuda(ours_tensor: dict[str, Any], require_gpu: bool) -> None:
    if not require_gpu:
        return
    metrics = ours_tensor.get("metrics", {})
    using_gpu = metrics.get("using_gpu")
    device = str(metrics.get("device", ""))
    if isinstance(using_gpu, bool):
        if using_gpu:
            return
    elif device.startswith("cuda:"):
        return
    raise RuntimeError(
        f"GPU required but our tensor run did not report CUDA usage: {ours_tensor}"
    )


def run_ours_scalar(args: argparse.Namespace, log_dir: pathlib.Path, dataset_path: pathlib.Path) -> dict[str, Any]:
    cmd = (
        "cargo run --release -- train --mode scalar --style classic "
        f"--steps {args.ours_steps} --data {shlex.quote(str(dataset_path))} --seed {args.seed}"
    )
    result = run_bash(cmd, cwd=REPO_ROOT)
    write_command_log(log_dir, "ours_scalar", result)

    line = ""
    for candidate in reversed(result["output"].splitlines()):
        if candidate.startswith("mode="):
            line = candidate.strip()
            break
    metrics = {k: maybe_float(v) for k, v in parse_key_values(line).items()}
    return {
        "status": "ok",
        "returncode": result["returncode"],
        "elapsed_sec": result["elapsed_sec"],
        "metrics": metrics,
    }


def ensure_uv(install_deps: bool, log_dir: pathlib.Path) -> None:
    if not install_deps:
        return
    result = run_bash("command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh")
    write_command_log(log_dir, "install_uv", result)


def run_nanochat(args: argparse.Namespace, log_dir: pathlib.Path, nanochat_repo: pathlib.Path) -> dict[str, Any]:
    ensure_uv(args.install_deps, log_dir)
    if args.install_deps:
        sync_cmd = "uv sync --extra gpu"
        if args.nanochat_device_type != "cuda":
            sync_cmd = "uv sync"
        sync = run_bash(sync_cmd, cwd=nanochat_repo)
        write_command_log(log_dir, "nanochat_uv_sync", sync)
    elif not (nanochat_repo / ".venv").exists():
        raise RuntimeError(
            "nanochat/.venv is missing. Re-run with --install-deps to set up nanochat dependencies."
        )

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["NANOCHAT_BASE_DIR"] = str(nanochat_repo / ".bench_cache")
    if args.nanochat_disable_compile:
        env["TORCH_COMPILE_DISABLE"] = "1"

    download = run_bash(
        f"uv run python -m nanochat.dataset -n {args.nanochat_num_shards} -w 2",
        cwd=nanochat_repo,
        env=env,
    )
    write_command_log(log_dir, "nanochat_dataset", download)

    tok_train = run_bash(
        f"uv run python -m scripts.tok_train --max-chars={args.nanochat_max_chars}",
        cwd=nanochat_repo,
        env=env,
    )
    write_command_log(log_dir, "nanochat_tok_train", tok_train)

    base_train_cmd = (
        "uv run python -m scripts.base_train "
        f"--device-type={shlex.quote(args.nanochat_device_type)} "
        f"--depth={args.nanochat_depth} "
        f"--head-dim={args.nanochat_head_dim} "
        "--window-pattern=L "
        f"--max-seq-len={args.nanochat_max_seq_len} "
        f"--device-batch-size={args.nanochat_device_batch_size} "
        f"--total-batch-size={args.nanochat_total_batch_size} "
        f"--eval-every={args.nanochat_eval_every} "
        f"--eval-tokens={args.nanochat_eval_tokens} "
        "--core-metric-every=-1 "
        "--sample-every=-1 "
        "--save-every=-1 "
        f"--target-flops={args.nanochat_target_flops} "
        f"--target-param-data-ratio={args.nanochat_target_param_data_ratio} "
        f"--num-iterations={args.nanochat_num_iterations} "
        "--run=dummy"
    )
    train = run_bash(
        base_train_cmd,
        cwd=nanochat_repo,
        env=env,
        check=False,
        timeout_sec=args.nanochat_train_timeout_sec,
    )
    write_command_log(log_dir, "nanochat_base_train", train)

    step_re = re.compile(
        r"step\s+(\d+)/(\d+).*loss:\s+([0-9.]+).*dt:\s+([0-9.]+)ms.*tok/sec:\s+([0-9,]+).*bf16_mfu:\s+([0-9.]+)"
    )
    val_re = re.compile(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([0-9.]+)")
    min_val_re = re.compile(r"Minimum validation bpb:\s+([0-9.]+)")

    steps = []
    for m in step_re.finditer(train["output"]):
        steps.append(
            {
                "step": int(m.group(1)),
                "num_iterations": int(m.group(2)),
                "loss": float(m.group(3)),
                "iter_ms": float(m.group(4)),
                "tok_per_sec": int(m.group(5).replace(",", "")),
                "bf16_mfu_percent": float(m.group(6)),
            }
        )
    vals = [
        {"step": int(m.group(1)), "val_bpb": float(m.group(2))}
        for m in val_re.finditer(train["output"])
    ]
    min_val = None
    for m in min_val_re.finditer(train["output"]):
        min_val = float(m.group(1))

    status = "ok"
    if train.get("timed_out"):
        status = "timeout"
    elif train["returncode"] != 0:
        status = "failed"

    return {
        "status": status,
        "returncode": train["returncode"],
        "elapsed_sec": train["elapsed_sec"],
        "timed_out": train.get("timed_out", False),
        "timeout_sec": train.get("timeout_sec"),
        "commit": git_commit(nanochat_repo),
        "steps_tail": steps[-5:],
        "last_step": steps[-1] if steps else None,
        "eval_rows": vals[-5:],
        "last_eval": vals[-1] if vals else None,
        "min_val_bpb": min_val,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append(f"- Timestamp (UTC): `{summary['timestamp_utc']}`")
    gpu_available = summary["gpu"]["available"]
    lines.append(f"- GPU available: `{gpu_available}`")
    if summary["gpu"]["devices"]:
        lines.append(f"- GPU(s): `{'; '.join(d['name'] for d in summary['gpu']['devices'])}`")
    lines.append(f"- Ours dataset: `{summary['ours_data_path']}`")
    lines.append(f"- Ours cargo features: `{summary['args']['ours_cargo_features']}`")
    lines.append(f"- nanochat ref: `{summary['args']['nanochat_ref']}`")
    lines.append(f"- nanochat device type: `{summary['args']['nanochat_device_type']}`")
    if summary.get("resolved_refs", {}).get("nanochat"):
        lines.append(f"- nanochat resolved ref: `{summary['resolved_refs']['nanochat']}`")
    if summary.get("python_torch"):
        pt = summary["python_torch"]
        if pt.get("ok"):
            lines.append(
                f"- Python torch: `version={pt.get('torch_version')} cuda_available={pt.get('cuda_available')} cuda_version={pt.get('cuda_version')}`"
            )
        else:
            lines.append(f"- Python torch: `unavailable error={pt.get('error')}`")
    lines.append(f"- Baseline mode: `{summary['args']['baseline']}`")
    lines.append("")

    ours = summary["runs"]["ours_tensor"]
    lines.append("## Ours (Tensor Mode)")
    lines.append("")
    lines.append(f"- Status: `{ours['status']}`")
    lines.append(f"- Elapsed: `{ours['elapsed_sec']:.3f}s`")
    lines.append(
        f"- Requested GPU backend: `{summary['ours_gpu_backend']['gpu_backend_detected']}`"
    )
    if summary["ours_gpu_backend"]["requested_features"]:
        lines.append(
            f"- Requested features: `{', '.join(summary['ours_gpu_backend']['requested_features'])}`"
        )
    if summary["ours_gpu_backend"]["backend_hints"]:
        lines.append(
            f"- Requested backend hints: `{', '.join(summary['ours_gpu_backend']['backend_hints'])}`"
        )
    lines.append(
        f"- Repo has GPU backend support: `{summary['ours_repo_gpu_backend']['gpu_backend_detected']}`"
    )
    if summary["ours_repo_gpu_backend"]["backend_hints"]:
        lines.append(
            f"- Repo backend hints: `{', '.join(summary['ours_repo_gpu_backend']['backend_hints'])}`"
        )
    ours_metrics = ours.get("metrics", {})
    if ours_metrics:
        lines.append(
            f"- Backend/device: `backend={ours_metrics.get('backend')} device={ours_metrics.get('device')} using_gpu={ours_metrics.get('using_gpu')}`"
        )
    lines.append(f"- Message: `{ours['message']}`")
    lines.append("")

    ours_scalar = summary["runs"].get("ours_scalar")
    if ours_scalar:
        lines.append("## Ours (Scalar Fallback)")
        lines.append("")
        metrics = ours_scalar.get("metrics", {})
        lines.append(f"- Status: `{ours_scalar['status']}`")
        lines.append(f"- Elapsed: `{ours_scalar['elapsed_sec']:.3f}s`")
        if metrics:
            lines.append(f"- Final loss: `{metrics.get('final_loss')}`")
            lines.append(f"- Steps/sec: `{metrics.get('steps_per_sec')}`")
            lines.append(f"- Tokens/sec: `{metrics.get('tokens_per_sec')}`")
        lines.append("")

    nanochat = summary["runs"].get("nanochat")
    if nanochat:
        lines.append("## nanochat")
        lines.append("")
        lines.append(f"- Status: `{nanochat['status']}`")
        if nanochat.get("timed_out"):
            lines.append(f"- Timeout: `{nanochat.get('timeout_sec')}s`")
        lines.append(f"- Commit: `{nanochat['commit']}`")
        lines.append(f"- Elapsed: `{nanochat['elapsed_sec']:.3f}s`")
        if nanochat.get("last_eval"):
            lines.append(
                f"- Last eval: `step={nanochat['last_eval']['step']} val_bpb={nanochat['last_eval']['val_bpb']:.6f}`"
            )
        if nanochat.get("min_val_bpb") is not None:
            lines.append(f"- Min val bpb: `{nanochat['min_val_bpb']:.6f}`")
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- JSON: `{summary['artifact_json']}`")
    lines.append(f"- Markdown: `{summary['artifact_md']}`")
    lines.append(f"- Logs dir: `{summary['artifact_logs_dir']}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-first benchmark against nanochat.")
    parser.add_argument("--baseline", choices=["nanochat"], default="nanochat")
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument(
        "--nanochat-ref",
        type=str,
        default=DEFAULT_NANOCHAT_PINNED_REF,
        help=(
            "git ref to checkout for nanochat (branch/tag/sha), "
            f"default pinned ref={DEFAULT_NANOCHAT_PINNED_REF}"
        ),
    )

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ours-steps", type=int, default=1000)
    parser.add_argument("--ours-data", type=str, default="input.txt")
    parser.add_argument("--ours-cargo-features", type=str, default="tch-backend")
    parser.add_argument("--torch-pip-index-url", type=str, default=DEFAULT_TORCH_INDEX)
    parser.add_argument(
        "--torch-pip-fallback-index-url",
        type=str,
        default=DEFAULT_TORCH_FALLBACK_INDEX,
    )
    parser.add_argument("--run-ours-scalar-fallback", action="store_true")

    parser.add_argument("--nanochat-num-shards", type=int, default=1)
    parser.add_argument("--nanochat-max-chars", type=int, default=20_000_000)
    parser.add_argument(
        "--nanochat-device-type",
        type=str,
        choices=["cuda", "cpu", "mps"],
        default="cuda",
    )
    parser.add_argument(
        "--nanochat-disable-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="set TORCH_COMPILE_DISABLE=1 for nanochat run (useful for lightweight CPU snapshots)",
    )
    parser.add_argument("--nanochat-depth", type=int, default=4)
    parser.add_argument("--nanochat-head-dim", type=int, default=64)
    parser.add_argument("--nanochat-max-seq-len", type=int, default=512)
    parser.add_argument("--nanochat-device-batch-size", type=int, default=2)
    parser.add_argument("--nanochat-total-batch-size", type=int, default=4096)
    parser.add_argument("--nanochat-eval-every", type=int, default=20)
    parser.add_argument("--nanochat-eval-tokens", type=int, default=32768)
    parser.add_argument("--nanochat-target-flops", type=float, default=-1.0)
    parser.add_argument("--nanochat-target-param-data-ratio", type=float, default=-1.0)
    parser.add_argument(
        "--nanochat-train-timeout-sec",
        type=float,
        default=0.0,
        help="timeout for nanochat base_train command; <=0 disables timeout",
    )
    parser.add_argument("--nanochat-num-iterations", type=int, default=60)
    args = parser.parse_args()
    if args.nanochat_train_timeout_sec <= 0:
        args.nanochat_train_timeout_sec = None
    return args


def main() -> int:
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"benchmark_karpathy_{stamp}"
    logs_dir = RESULTS_DIR / f"{run_name}_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpu_info = query_gpu_info(args.require_gpu)

    ours_data_path = pathlib.Path(args.ours_data)
    if not ours_data_path.is_absolute():
        ours_data_path = (REPO_ROOT / ours_data_path).resolve()
    if not ours_data_path.exists():
        raise RuntimeError(f"ours dataset path does not exist: {ours_data_path}")

    summary: dict[str, Any] = {
        "timestamp_utc": stamp,
        "repo_root": str(REPO_ROOT),
        "ours_data_path": str(ours_data_path),
        "ours_gpu_backend": detect_requested_gpu_backend(args.ours_cargo_features),
        "ours_repo_gpu_backend": detect_repo_gpu_backend(),
        "gpu": gpu_info,
        "args": vars(args),
        "resolved_refs": {},
        "runs": {},
    }

    ours_features = parse_ours_features(args.ours_cargo_features)
    needs_torch_runtime = args.require_gpu or ("tch-backend" in ours_features)
    python_torch = ensure_ours_tch_deps(
        args.install_deps and needs_torch_runtime,
        logs_dir,
        args.torch_pip_index_url,
        args.torch_pip_fallback_index_url,
    )
    summary["python_torch"] = python_torch
    if args.require_gpu and not python_torch.get("cuda_available", False):
        raise RuntimeError(
            f"GPU required but python torch CUDA is unavailable: {python_torch}"
        )
    summary["runs"]["ours_tensor"] = run_ours_tensor(args, logs_dir, ours_data_path)
    ensure_ours_used_cuda(summary["runs"]["ours_tensor"], args.require_gpu)
    if args.run_ours_scalar_fallback:
        summary["runs"]["ours_scalar"] = run_ours_scalar(args, logs_dir, ours_data_path)

    if args.baseline == "nanochat":
        nanochat_repo, nanochat_resolved_ref = ensure_repo(
            "https://github.com/karpathy/nanochat",
            "nanochat",
            args.nanochat_ref,
            logs_dir,
        )
        summary["resolved_refs"]["nanochat"] = nanochat_resolved_ref
        summary["runs"]["nanochat"] = run_nanochat(args, logs_dir, nanochat_repo)

    json_path = RESULTS_DIR / f"{run_name}.json"
    md_path = RESULTS_DIR / f"{run_name}.md"
    summary["artifact_json"] = str(json_path)
    summary["artifact_md"] = str(md_path)
    summary["artifact_logs_dir"] = str(logs_dir)

    write_text(json_path, json.dumps(summary, indent=2, sort_keys=True))
    write_text(md_path, render_markdown(summary))

    print(f"benchmark complete: {json_path}")
    print(f"summary markdown:   {md_path}")
    print(f"logs directory:     {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
