# microgpt-rs-lab

`microgpt-rs-lab` is a Rust CLI microGPT lab for training and sampling a small character-level language model end-to-end: it tokenizes text with a BOS token, trains a scalar autograd path and a tensor-style path, supports `classic`/`futuristic` styles for generation, and runs ablation sweeps (tied vs untied `lm_head`, with vs without input `rmsnorm`) that write comparable CSV/JSONL metrics in `results/`.

## Reference

This project is directly inspired by Karpathy's implementation:

- Gist: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
- Single-page version: https://karpathy.ai/microgpt.html

The intent is not to outgrow the reference quickly, but to preserve its teaching value while making controlled experiments easier in Rust.

## Planned Modes

- `scalar`: educational, explicit operations and gradients.
- `tensor`: backend-accelerated path for throughput experiments (CPU fallback + optional `tch` CUDA backend).

## Planned Experiments

- tied vs untied `lm_head`
- with vs without input `rmsnorm`

## Planned CLI

```bash
cargo run --release -- train --mode scalar --style futuristic --steps 500
cargo run --release -- train --mode tensor --steps 500
cargo run --release -- ablate --style futuristic --steps 500
cargo run --release -- sample --mode scalar --style classic --temperature 0.8 --max-new-tokens 120
cargo run --release -- sample --mode scalar --style futuristic --temperature 0.8 --max-new-tokens 120
cargo run --release -- sample --mode tensor --temperature 0.6
```

## Current Status

Implementation has started:

- Phase 0 baseline CLI skeleton is in place (`train`, `sample`, `ablate`).
- Phase 1 tokenizer/data utilities are implemented with unit tests.
- Phase 2 scalar autograd `Value` core is implemented with gradient tests.
- Phase 3 scalar path has a working bigram trainer and sampler.
- Tensor mode has working train/sample execution; with `--features tch-backend` it trains through `tch` (PyTorch/libtorch) and can use CUDA.
- Tensor train metrics now report `backend=... device=... using_gpu=...` so GPU usage is explicit.
- Scalar and tensor train/sample support style modes: `classic` and `futuristic` (default).
- Ablation now runs all 4 planned variants and persists style-tagged CSV/JSONL files in `results/`.

## Development

```bash
cargo check
cargo test

# tensor backend with tch/libtorch (uses GPU when available)
LIBTORCH_USE_PYTORCH=1 cargo run --release --features tch-backend -- train --mode tensor --steps 500
```

## NVIDIA-First Mode

For professional GPU runs, use the `tch` backend and pin CUDA wheel indexes:

```bash
# diagnose environment first
bash scripts/nvidia_doctor.sh

# run tensor training with nvidia backend
LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 cargo run --release --features tch-backend -- train --mode tensor --steps 500
```

`nvidia_doctor.sh` fails by default when CUDA is unavailable. For info-only checks on non-GPU laptops:

```bash
REQUIRE_GPU=0 bash scripts/nvidia_doctor.sh
```

On macOS, if you see `libtorch_cpu.dylib` load errors, prepend:

```bash
DYLD_LIBRARY_PATH="$(python3 - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)" LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 cargo run --release --features tch-backend -- train --mode tensor --steps 500
```

## GPU Benchmark (Colab-Friendly)

To benchmark this repo against Karpathy baselines (`nanochat` and/or `nanoGPT`) on a GPU runtime:

```bash
# default baseline/profile: nanogpt + auto (auto chooses quick/full from GPU memory)
bash scripts/colab_gpu_benchmark.sh

# faster benchmark against nanoGPT only
BASELINE=nanogpt bash scripts/colab_gpu_benchmark.sh

# run both baselines
BASELINE=both bash scripts/colab_gpu_benchmark.sh

# longer/more complete run profile
PROFILE=full BASELINE=both bash scripts/colab_gpu_benchmark.sh

# pin baseline refs for reproducible comparisons
NANOGPT_REF=master NANOCHAT_REF=main BASELINE=both bash scripts/colab_gpu_benchmark.sh
```

Raw benchmark artifacts are written to `results/`:

- `benchmark_karpathy_<timestamp>.json`
- `benchmark_karpathy_<timestamp>.md`
- `benchmark_karpathy_<timestamp>_logs/`

Direct script usage:

```bash
python3 scripts/benchmark_karpathy.py --baseline nanochat --install-deps --require-gpu --ours-cargo-features tch-backend --nanochat-ref auto --torch-pip-index-url https://download.pytorch.org/whl/cu128 --torch-pip-fallback-index-url https://download.pytorch.org/whl/cu126
python3 scripts/benchmark_karpathy.py --baseline nanogpt --install-deps --require-gpu --ours-cargo-features tch-backend --nanogpt-ref auto --torch-pip-index-url https://download.pytorch.org/whl/cu128 --torch-pip-fallback-index-url https://download.pytorch.org/whl/cu126
python3 scripts/benchmark_karpathy.py --baseline both --install-deps --require-gpu --ours-cargo-features tch-backend --nanogpt-ref auto --nanochat-ref auto --torch-pip-index-url https://download.pytorch.org/whl/cu128 --torch-pip-fallback-index-url https://download.pytorch.org/whl/cu126
```

### Deploy From Laptop to Colab

From this repo:

```bash
# 1) commit and push your current state
git add src/tensor/mod.rs README.md scripts/benchmark_karpathy.py scripts/colab_gpu_benchmark.sh scripts/nvidia_doctor.sh scripts/colab_url.py notebooks/colab_gpu_benchmark.ipynb
git commit -m "bench: colab gpu workflow"
git push origin HEAD

# 2) print Colab notebook URL for current branch
python3 scripts/colab_url.py
```

Open that URL, set GPU runtime, and run notebook `notebooks/colab_gpu_benchmark.ipynb`.

Notes:

- This harness is GPU-first and requires `nvidia-smi`.
- It probes `train --mode tensor` for this repo first, compiling with `--features tch-backend` by default.
- With `--require-gpu` (default), it fails unless our run reports `using_gpu=true` (no silent CPU fallback).
- It installs NVIDIA PyTorch wheels from a pinned index URL (with fallback) before running benchmarks.
- Baseline repos can be pinned by ref (`--nanogpt-ref`, `--nanochat-ref`) for reproducible runs.
- It reports whether a Rust GPU backend appears configured in `Cargo.toml` (heuristic).
- Optional scalar fallback can be added with `--run-ours-scalar-fallback` for a non-GPU reference run.
