# nanochat-rs-next

[![CI](https://github.com/ricardofrantz/nanochat-rs-next/actions/workflows/ci.yml/badge.svg)](https://github.com/ricardofrantz/nanochat-rs-next/actions/workflows/ci.yml)

`nanochat-rs-next` is a Rust-first continuation of [`AntigmaLabs/nanochat-rs`](https://github.com/AntigmaLabs/nanochat-rs), benchmarked against [`karpathy/nanochat`](https://github.com/karpathy/nanochat).

## Goal

Build a Rust repo that can match or beat `karpathy/nanochat` on:

1. Quality (comparable eval metrics at matched budget)
2. Speed (throughput and time-to-target)
3. Reliability (stability of eval/checkpoint behavior)
4. Reproducibility (pinned refs + machine-readable artifacts)

## Working Today

1. CLI flows are implemented: `train`, `sample`, `ablate`.
2. Scalar mode supports `--model-kind bigram` (default).
3. Scalar mode also supports `--model-kind mini-gpt` (opt-in, 1-layer GPT-style scalar path).
4. Tensor mode works (CPU native path + optional `tch` backend with CUDA when available).
5. Ablation sweep works for scalar bigram variants and writes artifacts to `results/`.
6. Eval prompt-length guard, eval memory-drift guard, and checkpoint-before-eval persistence guard are implemented and tested.
7. Benchmark harness against `nanochat` is in place (`scripts/benchmark_karpathy.py`) with pinned baseline ref support and timeout-safe runs.

## Not Working Yet / Known Gaps

1. No demonstrated end-to-end benchmark win vs `karpathy/nanochat` yet.
2. Full training/eval parity with upstream `nanochat` is not complete.
3. The new `mini-gpt` scalar path is minimal parity scaffolding, not a full production training stack.
4. No integrated distributed training path in this Rust codebase yet.
5. Checkpoint/eval guards exist, but full checkpoint lifecycle integration across all train loops is still incomplete.
6. Local CPU baseline runs of upstream `nanochat` can be slow; meaningful comparisons should use GPU runs.

## Lineage and Baseline

- Evolving from: `AntigmaLabs/nanochat-rs`
- Benchmark source of truth: `karpathy/nanochat`
- Current pinned baseline ref used by harness defaults:
`2f096867244e3d00a50284d1be05fa3f5dcfb84b` (observed 2026-02-12)

## Quick Start

```bash
cargo check
cargo test

# scalar bigram (default)
cargo run --release -- train --mode scalar --steps 500
cargo run --release -- sample --mode scalar --temperature 0.8 --max-new-tokens 120

# scalar mini-gpt (opt-in)
cargo run --release -- train --mode scalar --model-kind mini-gpt --style classic --steps 200
cargo run --release -- sample --mode scalar --model-kind mini-gpt --style classic --temperature 0.8 --max-new-tokens 120

# tensor (CPU fallback)
cargo run --release -- train --mode tensor --steps 500

# tensor with tch backend (uses CUDA when available)
LIBTORCH_USE_PYTORCH=1 cargo run --release --features tch-backend -- train --mode tensor --steps 500
```

## Benchmark Commands

```bash
# GPU-first benchmark
bash scripts/colab_gpu_benchmark.sh

# explicit profile
PROFILE=full BASELINE=nanochat bash scripts/colab_gpu_benchmark.sh

# local CPU smoke benchmark (expected to be limited)
python3 scripts/benchmark_karpathy.py \
  --baseline nanochat \
  --no-require-gpu \
  --install-deps \
  --ours-cargo-features "" \
  --nanochat-device-type cpu \
  --nanochat-disable-compile \
  --nanochat-target-flops -1 \
  --nanochat-target-param-data-ratio -1 \
  --nanochat-num-iterations 1 \
  --nanochat-eval-every -1 \
  --nanochat-max-chars 50000 \
  --nanochat-depth 1 \
  --nanochat-head-dim 16 \
  --nanochat-max-seq-len 64 \
  --nanochat-device-batch-size 1 \
  --nanochat-total-batch-size 8 \
  --nanochat-train-timeout-sec 120
```

Artifacts are written to `results/`.

## Sources

- https://github.com/AntigmaLabs/nanochat-rs
- https://github.com/karpathy/nanochat
