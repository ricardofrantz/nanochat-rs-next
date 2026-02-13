# nanochat-rs-next

[![CI](https://github.com/ricardofrantz/nanochat-rs-next/actions/workflows/ci.yml/badge.svg)](https://github.com/ricardofrantz/nanochat-rs-next/actions/workflows/ci.yml)

`nanochat-rs-next` is a small Rust project for training and evaluating tiny language models from a text corpus with a CLI-driven workflow: users pick text data and training/sampling options (mode, model kind, optimizer, scheduler, seed, steps, checkpoints, and output paths), and the tool returns trained checkpoints, generated continuations, and structured benchmark/ablation metrics so model behavior and throughput can be compared reproducibly.

`nanochat-rs-next` is a Rust CLI for training and sampling tiny language models from a text corpus, letting users choose between a pure-Rust scalar engine and a tensor engine, between bigram and mini-gpt model kinds, configure training controls (optimizer, schedule, checkpoints, sampling settings), and get structured outputs like training metrics, generated text, and ablation reports to compare performance, quality, and behavior in reproducible experiments.

`nanochat-rs-next` is a Rust-first continuation of [`AntigmaLabs/nanochat-rs`](https://github.com/AntigmaLabs/nanochat-rs), benchmarked against [`karpathy/nanochat`](https://github.com/karpathy/nanochat).

## Goal

Build a Rust repo that can match or beat `karpathy/nanochat` on:

1. Quality (comparable eval metrics at matched budget)
2. Speed (throughput and time-to-target)
3. Reliability (stability of eval/checkpoint behavior)
4. Reproducibility (pinned refs + machine-readable artifacts)

## Working Today

1. CLI flows: `train`, `sample`, `ablate` with full argument parsing.
2. Scalar mode: `--model-kind bigram` (default) and `--model-kind mini-gpt` (1-layer GPT).
3. Tensor mode: CPU-native path + optional `tch` backend with CUDA (`--features tch-backend`). Supports both bigram and mini-gpt model kinds.
4. Optimizers: SGD (default) and AdamW (`--optimizer adamw`) for both scalar and tensor paths.
5. Ablation sweep: 4 variants × (tie_lm_head, input_rmsnorm), writes artifacts to `results/`.
6. Guards: prompt-length guard, memory-drift guard, checkpoint-before-eval persistence guard.
7. Checkpointing: atomic persist-then-eval across tensor train loops with configurable interval.
8. Training utilities: shared LR scheduling (linear warmdown), train/val splitting, transition count building.
9. Benchmark harness against `nanochat` (`scripts/benchmark_karpathy.py`) with pinned baseline ref support, timeout-safe runs, and GPU profiling scripts.
10. 62 tests covering loss convergence, round-trip encoding, guard rejection, optimizer comparison, checkpoint ordering, and Python parity traces.

## Known Gaps

1. No demonstrated end-to-end benchmark win vs `karpathy/nanochat` yet.
2. Full training/eval parity with upstream `nanochat` is not complete.
3. Mini-gpt is minimal parity scaffolding, not a full production training stack.
4. No distributed training.
5. Char-level tokenizer only (no BPE/subword).
6. Ablation sweep is scalar-bigram only.
7. Local CPU baseline runs of upstream `nanochat` can be slow; meaningful comparisons should use GPU runs.

## Current benchmark status (as-of this environment)

- CPU scalar path (default): `cargo run --release -- train --mode scalar --model-kind bigram --steps 200 --optimizer adamw --seed 42` took ~0.40s.
- CPU tensor-native path: `cargo run --release -- train --mode tensor --model-kind bigram --steps 200 --optimizer sgd --seed 42` took ~0.26s.
- `tensor --optimizer adamw` is currently unsupported in CPU-native tensor mode in this codebase.
- The `tch` backend could not run in this environment yet because the local libtorch/PyTorch setup is incomplete (version mismatch and missing runtime `libtorch_cpu.dylib` lookup path).
- GeForce 4060 support is expected through the `tch` backend when a compatible libtorch/PyTorch/CUDA runtime is available.

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
