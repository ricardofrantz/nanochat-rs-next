# nanochat-rs-next

`nanochat-rs-next` is a Rust-first continuation of [`AntigmaLabs/nanochat-rs`](https://github.com/AntigmaLabs/nanochat-rs), built to go toe-to-toe with [`karpathy/nanochat`](https://github.com/karpathy/nanochat) on speed, quality, and reproducibility.

This is the guiding overview file for the project. Detailed execution scope lives in `SPECS.md`.

## What This Repo Is

- A practical engineering repo for training/eval/inference experiments in modern Rust.
- A benchmark-driven effort: every performance claim must be reproducible and comparable to `karpathy/nanochat`.
- A continuation, not a rewrite-from-scratch for the sake of rewriting.

## Lineage and Benchmark Anchors

- Evolving from: `AntigmaLabs/nanochat-rs` (Rust baseline and reusable modules).
- Benchmarking against: `karpathy/nanochat` (behavior and benchmark source of truth).
- Scope focus: **nanochat parity and improvements**.

## Objective

Build a repo that is:

1. `better`: equal or lower eval loss at matched training budget.
2. `faster`: higher throughput and lower time-to-target-loss on the same hardware class.
3. `safer`: stronger eval/checkpoint stability guarantees.
4. `reproducible`: pinned refs/seeds and machine-readable benchmark artifacts in `results/`.

## Core Strategy (Fork-and-Converge)

1. Reuse selected components from `nanochat-rs` where that accelerates delivery.
2. Implement missing training/eval parity contracts natively in this repo.
3. Keep `karpathy/nanochat` as the comparison baseline for correctness and benchmarks.
4. Reject changes that improve speed by changing the underlying problem definition.

## Gaps We Intend to Tackle

1. Eval stability and long-context safety.
- Example classes: RoPE/cache boundary failures and eval memory growth.

2. Checkpoint lifecycle reliability.
- Ensure save/eval ordering is safe under interruptions.

3. Portable runtime profiles.
- Make CPU/MPS/CUDA behavior explicit with predictable failure modes.

4. Rust-native tokenizer + parity infrastructure.
- Build robust tokenizer tooling with strict fixture parity.

5. Interop paths for non-Python serving stacks.
- Define clean export/runtime interfaces for broader deployment use.

6. Transparent benchmark governance.
- One-command benchmark with pass/fail scorecard (`quality`, `speed`, `reproducibility`).

## Repo Plan

### Phase 1: Parity Foundation

- Stabilize benchmark harness against pinned `karpathy/nanochat` refs.
- Land tokenizer/model/runtime parity tests.
- Add regression coverage for known eval/checkpoint failure classes.

### Phase 2: Performance Wins

- Profile hot paths and reduce overhead in training/inference loops.
- Report before/after results with matched budgets.
- Keep quality gates enforced.

### Phase 3: Practical Utility

- Expand hardware profiles and environment diagnostics.
- Improve contributor workflow and issue templates.
- Package reusable Rust components (tokenizer/metrics/parity helpers).

## Engineering Rules

- No feature is complete without tests.
- No optimization is accepted without benchmark evidence.
- No benchmark claim is accepted without reproducibility metadata.
- Prefer minimal, composable Rust implementations over framework-heavy designs.

## Success Criteria

A change is considered successful when:

1. Tests pass (`cargo test`).
2. Behavior remains compatible with target parity expectations.
3. Benchmark evidence is recorded when runtime-sensitive code changes.
4. Artifact outputs are auditable (`results/*.json`, `results/*.md`).

## Quick Start

```bash
cargo check
cargo test

# scalar path
cargo run --release -- train --mode scalar --steps 500
cargo run --release -- sample --mode scalar --temperature 0.8 --max-new-tokens 120

# scalar mini-gpt parity path (opt-in)
cargo run --release -- train --mode scalar --model-kind mini-gpt --style classic --steps 200
cargo run --release -- sample --mode scalar --model-kind mini-gpt --style classic --temperature 0.8 --max-new-tokens 120

# tensor path (CPU fallback, optional GPU with tch backend)
cargo run --release -- train --mode tensor --steps 500
LIBTORCH_USE_PYTORCH=1 cargo run --release --features tch-backend -- train --mode tensor --steps 500
```

## Benchmark Workflow

Default pinned `nanochat` baseline ref:
`2f096867244e3d00a50284d1be05fa3f5dcfb84b` (master head observed on 2026-02-12).

```bash
# GPU-first benchmark run
bash scripts/colab_gpu_benchmark.sh

# explicit profile and baseline
PROFILE=full BASELINE=nanochat bash scripts/colab_gpu_benchmark.sh

# local CPU snapshot (no GPU required)
python3 scripts/benchmark_karpathy.py --baseline nanochat --no-require-gpu --install-deps --ours-cargo-features "" --nanochat-device-type cpu --nanochat-disable-compile --nanochat-target-flops -1 --nanochat-target-param-data-ratio -1 --nanochat-num-iterations 1 --nanochat-eval-every -1 --nanochat-max-chars 50000 --nanochat-depth 1 --nanochat-head-dim 16 --nanochat-max-seq-len 64 --nanochat-device-batch-size 1 --nanochat-total-batch-size 8 --nanochat-train-timeout-sec 120

# direct script usage
python3 scripts/benchmark_karpathy.py --baseline nanochat --install-deps --require-gpu --ours-cargo-features tch-backend
```

Artifacts are stored in `results/`.

## Project Status (Current)

- CLI with `train`, `sample`, and `ablate` flows exists.
- Scalar and tensor codepaths are present.
- Scalar now supports `--model-kind bigram|mini-gpt` (default `bigram`).
- GPU-oriented benchmark scripts and Colab workflow are included.
- Next major push: stronger parity/stability tests and measurable speed wins.

## Sources

- https://github.com/AntigmaLabs/nanochat-rs
- https://github.com/karpathy/nanochat
