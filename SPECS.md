# SPECS.md

Date: 2026-02-12
Project: `nanochat-rs-next`

## 1. Purpose

`nanochat-rs-next` is a Rust-first continuation of `AntigmaLabs/nanochat-rs`, benchmarked directly against `karpathy/nanochat`.

The objective is to become:
1. Better: equal or lower eval loss at matched budget.
2. Faster: higher throughput and lower time-to-target-loss on the same hardware class.
3. Safer: stronger reliability for eval and checkpoint workflows.
4. Reproducible: pinned refs/seeds and machine-readable artifacts.

## 2. Lineage and Baseline

1. Upstream lineage to evolve from:
`https://github.com/AntigmaLabs/nanochat-rs`
2. Baseline to benchmark against:
`https://github.com/karpathy/nanochat`

Rule: benchmark semantics and behavior contracts are anchored to `karpathy/nanochat`.

## 3. Scope

In scope:
1. Training/eval/inference CLI with benchmark evidence.
2. Rust-native tokenizer/model/runtime evolution.
3. Parity and regression tests for known instability classes.
4. GPU-first benchmark harness with reproducible outputs.

Out of scope:
1. Features that cannot be measured against benchmark goals.
2. Optimizations that change problem definition.
3. Broad framework reimplementation without parity checkpoints.

## 4. Known Gaps to Tackle

1. Eval stability under long context and cache constraints.
2. Eval memory/time drift in longer runs.
3. Checkpoint lifecycle ordering and interruption safety.
4. Hardware profile portability and diagnostics.
5. Rust parity fixtures and tokenizer compatibility tooling.
6. Interop/export path for non-Python serving stacks.

## 5. Architecture Direction

Strategy: fork-and-converge.

1. Reuse selected `nanochat-rs` modules when they accelerate parity.
2. Build missing train/eval benchmark contracts in this repo.
3. Keep explicit parity tests and benchmark gates before claiming wins.

Current codebase anchors:
1. CLI and config: `src/cli.rs`, `src/config.rs`.
2. Scalar path: `src/scalar/*`.
3. Tensor path: `src/tensor/mod.rs`.
4. Data/tokenizer baseline: `src/data/*`.
5. Benchmark harness: `scripts/benchmark_karpathy.py`.

## 6. Benchmark Contract

Every runtime-sensitive change must report:
1. Hardware and runtime metadata.
2. Matched budget configuration.
3. Final quality metrics.
4. Throughput and elapsed time.
5. Pass/fail scorecard for:
`quality`, `speed`, `reproducibility`.

Artifacts are written under `results/` and must be machine-readable.

## 7. Start Here (Execution Order)

Sprint 0:
1. Finalize naming/repo hygiene and pass tests.
2. Lock benchmark harness to `nanochat` baseline only.
3. Add this `SPECS.md` and keep `README.md` as overview.

Sprint 1:
1. Add first regression tests for:
`eval prompt-length safety`, `eval memory drift guard`, `checkpoint-before-eval`.
2. Add parity fixture test scaffolding for tokenizer/model outputs.
3. Add scorecard fields to benchmark outputs if missing.

Sprint 2:
1. Profile tensor path and capture first optimization candidates.
2. Implement one measured speedup with no quality regression.
3. Publish before/after benchmark evidence.

## 8. Definition of Done

A task is done when:
1. Tests pass (`cargo test`).
2. Docs/spec updated if behavior changed.
3. Benchmark evidence recorded for runtime-sensitive work.
4. Reproducibility metadata is present.

## 9. Open Questions

1. Which `karpathy/nanochat` commit/tag should be the first pinned benchmark baseline?
2. What quality tolerance do we accept for speed-focused changes?
3. Which hardware profiles are mandatory in CI versus optional manual runs?
