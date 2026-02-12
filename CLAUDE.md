# nanochat-rs-next

## Overview

Rust-first continuation of `AntigmaLabs/nanochat-rs`, benchmarked against `karpathy/nanochat`. Goal: match or beat upstream on quality, speed, reliability, and reproducibility.

## Tech Stack

- **Language**: Rust 2024 edition
- **CLI**: clap 4.5 (derive)
- **RNG**: rand 0.8 (StdRng, deterministic seeding)
- **GPU**: tch 0.20 (optional, feature-gated `tch-backend`)
- **Benchmark harness**: Python 3 (`scripts/benchmark_karpathy.py`)

## Architecture

```
src/
├── main.rs          CLI entrypoint → dispatches to scalar/tensor
├── lib.rs           Module tree root
├── cli.rs           Clap-derived CLI parsing → AppCommand
├── config.rs        Domain config types (TrainConfig, SampleConfig, AblateConfig)
├── data/
│   ├── mod.rs       load_text, split_train_val
│   └── tokenizer.rs Char-level tokenizer with BOS, encode/decode
├── scalar/
│   ├── mod.rs       Scalar train/sample dispatcher, bigram integration
│   ├── bigram.rs    ScalarBigram model (autograd Value-based)
│   ├── minigpt.rs   1-layer GPT scalar path (multi-head attention + MLP)
│   └── value.rs     Autograd engine (Value with Rc<RefCell<Node>>)
├── tensor/
│   └── mod.rs       Tensor bigram: native CPU + optional tch/CUDA backend
├── eval.rs          Prompt-length guard, memory-drift guard
├── checkpoint.rs    Atomic persist-then-eval checkpoint pattern
└── experiments/
    └── mod.rs       Ablation sweep: 4 variants × (tie_lm_head, input_rmsnorm)
```

**Data flow**: CLI args → `AppCommand` → mode dispatch (scalar/tensor) → tokenizer → model train/sample → metrics/output.

**Autograd**: `value.rs` implements a scalar autograd engine (Add, Mul, Pow, Exp, Log, Relu) with topological-sort backward pass. Used by scalar bigram and mini-gpt.

**Tensor mode**: Dual-path — CPU-native `Vec<Vec<f64>>` implementation (default), or `tch` tensors with CUDA when `tch-backend` feature is enabled.

## Code Conventions

- Enums for modes/variants: `Mode`, `ModelKind`, `Style`
- Config structs carry all training/sampling parameters
- `pub(crate)` for internal model APIs; public for module-level `train`/`sample`
- Error types per module (`ScalarError`, `TensorError`, `AblationError`) with `From` impls
- Tests co-located in `#[cfg(test)] mod tests` at bottom of each file
- Deterministic seeding: each path XORs the base seed with a unique constant

## Testing

- **Framework**: built-in `#[test]`
- **Run**: `cargo test`
- **Coverage**: 51 tests across all modules
- **Patterns**: loss-drop verification, round-trip encoding, guard rejection, variant matrix coverage, finite-loss assertions
- **Test artifacts**: written to `results/test_artifacts/`, cleaned up after each test

## Quality Gates

- Formatting gate: `cargo fmt --check`
- Lint gate: `cargo clippy -- -D warnings`
- CI runs `cargo check --locked`, `cargo fmt --check`, `cargo clippy --locked -- -D warnings`, and tests with a 10-second command timeout.

## Build & Run

```bash
cargo check                    # type-check
cargo test                     # run all 46 tests
cargo build --release          # optimized binary

# Scalar bigram (default)
cargo run --release -- train --mode scalar --steps 500
cargo run --release -- sample --mode scalar --temperature 0.8

# Scalar mini-gpt
cargo run --release -- train --mode scalar --model-kind mini-gpt --style classic --steps 200

# Tensor CPU
cargo run --release -- train --mode tensor --steps 500

# Tensor tch + CUDA
LIBTORCH_USE_PYTORCH=1 cargo run --release --features tch-backend -- train --mode tensor --steps 500

# Ablation sweep
cargo run --release -- ablate --steps 500
```

## Git Conventions

- Branch: `master`
- Commit format: `type: description` (feat, fix, chore, docs)
- Stage named files only — no `git add -A`

## Boundaries

- No distributed training yet
- Mini-gpt is minimal parity scaffolding, not production
- Checkpoint lifecycle integration across all train loops is incomplete
- Char-level tokenizer only (no BPE/subword)
- Ablation sweep is scalar-bigram only

## Key Files

| File | Purpose |
|------|---------|
| `src/scalar/value.rs` | Autograd engine — the computational core |
| `src/scalar/bigram.rs` | Scalar bigram model with tied/untied LM head |
| `src/scalar/minigpt.rs` | 1-layer GPT: multi-head attention + ReGLU MLP |
| `src/tensor/mod.rs` | Tensor bigram with CPU/tch dual backend |
| `src/config.rs` | All config types and enums |
| `src/cli.rs` | CLI argument parsing |
| `scripts/benchmark_karpathy.py` | Benchmark harness vs upstream nanochat |
