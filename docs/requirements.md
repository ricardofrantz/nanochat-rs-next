# Requirements — nanochat-rs-next

## Purpose

**Problem**: Karpathy's `nanochat` (Python/PyTorch) is the reference implementation for small-scale language model training experiments. There is no equivalent Rust implementation that matches it on quality, speed, and reproducibility.

**Target users**: ML researchers and Rust enthusiasts who want a fast, reproducible, single-binary training CLI that can benchmark against `nanochat` on identical datasets.

**Success criteria**: Demonstrate parity or superiority vs `karpathy/nanochat` on matched training budgets, with machine-readable artifacts for every run.

---

## Functional Requirements

### FR-001: Scalar Bigram Training
**As a** researcher, **I want to** train a character-level bigram model using scalar autograd, **so that** I can validate the autograd engine and compare against count-based baselines.

**Acceptance criteria**:
- GIVEN `--mode scalar --model-kind bigram`, WHEN training completes, THEN loss drops below initial loss on repetitive text
- GIVEN any combination of `--tie-lm-head` and `--input-rmsnorm`, WHEN training runs, THEN all losses are finite
- GIVEN `--steps 0`, WHEN invoked, THEN reports baseline loss without training

**Priority**: P0 | **Status**: Implemented

### FR-002: Scalar Mini-GPT Training
**As a** researcher, **I want to** train a 1-layer GPT model using scalar autograd, **so that** I can validate attention/MLP paths at character level.

**Acceptance criteria**:
- GIVEN `--mode scalar --model-kind mini-gpt`, WHEN training completes, THEN loss drops below initial on repetitive text
- GIVEN a context window of `BLOCK_SIZE=8`, WHEN a training window is selected, THEN it respects the block size boundary

**Priority**: P0 | **Status**: Implemented (minimal parity scaffolding)

### FR-003: Tensor Bigram Training
**As a** researcher, **I want to** train a bigram model with native tensor operations, **so that** I can benchmark real throughput on CPU and GPU.

**Acceptance criteria**:
- GIVEN `--mode tensor` without `tch-backend` feature, WHEN training runs, THEN uses CPU-native `Vec<Vec<f64>>` path
- GIVEN `--mode tensor` with `tch-backend` feature and CUDA available, WHEN training runs, THEN uses GPU and reports `using_gpu=true`
- GIVEN either backend, WHEN training completes, THEN metrics report `backend`, `device`, and `using_gpu` fields

**Priority**: P0 | **Status**: Implemented

### FR-004: Text Sampling
**As a** user, **I want to** sample text from a trained model, **so that** I can inspect generation quality.

**Acceptance criteria**:
- GIVEN a prompt and `--max-new-tokens N`, WHEN sampling completes, THEN output starts with prompt and adds up to N tokens
- GIVEN `--temperature 0.0` or negative, WHEN invoked, THEN returns InvalidTemperature error
- GIVEN `--style futuristic`, WHEN sampling from bigram, THEN output contains characters beyond the base corpus

**Priority**: P0 | **Status**: Implemented

### FR-005: Ablation Sweep
**As a** researcher, **I want to** run all 4 architectural variants in one command, **so that** I can compare tied/untied heads and RMSNorm effects.

**Acceptance criteria**:
- GIVEN `ablate`, WHEN sweep completes, THEN produces CSV and JSONL artifacts in `results/`
- GIVEN the sweep, WHEN comparing variants, THEN untied variants report more parameters than tied
- GIVEN `--style futuristic`, WHEN sweep runs, THEN artifact filenames contain `style-futuristic`

**Priority**: P1 | **Status**: Implemented

### FR-006: Benchmark Harness
**As a** researcher, **I want to** benchmark this repo against `karpathy/nanochat`, **so that** I can measure relative speed and quality.

**Acceptance criteria**:
- GIVEN `scripts/benchmark_karpathy.py --baseline nanochat`, WHEN benchmark runs, THEN produces JSON + Markdown results in `results/`
- GIVEN `--no-require-gpu`, WHEN no GPU is available, THEN benchmark runs on CPU with degraded but valid results
- GIVEN a pinned baseline ref, WHEN cloning upstream, THEN checks out the exact commit

**Priority**: P1 | **Status**: Implemented (no demonstrated win yet)

### FR-007: Eval Safety Guards
**As a** developer, **I want** prompt-length and memory-drift guards, **so that** eval runs don't silently OOM or produce invalid results.

**Acceptance criteria**:
- GIVEN a prompt longer than `max_prompt_tokens`, WHEN sampling mini-gpt, THEN returns PromptTooLong error
- GIVEN memory observations exceeding `max_growth_ratio`, WHEN guard observes, THEN returns MemoryDrift error

**Priority**: P1 | **Status**: Implemented

### FR-008: Checkpoint Persistence
**As a** developer, **I want** atomic checkpoint-before-eval semantics, **so that** eval always runs against persisted state.

**Acceptance criteria**:
- GIVEN a checkpoint path and payload, WHEN `persist_then_eval` runs, THEN the file exists on disk before the eval callback executes
- GIVEN a write, WHEN persisting, THEN uses tmp-file + rename for atomicity

**Priority**: P2 | **Status**: Implemented (not yet wired into all train loops)

### FR-009: Demonstrated Benchmark Win
**As a** researcher, **I want** at least one configuration where nanochat-rs-next matches or beats upstream nanochat, **so that** the project validates its thesis.

**Acceptance criteria**:
- GIVEN a matched training budget (same dataset, comparable steps), WHEN comparing final eval loss, THEN Rust version is within 5% of Python baseline OR faster at matched quality
- GIVEN the benchmark harness, WHEN running on GPU, THEN produces a summary table showing the comparison

**Priority**: P0 | **Status**: Not implemented

### FR-010: Full Training/Eval Parity
**As a** researcher, **I want** the Rust training loop to match upstream nanochat's eval methodology, **so that** comparisons are apples-to-apples.

**Acceptance criteria**:
- GIVEN matched hyperparameters, WHEN training on the same dataset, THEN eval loss curves are comparable within noise
- GIVEN upstream's eval schedule, WHEN our training loop evaluates, THEN uses the same eval set split and metrics

**Priority**: P1 | **Status**: Not implemented

---

## Non-Functional Requirements

### NFR-001: Deterministic Reproducibility
All training runs with the same seed must produce identical loss trajectories and outputs. Each code path XORs the base seed with a unique constant to ensure independent RNG streams.

### NFR-002: Single Binary
The tool must compile to a single static binary (modulo optional `tch` dynamic linking). No runtime config files required.

### NFR-003: Machine-Readable Artifacts
Every experiment must produce structured output (CSV, JSONL, or JSON) suitable for automated analysis. Human-readable summaries are secondary.

### NFR-004: Compilation Without GPU
`cargo check` and `cargo test` must succeed without CUDA/libtorch installed (i.e., without `tch-backend` feature).

### NFR-005: Test Suite < 10s
The full `cargo test` suite must complete in under 10 seconds on a modern laptop.

---

## Out of Scope

- Distributed training (multi-node, multi-GPU)
- Subword/BPE tokenization
- Model serialization/loading across runs (beyond checkpoint bytes)
- Web UI or REST API
- Training on datasets larger than fits in memory

---

## Open Questions

1. **OQ-001**: What is the target dataset for the definitive benchmark comparison? (Shakespeare? TinyStories? Something else?)
2. **OQ-002**: Should the tensor mode support mini-gpt, or is scalar-only sufficient for the GPT path?
3. **OQ-003**: What GPU target is primary for benchmark comparisons? (A100? T4/Colab? Consumer RTX?)
4. **OQ-004**: Should the project adopt a proper optimizer (Adam/AdamW) for the scalar autograd path, or keep vanilla SGD?
