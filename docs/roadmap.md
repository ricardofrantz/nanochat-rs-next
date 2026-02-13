# nanochat-rs-next Roadmap

Status as of 2026-02-13: foundation complete (scalar autograd, tensor dual-backend, CLI, ablation, checkpointing, 62 tests passing, all quality gates green).

## Track A: Benchmark Win

The project's thesis — demonstrate Rust parity or superiority vs `karpathy/nanochat`.

### A.1 GPU profiling run
- Run tensor training with `tch-backend` on GPU (Colab T4 or local)
- Collect: steps/sec, tokens/sec, memory usage
- Compare against upstream at matched budget
- Files: `scripts/profile_gpu.sh`, results in `results/`

### A.2 1-sqrt warmdown LR schedule
- Current: linear warmdown
- Upstream moving to `LR = 1 - sqrt(progress)` (faster initial decay, longer tail)
- Implement as option in `training.rs`, wire through CLI as `--lr-schedule linear|sqrt`
- Files: `src/training.rs`, `src/cli.rs`, `src/config.rs`
- Ref: upstream PR #513

### A.3 BPB (bits-per-byte) validation metric
- Current: mean NLL loss
- Upstream uses BPB: `total_nats / (ln(2) * total_bytes)`
- Needed for apples-to-apples comparison
- See: `docs/upstream_analysis.md` section 1
- Files: `src/training.rs`, `src/scalar/mod.rs`, `src/tensor/mod.rs`

### A.4 Definitive benchmark comparison
- Run `scripts/benchmark_karpathy.py` on GPU with matched settings
- Produce comparison report, update README
- Target: Rust within 5% of Python quality OR faster at matched quality

## Track B: Tensor Mini-GPT

Currently `--mode tensor --model-kind mini-gpt` only works with `tch-backend`. The CPU-native path rejects it.

### B.1 CPU-native tensor mini-gpt
- Port scalar mini-gpt architecture (multi-head attention + ReGLU MLP) to `Vec<Vec<f64>>` ops
- Or: accept CPU-native is bigram-only and document clearly
- Files: `src/tensor/mod.rs`

### B.2 Tensor mini-gpt sampling
- `sample --mode tensor --model-kind mini-gpt` needs to work
- Files: `src/tensor/mod.rs`

### B.3 Scalar-vs-tensor parity test
- Same small input → scalar and tensor mini-gpt loss trajectories within tolerance
- Files: test in `src/tensor/mod.rs` or integration test

## Track C: Quality & Polish

### C.1 Checkpoint-before-eval everywhere
- Pattern exists in `checkpoint.rs` — upstream has this as open issue #446
- Verify wired into ALL train loops (scalar bigram, scalar mini-gpt, tensor)
- Differentiator: we solved a problem upstream hasn't yet

### C.2 Eval memory safety
- `MemoryDriftGuard` exists — upstream has memory leak in hellaswag eval (#427)
- Wire guard into actual eval loops (currently only tested in isolation)

### C.3 Error handling audit
- Ensure all error paths tested
- Check for `unwrap()` calls that could panic in production

### C.4 Integration tests
- Add end-to-end tests exercising the CLI binary
- Consider `assert_cmd` crate

### C.5 README benchmark section
- Document what CAN be benchmarked today
- Show sample output from CPU run

## Priority Order

1. **C.1 + C.2** — wire existing guards into train loops (already built, just need to use)
2. **A.2** — 1-sqrt warmdown (small, high-value, matches upstream direction)
3. **A.3** — BPB metric (needed before any benchmark claim)
4. **B.1/B.2** — tensor mini-gpt (extends capability)
5. **A.1 → A.4** — GPU profiling and benchmark (needs hardware access)
