# Tasks — nanochat-rs-next

## Phase 1: Training Parity [FR-010, TD-001, TD-002]

### 1.1: Study upstream nanochat training methodology [FR-010]
- **What**: Read `karpathy/nanochat` source at pinned ref, document: loss function, optimizer, learning rate schedule, train/val split, sequence construction, and reporting format
- **Files**: `docs/upstream_analysis.md` (new)
- **Tests**: N/A (research task)
- **Done when**: Documented upstream methodology with specific code references
- [x] Clone and read upstream at pinned ref
- [x] Document loss computation (cross-entropy details, averaging)
- [x] Document optimizer config (Adam? AdamW? LR schedule?)
- [x] Document train/val split ratio and eval schedule
- [x] Document sequence construction (block size, batching)

### 1.2: Align tensor training loop with upstream methodology [FR-010, TD-002]
- **What**: Modify tensor training to use upstream-compatible loss computation, train/val split, and eval schedule
- **Files**: `src/tensor/mod.rs`, `src/data/mod.rs`
- **Tests**: Parity test comparing Rust vs Python loss on identical small dataset
- **Done when**: On a 10K-char dataset with matched hyperparameters, Rust and Python produce eval losses within 5% after N steps
- [x] Write parity comparison test
- [x] Implement eval schedule (eval every N steps)
- [x] Use `split_train_val` in training loop (currently unused)
- [x] Align optimizer and LR with upstream defaults
- [x] Implement

### 1.3: Add train/val eval reporting to metrics [FR-010]
- **What**: Report both training loss and validation loss in `TrainMetrics`
- **Files**: `src/tensor/mod.rs`, `src/scalar/mod.rs`, `src/config.rs`
- **Tests**: Metrics include val_loss field; val_loss is computed on held-out data
- **Done when**: `train` output includes `val_loss=X.XXXX` for tensor mode
- [x] Write test for val_loss in metrics
- [x] Add val_loss to TrainMetrics
- [x] Compute val_loss in tensor training loop
- [x] Implement

---

## Phase 2: Benchmark Win [FR-009, FR-006]

### 2.1: Profile tensor training throughput on GPU [FR-009]
- **What**: Run tensor training with tch-backend on GPU, collect throughput metrics, identify bottlenecks
- **Files**: `scripts/profile_gpu.sh` (new)
- **Tests**: N/A (profiling task)
- **Done when**: Throughput numbers (tokens/sec) collected for GPU tensor path; bottleneck identified
- [ ] Run tch-backend training on GPU with timing
- [ ] Compare steps/sec vs upstream nanochat
- [ ] Profile with `perf` or NVTX if throughput gap is large
- [ ] Document findings

### 2.2: Optimize tensor hot loops [FR-009, TD-002]
- **What**: Based on profiling, optimize the most expensive operations in tensor training
- **Files**: `src/tensor/mod.rs`
- **Tests**: Existing tests still pass; throughput improves measurably
- **Done when**: At least 10% throughput improvement on GPU, or documented why the gap is framework-level
- [ ] Write throughput benchmark test
- [ ] Implement optimizations
- [ ] Re-run benchmark comparison

### 2.3: Run definitive benchmark comparison [FR-009, FR-006]
- **What**: Execute full benchmark harness on GPU with matched settings, produce comparison report
- **Files**: `scripts/benchmark_karpathy.py`, `results/` artifacts
- **Tests**: Benchmark completes without errors; JSON/Markdown reports are valid
- **Done when**: Published comparison showing Rust within 5% of Python quality OR faster at matched quality
- [ ] Select target dataset and GPU
- [ ] Run benchmark with `--baseline nanochat`
- [ ] Analyze results
- [ ] Update README with findings

---

## Phase 3: Tensor Mini-GPT [FR-002, TD-002]

### 3.1: Port mini-gpt architecture to tensor operations [FR-002]
- **What**: Implement `TensorMiniGpt` using either CPU-native or tch tensors, matching scalar architecture
- **Files**: `src/tensor/mod.rs` or `src/tensor/minigpt.rs` (new)
- **Tests**: Tensor mini-gpt loss drops on repetitive text; matches scalar mini-gpt loss trajectory within tolerance
- **Done when**: `--mode tensor --model-kind mini-gpt` trains and produces metrics
- [ ] Write parity test: scalar vs tensor mini-gpt loss on same input
- [ ] Implement TensorMiniGpt struct
- [ ] Implement forward pass (multi-head attention + MLP)
- [ ] Implement train_step with backprop
- [ ] Wire into CLI dispatch
- [ ] Implement and verify tests

### 3.2: Add mini-gpt sampling to tensor mode [FR-004]
- **What**: Implement autoregressive sampling for tensor mini-gpt
- **Files**: `src/tensor/mod.rs` or `src/tensor/minigpt.rs`
- **Tests**: Sampling produces valid text starting with prompt
- **Done when**: `sample --mode tensor --model-kind mini-gpt` works
- [ ] Write sampling test
- [ ] Implement sample function
- [ ] Verify prompt and length constraints

---

## Phase 4: Optimizer & Convergence [FR-010, TD-001]

### 4.1: Implement AdamW for scalar autograd path [FR-010]
- **What**: Add AdamW optimizer state (momentum, variance, weight decay) for the scalar `Value` parameter updates
- **Files**: `src/scalar/optimizer.rs` (new), `src/scalar/mod.rs`
- **Tests**: AdamW converges faster than SGD on same dataset; momentum/variance state updates correctly
- **Done when**: Scalar training with AdamW shows improved convergence curve
- [x] Write convergence comparison test (SGD vs AdamW)
- [x] Implement AdamW struct with step()
- [x] Wire into scalar training loop (optional flag)
- [x] Implement and verify

### 4.2: Add learning rate schedule [FR-010]
- **What**: Implement warmup + linear warmdown schedule matching upstream
- **Files**: `src/scalar/mod.rs`, `src/tensor/mod.rs`
- **Tests**: LR at step 0 is warmup value; LR at final step matches min LR
- **Done when**: LR schedule matches upstream nanochat's schedule
- [x] Write LR schedule unit test
- [x] Implement schedule function
- [x] Wire into both scalar and tensor training loops

---

## Infrastructure (parallel with any phase)

### I.1: Add CI pipeline [NFR-004, NFR-005]
- **What**: GitHub Actions workflow: `cargo check`, `cargo test`, `cargo clippy`
- **Files**: `.github/workflows/ci.yml` (new)
- **Tests**: CI passes on push to master
- **Done when**: Green badge on README
- [x] Write workflow file
- [x] Add clippy lint step
- [x] Add test timeout (10s)
- [x] Add badge to README

### I.2: Wire checkpoint into tensor training loop [FR-008]
- **What**: Call `persist_then_eval` at checkpoint intervals during tensor training
- **Files**: `src/tensor/mod.rs`, `src/checkpoint.rs`
- **Tests**: Checkpoint file exists on disk at expected intervals during training
- **Done when**: Tensor training produces checkpoint files and runs callback
- [x] Write test for checkpoint creation during training
- [x] Add checkpoint interval config to TrainConfig
- [x] Wire persist_then_eval into tensor train loop
- [x] Implement

### I.3: Add clippy + fmt enforcement [NFR-004]
- **What**: Ensure `cargo clippy -- -D warnings` and `cargo fmt --check` pass
- **Files**: project-wide
- **Tests**: Clean clippy + fmt
- **Done when**: No warnings
- [x] Run clippy and fix any warnings
- [x] Run fmt and fix any formatting
- [x] Document in CLAUDE.md
