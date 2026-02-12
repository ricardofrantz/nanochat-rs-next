# Plan — nanochat-rs-next

## Architecture Diagram

```
┌──────────────┐
│   CLI (clap)  │  cli.rs → AppCommand
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌────────────┐
│  Config types │────▶│  Data layer │  data/mod.rs, tokenizer.rs
│  config.rs    │     │  load, split│  Char-level tokenizer + BOS
└──────┬───────┘     └─────┬──────┘
       │                   │
       ▼                   ▼
┌──────────────────────────────────────────┐
│           Mode Dispatch                   │
│  ┌─────────────┐    ┌─────────────────┐  │
│  │  Scalar mode │    │   Tensor mode   │  │
│  │ ┌─────────┐ │    │ ┌─────────────┐ │  │
│  │ │ Bigram  │ │    │ │ CPU-native  │ │  │
│  │ │(autograd)│ │    │ │ Vec<f64>    │ │  │
│  │ └─────────┘ │    │ └─────────────┘ │  │
│  │ ┌─────────┐ │    │ ┌─────────────┐ │  │
│  │ │Mini-GPT │ │    │ │ tch backend │ │  │
│  │ │(autograd)│ │    │ │ (optional)  │ │  │
│  │ └─────────┘ │    │ └─────────────┘ │  │
│  └─────────────┘    └─────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│          Cross-cutting concerns           │
│  guards    ─ prompt/memory safety         │
│  checkpoint ─ atomic persist-then-run     │
│  experiments ─ ablation sweep framework   │
└──────────────────────────────────────────┘
               │
               ▼
       results/ (CSV, JSONL, JSON, Markdown)
```

## Component Breakdown

| Component | Responsibility | Key Interfaces | Dependencies |
|-----------|---------------|----------------|-------------|
| `cli` | Parse CLI args → `AppCommand` | `parse_command()`, `try_command_from_iter()` | clap, config |
| `config` | Domain types for all operations | `TrainConfig`, `SampleConfig`, `AblateConfig`, `Mode`, `ModelKind`, `Style` | — |
| `data` | Load text, tokenize | `load_text()`, `Tokenizer`, `split_train_val()` | std::fs |
| `scalar::value` | Autograd engine | `Value` (new, add, mul, powf, exp, log, relu, backward) | Rc, RefCell |
| `scalar::bigram` | Scalar bigram model | `ScalarBigram` (new, nll_loss, train_step, parameters) | value |
| `scalar::minigpt` | 1-layer GPT model | `train_from_text()`, `sample_from_text()` | value, data, guards |
| `tensor` | Tensor bigram (dual-backend) | `train()`, `sample()` | data, optional tch |
| `guards` | Safety guards | `validate_prompt_length()`, `MemoryDriftGuard` | — |
| `checkpoint` | Atomic checkpoint+run | `persist_checkpoint()`, `persist_then_eval()` | std::fs |
| `experiments` | Ablation sweep | `run_ablation()` | scalar, config |

## Data Model

**Core entities**:
- `Tokenizer`: char→id mapping, BOS at id 0, sorted BTreeSet for determinism
- `Value`: autograd node with `Rc<RefCell<Node>>` — data, grad, op, prev children
- Token pairs: `Vec<(usize, usize)>` — (context_id, target_id) for bigram training
- Transition counts: `Vec<Vec<u64>>` — bigram frequency table with Laplace smoothing

**Artifacts**:
- `TrainMetrics` (scalar/tensor): model_kind, style, params, loss, throughput
- `AblationReport`: Vec of records + paths to CSV/JSONL files

## Technical Decisions

### TD-001: Scalar Autograd as Foundation
**Context**: Need to validate model logic independently of tensor framework bugs.
**Decision**: Implement a `Value`-based autograd engine (inspired by micrograd) for scalar training paths.
**Alternatives**: Use `tch` everywhere; use `ndarray` for CPU tensors.
**Consequences**: Scalar path is slow but transparent. Good for correctness validation, bad for throughput benchmarks. Mini-gpt on scalar autograd is extremely slow due to graph construction per step.

### TD-002: Dual Tensor Backend
**Context**: Want GPU benchmarks without mandating CUDA for development.
**Decision**: Feature-gate `tch` behind `tch-backend`. CPU-native path uses plain `Vec<Vec<f64>>`.
**Alternatives**: Always require tch; use burn/candle.
**Consequences**: Clean separation — `cargo test` works everywhere. GPU path requires libtorch.

### TD-003: Char-Level Tokenizer with BOS
**Context**: Matching Karpathy's character-level approach for fair comparison.
**Decision**: Custom char-level tokenizer, BOS token at id 0, sorted vocab for determinism.
**Alternatives**: Use `tokenizers` crate; implement BPE.
**Consequences**: Simple and deterministic. Limits model capability but matches upstream methodology.

### TD-004: Ablation as First-Class CLI Command
**Context**: Comparing architectural variants (tied/untied head, RMSNorm) is a core use case.
**Decision**: `ablate` subcommand that runs all 4 combinations and writes structured artifacts.
**Alternatives**: Shell scripts; separate binary.
**Consequences**: Reproducible sweeps in one command. Currently scalar-bigram only.

### TD-005: Deterministic Seeding with XOR Constants
**Context**: Multiple RNG streams needed (model init, training sampling, generation).
**Decision**: Each code path XORs the user-provided seed with a unique hex constant.
**Alternatives**: Sequential seed increments; independent seed args.
**Consequences**: Single `--seed` flag controls all randomness. Reproducible across runs.

## Implementation Phases

### Phase 1: Training Parity (validates FR-010)
**Goal**: Make the Rust training loop produce comparable metrics to upstream nanochat on matched budgets.
**Approach**: Implement upstream's schedule, train/val split, and loss computation methodology in the tensor path. Align hyperparameters.
**Complexity**: Medium — requires studying upstream's exact methodology.

### Phase 2: Benchmark Win (validates FR-009)
**Goal**: Demonstrate at least one config where Rust matches or beats upstream.
**Approach**: Profile tensor path on GPU. Optimize hot loops. Run benchmark harness with matched settings. Target throughput advantage from compiled Rust vs Python interpreter overhead.
**Complexity**: High — depends on Phase 1 parity being established first.

### Phase 3: Tensor Mini-GPT (validates FR-002 extension, OQ-002)
**Goal**: Extend the mini-gpt architecture to the tensor path for meaningful GPU benchmarks.
**Approach**: Port `ScalarMiniGpt` to tensor operations. Reuse the same architecture constants.
**Complexity**: Medium — architecture is defined, needs tensor implementation.

### Phase 4: Optimizer & Convergence (validates FR-010, OQ-004)
**Goal**: Improve training convergence with a proper optimizer.
**Approach**: Implement AdamW in the scalar autograd path. Use tch's AdamW for tensor path (already present but only for tch backend).
**Complexity**: Low-Medium — scalar AdamW needs momentum/variance state in autograd.

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Scalar autograd too slow for meaningful benchmarks | High | Medium | Use scalar only for correctness; benchmark on tensor path |
| tch backend version incompatibility with CUDA | Medium | High | Pin tch version; document exact libtorch version in CI |
| Upstream nanochat changes break pinned ref | Low | Medium | Pin to specific commit hash (already done) |
| Char-level tokenizer limits model quality ceiling | Medium | Low | Acceptable for benchmark scope; document limitation |
| No CI means regressions go undetected | High | Medium | Add GitHub Actions with `cargo test` gate |
| Tensor mini-gpt may not converge comparably to scalar | Medium | Medium | Validate numerically against scalar path on small dataset first |
