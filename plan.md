# microgpt-rs-lab: Full Implementation Plan

Date: 2026-02-12

## 1. Project Goal

Implement a Rust microGPT project that stays faithful to Karpathy's pedagogical style while adding a structured experimental workflow.

Primary outcomes:

1. A readable scalar implementation that is easy to inspect and reason about.
2. A tensor-backed implementation for speed and larger experiments.
3. A reproducible ablation harness for architecture choices.

## 2. Success Criteria

The project is considered ready for "coding phase complete" when all items below are true:

1. `train` works in both `scalar` and `tensor` modes.
2. `sample` works in both modes with temperature control.
3. `ablate` runs all 4 combinations:
   - tied `lm_head`, no input `rmsnorm`
   - tied `lm_head`, input `rmsnorm`
   - untied `lm_head`, no input `rmsnorm`
   - untied `lm_head`, input `rmsnorm`
4. Metrics are printed and persisted:
   - final loss
   - mean loss over last N steps
   - throughput (`steps/s` and/or `tokens/s`)
5. Test suite covers math/autograd correctness and basic training sanity.
6. Runs are reproducible with fixed seed and recorded config.

## 3. Scope and Non-Goals

In scope:

- Character-level tokenizer and names-style dataset workflow (`input.txt` compatible).
- Minimal GPT-style model matching the reference spirit.
- Adam optimizer and training loop.
- CLI commands for training, sampling, and ablation.
- Repeatable experiment output.

Out of scope (initial phase):

- SOTA performance tuning.
- Distributed training/multi-node orchestration.
- Feature-heavy framework abstractions.

## 4. Technical Strategy

## 4.1 Scalar Path (correctness + pedagogy)

- Implement scalar `Value` type with reverse-mode autodiff.
- Maintain simple, explicit graph operations.
- Favor clarity over micro-optimizations.

Why first:

- It defines the correctness baseline.
- It provides a reference for tensor parity checks.

## 4.2 Tensor Path (speed)

- Start with one backend (candidate: Candle).
- Mirror model structure and initialization from scalar path.
- Validate with shape checks and trend-level loss parity.

## 4.3 Shared Configuration Layer

- Single config model used by both paths.
- Shared seed handling and dataset split logic.
- Shared metric reporting format.

## 5. Work Breakdown

## Milestone 0: Project Setup

- Update crate naming and docs.
- Finalize baseline folder structure.
- Add lint/test commands to README.

Acceptance:

- `cargo check` and `cargo test` run in clean repo.

## Milestone 1: Data + Tokenization

- Load docs from local `input.txt`.
- Add auto-fetch option compatible with reference dataset URL.
- Build char vocabulary with `<BOS>` token.
- Add encode/decode helpers.

Acceptance:

- Deterministic vocabulary and token mapping with fixed input.
- Unit tests for tokenizer round-trips.

## Milestone 2: Scalar Autograd Core

- Implement operations: add, mul, pow, exp, log, relu.
- Implement graph traversal and `backward()`.
- Add basic gradient regression tests.

Acceptance:

- Tests verify gradients for small analytic cases.

## Milestone 3: Scalar microGPT Training

- Implement embeddings, attention, MLP, norm, output head.
- Implement optimizer and training step loop.
- Implement sample generation.

Acceptance:

- Loss trend decreases on toy run.
- `sample` outputs valid decoded strings.

## Milestone 4: Tensor Backend Path

- Add backend dependency and abstraction boundary.
- Implement equivalent forward pass and training loop.
- Reuse tokenizer/config/CLI.

Acceptance:

- Tensor mode completes train + sample end-to-end.
- Basic throughput improvement over scalar on same machine is measurable.

## Milestone 5: Ablation Harness

- Add toggles for `tie_lm_head` and `input_rmsnorm`.
- Run all combinations with fixed seed and common data split.
- Print table and save machine-readable results.

Acceptance:

- One command executes all 4 variants.
- Output includes loss and throughput per variant.

## Milestone 6: Reporting + Reproducibility

- Persist run metadata (seed, config, commit hash, backend).
- Add simple result file format (e.g., JSON/CSV in `results/`).
- Document exact commands for reproducing baseline experiments.

Acceptance:

- Re-running same config with same seed gives consistent trend.

## 6. Proposed Initial Repository Layout

```text
src/
  main.rs
  cli.rs
  config.rs
  data/
    mod.rs
    tokenizer.rs
  scalar/
    mod.rs
    value.rs
    model.rs
    optim.rs
  tensor/
    mod.rs
    model.rs
    optim.rs
  experiments/
    mod.rs
    ablate.rs
results/
```

(Structure may be adjusted if a simpler layout proves clearer.)

## 7. Testing Strategy

Unit tests:

- tokenizer encode/decode and BOS behavior
- scalar operation gradients
- softmax normalization
- config parsing/validation

Integration tests (lightweight):

- one short scalar training run
- one short tensor training run
- ablation command smoke test

Regression tests:

- add a focused test for every fixed bug when practical

## 8. Benchmark Protocol

For fair scalar vs tensor comparison:

1. Same dataset subset.
2. Same seed and model hyperparameters.
3. Same number of steps.
4. Warm-up run before timing.
5. Report hardware info and backend.

## 9. Risks and Mitigations

Risk: scalar/tensor mismatch due to numeric differences.
Mitigation: assert trend-level parity, not exact value equality.

Risk: complexity drift reduces readability.
Mitigation: keep scalar path minimal and documented; isolate backend code.

Risk: misleading performance claims.
Mitigation: publish benchmark protocol with run metadata.

## 10. Definition of Done (pre-feature expansion)

Done means:

1. All milestones above meet acceptance criteria.
2. README documents setup, commands, and experiment flow.
3. `cargo check`, `cargo test`, and at least one end-to-end run succeed.
4. Clean git history with reviewable commits.

## 11. First Coding Tasks (next session)

1. Build CLI skeleton (`train`, `sample`, `ablate`).
2. Add tokenizer/data loader module with tests.
3. Implement scalar `Value` core and gradient tests.
4. Wire first scalar train step.
