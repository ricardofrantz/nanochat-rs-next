# microgpt-rs-dual

A minimal Rust implementation of microGPT with two execution modes:

1. `scalar` mode for educational clarity (close to the original pure-Python structure).
2. `tensor` mode for practical speed using a Rust tensor backend.

## Why this repo

The goal is to keep the "readable from top to bottom" learning value while also making it useful for measurement and iteration.

## Scope

- Implement character-level GPT training and inference in Rust.
- Support side-by-side architectural ablations:
  - tied vs untied `lm_head`
  - with vs without input `rmsnorm`
- Report both quality and speed metrics.

## Non-goals (for v1)

- Chasing SOTA benchmarks.
- Building a full training framework.
- Multi-node distributed training.

## Planned CLI shape

```bash
cargo run --release -- train --mode scalar --steps 500
cargo run --release -- train --mode tensor --steps 500
cargo run --release -- ablate --steps 500
cargo run --release -- sample --mode tensor --temperature 0.6
```

## Status

Scaffolded repository. Implementation starts in `plan_2026-02-12.md`.

## Development

```bash
cargo check
cargo test
```

## License

TBD
