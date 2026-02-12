# microgpt-rs-lab

`microgpt-rs-lab` is a Rust-first reimplementation of Andrej Karpathy's microGPT idea, with two goals:

1. Keep the code readable enough to study end-to-end.
2. Add a practical path for speed experiments and architecture ablations.

## Reference

This project is directly inspired by Karpathy's implementation:

- Gist: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
- Single-page version: https://karpathy.ai/microgpt.html

The intent is not to outgrow the reference quickly, but to preserve its teaching value while making controlled experiments easier in Rust.

## Planned Modes

- `scalar`: educational, explicit operations and gradients.
- `tensor`: backend-accelerated path for throughput and scaling experiments.

## Planned Experiments

- tied vs untied `lm_head`
- with vs without input `rmsnorm`

## Planned CLI

```bash
cargo run --release -- train --mode scalar --steps 500
cargo run --release -- train --mode tensor --steps 500
cargo run --release -- ablate --steps 500
cargo run --release -- sample --mode tensor --temperature 0.6
```

## Current Status

Repository and planning are ready. See `plan.md` before implementation starts.

## Development

```bash
cargo check
cargo test
```
