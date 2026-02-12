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
cargo run --release -- train --mode scalar --style futuristic --steps 500
cargo run --release -- train --mode tensor --steps 500
cargo run --release -- ablate --style futuristic --steps 500
cargo run --release -- sample --mode scalar --style classic --temperature 0.8 --max-new-tokens 120
cargo run --release -- sample --mode scalar --style futuristic --temperature 0.8 --max-new-tokens 120
cargo run --release -- sample --mode tensor --temperature 0.6
```

## Current Status

Implementation has started:

- Phase 0 baseline CLI skeleton is in place (`train`, `sample`, `ablate`).
- Phase 1 tokenizer/data utilities are implemented with unit tests.
- Phase 2 scalar autograd `Value` core is implemented with gradient tests.
- Phase 3 scalar path now has a working bigram trainer and sampler.
- Scalar train/sample support style modes: `classic` and `futuristic` (default).
- Ablation now runs all 4 planned variants and persists style-tagged CSV/JSONL files in `results/`.

Tensor mode is still a stub.

## Development

```bash
cargo check
cargo test
```
