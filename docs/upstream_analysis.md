# Upstream Training Methodology Analysis (`karpathy/nanochat`)

Pinned ref analyzed: `2f096867244e3d00a50284d1be05fa3f5dcfb84b`.

Primary sources:
- `references/nanochat/scripts/base_train.py`
- `references/nanochat/nanochat/gpt.py`
- `references/nanochat/nanochat/optim.py`
- `references/nanochat/nanochat/dataloader.py`
- `references/nanochat/nanochat/dataset.py`
- `references/nanochat/nanochat/loss_eval.py`

## 1) Loss computation

### Training loss
- `GPT.forward(..., targets, loss_reduction='mean')` computes token cross-entropy with `ignore_index=-1`:
  - `F.cross_entropy(logits.view(-1, vocab), targets.view(-1), ignore_index=-1, reduction=loss_reduction)`
  - Source: `references/nanochat/nanochat/gpt.py:388-420`
- In `base_train.py`, the train step calls `loss = model(x, y)` (mean reduction), then divides by gradient accumulation steps before backprop:
  - `loss = loss / grad_accum_steps`
  - Source: `references/nanochat/scripts/base_train.py:490-496`
- Logged training loss is an EMA-smoothed scalar (`ema_beta=0.9`), not raw per-step loss:
  - Source: `references/nanochat/scripts/base_train.py:515-517`

### Validation metric
- Validation is **bits per byte (BPB)**, not plain mean CE.
- `evaluate_bpb` computes unreduced per-token loss (`loss_reduction='none'`), masks ignored tokens and tokens with zero byte weight, then returns:
  - `total_nats / (log(2) * total_bytes)`
  - Source: `references/nanochat/nanochat/loss_eval.py:9-65`

Implication for parity in `nanochat-rs-next`:
- Rust tensor path must report a metric equivalent to BPB (or explicitly map CE to BPB) to claim apples-to-apples parity.

## 2) Optimizer and schedules

### Optimizer type
- Training uses a **combined Muon + AdamW** optimizer via `model.setup_optimizer(...)`:
  - Source: `references/nanochat/scripts/base_train.py:300-310`
- Parameter grouping and defaults:
  - AdamW groups: `lm_head`, token/value embeddings, scalar parameters
  - Muon groups: transformer matrix parameters grouped by shape
  - Source: `references/nanochat/nanochat/gpt.py:348-386`
- AdamW defaults in groups include `eps=1e-10`, configurable `betas` (`--adam-beta1`, `--adam-beta2`), and `weight_decay=0.0` for AdamW groups; Muon carries weight decay.
  - Source: `references/nanochat/nanochat/gpt.py:368-380`

### Learning-rate schedule
- LR multiplier schedule is piecewise:
  1. linear warmup (`warmup_ratio`)
  2. constant plateau
  3. linear warmdown to `final_lr_frac`
- Source: `references/nanochat/scripts/base_train.py:347-357`
- Applied every step as `group['lr'] = group['initial_lr'] * lrm`:
  - Source: `references/nanochat/scripts/base_train.py:498-503`

### Additional optimizer schedules
- Muon momentum ramps linearly from `0.85` to `0.95` over first 300 steps:
  - Source: `references/nanochat/scripts/base_train.py:359-363`
- Muon weight decay decays linearly to zero across training:
  - Source: `references/nanochat/scripts/base_train.py:365-367`

## 3) Train/val split and eval cadence

### Split logic
- Dataset split is file-based:
  - train = all parquet files except last
  - val = last parquet file only
- Source: `references/nanochat/nanochat/dataset.py:43-52`, `references/nanochat/nanochat/dataloader.py:35-38`
- This is **not** a fixed percentage split; ratio depends on number/size of shards present locally.

### Eval schedule
- Validation BPB runs when `step % eval_every == 0` and also at final step:
  - Source: `references/nanochat/scripts/base_train.py:398-408`
- Number of eval batches:
  - `eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)`
  - Source: `references/nanochat/scripts/base_train.py:405`

## 4) Sequence construction and batching

### Core shapes and accumulation
- Micro-batch token count per rank: `device_batch_size * max_seq_len`
- Global tokens per micro-step: above multiplied by `ddp_world_size`
- `grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd`
- Source: `references/nanochat/scripts/base_train.py:387-394`

### Dataloader semantics
- BOS-aligned best-fit packing dataloader:
  - each row capacity is `T + 1`
  - each row starts with BOS
  - packs full docs greedily by largest-fit, otherwise crops shortest doc to exactly fill remainder
  - returns `inputs=row[:, :-1]`, `targets=row[:, 1:]` (both shape `[B, T]`)
- Source: `references/nanochat/nanochat/dataloader.py:73-160`

### Important parity nuance
- This loader is not simple contiguous chunking; it is BOS-anchored packing + cropping and may drop tokens.
- Any Rust parity attempt must either replicate this loader behavior or clearly define a different protocol and compare under that protocol.

## 5) Reporting format

- Console status line per step includes: smoothed loss, LR multiplier, step time, tok/sec, BF16 MFU, epoch.
  - Source: `references/nanochat/scripts/base_train.py:533-535`
- W&B logging includes:
  - train metrics every 100 steps (`train/loss`, `train/lrm`, `train/tok_per_sec`, etc.)
  - validation metric at eval points (`val/bpb`)
  - Source: `references/nanochat/scripts/base_train.py:411-416`, `references/nanochat/scripts/base_train.py:535-547`

## 6) Parity checklist for `nanochat-rs-next` (Phase 1.2)

1. Match objective semantics: CE with ignore-index behavior equivalent to upstream.
2. Match validation metric semantics: BPB (byte-weighted nats to bits conversion).
3. Match eval cadence and eval token budgeting formula.
4. Match effective batch semantics (`device_batch_size`, `max_seq_len`, `total_batch_size`, grad accumulation).
5. Decide whether to replicate BOS-best-fit packing exactly or define an explicit alternative baseline.
6. Report both training loss and validation metric in comparable units.
