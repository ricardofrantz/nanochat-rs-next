use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::config::{ModelKind, Optimizer, Style};
use crate::data::{self, Tokenizer, TokenizerError};
use crate::eval::validate_prompt_length;

use super::optimizer::{AdamW, AdamWConfig};
use super::value::Value;
use super::{ScalarError, TrainMetrics, VAL_FRACTION, build_pairs, should_eval, styled_corpus};

const MINI_GPT_EMBD: usize = 16;
const MINI_GPT_HEADS: usize = 4;
const MINI_GPT_HEAD_DIM: usize = MINI_GPT_EMBD / MINI_GPT_HEADS;
const MINI_GPT_LAYERS: usize = 1;
const MINI_GPT_BLOCK_SIZE: usize = 8;
const MINI_GPT_INIT_STD: f64 = 0.02;
const MINI_GPT_ZERO_STD: f64 = 0.0;
const MINI_GPT_LOG_EPS: f64 = 1e-12;
const MINI_GPT_RMS_EPS: f64 = 1e-5;
const MINI_GPT_LEARNING_RATE: f64 = 0.02;
const MINI_GPT_ADAMW_LEARNING_RATE: f64 = 0.01;
const MINI_GPT_LOSS_WINDOW: usize = 50;
const MINI_GPT_EVAL_EVERY: usize = 20;
const MINI_GPT_EVAL_WINDOWS: usize = 32;

pub(super) fn train_from_text(
    text: &str,
    steps: usize,
    seed: u64,
    optimizer: Optimizer,
    style: Style,
    tie_lm_head: bool,
    input_rmsnorm: bool,
) -> Result<TrainMetrics, ScalarError> {
    let corpus = styled_corpus(text, style);
    let tokenizer = Tokenizer::from_text(&corpus);
    let token_ids = tokenizer.encode_with_bos(&corpus)?;
    let (train_token_ids, val_token_ids) = split_train_val_tokens(&token_ids)?;
    let train_pairs = build_pairs(&train_token_ids)?;

    let mut model = ScalarMiniGpt::new(tokenizer.vocab_size(), seed, tie_lm_head, input_rmsnorm);
    let parameters = model.parameters();
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA36C_D6F2_15B3_9241);
    let mut adamw =
        matches!(optimizer, Optimizer::AdamW).then(|| AdamW::new(AdamWConfig::default()));

    let mut losses = Vec::with_capacity(steps.max(1));
    let mut val_loss = Some(mean_window_loss(&mut model, &val_token_ids));
    let started = Instant::now();

    if steps == 0 {
        losses.push(mean_window_loss(&mut model, &train_token_ids));
    } else {
        for step in 0..steps {
            let window = select_training_window(&train_token_ids, &mut rng);
            let loss = if let Some(opt) = adamw.as_mut() {
                for parameter in &parameters {
                    parameter.zero_grad();
                }
                let loss_node = model.loss_for_tokens(window);
                let loss = loss_node.data();
                loss_node.backward();
                opt.step(&parameters, MINI_GPT_ADAMW_LEARNING_RATE);
                loss
            } else {
                model.train_step(window, MINI_GPT_LEARNING_RATE, &parameters)
            };
            losses.push(loss);
            let step_idx = step + 1;
            if should_eval(step_idx, steps, MINI_GPT_EVAL_EVERY) {
                val_loss = Some(mean_window_loss(&mut model, &val_token_ids));
            }
        }
    }

    let elapsed_seconds = started.elapsed().as_secs_f64();
    let steps_per_sec = if steps == 0 || elapsed_seconds <= 0.0 {
        0.0
    } else {
        (steps as f64) / elapsed_seconds
    };
    let tokens_per_sec = steps_per_sec * (MINI_GPT_BLOCK_SIZE as f64);

    let last_n = losses.len().min(MINI_GPT_LOSS_WINDOW);
    let tail = &losses[losses.len() - last_n..];
    let mean_loss_last_n = tail.iter().sum::<f64>() / (tail.len() as f64);
    let final_loss = *losses
        .last()
        .expect("losses contains one element when steps=0");

    Ok(TrainMetrics {
        model_kind: ModelKind::MiniGpt,
        style,
        tie_lm_head,
        input_rmsnorm,
        parameter_count: parameters.len(),
        steps,
        final_loss,
        val_loss,
        mean_loss_last_n,
        last_n,
        steps_per_sec,
        tokens_per_sec,
        vocab_size: tokenizer.vocab_size(),
        train_tokens: train_pairs.len(),
    })
}

pub(super) fn sample_from_text(
    text: &str,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f64,
    seed: u64,
    style: Style,
) -> Result<String, ScalarError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(ScalarError::InvalidTemperature(temperature));
    }

    let corpus = styled_corpus(text, style);
    let tokenizer = data::Tokenizer::from_text(&corpus);
    let token_ids = tokenizer.encode_with_bos(&corpus)?;
    let _ = build_pairs(&token_ids)?;

    let prompt_ids = tokenizer.encode(prompt)?;
    validate_prompt_length(prompt_ids.len(), MINI_GPT_BLOCK_SIZE.saturating_sub(1))?;

    let mut model = ScalarMiniGpt::new(tokenizer.vocab_size(), seed, true, true);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xE61A_4BF0_2D89_7C53);
    let mut generated = prompt.to_string();
    let bos_id = tokenizer.bos_id();

    let mut keys: Vec<Vec<Vec<Value>>> = (0..MINI_GPT_LAYERS).map(|_| Vec::new()).collect();
    let mut values: Vec<Vec<Vec<Value>>> = (0..MINI_GPT_LAYERS).map(|_| Vec::new()).collect();
    let mut current_token = bos_id;
    let mut pos = 0;

    for token_id in prompt_ids {
        let _ = model.forward_token(current_token, pos, &mut keys, &mut values);
        current_token = token_id;
        pos += 1;
    }

    for _ in 0..max_new_tokens {
        if pos >= MINI_GPT_BLOCK_SIZE {
            break;
        }

        let logits = model.forward_token(current_token, pos, &mut keys, &mut values);
        let next_id = sample_from_logits(&logits, temperature, bos_id, &mut rng);
        if next_id == bos_id {
            continue;
        }

        let next_char = tokenizer
            .char_for_id(next_id)
            .ok_or(TokenizerError::UnknownId(next_id))?;
        generated.push(next_char);
        current_token = next_id;
        pos += 1;
    }

    Ok(generated)
}

struct ScalarMiniGpt {
    tie_lm_head: bool,
    input_rmsnorm: bool,
    token_embedding: Vec<Vec<Value>>,
    position_embedding: Vec<Vec<Value>>,
    lm_head: Option<Vec<Vec<Value>>>,
    layers: Vec<MiniGptLayer>,
}

struct MiniGptLayer {
    attn_wq: Vec<Vec<Value>>,
    attn_wk: Vec<Vec<Value>>,
    attn_wv: Vec<Vec<Value>>,
    attn_wo: Vec<Vec<Value>>,
    mlp_fc1: Vec<Vec<Value>>,
    mlp_fc2: Vec<Vec<Value>>,
}

impl ScalarMiniGpt {
    fn new(vocab_size: usize, seed: u64, tie_lm_head: bool, input_rmsnorm: bool) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let token_embedding = init_matrix(vocab_size, MINI_GPT_EMBD, MINI_GPT_INIT_STD, &mut rng);
        let position_embedding = init_matrix(
            MINI_GPT_BLOCK_SIZE,
            MINI_GPT_EMBD,
            MINI_GPT_INIT_STD,
            &mut rng,
        );
        let lm_head = if tie_lm_head {
            None
        } else {
            Some(init_matrix(
                vocab_size,
                MINI_GPT_EMBD,
                MINI_GPT_INIT_STD,
                &mut rng,
            ))
        };

        let mut layers = Vec::with_capacity(MINI_GPT_LAYERS);
        for _ in 0..MINI_GPT_LAYERS {
            layers.push(MiniGptLayer {
                attn_wq: init_matrix(MINI_GPT_EMBD, MINI_GPT_EMBD, MINI_GPT_INIT_STD, &mut rng),
                attn_wk: init_matrix(MINI_GPT_EMBD, MINI_GPT_EMBD, MINI_GPT_INIT_STD, &mut rng),
                attn_wv: init_matrix(MINI_GPT_EMBD, MINI_GPT_EMBD, MINI_GPT_INIT_STD, &mut rng),
                attn_wo: init_matrix(MINI_GPT_EMBD, MINI_GPT_EMBD, MINI_GPT_ZERO_STD, &mut rng),
                mlp_fc1: init_matrix(
                    4 * MINI_GPT_EMBD,
                    MINI_GPT_EMBD,
                    MINI_GPT_INIT_STD,
                    &mut rng,
                ),
                mlp_fc2: init_matrix(
                    MINI_GPT_EMBD,
                    4 * MINI_GPT_EMBD,
                    MINI_GPT_ZERO_STD,
                    &mut rng,
                ),
            });
        }

        Self {
            tie_lm_head,
            input_rmsnorm,
            token_embedding,
            position_embedding,
            lm_head,
            layers,
        }
    }

    fn parameters(&self) -> Vec<Value> {
        let mut params: Vec<Value> = self
            .token_embedding
            .iter()
            .chain(self.position_embedding.iter())
            .flat_map(|row| row.iter().cloned())
            .collect();
        if let Some(lm_head) = &self.lm_head {
            params.extend(lm_head.iter().flat_map(|row| row.iter().cloned()));
        }
        for layer in &self.layers {
            params.extend(layer.attn_wq.iter().flat_map(|row| row.iter().cloned()));
            params.extend(layer.attn_wk.iter().flat_map(|row| row.iter().cloned()));
            params.extend(layer.attn_wv.iter().flat_map(|row| row.iter().cloned()));
            params.extend(layer.attn_wo.iter().flat_map(|row| row.iter().cloned()));
            params.extend(layer.mlp_fc1.iter().flat_map(|row| row.iter().cloned()));
            params.extend(layer.mlp_fc2.iter().flat_map(|row| row.iter().cloned()));
        }
        params
    }

    fn loss_for_tokens(&mut self, token_ids: &[usize]) -> Value {
        let n = token_ids.len().saturating_sub(1).min(MINI_GPT_BLOCK_SIZE);
        let mut keys: Vec<Vec<Vec<Value>>> = (0..self.layers.len()).map(|_| Vec::new()).collect();
        let mut values: Vec<Vec<Vec<Value>>> = (0..self.layers.len()).map(|_| Vec::new()).collect();
        let mut losses = Vec::with_capacity(n);

        for pos_id in 0..n {
            let token_id = token_ids[pos_id];
            let target_id = token_ids[pos_id + 1];
            let logits = self.forward_token(token_id, pos_id, &mut keys, &mut values);
            let probs = softmax(&logits);
            let target_prob = probs[target_id]
                .add(&Value::new(MINI_GPT_LOG_EPS))
                .log()
                .mul(&Value::new(-1.0));
            losses.push(target_prob);
        }

        sum_values(&losses).mul(&Value::new(1.0 / (n as f64)))
    }

    fn train_step(&mut self, token_ids: &[usize], learning_rate: f64, params: &[Value]) -> f64 {
        for parameter in params {
            parameter.zero_grad();
        }

        let loss = self.loss_for_tokens(token_ids);
        let loss_value = loss.data();
        loss.backward();

        for parameter in params {
            let updated = parameter.data() - learning_rate * parameter.grad();
            parameter.set_data(updated);
        }
        loss_value
    }

    fn forward_token(
        &mut self,
        token_id: usize,
        pos_id: usize,
        keys: &mut [Vec<Vec<Value>>],
        values: &mut [Vec<Vec<Value>>],
    ) -> Vec<Value> {
        let mut x = add_vectors(
            self.token_embedding[token_id].clone(),
            self.position_embedding[pos_id].clone(),
        );
        if self.input_rmsnorm {
            x = rmsnorm(x);
        }

        let inv_sqrt_head_dim = Value::new(1.0 / (MINI_GPT_HEAD_DIM as f64).sqrt());

        for layer_id in 0..self.layers.len() {
            let layer = &self.layers[layer_id];

            let x_residual = x.clone();
            let x_norm = rmsnorm(x);
            let q = linear(&x_norm, &layer.attn_wq);
            let k = linear(&x_norm, &layer.attn_wk);
            let v = linear(&x_norm, &layer.attn_wv);
            keys[layer_id].push(k);
            values[layer_id].push(v);

            let mut x_attn = Vec::with_capacity(MINI_GPT_EMBD);
            for h in 0..MINI_GPT_HEADS {
                let hs = h * MINI_GPT_HEAD_DIM;
                let he = hs + MINI_GPT_HEAD_DIM;
                let q_h = q[hs..he].to_vec();
                let mut attn_logits = Vec::with_capacity(keys[layer_id].len());
                for key_state in &keys[layer_id] {
                    let k_h = key_state[hs..he].to_vec();
                    attn_logits.push(dot(&q_h, &k_h).mul(&inv_sqrt_head_dim));
                }
                let attn_weights = softmax(&attn_logits);
                for j in 0..MINI_GPT_HEAD_DIM {
                    let mut head_terms = Vec::with_capacity(attn_weights.len());
                    for (weight, value_state) in attn_weights.iter().zip(values[layer_id].iter()) {
                        head_terms.push(weight.mul(&value_state[hs + j]));
                    }
                    x_attn.push(sum_values(&head_terms));
                }
            }

            x = add_vectors(linear(&x_attn, &layer.attn_wo), x_residual);

            let x_residual = x.clone();
            let x_norm = rmsnorm(x);
            let x_mlp = linear(&x_norm, &layer.mlp_fc1);
            let x_mlp = x_mlp
                .into_iter()
                .map(|value| value.relu().powf(2.0))
                .collect::<Vec<_>>();
            x = add_vectors(linear(&x_mlp, &layer.mlp_fc2), x_residual);
        }

        let projection = if self.tie_lm_head {
            &self.token_embedding
        } else {
            self.lm_head
                .as_ref()
                .expect("lm_head exists when tie_lm_head is false")
        };
        linear(&x, projection)
    }
}

fn select_training_window<'a>(token_ids: &'a [usize], rng: &mut StdRng) -> &'a [usize] {
    if token_ids.len() <= MINI_GPT_BLOCK_SIZE + 1 {
        return token_ids;
    }
    let max_start = token_ids.len() - (MINI_GPT_BLOCK_SIZE + 1);
    let start = rng.gen_range(0..=max_start);
    &token_ids[start..start + MINI_GPT_BLOCK_SIZE + 1]
}

fn mean_window_loss(model: &mut ScalarMiniGpt, token_ids: &[usize]) -> f64 {
    if token_ids.len() <= MINI_GPT_BLOCK_SIZE + 1 {
        return model.loss_for_tokens(token_ids).data();
    }
    let max_start = token_ids.len() - (MINI_GPT_BLOCK_SIZE + 1);
    let step = (max_start / MINI_GPT_EVAL_WINDOWS.max(1)).max(1);

    let mut total = 0.0;
    let mut count = 0usize;
    let mut start = 0usize;
    while start <= max_start && count < MINI_GPT_EVAL_WINDOWS {
        total += model
            .loss_for_tokens(&token_ids[start..start + MINI_GPT_BLOCK_SIZE + 1])
            .data();
        count += 1;
        start += step;
    }
    total / (count as f64)
}

fn split_train_val_tokens(token_ids: &[usize]) -> Result<(Vec<usize>, Vec<usize>), ScalarError> {
    let _ = build_pairs(token_ids)?;
    let (train_tokens, val_tokens) = data::split_train_val(token_ids, VAL_FRACTION);
    let train_tokens = if build_pairs(&train_tokens).is_ok() {
        train_tokens
    } else {
        token_ids.to_vec()
    };
    let val_tokens = if build_pairs(&val_tokens).is_ok() {
        val_tokens
    } else {
        train_tokens.clone()
    };
    Ok((train_tokens, val_tokens))
}

fn sample_from_logits(
    logits: &[Value],
    temperature: f64,
    bos_id: usize,
    rng: &mut StdRng,
) -> usize {
    let inv_temp = Value::new(1.0 / temperature);
    let scaled_logits: Vec<Value> = logits.iter().map(|logit| logit.mul(&inv_temp)).collect();
    let probs = softmax(&scaled_logits);
    let mut weights = Vec::with_capacity(probs.len());
    for (idx, prob) in probs.iter().enumerate() {
        if idx == bos_id {
            weights.push(0.0);
        } else {
            weights.push(prob.data().max(0.0));
        }
    }
    weighted_choice(&weights, rng)
}

fn init_matrix(rows: usize, cols: usize, std: f64, rng: &mut StdRng) -> Vec<Vec<Value>> {
    let mut matrix = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            let value = if std == 0.0 {
                0.0
            } else {
                rng.gen_range(-std..std)
            };
            row.push(Value::new(value));
        }
        matrix.push(row);
    }
    matrix
}

fn linear(x: &[Value], w: &[Vec<Value>]) -> Vec<Value> {
    w.iter().map(|row| dot(row, x)).collect()
}

fn dot(lhs: &[Value], rhs: &[Value]) -> Value {
    assert_eq!(lhs.len(), rhs.len(), "dot expects matching vector lengths");
    let mut terms = lhs.iter().zip(rhs).map(|(a, b)| a.mul(b));
    let first = terms.next().expect("dot expects non-empty vectors");
    terms.fold(first, |acc, term| acc.add(&term))
}

fn add_vectors(lhs: Vec<Value>, rhs: Vec<Value>) -> Vec<Value> {
    assert_eq!(lhs.len(), rhs.len(), "add_vectors expects matching lengths");
    lhs.into_iter().zip(rhs).map(|(a, b)| a.add(&b)).collect()
}

fn rmsnorm(values: Vec<Value>) -> Vec<Value> {
    let inv_count = Value::new(1.0 / (values.len() as f64));
    let eps = Value::new(MINI_GPT_RMS_EPS);
    let squares: Vec<Value> = values.iter().map(|value| value.powf(2.0)).collect();
    let mean_sq = sum_values(&squares).mul(&inv_count);
    let inv_rms = mean_sq.add(&eps).powf(-0.5);
    values
        .into_iter()
        .map(|value| value.mul(&inv_rms))
        .collect()
}

fn softmax(logits: &[Value]) -> Vec<Value> {
    let max_logit = logits
        .iter()
        .map(Value::data)
        .fold(f64::NEG_INFINITY, f64::max);
    let shift = Value::new(-max_logit);
    let shifted: Vec<Value> = logits.iter().map(|logit| logit.add(&shift)).collect();
    let exp_logits: Vec<Value> = shifted.iter().map(Value::exp).collect();
    let sum_exp = sum_values(&exp_logits);
    let inv_sum = sum_exp.powf(-1.0);
    exp_logits
        .into_iter()
        .map(|exp_logit| exp_logit.mul(&inv_sum))
        .collect()
}

fn sum_values(values: &[Value]) -> Value {
    let mut iter = values.iter();
    let first = iter
        .next()
        .expect("sum_values requires at least one value")
        .clone();
    iter.fold(first, |acc, value| acc.add(value))
}

fn weighted_choice(weights: &[f64], rng: &mut StdRng) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0;
    }

    let mut sample = rng.gen_range(0.0..total);
    let mut last_nonzero = 0;
    for (idx, weight) in weights.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        last_nonzero = idx;
        if sample <= *weight {
            return idx;
        }
        sample -= *weight;
    }
    last_nonzero
}

#[cfg(test)]
mod tests {
    use crate::config::{Optimizer, Style};

    use super::{MINI_GPT_BLOCK_SIZE, ScalarError, sample_from_text, train_from_text};

    #[test]
    fn mini_gpt_train_loss_drops_on_repetitive_text() {
        let baseline = train_from_text(
            "abababababababab",
            0,
            17,
            Optimizer::Sgd,
            Style::Classic,
            true,
            true,
        )
        .expect("baseline metrics");
        let trained = train_from_text(
            "abababababababab",
            50,
            17,
            Optimizer::Sgd,
            Style::Classic,
            true,
            true,
        )
        .expect("trained metrics");

        assert!(trained.mean_loss_last_n < baseline.final_loss);
        assert!(trained.final_loss.is_finite());
    }

    #[test]
    fn mini_gpt_sample_respects_prompt_and_length() {
        let output =
            sample_from_text("abababababab", "ab", 4, 1.0, 3, Style::Classic).expect("sample text");
        assert!(output.starts_with("ab"));
        assert!(output.len() >= 2);
        assert!(output.len() <= 6);
    }

    #[test]
    fn mini_gpt_rejects_prompt_longer_than_block_size() {
        let too_long_prompt = "a".repeat(MINI_GPT_BLOCK_SIZE);
        let err = sample_from_text("abababababab", &too_long_prompt, 2, 1.0, 3, Style::Classic)
            .expect_err("prompt should exceed mini-gpt block-size guard");
        match err {
            ScalarError::EvalGuard(_) => {}
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn mini_gpt_reports_validation_loss() {
        let text = "abcdefghijklmnopqrstuvwxyz0123456789";
        let metrics = train_from_text(text, 5, 19, Optimizer::Sgd, Style::Classic, true, true)
            .expect("metrics");

        let tokenizer = crate::data::Tokenizer::from_text(text);
        let token_ids = tokenizer
            .encode_with_bos(text)
            .expect("token ids should encode");
        let total_pairs = token_ids.len() - 1;

        assert!(
            metrics.train_tokens < total_pairs,
            "train split should hold out validation tokens"
        );
        assert!(
            metrics.val_loss.is_some(),
            "validation loss should be reported"
        );
        assert!(
            metrics.val_loss.expect("val loss exists").is_finite(),
            "validation loss should be finite"
        );
    }
}
