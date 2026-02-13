use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::value::Value;
use crate::training;

const RMS_EPS: f64 = 1e-8;
const LOG_EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScalarBigramConfig {
    pub tie_lm_head: bool,
    pub input_rmsnorm: bool,
}

pub(crate) struct ScalarBigram {
    config: ScalarBigramConfig,
    token_embedding: Vec<Vec<Value>>,
    lm_head: Option<Vec<Vec<Value>>>,
}

impl ScalarBigram {
    pub(crate) fn new(
        vocab_size: usize,
        seed: u64,
        tie_lm_head: bool,
        input_rmsnorm: bool,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let token_embedding = init_matrix(vocab_size, &mut rng);
        let lm_head = if tie_lm_head {
            None
        } else {
            Some(init_matrix(vocab_size, &mut rng))
        };

        Self {
            config: ScalarBigramConfig {
                tie_lm_head,
                input_rmsnorm,
            },
            token_embedding,
            lm_head,
        }
    }

    pub(crate) fn parameters(&self) -> Vec<Value> {
        let mut params: Vec<Value> = self
            .token_embedding
            .iter()
            .flat_map(|row| row.iter().cloned())
            .collect();
        if let Some(lm_head) = &self.lm_head {
            params.extend(lm_head.iter().flat_map(|row| row.iter().cloned()));
        }
        params
    }

    pub(crate) fn nll_loss(&self, context_id: usize, target_id: usize) -> Value {
        let mut hidden = self.token_embedding[context_id].clone();
        if self.config.input_rmsnorm {
            hidden = rmsnorm(hidden);
        }

        let projection_rows = if self.config.tie_lm_head {
            &self.token_embedding
        } else {
            self.lm_head
                .as_ref()
                .expect("lm_head exists when tie_lm_head is false")
        };
        let logits: Vec<Value> = projection_rows
            .iter()
            .map(|row| dot(&hidden, row))
            .collect();

        // Numerical stability: subtract max(logits) before exponentiation.
        let max_logit = logits
            .iter()
            .map(Value::data)
            .fold(f64::NEG_INFINITY, f64::max);
        let shift = Value::new(-max_logit);
        let shifted_logits: Vec<Value> = logits.iter().map(|logit| logit.add(&shift)).collect();

        let exp_logits: Vec<Value> = shifted_logits.iter().map(Value::exp).collect();
        let sum_exp = sum_values(&exp_logits);
        let inv_sum = sum_exp.powf(-1.0);
        let target_prob = exp_logits[target_id].mul(&inv_sum);
        target_prob
            .add(&Value::new(LOG_EPS))
            .log()
            .mul(&Value::new(-1.0))
    }

    pub(crate) fn train_step(
        &mut self,
        context_id: usize,
        target_id: usize,
        learning_rate: f64,
        params: &[Value],
    ) -> f64 {
        for parameter in params {
            parameter.zero_grad();
        }

        let loss = self.nll_loss(context_id, target_id);
        let loss_value = loss.data();
        loss.backward();

        for parameter in params {
            let updated = parameter.data() - learning_rate * parameter.grad();
            parameter.set_data(updated);
        }

        loss_value
    }
}

fn init_matrix(vocab_size: usize, rng: &mut StdRng) -> Vec<Vec<Value>> {
    let mut matrix = Vec::with_capacity(vocab_size);
    for _ in 0..vocab_size {
        let mut row = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            // Small symmetric init keeps logits near zero at start.
            let value = rng.gen_range(-0.01..0.01);
            row.push(Value::new(value));
        }
        matrix.push(row);
    }
    matrix
}

fn dot(lhs: &[Value], rhs: &[Value]) -> Value {
    assert_eq!(lhs.len(), rhs.len(), "dot expects matching vector lengths");
    let mut terms = lhs.iter().zip(rhs).map(|(a, b)| a.mul(b));
    let first = terms.next().expect("dot expects non-empty vectors");
    terms.fold(first, |acc, term| acc.add(&term))
}

fn rmsnorm(values: Vec<Value>) -> Vec<Value> {
    let inv_count = Value::new(1.0 / (values.len() as f64));
    let eps = Value::new(RMS_EPS);

    let squares: Vec<Value> = values.iter().map(|value| value.powf(2.0)).collect();
    let mean_sq = sum_values(&squares).mul(&inv_count);
    let inv_rms = mean_sq.add(&eps).powf(-0.5);

    values
        .into_iter()
        .map(|value| value.mul(&inv_rms))
        .collect()
}

pub(crate) fn sample_index(
    transition_counts: &[Vec<u64>],
    context_id: usize,
    bos_id: usize,
    temperature: f64,
    rng: &mut StdRng,
) -> usize {
    training::sample_index(transition_counts, context_id, bos_id, temperature, rng)
}

pub(crate) fn build_transition_counts(
    vocab_size: usize,
    pairs: &[(usize, usize)],
) -> Vec<Vec<u64>> {
    training::build_transition_counts(vocab_size, pairs)
}

fn sum_values(values: &[Value]) -> Value {
    let mut iter = values.iter();
    let first = iter
        .next()
        .expect("sum_values requires at least one value")
        .clone();
    iter.fold(first, |acc, value| acc.add(value))
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::{ScalarBigram, build_transition_counts, sample_index};

    #[test]
    fn train_step_reduces_loss_for_repeated_pair() {
        let mut model = ScalarBigram::new(3, 7, true, false);
        let params = model.parameters();

        let mut previous = model.nll_loss(1, 2).data();
        for _ in 0..200 {
            model.train_step(1, 2, 0.2, &params);
        }
        let current = model.nll_loss(1, 2).data();

        assert!(current < previous, "expected loss to decrease");
        previous = current;
        assert!(previous.is_finite());
    }

    #[test]
    fn untied_has_more_parameters_than_tied() {
        let tied = ScalarBigram::new(4, 1, true, false);
        let untied = ScalarBigram::new(4, 1, false, false);
        assert!(untied.parameters().len() > tied.parameters().len());
    }

    #[test]
    fn rmsnorm_changes_loss_value() {
        let tied_plain = ScalarBigram::new(4, 2, true, false);
        let tied_norm = ScalarBigram::new(4, 2, true, true);
        let plain = tied_plain.nll_loss(1, 2).data();
        let normed = tied_norm.nll_loss(1, 2).data();
        assert!((plain - normed).abs() > 1e-12);
    }

    #[test]
    fn all_variant_losses_are_finite() {
        let configs = [(true, false), (true, true), (false, false), (false, true)];
        for (tie_lm_head, input_rmsnorm) in configs {
            let model = ScalarBigram::new(8, 11, tie_lm_head, input_rmsnorm);
            let loss = model.nll_loss(3, 5).data();
            assert!(
                loss.is_finite(),
                "loss should be finite for tie_lm_head={tie_lm_head} input_rmsnorm={input_rmsnorm}, got {loss}"
            );
        }
    }

    #[test]
    fn sampling_avoids_bos_when_weights_present() {
        let counts = vec![vec![1, 10, 1], vec![1, 1, 1], vec![1, 1, 1]];
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);

        let sampled = sample_index(&counts, 0, 0, 1.0, &mut rng);
        assert_ne!(sampled, 0);
    }

    #[test]
    fn transition_counts_apply_laplace_smoothing() {
        let counts = build_transition_counts(3, &[(1, 2)]);
        assert_eq!(counts[1][2], 2);
        assert_eq!(counts[1][1], 1);
        assert_eq!(counts[0][0], 1);
    }
}
