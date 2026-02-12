use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::value::Value;

pub(crate) struct ScalarBigram {
    weights: Vec<Vec<Value>>,
}

impl ScalarBigram {
    pub(crate) fn new(vocab_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let mut row = Vec::with_capacity(vocab_size);
            for _ in 0..vocab_size {
                // Small symmetric init keeps logits near zero at start.
                let value = rng.gen_range(-0.01..0.01);
                row.push(Value::new(value));
            }
            weights.push(row);
        }
        Self { weights }
    }

    pub(crate) fn parameters(&self) -> Vec<Value> {
        self.weights
            .iter()
            .flat_map(|row| row.iter().cloned())
            .collect()
    }

    pub(crate) fn nll_loss(&self, context_id: usize, target_id: usize) -> Value {
        let logits = self.weights[context_id].clone();
        let exp_logits: Vec<Value> = logits.iter().map(Value::exp).collect();
        let sum_exp = sum_values(&exp_logits);
        let inv_sum = sum_exp.powf(-1.0);
        let target_prob = exp_logits[target_id].mul(&inv_sum);
        target_prob.log().mul(&Value::new(-1.0))
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

pub(crate) fn sample_index(
    transition_counts: &[Vec<u64>],
    context_id: usize,
    bos_id: usize,
    temperature: f64,
    rng: &mut StdRng,
) -> usize {
    let exponent = 1.0 / temperature;
    let mut weights = Vec::with_capacity(transition_counts[context_id].len());
    for (token_id, count) in transition_counts[context_id].iter().enumerate() {
        if token_id == bos_id {
            weights.push(0.0);
            continue;
        }
        let count = (*count).max(1) as f64;
        weights.push(count.powf(exponent));
    }
    weighted_choice(&weights, rng)
}

pub(crate) fn build_transition_counts(
    vocab_size: usize,
    pairs: &[(usize, usize)],
) -> Vec<Vec<u64>> {
    let mut counts = vec![vec![1_u64; vocab_size]; vocab_size];
    for (context_id, target_id) in pairs {
        counts[*context_id][*target_id] += 1;
    }
    counts
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
    let mut total = 0.0;
    for weight in weights {
        total += *weight;
    }

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
    use rand::SeedableRng;

    use super::{ScalarBigram, build_transition_counts, sample_index};

    #[test]
    fn train_step_reduces_loss_for_repeated_pair() {
        let mut model = ScalarBigram::new(3, 7);
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
