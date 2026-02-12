use std::collections::HashMap;

use super::value::Value;

#[derive(Debug, Clone, Copy)]
pub(crate) struct AdamWConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct AdamWState {
    step: u64,
    exp_avg: f64,
    exp_avg_sq: f64,
}

impl Default for AdamWState {
    fn default() -> Self {
        Self {
            step: 0,
            exp_avg: 0.0,
            exp_avg_sq: 0.0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct AdamW {
    config: AdamWConfig,
    state: HashMap<usize, AdamWState>,
}

impl AdamW {
    pub(crate) fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            state: HashMap::new(),
        }
    }

    pub(crate) fn step(&mut self, params: &[Value], learning_rate: f64) {
        for parameter in params {
            let grad = parameter.grad();
            let state = self.state.entry(parameter.id()).or_default();

            state.step = state.step.saturating_add(1);
            state.exp_avg = self.config.beta1 * state.exp_avg + (1.0 - self.config.beta1) * grad;
            state.exp_avg_sq =
                self.config.beta2 * state.exp_avg_sq + (1.0 - self.config.beta2) * grad * grad;

            let step_f = state.step as f64;
            let bias_correction1 = 1.0 - self.config.beta1.powf(step_f);
            let bias_correction2 = 1.0 - self.config.beta2.powf(step_f);
            let m_hat = state.exp_avg / bias_correction1.max(f64::MIN_POSITIVE);
            let v_hat = state.exp_avg_sq / bias_correction2.max(f64::MIN_POSITIVE);
            let adam_update = m_hat / (v_hat.sqrt() + self.config.eps);

            let data = parameter.data();
            let update = adam_update + self.config.weight_decay * data;
            parameter.set_data(data - learning_rate * update);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AdamW, AdamWConfig};
    use crate::scalar::value::Value;

    #[test]
    fn adamw_updates_match_reference_for_repeated_gradient() {
        let parameter = Value::new(1.0);
        let half = Value::new(0.5);
        let mut optimizer = AdamW::new(AdamWConfig {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        });

        parameter.zero_grad();
        let loss1 = parameter.mul(&half);
        loss1.backward();
        optimizer.step(std::slice::from_ref(&parameter), 0.1);
        assert!((parameter.data() - 0.9).abs() < 1e-8);

        parameter.zero_grad();
        let loss2 = parameter.mul(&half);
        loss2.backward();
        optimizer.step(std::slice::from_ref(&parameter), 0.1);
        assert!((parameter.data() - 0.8).abs() < 1e-8);
    }
}
