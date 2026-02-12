use std::collections::VecDeque;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum EvalGuardError {
    PromptTooLong {
        prompt_tokens: usize,
        max_prompt_tokens: usize,
    },
    MemoryDrift {
        baseline_bytes: usize,
        current_bytes: usize,
        growth_ratio: f64,
        max_growth_ratio: f64,
    },
}

impl fmt::Display for EvalGuardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PromptTooLong {
                prompt_tokens,
                max_prompt_tokens,
            } => write!(
                f,
                "prompt token count {prompt_tokens} exceeds max prompt tokens {max_prompt_tokens}"
            ),
            Self::MemoryDrift {
                baseline_bytes,
                current_bytes,
                growth_ratio,
                max_growth_ratio,
            } => write!(
                f,
                "eval memory drift detected: baseline={baseline_bytes}B current={current_bytes}B growth_ratio={growth_ratio:.3} max_growth_ratio={max_growth_ratio:.3}"
            ),
        }
    }
}

impl std::error::Error for EvalGuardError {}

pub fn validate_prompt_length(
    prompt_tokens: usize,
    max_prompt_tokens: usize,
) -> Result<(), EvalGuardError> {
    if prompt_tokens > max_prompt_tokens {
        return Err(EvalGuardError::PromptTooLong {
            prompt_tokens,
            max_prompt_tokens,
        });
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct MemoryDriftGuard {
    window_len: usize,
    max_growth_ratio: f64,
    history: VecDeque<usize>,
}

impl MemoryDriftGuard {
    pub fn new(window_len: usize, max_growth_ratio: f64) -> Self {
        assert!(window_len > 0, "window_len must be > 0");
        assert!(
            max_growth_ratio.is_finite() && max_growth_ratio >= 1.0,
            "max_growth_ratio must be finite and >= 1.0"
        );
        Self {
            window_len,
            max_growth_ratio,
            history: VecDeque::with_capacity(window_len),
        }
    }

    pub fn observe(&mut self, current_bytes: usize) -> Result<(), EvalGuardError> {
        if let Some(baseline_bytes) = self.history.iter().copied().min() {
            let growth_ratio = if baseline_bytes == 0 {
                if current_bytes == 0 {
                    1.0
                } else {
                    f64::INFINITY
                }
            } else {
                (current_bytes as f64) / (baseline_bytes as f64)
            };

            if growth_ratio > self.max_growth_ratio {
                return Err(EvalGuardError::MemoryDrift {
                    baseline_bytes,
                    current_bytes,
                    growth_ratio,
                    max_growth_ratio: self.max_growth_ratio,
                });
            }
        }

        self.history.push_back(current_bytes);
        if self.history.len() > self.window_len {
            self.history.pop_front();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{EvalGuardError, MemoryDriftGuard, validate_prompt_length};

    #[test]
    fn eval_prompt_length_safety_rejects_over_limit() {
        let err =
            validate_prompt_length(1025, 1024).expect_err("expected prompt-length safety error");
        match err {
            EvalGuardError::PromptTooLong {
                prompt_tokens,
                max_prompt_tokens,
            } => {
                assert_eq!(prompt_tokens, 1025);
                assert_eq!(max_prompt_tokens, 1024);
            }
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn eval_memory_drift_detection_flags_growth_beyond_limit() {
        let mut guard = MemoryDriftGuard::new(4, 1.20);
        guard.observe(100).expect("warm-up sample");
        guard.observe(102).expect("second sample");
        guard.observe(103).expect("third sample");

        let err = guard
            .observe(150)
            .expect_err("expected memory-drift guard to trigger");
        match err {
            EvalGuardError::MemoryDrift {
                baseline_bytes,
                current_bytes,
                growth_ratio,
                max_growth_ratio,
            } => {
                assert_eq!(baseline_bytes, 100);
                assert_eq!(current_bytes, 150);
                assert!(growth_ratio > max_growth_ratio);
            }
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn eval_memory_drift_guard_allows_small_variation() {
        let mut guard = MemoryDriftGuard::new(3, 1.30);
        guard.observe(100).expect("warm-up sample");
        guard.observe(110).expect("small increase");
        guard.observe(120).expect("small increase");
        guard.observe(125).expect("still below threshold");
    }
}
