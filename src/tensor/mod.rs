#![cfg_attr(feature = "tch-backend", allow(dead_code))]

use std::fmt;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::config::{SampleConfig, Style, TrainConfig};
use crate::data::{self, TokenizerError};

#[cfg(feature = "tch-backend")]
use tch::nn::{self, OptimizerConfig, VarStore};
#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

const DEFAULT_LEARNING_RATE: f64 = 0.1;
const LOSS_WINDOW: usize = 50;
const RMS_EPS: f64 = 1e-8;
const LOG_EPS: f64 = 1e-12;
const FUTURISTIC_CORPUS: &str = "\
neon skylines hum over orbital transit lanes.\n\
quantum couriers sync with neural uplinks at dawn.\n\
autonomous swarms calibrate reactor lattices in silence.\n\
synthetic pilots chart wormhole routes beyond saturn.\n";

#[derive(Debug, Clone)]
pub struct TrainMetrics {
    pub style: Style,
    pub tie_lm_head: bool,
    pub input_rmsnorm: bool,
    pub backend: &'static str,
    pub device: String,
    pub using_gpu: bool,
    pub parameter_count: usize,
    pub steps: usize,
    pub final_loss: f64,
    pub mean_loss_last_n: f64,
    pub last_n: usize,
    pub steps_per_sec: f64,
    pub tokens_per_sec: f64,
    pub vocab_size: usize,
    pub train_tokens: usize,
}

impl fmt::Display for TrainMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mode=tensor style={} tie_lm_head={} input_rmsnorm={} backend={} device={} using_gpu={} params={} steps={} final_loss={:.6} mean_loss_last_{}={:.6} steps_per_sec={:.2} tokens_per_sec={:.2} vocab_size={} train_tokens={}",
            self.style,
            self.tie_lm_head,
            self.input_rmsnorm,
            self.backend,
            self.device,
            self.using_gpu,
            self.parameter_count,
            self.steps,
            self.final_loss,
            self.last_n,
            self.mean_loss_last_n,
            self.steps_per_sec,
            self.tokens_per_sec,
            self.vocab_size,
            self.train_tokens
        )
    }
}

#[derive(Debug)]
pub enum TensorError {
    Io(std::io::Error),
    Tokenizer(TokenizerError),
    EmptyDataset,
    InvalidTemperature(f64),
    #[cfg(feature = "tch-backend")]
    Tch(tch::TchError),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Tokenizer(err) => write!(f, "{err}"),
            Self::EmptyDataset => write!(f, "dataset must contain at least one token pair"),
            Self::InvalidTemperature(value) => {
                write!(f, "temperature must be finite and > 0, got {value}")
            }
            #[cfg(feature = "tch-backend")]
            Self::Tch(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for TensorError {}

impl From<std::io::Error> for TensorError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<TokenizerError> for TensorError {
    fn from(err: TokenizerError) -> Self {
        Self::Tokenizer(err)
    }
}

#[cfg(feature = "tch-backend")]
impl From<tch::TchError> for TensorError {
    fn from(err: tch::TchError) -> Self {
        Self::Tch(err)
    }
}

struct TensorBigram {
    tie_lm_head: bool,
    input_rmsnorm: bool,
    token_embedding: Vec<Vec<f64>>,
    lm_head: Option<Vec<Vec<f64>>>,
}

impl TensorBigram {
    fn new(vocab_size: usize, seed: u64, tie_lm_head: bool, input_rmsnorm: bool) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let token_embedding = init_matrix(vocab_size, &mut rng);
        let lm_head = if tie_lm_head {
            None
        } else {
            Some(init_matrix(vocab_size, &mut rng))
        };
        Self {
            tie_lm_head,
            input_rmsnorm,
            token_embedding,
            lm_head,
        }
    }

    fn parameter_count(&self) -> usize {
        let token_params = self.token_embedding.len() * self.token_embedding.len();
        let head_params = self
            .lm_head
            .as_ref()
            .map_or(0, |head| head.len() * head.len());
        token_params + head_params
    }

    fn nll_loss(&self, context_id: usize, target_id: usize) -> f64 {
        let hidden = self.token_embedding[context_id].clone();
        let hidden_norm = if self.input_rmsnorm {
            rmsnorm(&hidden)
        } else {
            hidden
        };
        let projection_rows = if self.tie_lm_head {
            &self.token_embedding
        } else {
            self.lm_head
                .as_ref()
                .expect("lm_head exists when tie_lm_head is false")
        };
        let logits: Vec<f64> = projection_rows
            .iter()
            .map(|row| dot(&hidden_norm, row))
            .collect();
        let (_, loss) = softmax_probs_and_loss(&logits, target_id);
        loss
    }

    fn train_step(&mut self, context_id: usize, target_id: usize, learning_rate: f64) -> f64 {
        let vocab_size = self.token_embedding.len();
        let hidden = self.token_embedding[context_id].clone();
        let hidden_norm = if self.input_rmsnorm {
            rmsnorm(&hidden)
        } else {
            hidden.clone()
        };

        let projection_rows = if self.tie_lm_head {
            &self.token_embedding
        } else {
            self.lm_head
                .as_ref()
                .expect("lm_head exists when tie_lm_head is false")
        };
        let logits: Vec<f64> = projection_rows
            .iter()
            .map(|row| dot(&hidden_norm, row))
            .collect();
        let (probs, loss) = softmax_probs_and_loss(&logits, target_id);

        let mut logits_grad = probs;
        logits_grad[target_id] -= 1.0;

        let mut hidden_grad_norm = vec![0.0; vocab_size];
        for (row_idx, row) in projection_rows.iter().enumerate() {
            let grad = logits_grad[row_idx];
            if grad == 0.0 {
                continue;
            }
            for (col_idx, weight) in row.iter().enumerate() {
                hidden_grad_norm[col_idx] += grad * weight;
            }
        }

        if self.tie_lm_head {
            for (row_idx, grad) in logits_grad.iter().copied().enumerate() {
                if grad == 0.0 {
                    continue;
                }
                for (col_idx, hidden_value) in hidden_norm.iter().enumerate() {
                    self.token_embedding[row_idx][col_idx] -= learning_rate * grad * hidden_value;
                }
            }
        } else if let Some(lm_head) = self.lm_head.as_mut() {
            for (row_idx, grad) in logits_grad.iter().copied().enumerate() {
                if grad == 0.0 {
                    continue;
                }
                for (col_idx, hidden_value) in hidden_norm.iter().enumerate() {
                    lm_head[row_idx][col_idx] -= learning_rate * grad * hidden_value;
                }
            }
        }

        let hidden_grad = if self.input_rmsnorm {
            rmsnorm_backward(&hidden, &hidden_grad_norm)
        } else {
            hidden_grad_norm
        };
        for (col_idx, grad) in hidden_grad.iter().enumerate() {
            self.token_embedding[context_id][col_idx] -= learning_rate * grad;
        }

        loss
    }
}

pub fn train(config: &TrainConfig) -> Result<TrainMetrics, TensorError> {
    let base_text = data::load_text(&config.data_path)?;
    train_from_text(
        &base_text,
        config.steps,
        config.seed,
        config.style,
        config.tie_lm_head,
        config.input_rmsnorm,
    )
}

pub fn sample(config: &SampleConfig) -> Result<String, TensorError> {
    let base_text = data::load_text(&config.data_path)?;
    sample_from_text(
        &base_text,
        &config.prompt,
        config.max_new_tokens,
        config.temperature,
        config.seed,
        config.style,
    )
}

fn train_from_text(
    text: &str,
    steps: usize,
    seed: u64,
    style: Style,
    tie_lm_head: bool,
    input_rmsnorm: bool,
) -> Result<TrainMetrics, TensorError> {
    #[cfg(feature = "tch-backend")]
    {
        train_from_text_tch(text, steps, seed, style, tie_lm_head, input_rmsnorm)
    }
    #[cfg(not(feature = "tch-backend"))]
    {
        train_from_text_cpu(text, steps, seed, style, tie_lm_head, input_rmsnorm)
    }
}

fn train_from_text_cpu(
    text: &str,
    steps: usize,
    seed: u64,
    style: Style,
    tie_lm_head: bool,
    input_rmsnorm: bool,
) -> Result<TrainMetrics, TensorError> {
    let corpus = styled_corpus(text, style);
    let tokenizer = data::Tokenizer::from_text(&corpus);
    let token_ids = tokenizer.encode_with_bos(&corpus)?;
    let pairs = build_pairs(&token_ids)?;

    let mut model = TensorBigram::new(tokenizer.vocab_size(), seed, tie_lm_head, input_rmsnorm);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xD1B5_4A32_44C1_AA77);

    let mut losses = Vec::with_capacity(steps.max(1));
    let started = Instant::now();
    if steps == 0 {
        let (context_id, target_id) = pairs[0];
        losses.push(model.nll_loss(context_id, target_id));
    } else {
        for _ in 0..steps {
            let pair_idx = rng.gen_range(0..pairs.len());
            let (context_id, target_id) = pairs[pair_idx];
            let loss = model.train_step(context_id, target_id, DEFAULT_LEARNING_RATE);
            losses.push(loss);
        }
    }

    let elapsed_seconds = started.elapsed().as_secs_f64();
    let steps_per_sec = if steps == 0 || elapsed_seconds <= 0.0 {
        0.0
    } else {
        (steps as f64) / elapsed_seconds
    };
    let tokens_per_sec = steps_per_sec;

    let last_n = losses.len().min(LOSS_WINDOW);
    let tail = &losses[losses.len() - last_n..];
    let mean_loss_last_n = tail.iter().sum::<f64>() / (tail.len() as f64);
    let final_loss = *losses
        .last()
        .expect("losses contains one element when steps=0");

    Ok(TrainMetrics {
        style,
        tie_lm_head,
        input_rmsnorm,
        backend: "cpu-native",
        device: "cpu".to_string(),
        using_gpu: false,
        parameter_count: model.parameter_count(),
        steps,
        final_loss,
        mean_loss_last_n,
        last_n,
        steps_per_sec,
        tokens_per_sec,
        vocab_size: tokenizer.vocab_size(),
        train_tokens: pairs.len(),
    })
}

#[cfg(feature = "tch-backend")]
fn train_from_text_tch(
    text: &str,
    steps: usize,
    seed: u64,
    style: Style,
    tie_lm_head: bool,
    input_rmsnorm: bool,
) -> Result<TrainMetrics, TensorError> {
    let corpus = styled_corpus(text, style);
    let tokenizer = data::Tokenizer::from_text(&corpus);
    let token_ids = tokenizer.encode_with_bos(&corpus)?;
    let pairs = build_pairs(&token_ids)?;

    let device = resolve_tch_device();
    let device_label = tch_device_label(device);
    let using_gpu = matches!(device, Device::Cuda(_));
    let vs = VarStore::new(device);
    let root = &vs.root();
    let vocab_size = i64::try_from(tokenizer.vocab_size()).expect("vocab size fits i64");
    let token_embedding = root.var(
        "token_embedding",
        &[vocab_size, vocab_size],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.01,
        },
    );
    let lm_head = if tie_lm_head {
        None
    } else {
        Some(root.var(
            "lm_head",
            &[vocab_size, vocab_size],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        ))
    };

    let parameter_count: usize = vs.trainable_variables().iter().map(Tensor::numel).sum();
    let mut optimizer = nn::AdamW::default().build(&vs, DEFAULT_LEARNING_RATE)?;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xD1B5_4A32_44C1_AA77);
    let mut losses = Vec::with_capacity(steps.max(1));
    let started = Instant::now();

    if steps == 0 {
        let (context_id, target_id) = pairs[0];
        let context = Tensor::from_slice(&[i64::try_from(context_id).expect("context id fits i64")])
            .to(device);
        let target = Tensor::from_slice(&[i64::try_from(target_id).expect("target id fits i64")])
            .to(device);
        let loss = tch_forward_loss(
            &token_embedding,
            lm_head.as_ref(),
            &context,
            &target,
            input_rmsnorm,
        );
        losses.push(loss.double_value(&[]));
    } else {
        for _ in 0..steps {
            let pair_idx = rng.gen_range(0..pairs.len());
            let (context_id, target_id) = pairs[pair_idx];
            let context =
                Tensor::from_slice(&[i64::try_from(context_id).expect("context id fits i64")])
                    .to(device);
            let target =
                Tensor::from_slice(&[i64::try_from(target_id).expect("target id fits i64")])
                    .to(device);

            let loss = tch_forward_loss(
                &token_embedding,
                lm_head.as_ref(),
                &context,
                &target,
                input_rmsnorm,
            );
            let loss_value = loss.double_value(&[]);
            optimizer.backward_step(&loss);
            losses.push(loss_value);
        }
    }

    let elapsed_seconds = started.elapsed().as_secs_f64();
    let steps_per_sec = if steps == 0 || elapsed_seconds <= 0.0 {
        0.0
    } else {
        (steps as f64) / elapsed_seconds
    };
    let tokens_per_sec = steps_per_sec;

    let last_n = losses.len().min(LOSS_WINDOW);
    let tail = &losses[losses.len() - last_n..];
    let mean_loss_last_n = tail.iter().sum::<f64>() / (tail.len() as f64);
    let final_loss = *losses
        .last()
        .expect("losses contains one element when steps=0");

    Ok(TrainMetrics {
        style,
        tie_lm_head,
        input_rmsnorm,
        backend: "tch",
        device: device_label,
        using_gpu,
        parameter_count,
        steps,
        final_loss,
        mean_loss_last_n,
        last_n,
        steps_per_sec,
        tokens_per_sec,
        vocab_size: tokenizer.vocab_size(),
        train_tokens: pairs.len(),
    })
}

#[cfg(feature = "tch-backend")]
fn resolve_tch_device() -> Device {
    if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

#[cfg(feature = "tch-backend")]
fn tch_device_label(device: Device) -> String {
    match device {
        Device::Cuda(index) => format!("cuda:{index}"),
        _ => format!("{device:?}").to_lowercase(),
    }
}

#[cfg(feature = "tch-backend")]
fn tch_forward_loss(
    token_embedding: &Tensor,
    lm_head: Option<&Tensor>,
    context: &Tensor,
    target: &Tensor,
    input_rmsnorm: bool,
) -> Tensor {
    let hidden = token_embedding.index_select(0, context);
    let hidden = if input_rmsnorm {
        let mean_sq = hidden
            .pow_tensor_scalar(2.0)
            .mean_dim(&[1_i64][..], true, Kind::Float);
        let inv_rms = (mean_sq + RMS_EPS).rsqrt();
        hidden * inv_rms
    } else {
        hidden
    };

    let projection = match lm_head {
        Some(matrix) => matrix,
        None => token_embedding,
    };
    let logits = hidden.matmul(&projection.transpose(0, 1));
    logits.cross_entropy_for_logits(target)
}

fn sample_from_text(
    text: &str,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f64,
    seed: u64,
    style: Style,
) -> Result<String, TensorError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(TensorError::InvalidTemperature(temperature));
    }

    let corpus = styled_corpus(text, style);
    let tokenizer = data::Tokenizer::from_text(&corpus);
    let token_ids = tokenizer.encode_with_bos(&corpus)?;
    let pairs = build_pairs(&token_ids)?;
    let mut transition_counts = build_transition_counts(tokenizer.vocab_size(), &pairs);
    if style == Style::Futuristic {
        apply_futuristic_boost(&mut transition_counts, &tokenizer, tokenizer.bos_id());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let bos_id = tokenizer.bos_id();
    let mut generated = prompt.to_string();

    let mut context_id = if prompt.is_empty() {
        bos_id
    } else {
        let prompt_ids = tokenizer.encode(prompt)?;
        *prompt_ids
            .last()
            .expect("prompt ids non-empty when prompt text is non-empty")
    };

    for _ in 0..max_new_tokens {
        let next_id = sample_index(
            &transition_counts,
            context_id,
            bos_id,
            temperature,
            &mut rng,
        );
        if next_id == bos_id {
            continue;
        }
        let next_char = tokenizer
            .char_for_id(next_id)
            .ok_or(TokenizerError::UnknownId(next_id))?;
        generated.push(next_char);
        context_id = next_id;
    }
    Ok(generated)
}

fn init_matrix(vocab_size: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut matrix = Vec::with_capacity(vocab_size);
    for _ in 0..vocab_size {
        let mut row = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            row.push(rng.gen_range(-0.01..0.01));
        }
        matrix.push(row);
    }
    matrix
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len(), "dot expects matching vector lengths");
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

fn rmsnorm(values: &[f64]) -> Vec<f64> {
    let inv_count = 1.0 / (values.len() as f64);
    let mean_sq = values.iter().map(|value| value * value).sum::<f64>() * inv_count;
    let inv_rms = (mean_sq + RMS_EPS).powf(-0.5);
    values.iter().map(|value| value * inv_rms).collect()
}

fn rmsnorm_backward(input: &[f64], grad_output: &[f64]) -> Vec<f64> {
    let count = input.len() as f64;
    let mean_sq = input.iter().map(|value| value * value).sum::<f64>() / count;
    let inv_rms = (mean_sq + RMS_EPS).powf(-0.5);
    let inv_rms_cubed = inv_rms * inv_rms * inv_rms;
    let dot_grad_input: f64 = grad_output.iter().zip(input).map(|(g, x)| g * x).sum();

    grad_output
        .iter()
        .zip(input)
        .map(|(grad, x)| grad * inv_rms - x * dot_grad_input * inv_rms_cubed / count)
        .collect()
}

fn softmax_probs_and_loss(logits: &[f64], target_id: usize) -> (Vec<f64>, f64) {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<f64> = logits.iter().map(|logit| logit - max_logit).collect();
    let exp_logits: Vec<f64> = shifted.iter().map(|logit| logit.exp()).collect();
    let exp_sum: f64 = exp_logits.iter().sum();
    let inv_sum = 1.0 / exp_sum;
    let probs: Vec<f64> = exp_logits.iter().map(|value| value * inv_sum).collect();
    let target_prob = probs[target_id];
    let loss = -(target_prob + LOG_EPS).ln();
    (probs, loss)
}

fn build_transition_counts(vocab_size: usize, pairs: &[(usize, usize)]) -> Vec<Vec<u64>> {
    let mut counts = vec![vec![1_u64; vocab_size]; vocab_size];
    for (context_id, target_id) in pairs {
        counts[*context_id][*target_id] += 1;
    }
    counts
}

fn sample_index(
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

fn styled_corpus(base: &str, style: Style) -> String {
    match style {
        Style::Classic => base.to_string(),
        Style::Futuristic => format!("{base}\n{FUTURISTIC_CORPUS}"),
    }
}

fn apply_futuristic_boost(
    transition_counts: &mut [Vec<u64>],
    tokenizer: &data::Tokenizer,
    bos_id: usize,
) {
    const BOOST: u64 = 400;
    const PHRASES: [&str; 8] = [
        "neon",
        "quantum",
        "neural",
        "orbital",
        "reactor",
        "wormhole",
        "autonomous",
        "synthetic",
    ];

    for phrase in PHRASES {
        let Ok(ids) = tokenizer.encode(phrase) else {
            continue;
        };
        if ids.is_empty() {
            continue;
        }
        transition_counts[bos_id][ids[0]] += BOOST;
        for pair in ids.windows(2) {
            transition_counts[pair[0]][pair[1]] += BOOST;
        }
    }
}

fn build_pairs(token_ids: &[usize]) -> Result<Vec<(usize, usize)>, TensorError> {
    if token_ids.len() < 2 {
        return Err(TensorError::EmptyDataset);
    }
    Ok(token_ids
        .windows(2)
        .map(|window| (window[0], window[1]))
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::config::Style;

    use super::{TensorError, sample_from_text, train_from_text};

    #[test]
    fn train_loss_drops_on_repetitive_text() {
        let baseline = train_from_text("abababababababab", 0, 13, Style::Classic, true, false)
            .expect("baseline metrics");
        let trained = train_from_text("abababababababab", 400, 13, Style::Classic, true, false)
            .expect("trained metrics");
        assert!(trained.mean_loss_last_n < baseline.final_loss);
        assert!(trained.final_loss.is_finite());
    }

    #[test]
    fn sample_respects_prompt_and_length() {
        let output = sample_from_text("abababababab", "ab", 12, 1.0, 3, Style::Classic)
            .expect("sample text");
        assert!(output.starts_with("ab"));
        assert_eq!(output.len(), 14);
    }

    #[test]
    fn sample_rejects_invalid_temperature() {
        let err = sample_from_text("abab", "", 5, 0.0, 1, Style::Classic)
            .expect_err("invalid temperature");
        match err {
            TensorError::InvalidTemperature(value) => assert_eq!(value, 0.0),
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn empty_input_errors() {
        let err =
            train_from_text("", 1, 0, Style::Classic, true, false).expect_err("empty dataset");
        match err {
            TensorError::EmptyDataset => {}
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn futuristic_style_changes_sampling_distribution() {
        let futuristic = sample_from_text("abababababab", "", 16, 1.0, 9, Style::Futuristic)
            .expect("futuristic sample");
        assert!(
            futuristic.chars().any(|ch| ch != 'a' && ch != 'b'),
            "futuristic output should include non-classic symbols, got {futuristic:?}"
        );
    }

    #[test]
    fn untied_metrics_report_more_parameters() {
        let tied = train_from_text("abababababababab", 0, 5, Style::Classic, true, false)
            .expect("tied metrics");
        let untied = train_from_text("abababababababab", 0, 5, Style::Classic, false, false)
            .expect("untied metrics");
        assert!(untied.parameter_count > tied.parameter_count);
    }

    #[test]
    fn train_reports_backend_and_device() {
        let metrics =
            train_from_text("abababababababab", 1, 5, Style::Classic, true, false)
                .expect("metrics");
        assert!(!metrics.backend.is_empty());
        assert!(!metrics.device.is_empty());
        if metrics.using_gpu {
            assert!(
                metrics.device.starts_with("cuda:"),
                "gpu runs should report cuda device, got {}",
                metrics.device
            );
        } else {
            assert!(
                !metrics.device.starts_with("cuda:"),
                "cpu runs should not report cuda device, got {}",
                metrics.device
            );
        }
    }
}
