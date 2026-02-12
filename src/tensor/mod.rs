#![cfg_attr(feature = "tch-backend", allow(dead_code))]

use std::convert::Infallible;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::checkpoint::{CheckpointEvalError, persist_then_eval};
use crate::config::{ModelKind, Optimizer, SampleConfig, Style, TrainConfig};
use crate::eval::EvalGuardError;
use crate::data::{self, TokenizerError};
use crate::training;

#[cfg(feature = "tch-backend")]
use tch::nn::{self, OptimizerConfig, VarStore};
#[cfg(feature = "tch-backend")]
use tch::{Device, Kind, Tensor};

const DEFAULT_LEARNING_RATE: f64 = 0.1;
const LOSS_WINDOW: usize = 50;
const RMS_EPS: f64 = 1e-8;
const LOG_EPS: f64 = 1e-12;
const EVAL_EVERY: usize = training::EVAL_EVERY;
#[cfg(feature = "tch-backend")]
const MINI_GPT_EMBED_DIM: i64 = 16;
#[cfg(feature = "tch-backend")]
const MINI_GPT_BLOCK_SIZE: usize = 8;
#[cfg(feature = "tch-backend")]
const MINI_GPT_LEARNING_RATE: f64 = 0.02;
#[cfg(feature = "tch-backend")]
const MINI_GPT_ADAMW_LEARNING_RATE: f64 = 0.01;
#[cfg(feature = "tch-backend")]
const MINI_GPT_LOSS_WINDOW: usize = 50;

type TokenPair = training::TokenPair;
type PairSplits = training::PairSplits;

#[derive(Clone, Copy)]
struct TensorTrainRuntime<'a> {
    checkpoint_every: usize,
    checkpoint_dir: &'a Path,
}

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
    pub val_loss: Option<f64>,
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
            "mode=tensor style={} tie_lm_head={} input_rmsnorm={} backend={} device={} using_gpu={} params={} steps={} final_loss={:.6} val_loss={} mean_loss_last_{}={:.6} steps_per_sec={:.2} tokens_per_sec={:.2} vocab_size={} train_tokens={}",
            self.style,
            self.tie_lm_head,
            self.input_rmsnorm,
            self.backend,
            self.device,
            self.using_gpu,
            self.parameter_count,
            self.steps,
            self.final_loss,
            self.val_loss
                .map(|loss| format!("{loss:.6}"))
                .unwrap_or_else(|| "none".to_string()),
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
    EvalGuard(EvalGuardError),
    UnsupportedModelKind(ModelKind),
    UnsupportedOptimizer(Optimizer),
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
            Self::EvalGuard(err) => write!(f, "{err}"),
            Self::UnsupportedModelKind(model_kind) => {
                write!(f, "model kind {model_kind} is not supported by tensor mode")
            }
            Self::UnsupportedOptimizer(optimizer) => {
                write!(f, "optimizer {optimizer} is not supported by tensor mode")
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

impl From<EvalGuardError> for TensorError {
    fn from(err: EvalGuardError) -> Self {
        Self::EvalGuard(err)
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
        let (hidden_norm, hidden_for_rmsnorm_backward) = if self.input_rmsnorm {
            (rmsnorm(&hidden), Some(hidden))
        } else {
            (hidden, None)
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
            apply_projection_gradient(
                &mut self.token_embedding,
                &logits_grad,
                &hidden_norm,
                learning_rate,
            );
        } else if let Some(lm_head) = self.lm_head.as_mut() {
            apply_projection_gradient(lm_head, &logits_grad, &hidden_norm, learning_rate);
        }

        let hidden_grad = if let Some(hidden_original) = hidden_for_rmsnorm_backward.as_ref() {
            rmsnorm_backward(hidden_original, &hidden_grad_norm)
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
    let runtime = &config.runtime;
    ensure_tensor_support(runtime.model_kind)?;
    ensure_tensor_optimizer_support(config.optimizer)?;
    let base_text = data::load_text(&runtime.data_path)?;
    train_from_text_with_checkpoints(
        &base_text,
        config.steps,
        runtime.seed,
        runtime.model_kind,
        config.optimizer,
        runtime.style,
        config.tie_lm_head,
        config.input_rmsnorm,
        TensorTrainRuntime {
            checkpoint_every: config.checkpoint_every,
            checkpoint_dir: &config.checkpoint_dir,
        },
    )
}

pub fn sample(config: &SampleConfig) -> Result<String, TensorError> {
    let runtime = &config.runtime;
    ensure_tensor_support(runtime.model_kind)?;
    let base_text = data::load_text(&runtime.data_path)?;
    sample_from_text_with_model_kind(
        &base_text,
        runtime.model_kind,
        &config.prompt,
        config.max_new_tokens,
        config.temperature,
        runtime.seed,
        runtime.style,
    )
}

fn sample_from_text_with_model_kind(
    text: &str,
    model_kind: ModelKind,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f64,
    seed: u64,
    style: Style,
) -> Result<String, TensorError> {
    match model_kind {
        ModelKind::Bigram => sample_from_text(text, prompt, max_new_tokens, temperature, seed, style),
        #[cfg(feature = "tch-backend")]
        ModelKind::MiniGpt => sample_from_text_tch_minigpt(
            text,
            prompt,
            max_new_tokens,
            temperature,
            seed,
            style,
        ),
        #[cfg(not(feature = "tch-backend"))]
        ModelKind::MiniGpt => Err(TensorError::UnsupportedModelKind(model_kind)),
    }
}

#[cfg(test)]
fn train_from_text(
    text: &str,
    steps: usize,
    seed: u64,
    model_kind: ModelKind,
    optimizer: Optimizer,
    style: Style,
    tie_lm_head: bool,
    input_rmsnorm: bool,
) -> Result<TrainMetrics, TensorError> {
    ensure_tensor_support(model_kind)?;
    ensure_tensor_optimizer_support(optimizer)?;
    train_from_text_with_checkpoints(
        text,
        steps,
        seed,
        model_kind,
        optimizer,
        style,
        tie_lm_head,
        input_rmsnorm,
        TensorTrainRuntime {
            checkpoint_every: 0,
            checkpoint_dir: Path::new(training::DEFAULT_CHECKPOINT_DIR),
        },
    )
}

fn train_from_text_with_checkpoints(
    text: &str,
    steps: usize,
    seed: u64,
    model_kind: ModelKind,
    optimizer: Optimizer,
    style: Style,
    tie_lm_head: bool,
    input_rmsnorm: bool,
    runtime: TensorTrainRuntime<'_>,
) -> Result<TrainMetrics, TensorError> {
    #[cfg(feature = "tch-backend")]
    match model_kind {
        ModelKind::Bigram => train_from_text_tch(
            text,
            steps,
            seed,
            model_kind,
            optimizer,
            style,
            tie_lm_head,
            input_rmsnorm,
            runtime,
        ),
        ModelKind::MiniGpt => train_from_text_tch_minigpt(
            text,
            steps,
            seed,
            optimizer,
            style,
            tie_lm_head,
            input_rmsnorm,
            runtime,
        ),
    }
    #[cfg(not(feature = "tch-backend"))]
    {
        if !matches!(model_kind, ModelKind::Bigram) {
            return Err(TensorError::UnsupportedModelKind(model_kind));
        }
        train_from_text_cpu(
            text,
            steps,
            seed,
            model_kind,
            optimizer,
            style,
            tie_lm_head,
            input_rmsnorm,
            runtime,
        )
    }
}

fn train_from_text_cpu(
    text: &str,
    steps: usize,
    seed: u64,
    model_kind: ModelKind,
    optimizer: Optimizer,
    style: Style,
    tie_lm_head: bool,
    input_rmsnorm: bool,
    runtime: TensorTrainRuntime<'_>,
) -> Result<TrainMetrics, TensorError> {
    if !matches!(model_kind, ModelKind::Bigram) {
        return Err(TensorError::UnsupportedModelKind(model_kind));
    }
    ensure_tensor_optimizer_support(optimizer)?;
    let corpus = styled_corpus(text, style);
    let tokenizer = data::Tokenizer::from_text(&corpus);
    let token_ids = tokenizer.encode_with_bos(&corpus)?;
    let (train_pairs, val_pairs) = split_train_val_pairs(&token_ids)?;

    let mut model = TensorBigram::new(tokenizer.vocab_size(), seed, tie_lm_head, input_rmsnorm);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xD1B5_4A32_44C1_AA77);

    let mut losses = Vec::with_capacity(steps.max(1));
    let mut val_loss = Some(mean_nll_loss_cpu(&model, eval_pairs_slice(&val_pairs)));
    let started = Instant::now();
    if steps == 0 {
        losses.push(mean_nll_loss_cpu(&model, eval_pairs_slice(&train_pairs)));
        maybe_persist_checkpoint(
            runtime.checkpoint_every,
            runtime.checkpoint_dir,
            "cpu-native",
            0,
            steps,
            losses[0],
            val_loss,
        )?;
    } else {
        for step in 0..steps {
            let pair_idx = rng.gen_range(0..train_pairs.len());
            let (context_id, target_id) = train_pairs[pair_idx];
            let lr = DEFAULT_LEARNING_RATE * lr_multiplier(step, steps);
            let loss = model.train_step(context_id, target_id, lr);
            losses.push(loss);
            let step_idx = step + 1;
            if should_eval(step_idx, steps, EVAL_EVERY) {
                val_loss = Some(mean_nll_loss_cpu(&model, eval_pairs_slice(&val_pairs)));
            }
            maybe_persist_checkpoint(
                runtime.checkpoint_every,
                runtime.checkpoint_dir,
                "cpu-native",
                step_idx,
                steps,
                loss,
                val_loss,
            )?;
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
        val_loss,
        mean_loss_last_n,
        last_n,
        steps_per_sec,
        tokens_per_sec,
        vocab_size: tokenizer.vocab_size(),
        train_tokens: train_pairs.len(),
    })
}

#[cfg(feature = "tch-backend")]
fn train_from_text_tch(
    text: &str,
    steps: usize,
    seed: u64,
    model_kind: ModelKind,
    optimizer: Optimizer,
    style: Style,
    tie_lm_head: bool,
    input_rmsnorm: bool,
    runtime: TensorTrainRuntime<'_>,
) -> Result<TrainMetrics, TensorError> {
    if !matches!(model_kind, ModelKind::Bigram) {
        return Err(TensorError::UnsupportedModelKind(model_kind));
    }
    ensure_tensor_optimizer_support(optimizer)?;
    let corpus = styled_corpus(text, style);
    let tokenizer = data::Tokenizer::from_text(&corpus);
    let token_ids = tokenizer.encode_with_bos(&corpus)?;
    let (train_pairs, val_pairs) = split_train_val_pairs(&token_ids)?;

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
    let mut optimizer = match optimizer {
        Optimizer::AdamW => nn::AdamW::default().build(&vs, DEFAULT_LEARNING_RATE)?,
        Optimizer::Sgd => nn::Sgd::default().build(&vs, DEFAULT_LEARNING_RATE)?,
    };
    let mut rng = StdRng::seed_from_u64(seed ^ 0xD1B5_4A32_44C1_AA77);
    let mut losses = Vec::with_capacity(steps.max(1));
    let mut val_loss = Some(mean_nll_loss_tch(
        &token_embedding,
        lm_head.as_ref(),
        eval_pairs_slice(&val_pairs),
        input_rmsnorm,
        device,
    ));
    let started = Instant::now();

    if steps == 0 {
        let loss = tch_forward_loss(
            &token_embedding,
            lm_head.as_ref(),
            &pairs_to_tensor(eval_pairs_slice(&train_pairs), device, PairField::Context),
            &pairs_to_tensor(eval_pairs_slice(&train_pairs), device, PairField::Target),
            input_rmsnorm,
        );
        losses.push(loss.double_value(&[]));
        maybe_persist_checkpoint(
            runtime.checkpoint_every,
            runtime.checkpoint_dir,
            "tch",
            0,
            steps,
            losses[0],
            val_loss,
        )?;
    } else {
        for step in 0..steps {
            let pair_idx = rng.gen_range(0..train_pairs.len());
            let (context_id, target_id) = train_pairs[pair_idx];
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
            let lr = DEFAULT_LEARNING_RATE * lr_multiplier(step, steps);
            optimizer.set_lr(lr);
            let loss_value = loss.double_value(&[]);
            optimizer.backward_step(&loss);
            losses.push(loss_value);
            let step_idx = step + 1;
            if should_eval(step_idx, steps, EVAL_EVERY) {
                val_loss = Some(mean_nll_loss_tch(
                    &token_embedding,
                    lm_head.as_ref(),
                    eval_pairs_slice(&val_pairs),
                    input_rmsnorm,
                    device,
                ));
            }
            maybe_persist_checkpoint(
                runtime.checkpoint_every,
                runtime.checkpoint_dir,
                "tch",
                step_idx,
                steps,
                loss_value,
                val_loss,
            )?;
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
        val_loss,
        mean_loss_last_n,
        last_n,
        steps_per_sec,
        tokens_per_sec,
        vocab_size: tokenizer.vocab_size(),
        train_tokens: train_pairs.len(),
    })
}

#[cfg(feature = "tch-backend")]
#[derive(Clone)]
struct TensorMiniGptWindow {
    context: Vec<i64>,
    target: i64,
}

#[cfg(feature = "tch-backend")]
fn train_from_text_tch_minigpt(
    text: &str,
    steps: usize,
    seed: u64,
    optimizer: Optimizer,
    style: Style,
    tie_lm_head: bool,
    input_rmsnorm: bool,
    runtime: TensorTrainRuntime<'_>,
) -> Result<TrainMetrics, TensorError> {
    ensure_tensor_optimizer_support(optimizer)?;
    let corpus = styled_corpus(text, style);
    let tokenizer = data::Tokenizer::from_text(&corpus);
    let token_ids = tokenizer.encode_with_bos(&corpus)?;
    let (train_tokens, val_tokens) =
        training::split_train_val_tokens(&token_ids).map_err(|_| TensorError::EmptyDataset)?;
    let train_windows = build_minigpt_windows(&train_tokens, MINI_GPT_BLOCK_SIZE);
    let mut val_windows = build_minigpt_windows(&val_tokens, MINI_GPT_BLOCK_SIZE);
    if val_windows.is_empty() {
        val_windows = train_windows.clone();
    }
    if train_windows.is_empty() {
        return Err(TensorError::EmptyDataset);
    }

    let device = resolve_tch_device();
    let device_label = tch_device_label(device);
    let using_gpu = matches!(device, Device::Cuda(_));
    let vs = VarStore::new(device);
    let root = &vs.root();
    let vocab_size = i64::try_from(tokenizer.vocab_size()).expect("vocab size fits i64");
    let block_size = i64::try_from(MINI_GPT_BLOCK_SIZE).expect("block size fits i64");
    let token_embedding = root.var(
        "token_embedding",
        &[vocab_size, MINI_GPT_EMBED_DIM],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        },
    );
    let position_embedding = root.var(
        "position_embedding",
        &[block_size, MINI_GPT_EMBED_DIM],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        },
    );
    let lm_head = if tie_lm_head {
        None
    } else {
        Some(root.var(
            "lm_head",
            &[vocab_size, MINI_GPT_EMBED_DIM],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        ))
    };

    let parameter_count: usize = vs.trainable_variables().iter().map(Tensor::numel).sum();
    let base_learning_rate = match optimizer {
        Optimizer::AdamW => MINI_GPT_ADAMW_LEARNING_RATE,
        Optimizer::Sgd => MINI_GPT_LEARNING_RATE,
    };
    let mut optimizer = match optimizer {
        Optimizer::AdamW => nn::AdamW::default().build(&vs, MINI_GPT_ADAMW_LEARNING_RATE)?,
        Optimizer::Sgd => nn::Sgd::default().build(&vs, MINI_GPT_LEARNING_RATE)?,
    };
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA0E3_3D77_22A9_F14C);
    let mut losses = Vec::with_capacity(steps.max(1));
    let mut val_loss = Some(mean_nll_loss_tch_minigpt(
        &token_embedding,
        lm_head.as_ref(),
        &position_embedding,
        &val_windows,
        input_rmsnorm,
        device,
    ));
    let started = Instant::now();

    if steps == 0 {
        let loss = mean_nll_loss_tch_minigpt(
            &token_embedding,
            lm_head.as_ref(),
            &position_embedding,
            &train_windows,
            input_rmsnorm,
            device,
        );
        losses.push(loss);
        maybe_persist_checkpoint(
            runtime.checkpoint_every,
            runtime.checkpoint_dir,
            "tch-minigpt",
            0,
            steps,
            losses[0],
            val_loss,
        )?;
    } else {
        for step in 0..steps {
            let window_idx = rng.gen_range(0..train_windows.len());
            let window = &train_windows[window_idx];
            let context = Tensor::from_slice(&window.context).to(device);
            let target = Tensor::from_slice(&[window.target]).to(device);

            let loss = tch_minigpt_forward_loss_fn(
                &token_embedding,
                lm_head.as_ref(),
                &position_embedding,
                &context,
                &target,
                input_rmsnorm,
            );
            let lr = base_learning_rate * lr_multiplier(step, steps);
            optimizer.set_lr(lr);
            let loss_value = loss.double_value(&[]);
            optimizer.backward_step(&loss);
            losses.push(loss_value);
            let step_idx = step + 1;
            if should_eval(step_idx, steps, EVAL_EVERY) {
                val_loss = Some(mean_nll_loss_tch_minigpt(
                    &token_embedding,
                    lm_head.as_ref(),
                    &position_embedding,
                    &val_windows,
                    input_rmsnorm,
                    device,
                ));
            }
            maybe_persist_checkpoint(
                runtime.checkpoint_every,
                runtime.checkpoint_dir,
                "tch-minigpt",
                step_idx,
                steps,
                loss_value,
                val_loss,
            )?;
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
        style,
        tie_lm_head,
        input_rmsnorm,
        backend: "tch",
        device: device_label,
        using_gpu,
        parameter_count,
        steps,
        final_loss,
        val_loss,
        mean_loss_last_n,
        last_n,
        steps_per_sec,
        tokens_per_sec,
        vocab_size: tokenizer.vocab_size(),
        train_tokens: train_windows.len(),
    })
}

#[cfg(feature = "tch-backend")]
fn build_minigpt_windows(
    token_ids: &[usize],
    block_size: usize,
) -> Vec<TensorMiniGptWindow> {
    if token_ids.len() < 2 {
        return Vec::new();
    }
    let mut windows = Vec::with_capacity(token_ids.len().saturating_sub(1));
    for target_idx in 1..token_ids.len() {
        let start = target_idx.saturating_sub(block_size);
        let context = token_ids[start..target_idx]
            .iter()
            .map(|value| i64::try_from(*value).expect("token id fits i64"))
            .collect::<Vec<_>>();
        if context.is_empty() {
            continue;
        }
        windows.push(TensorMiniGptWindow {
            context,
            target: i64::try_from(token_ids[target_idx]).expect("token id fits i64"),
        });
    }
    windows
}

#[cfg(feature = "tch-backend")]
fn tch_minigpt_forward_loss(
    token_embedding: &Tensor,
    lm_head: Option<&Tensor>,
    position_embedding: &Tensor,
    context: &Tensor,
    input_rmsnorm: bool,
) -> Tensor {
    let window_len = context.size()[0];
    let position_ids = Tensor::arange_start(
        0,
        window_len,
        (Kind::Int64, context.device()),
    );
    let token_emb = token_embedding.index_select(0, context);
    let position_emb = position_embedding.index_select(0, &position_ids);
    let hidden = token_emb + position_emb;
    let hidden = hidden.mean_dim(&[0_i64][..], false, Kind::Float);
    let hidden = if input_rmsnorm {
        let mean_sq = hidden.pow_tensor_scalar(2.0).mean(Kind::Float);
        let inv_rms = (mean_sq + RMS_EPS).rsqrt();
        hidden * inv_rms
    } else {
        hidden
    };
    let projection = match lm_head {
        Some(matrix) => matrix,
        None => token_embedding,
    };
    let logits = hidden.unsqueeze(0).matmul(&projection.transpose(0, 1));
    logits
}

#[cfg(feature = "tch-backend")]
fn tch_minigpt_forward_loss_fn(
    token_embedding: &Tensor,
    lm_head: Option<&Tensor>,
    position_embedding: &Tensor,
    context: &Tensor,
    target: &Tensor,
    input_rmsnorm: bool,
) -> Tensor {
    tch_minigpt_forward_loss(token_embedding, lm_head, position_embedding, context, input_rmsnorm)
        .cross_entropy_for_logits(target)
}

#[cfg(feature = "tch-backend")]
fn mean_nll_loss_tch_minigpt(
    token_embedding: &Tensor,
    lm_head: Option<&Tensor>,
    position_embedding: &Tensor,
    windows: &[TensorMiniGptWindow],
    input_rmsnorm: bool,
    device: Device,
) -> f64 {
    if windows.is_empty() {
        return 0.0;
    }
    let mut total_loss = 0.0;
    tch::no_grad(|| {
        for window in windows {
            let context = Tensor::from_slice(&window.context).to(device);
            let target = Tensor::from_slice(&[window.target]).to(device);
            let loss = tch_minigpt_forward_loss_fn(
                token_embedding,
                lm_head,
                position_embedding,
                &context,
                &target,
                input_rmsnorm,
            );
            total_loss += loss.double_value(&[]);
        }
    });
    total_loss / (windows.len() as f64)
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

#[cfg(feature = "tch-backend")]
#[derive(Clone, Copy)]
enum PairField {
    Context,
    Target,
}

#[cfg(feature = "tch-backend")]
fn pairs_to_tensor(pairs: &[(usize, usize)], device: Device, field: PairField) -> Tensor {
    let values: Vec<i64> = pairs
        .iter()
        .map(|(context_id, target_id)| {
            let value = match field {
                PairField::Context => *context_id,
                PairField::Target => *target_id,
            };
            i64::try_from(value).expect("token id fits i64")
        })
        .collect();
    Tensor::from_slice(&values).to(device)
}

#[cfg(feature = "tch-backend")]
fn mean_nll_loss_tch(
    token_embedding: &Tensor,
    lm_head: Option<&Tensor>,
    pairs: &[(usize, usize)],
    input_rmsnorm: bool,
    device: Device,
) -> f64 {
    let context = pairs_to_tensor(pairs, device, PairField::Context);
    let target = pairs_to_tensor(pairs, device, PairField::Target);
    tch::no_grad(|| {
        let loss = tch_forward_loss(token_embedding, lm_head, &context, &target, input_rmsnorm);
        loss.double_value(&[])
    })
}

#[cfg(feature = "tch-backend")]
fn sample_from_text_tch_minigpt(
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
    let prompt_ids = tokenizer.encode(prompt)?;
    crate::eval::validate_prompt_length(prompt_ids.len(), MINI_GPT_BLOCK_SIZE.saturating_sub(1))
        .map_err(TensorError::from)?;

    if token_ids.len() < 2 {
        return Err(TensorError::EmptyDataset);
    }

    let device = resolve_tch_device();
    let _ = tch::manual_seed(seed as i64);
    let vocab_size = i64::try_from(tokenizer.vocab_size()).expect("vocab size fits i64");
    let block_size = i64::try_from(MINI_GPT_BLOCK_SIZE).expect("block size fits i64");
    let vs = VarStore::new(device);
    let root = &vs.root();
    let token_embedding = root.var(
        "token_embedding",
        &[vocab_size, MINI_GPT_EMBED_DIM],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        },
    );
    let position_embedding = root.var(
        "position_embedding",
        &[block_size, MINI_GPT_EMBED_DIM],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        },
    );

    let mut generated = prompt.to_string();
    let mut context = vec![i64::try_from(tokenizer.bos_id()).expect("bos id fits i64")];
    context.extend(prompt_ids.iter().map(|value| i64::try_from(*value).expect("token id fits i64")));

    for _ in 0..max_new_tokens {
        let start = context.len().saturating_sub(MINI_GPT_BLOCK_SIZE);
        let context_window = &context[start..];
        let context_tensor = Tensor::from_slice(context_window).to(device);
        let logits = tch_minigpt_forward_loss(
            &token_embedding,
            None,
            &position_embedding,
            &context_tensor,
            false,
        );
        let scaled_logits = logits / temperature;
        let sampled_id = scaled_logits
            .softmax(-1, Kind::Float)
            .squeeze()
            .multinomial(1, false)
            .int64_value(&[0]);
        if sampled_id == i64::try_from(tokenizer.bos_id()).expect("bos id fits i64") {
            continue;
        }
        let next_id = usize::try_from(sampled_id).expect("sampled id fits usize");
        let next_char = tokenizer
            .char_for_id(next_id)
            .ok_or(TokenizerError::UnknownId(next_id))?;
        generated.push(next_char);
        context.push(sampled_id);
        if context.len() > MINI_GPT_BLOCK_SIZE {
            let _ = context.remove(0);
        }
    }

    Ok(generated)
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
    let mut probs = Vec::with_capacity(logits.len());
    let mut exp_sum = 0.0;
    for logit in logits {
        let exp_value = (logit - max_logit).exp();
        probs.push(exp_value);
        exp_sum += exp_value;
    }
    let inv_sum = 1.0 / exp_sum;
    for value in &mut probs {
        *value *= inv_sum;
    }
    let target_prob = probs[target_id];
    let loss = -(target_prob + LOG_EPS).ln();
    (probs, loss)
}

fn apply_projection_gradient(
    projection_rows: &mut [Vec<f64>],
    logits_grad: &[f64],
    hidden_norm: &[f64],
    learning_rate: f64,
) {
    for (row, grad) in projection_rows.iter_mut().zip(logits_grad.iter().copied()) {
        if grad == 0.0 {
            continue;
        }
        let scaled_lr = learning_rate * grad;
        for (weight, hidden_value) in row.iter_mut().zip(hidden_norm.iter().copied()) {
            *weight -= scaled_lr * hidden_value;
        }
    }
}

fn build_transition_counts(vocab_size: usize, pairs: &[(usize, usize)]) -> Vec<Vec<u64>> {
    training::build_transition_counts(vocab_size, pairs)
}

fn sample_index(
    transition_counts: &[Vec<u64>],
    context_id: usize,
    bos_id: usize,
    temperature: f64,
    rng: &mut StdRng,
) -> usize {
    training::sample_index(transition_counts, context_id, bos_id, temperature, rng)
}

fn styled_corpus(base: &str, style: Style) -> String {
    training::styled_corpus(base, style)
}

fn apply_futuristic_boost(
    transition_counts: &mut [Vec<u64>],
    tokenizer: &data::Tokenizer,
    bos_id: usize,
) {
    training::apply_futuristic_boost(transition_counts, tokenizer, bos_id)
}

fn eval_pairs_slice(pairs: &[(usize, usize)]) -> &[(usize, usize)] {
    training::eval_pairs_slice(pairs)
}

fn mean_nll_loss_cpu(model: &TensorBigram, pairs: &[(usize, usize)]) -> f64 {
    let total = pairs
        .iter()
        .map(|(context_id, target_id)| model.nll_loss(*context_id, *target_id))
        .sum::<f64>();
    total / (pairs.len() as f64)
}

fn split_train_val_pairs(token_ids: &[usize]) -> Result<PairSplits, TensorError> {
    training::split_train_val_pairs(token_ids).map_err(|_| TensorError::EmptyDataset)
}

fn ensure_tensor_support(model_kind: ModelKind) -> Result<(), TensorError> {
    match model_kind {
        ModelKind::Bigram => Ok(()),
        #[cfg(feature = "tch-backend")]
        ModelKind::MiniGpt => Ok(()),
        #[cfg(not(feature = "tch-backend"))]
        ModelKind::MiniGpt => Err(TensorError::UnsupportedModelKind(model_kind)),
    }
}

fn ensure_tensor_optimizer_support(optimizer: Optimizer) -> Result<(), TensorError> {
    if cfg!(not(feature = "tch-backend")) && matches!(optimizer, Optimizer::AdamW) {
        return Err(TensorError::UnsupportedOptimizer(optimizer));
    }
    Ok(())
}

fn should_eval(step: usize, total_steps: usize, eval_every: usize) -> bool {
    training::should_eval(step, total_steps, eval_every)
}

fn lr_multiplier(step: usize, total_steps: usize) -> f64 {
    training::lr_multiplier(step, total_steps)
}

fn checkpoint_path(checkpoint_dir: &Path, backend: &str, step: usize) -> PathBuf {
    checkpoint_dir.join(format!("tensor_{backend}_step_{step:06}.ckpt"))
}

fn checkpoint_payload(step: usize, loss: f64, val_loss: Option<f64>) -> Vec<u8> {
    format!(
        "step={step}\nloss={loss:.12}\nval_loss={}\n",
        val_loss
            .map(|value| format!("{value:.12}"))
            .unwrap_or_else(|| "none".to_string())
    )
    .into_bytes()
}

fn should_checkpoint(step: usize, total_steps: usize, checkpoint_every: usize) -> bool {
    if step == total_steps {
        return true;
    }
    checkpoint_every > 0 && step.is_multiple_of(checkpoint_every)
}

fn maybe_persist_checkpoint(
    checkpoint_every: usize,
    checkpoint_dir: &Path,
    backend: &str,
    step: usize,
    total_steps: usize,
    loss: f64,
    val_loss: Option<f64>,
) -> Result<(), TensorError> {
    if checkpoint_every == 0 || !should_checkpoint(step, total_steps, checkpoint_every) {
        return Ok(());
    }
    let payload = checkpoint_payload(step, loss, val_loss);
    let path = checkpoint_path(checkpoint_dir, backend, step);
    persist_then_eval(
        &path,
        &payload,
        |checkpoint_path| -> Result<(), Infallible> {
            debug_assert!(checkpoint_path.exists());
            Ok(())
        },
    )
    .map_err(|err| match err {
        CheckpointEvalError::Io(io_err) => TensorError::Io(io_err),
        CheckpointEvalError::Eval(infallible) => match infallible {},
    })?;
    Ok(())
}

fn build_pairs(token_ids: &[usize]) -> Result<Vec<TokenPair>, TensorError> {
    training::build_pairs(token_ids).map_err(|_| TensorError::EmptyDataset)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::{cli, training};
    use crate::config::{AppCommand, Mode, ModelKind, Optimizer, RuntimeConfig, SampleConfig, Style, TrainConfig};

    use super::{
        TensorBigram, TensorError, eval_pairs_slice, lr_multiplier, mean_nll_loss_cpu,
        sample_from_text, should_eval, split_train_val_pairs, train_from_text,
    };

    struct TempTextFile {
        path: PathBuf,
    }

    impl TempTextFile {
        fn as_path(&self) -> &Path {
            &self.path
        }

        fn as_path_string(&self) -> String {
            self.as_path().to_string_lossy().to_string()
        }
    }

    impl Drop for TempTextFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn temp_text_file() -> TempTextFile {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = env::temp_dir().join(format!("nanochat_tensor_cli_{unique}.txt"));
        fs::write(&path, "abababababab").expect("temp text fixture");
        TempTextFile { path }
    }

    fn train_config_from_cli(mut args: Vec<String>) -> TrainConfig {
        let mut iter = vec!["nanochat-rs-next".to_string(), "train".to_string()];
        iter.append(&mut args);
        let command = cli::try_command_from_iter(iter).expect("valid train command");
        let AppCommand::Train(config) = command else {
            panic!("expected train command");
        };
        config
    }

    fn sample_config_from_cli(mut args: Vec<String>) -> SampleConfig {
        let mut iter = vec!["nanochat-rs-next".to_string(), "sample".to_string()];
        iter.append(&mut args);
        let command = cli::try_command_from_iter(iter).expect("valid sample command");
        let AppCommand::Sample(config) = command else {
            panic!("expected sample command");
        };
        config
    }

    fn train_config(
        model_kind: ModelKind,
        optimizer: Optimizer,
    ) -> TrainConfig {
        TrainConfig {
            runtime: RuntimeConfig {
                mode: Mode::Tensor,
                model_kind,
                style: Style::Classic,
                data_path: PathBuf::from("input.txt"),
                seed: 1,
            },
            optimizer,
            tie_lm_head: true,
            input_rmsnorm: false,
            steps: 1,
            checkpoint_every: 0,
            checkpoint_dir: PathBuf::from(training::DEFAULT_CHECKPOINT_DIR),
        }
    }

    fn sample_config(model_kind: ModelKind) -> SampleConfig {
        SampleConfig {
            runtime: RuntimeConfig {
                mode: Mode::Tensor,
                model_kind,
                style: Style::Classic,
                data_path: PathBuf::from("input.txt"),
                seed: 7,
            },
            temperature: 1.0,
            max_new_tokens: 8,
            prompt: String::from("a"),
        }
    }

    #[test]
    fn train_loss_drops_on_repetitive_text() {
        let baseline = train_from_text(
            "abababababababab",
            0,
            13,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            true,
            false,
        )
            .expect("baseline metrics");
        let trained = train_from_text(
            "abababababababab",
            400,
            13,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            true,
            false,
        )
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

    #[derive(Clone, Copy)]
    enum RejectionExpectation {
        ModelKind(ModelKind),
        Optimizer(Optimizer),
    }

    impl RejectionExpectation {
        fn assert_error(self, err: TensorError) {
            match (self, err) {
                (Self::ModelKind(expected), TensorError::UnsupportedModelKind(actual)) => {
                    assert_eq!(actual, expected);
                }
                (Self::Optimizer(expected), TensorError::UnsupportedOptimizer(actual)) => {
                    assert_eq!(actual, expected);
                }
                (_, err) => panic!("unexpected error type: {err:?}"),
            }
        }
    }

    fn tensor_cli_base_args(path: &str) -> Vec<String> {
        vec![
            "--mode".to_string(),
            "tensor".to_string(),
            "--data".to_string(),
            path.to_string(),
        ]
    }

    #[derive(Clone, Copy)]
    enum PublicRejectionCase {
        Train {
            label: &'static str,
            model_kind: ModelKind,
            optimizer: Optimizer,
            expected: RejectionExpectation,
        },
        Sample {
            label: &'static str,
            model_kind: ModelKind,
            expected: RejectionExpectation,
        },
    }

    #[derive(Clone, Copy)]
    enum CliRejectionCase {
        Train {
            label: &'static str,
            model_kind: ModelKind,
            optimizer: Optimizer,
            expected: RejectionExpectation,
        },
        Sample {
            label: &'static str,
            model_kind: ModelKind,
            expected: RejectionExpectation,
        },
    }

    #[test]
    fn tensor_public_rejections_are_reported_consistently() {
        let cases = [
            #[cfg(not(feature = "tch-backend"))]
            PublicRejectionCase::Train {
                label: "train should reject mini-gpt in tensor mode",
                model_kind: ModelKind::MiniGpt,
                optimizer: Optimizer::Sgd,
                expected: RejectionExpectation::ModelKind(ModelKind::MiniGpt),
            },
            #[cfg(not(feature = "tch-backend"))]
            PublicRejectionCase::Sample {
                label: "sample should reject mini-gpt in tensor mode",
                model_kind: ModelKind::MiniGpt,
                expected: RejectionExpectation::ModelKind(ModelKind::MiniGpt),
            },
            #[cfg(not(feature = "tch-backend"))]
            PublicRejectionCase::Train {
                label: "train should reject adamw without tch backend",
                model_kind: ModelKind::Bigram,
                optimizer: Optimizer::AdamW,
                expected: RejectionExpectation::Optimizer(Optimizer::AdamW),
            },
        ];

        for case in cases {
            match case {
                PublicRejectionCase::Train {
                    label,
                    model_kind,
                    optimizer,
                    expected,
                } => expected.assert_error(
                    super::train(&train_config(model_kind, optimizer)).expect_err(label),
                ),
                PublicRejectionCase::Sample {
                    label,
                    model_kind,
                    expected,
                } => expected.assert_error(
                    super::sample(&sample_config(model_kind)).expect_err(label),
                ),
            }
        }
    }

    #[test]
    fn tensor_cli_rejections_are_reported_consistently() {
        let path = temp_text_file();
        let path_string = path.as_path_string();

        let cases = [
            #[cfg(not(feature = "tch-backend"))]
            CliRejectionCase::Train {
                label: "train cli should reject mini-gpt in tensor mode",
                model_kind: ModelKind::MiniGpt,
                optimizer: Optimizer::Sgd,
                expected: RejectionExpectation::ModelKind(ModelKind::MiniGpt),
            },
            #[cfg(not(feature = "tch-backend"))]
            CliRejectionCase::Sample {
                label: "sample cli should reject mini-gpt in tensor mode",
                model_kind: ModelKind::MiniGpt,
                expected: RejectionExpectation::ModelKind(ModelKind::MiniGpt),
            },
            #[cfg(not(feature = "tch-backend"))]
            CliRejectionCase::Train {
                label: "train cli should reject adamw without tch backend",
                model_kind: ModelKind::Bigram,
                optimizer: Optimizer::AdamW,
                expected: RejectionExpectation::Optimizer(Optimizer::AdamW),
            },
        ];

        for case in cases {
            match case {
                CliRejectionCase::Train {
                    label,
                    model_kind,
                    optimizer,
                    expected,
                } => {
                    let mut args = tensor_cli_base_args(&path_string);
                    args.extend([
                        "--model-kind".to_string(),
                        model_kind.to_string(),
                        "--optimizer".to_string(),
                        optimizer.to_string(),
                    ]);
                    expected.assert_error(
                        super::train(&train_config_from_cli(args)).expect_err(label),
                    );
                }
                CliRejectionCase::Sample {
                    label,
                    model_kind,
                    expected,
                } => {
                    let mut args = tensor_cli_base_args(&path_string);
                    args.extend(["--model-kind".to_string(), model_kind.to_string()]);
                    expected.assert_error(
                        super::sample(&sample_config_from_cli(args)).expect_err(label),
                    );
                }
            }
        }
    }

    #[cfg(feature = "tch-backend")]
    #[test]
    fn tensor_public_mini_gpt_train_runs_in_tch() {
        let metrics = super::train(&train_config(ModelKind::MiniGpt, Optimizer::Sgd))
            .expect("mini-gpt tensor training should run with tch backend");
        assert_eq!(metrics.backend, "tch");
        assert_eq!(metrics.steps, 1);
        assert!(metrics.final_loss.is_finite());
        assert!(metrics.train_tokens > 0);
        assert!(metrics.val_loss.is_some());
    }

    #[cfg(feature = "tch-backend")]
    #[test]
    fn tensor_public_mini_gpt_zero_step_train_runs_in_tch() {
        let mut config = train_config(ModelKind::MiniGpt, Optimizer::Sgd);
        config.steps = 0;
        let metrics = super::train(&config).expect("mini-gpt tensor zero-step training should run");
        assert_eq!(metrics.backend, "tch");
        assert_eq!(metrics.steps, 0);
        assert!(metrics.final_loss.is_finite());
        assert!(metrics.train_tokens > 0);
    }

    #[cfg(feature = "tch-backend")]
    #[test]
    fn tensor_public_mini_gpt_sample_runs_in_tch() {
        let output = super::sample(&sample_config(ModelKind::MiniGpt))
            .expect("mini-gpt tensor sampling should run with tch backend");
        assert!(!output.is_empty());
        assert!(output.starts_with("a"));
    }

    #[cfg(feature = "tch-backend")]
    #[test]
    fn tensor_public_mini_gpt_rejects_overlong_prompt() {
        let mut config = sample_config(ModelKind::MiniGpt);
        config.prompt = "aaaaaaaa".to_string();
        let err = super::sample(&config).expect_err("overlong prompt should fail");
        match err {
            TensorError::EvalGuard(_) => {}
            _ => panic!("unexpected error type: {err:?}"),
        }
    }

    #[cfg(feature = "tch-backend")]
    #[test]
    fn tensor_public_mini_gpt_rejects_invalid_temperature() {
        let mut config = sample_config(ModelKind::MiniGpt);
        config.temperature = 0.0;
        let err = super::sample(&config).expect_err("invalid temperature should fail");
        match err {
            TensorError::InvalidTemperature(value) => assert_eq!(value, 0.0),
            _ => panic!("unexpected error type: {err:?}"),
        }
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
    fn sample_rejects_non_finite_temperature() {
        let invalid_temperatures = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for temperature in invalid_temperatures {
            let err = sample_from_text("abab", "", 5, temperature, 1, Style::Classic)
                .expect_err("invalid temperature should fail");
            match err {
                TensorError::InvalidTemperature(value) => {
                    assert!(value.is_infinite() || value.is_nan())
                }
                _ => panic!("unexpected error type"),
            }
        }
    }

    #[test]
    fn empty_input_errors() {
        let err = train_from_text(
            "",
            1,
            0,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            true,
            false,
        )
        .expect_err("empty dataset");
        match err {
            TensorError::EmptyDataset => {}
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn train_handles_minimal_dataset_split_fallback() {
        let metrics = train_from_text(
            "a",
            1,
            1,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            true,
            false,
        )
        .expect("metrics should be produced for minimal text");
        assert_eq!(metrics.train_tokens, 1);
        let val_loss = metrics.val_loss.expect("validation loss should be computed");
        assert!(val_loss.is_finite());
        assert!(metrics.final_loss.is_finite());
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
        let tied = train_from_text(
            "abababababababab",
            0,
            5,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            true,
            false,
        )
            .expect("tied metrics");
        let untied = train_from_text(
            "abababababababab",
            0,
            5,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            false,
            false,
        )
            .expect("untied metrics");
        assert!(untied.parameter_count > tied.parameter_count);
    }

    #[test]
    fn train_reports_backend_and_device() {
        let metrics = train_from_text(
            "abababababababab",
            1,
            5,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            true,
            false,
        )
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

    #[test]
    fn train_reports_finite_throughput_metrics() {
        let zero_step = train_from_text(
            "abababababababab",
            0,
            11,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            true,
            false,
        )
            .expect("zero-step metrics");
        assert_eq!(zero_step.steps_per_sec, 0.0);
        assert_eq!(zero_step.tokens_per_sec, 0.0);

        let trained = train_from_text(
            "abababababababab",
            8,
            11,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            true,
            false,
        )
            .expect("trained metrics");
        assert!(trained.steps_per_sec.is_finite());
        assert!(trained.tokens_per_sec.is_finite());
        assert!(trained.steps_per_sec > 0.0);
        assert!(trained.tokens_per_sec > 0.0);
    }

    #[test]
    fn train_uses_split_and_reports_validation_loss() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let metrics = train_from_text(
            text,
            5,
            29,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            true,
            false,
        )
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

    #[test]
    fn lr_schedule_matches_upstream_linear_warmdown_shape() {
        let total_steps = 10;
        assert!((lr_multiplier(0, total_steps) - 1.0).abs() < 1e-12);
        assert!((lr_multiplier(5, total_steps) - 1.0).abs() < 1e-12);
        assert!((lr_multiplier(8, total_steps) - 0.4).abs() < 1e-12);
        assert!((lr_multiplier(9, total_steps) - 0.2).abs() < 1e-12);
        assert!((lr_multiplier(10, total_steps) - 0.0).abs() < 1e-12);

        assert!(should_eval(total_steps, total_steps, 20));
        assert!(!should_eval(7, total_steps, 20));
    }

    #[test]
    fn train_creates_checkpoints_at_interval_and_final_step() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let checkpoint_dir = PathBuf::from(format!("results/test_artifacts/tensor_ckpt_{unique}"));

        let _ = super::train_from_text_with_checkpoints(
            "abababababababab",
            5,
            41,
            ModelKind::Bigram,
            Optimizer::Sgd,
            Style::Classic,
            true,
            false,
            super::TensorTrainRuntime {
                checkpoint_every: 2,
                checkpoint_dir: &checkpoint_dir,
            },
        )
        .expect("training should succeed with checkpoints");

        let backend = if cfg!(feature = "tch-backend") {
            "tch"
        } else {
            "cpu-native"
        };

        let step_000002 = format!("tensor_{backend}_step_000002.ckpt");
        let step_000004 = format!("tensor_{backend}_step_000004.ckpt");
        let step_000005 = format!("tensor_{backend}_step_000005.ckpt");

        assert!(
            checkpoint_dir.join(step_000002).exists()
        );
        assert!(
            checkpoint_dir.join(step_000004).exists()
        );
        assert!(
            checkpoint_dir.join(step_000005).exists()
        );

        let _ = fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    fn python_parity_matches_rust_tensor_loss_trace() {
        let has_python = Command::new("python3").arg("--version").output().is_ok();
        if !has_python {
            eprintln!("skipping python parity test: python3 not available");
            return;
        }

        let text = "abracadabra abracadabra";
        let tokenizer = crate::data::Tokenizer::from_text(text);
        let token_ids = tokenizer
            .encode_with_bos(text)
            .expect("token ids should encode");
        let (train_pairs, val_pairs) = split_train_val_pairs(&token_ids).expect("pair splits");
        let eval_val_pairs = eval_pairs_slice(&val_pairs).to_vec();

        let vocab_size = tokenizer.vocab_size();
        let initial_embedding: Vec<Vec<f64>> = (0..vocab_size)
            .map(|i| {
                (0..vocab_size)
                    .map(|j| ((i * 31 + j * 17) as f64) / 1000.0 - 0.1)
                    .collect()
            })
            .collect();

        let steps = 24usize;
        let mut rng = StdRng::seed_from_u64(17 ^ 0xD1B5_4A32_44C1_AA77);
        let pair_indices: Vec<usize> = (0..steps)
            .map(|_| rng.gen_range(0..train_pairs.len()))
            .collect();

        let mut rust_model = TensorBigram {
            tie_lm_head: true,
            input_rmsnorm: false,
            token_embedding: initial_embedding.clone(),
            lm_head: None,
        };
        let mut rust_losses = Vec::with_capacity(steps);
        for (step, pair_idx) in pair_indices.iter().copied().enumerate() {
            let (context_id, target_id) = train_pairs[pair_idx];
            let lr = super::DEFAULT_LEARNING_RATE * lr_multiplier(step, steps);
            let loss = rust_model.train_step(context_id, target_id, lr);
            rust_losses.push(loss);
        }
        let rust_val_loss = mean_nll_loss_cpu(&rust_model, &eval_val_pairs);

        let script = format!(
            r#"
import math
train_pairs = {train_pairs:?}
val_pairs = {eval_val_pairs:?}
pair_indices = {pair_indices:?}
token_embedding = {initial_embedding:?}
steps = {steps}
default_lr = {default_lr}
log_eps = {log_eps}

def dot(lhs, rhs):
    return sum(a * b for a, b in zip(lhs, rhs))

def softmax_probs_and_loss(logits, target_id):
    max_logit = max(logits)
    shifted = [logit - max_logit for logit in logits]
    exp_logits = [math.exp(logit) for logit in shifted]
    exp_sum = sum(exp_logits)
    probs = [value / exp_sum for value in exp_logits]
    loss = -math.log(probs[target_id] + log_eps)
    return probs, loss

def nll_loss(context_id, target_id):
    hidden = token_embedding[context_id][:]
    logits = [dot(hidden, row) for row in token_embedding]
    _, loss = softmax_probs_and_loss(logits, target_id)
    return loss

losses = []
for step, pair_idx in enumerate(pair_indices):
    context_id, target_id = train_pairs[pair_idx]
    hidden = token_embedding[context_id][:]
    logits = [dot(hidden, row) for row in token_embedding]
    probs, loss = softmax_probs_and_loss(logits, target_id)

    logits_grad = probs[:]
    logits_grad[target_id] -= 1.0

    vocab_size = len(token_embedding)
    hidden_grad = [0.0] * vocab_size
    for row_idx, row in enumerate(token_embedding):
        grad = logits_grad[row_idx]
        if grad == 0.0:
            continue
        for col_idx, weight in enumerate(row):
            hidden_grad[col_idx] += grad * weight

    warmdown_iters = round(0.5 * steps)
    if warmdown_iters == 0 or step <= steps - warmdown_iters:
        lr_mult = 1.0
    else:
        progress = (steps - step) / warmdown_iters
        lr_mult = progress
    lr = default_lr * lr_mult

    for row_idx, grad in enumerate(logits_grad):
        if grad == 0.0:
            continue
        for col_idx, hidden_value in enumerate(hidden):
            token_embedding[row_idx][col_idx] -= lr * grad * hidden_value

    for col_idx, grad in enumerate(hidden_grad):
        token_embedding[context_id][col_idx] -= lr * grad

    losses.append(loss)

val_loss = sum(nll_loss(context_id, target_id) for context_id, target_id in val_pairs) / len(val_pairs)
print('losses=' + ','.join(f'{{loss:.17g}}' for loss in losses))
print(f'val={{val_loss:.17g}}')
"#,
            train_pairs = train_pairs,
            eval_val_pairs = eval_val_pairs,
            pair_indices = pair_indices,
            initial_embedding = initial_embedding,
            steps = steps,
            default_lr = super::DEFAULT_LEARNING_RATE,
            log_eps = super::LOG_EPS,
        );

        let output = Command::new("python3")
            .arg("-c")
            .arg(script)
            .output()
            .expect("python3 process should run");
        assert!(
            output.status.success(),
            "python script failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8(output.stdout).expect("python output should be utf-8");
        let mut python_losses = None::<Vec<f64>>;
        let mut python_val = None::<f64>;
        for line in stdout.lines() {
            if let Some(raw_losses) = line.strip_prefix("losses=") {
                let parsed = raw_losses
                    .split(',')
                    .map(|token| token.parse::<f64>().expect("loss token should parse"))
                    .collect::<Vec<_>>();
                python_losses = Some(parsed);
            }
            if let Some(raw_val) = line.strip_prefix("val=") {
                python_val = Some(raw_val.parse::<f64>().expect("val token should parse"));
            }
        }
        let python_losses = python_losses.expect("python losses should be reported");
        let python_val = python_val.expect("python val should be reported");

        assert_eq!(rust_losses.len(), python_losses.len());
        for (rust_loss, python_loss) in rust_losses.iter().zip(python_losses.iter()) {
            let abs_diff = (rust_loss - python_loss).abs();
            assert!(
                abs_diff < 1e-10,
                "loss mismatch: rust={rust_loss} python={python_loss} abs_diff={abs_diff}"
            );
        }

        let val_abs_diff = (rust_val_loss - python_val).abs();
        assert!(
            val_abs_diff < 1e-10,
            "val loss mismatch: rust={rust_val_loss} python={python_val} abs_diff={val_abs_diff}"
        );
    }
}
