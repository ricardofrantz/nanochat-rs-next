mod bigram;
mod minigpt;
pub mod value;

use std::fmt;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::config::{ModelKind, SampleConfig, Style, TrainConfig};
use crate::data::{self, TokenizerError};
use crate::eval::EvalGuardError;

use self::bigram::{ScalarBigram, build_transition_counts, sample_index};

const DEFAULT_LEARNING_RATE: f64 = 0.1;
const LOSS_WINDOW: usize = 50;
pub(super) const VAL_FRACTION: f32 = 0.1;
const EVAL_EVERY: usize = 20;
const EVAL_PAIRS: usize = 1024;
const FUTURISTIC_CORPUS: &str = "\
neon skylines hum over orbital transit lanes.\n\
quantum couriers sync with neural uplinks at dawn.\n\
autonomous swarms calibrate reactor lattices in silence.\n\
synthetic pilots chart wormhole routes beyond saturn.\n";

#[derive(Debug, Clone)]
pub struct TrainMetrics {
    pub model_kind: ModelKind,
    pub style: Style,
    pub tie_lm_head: bool,
    pub input_rmsnorm: bool,
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
            "mode=scalar model_kind={} style={} tie_lm_head={} input_rmsnorm={} params={} steps={} final_loss={:.6} val_loss={} mean_loss_last_{}={:.6} steps_per_sec={:.2} tokens_per_sec={:.2} vocab_size={} train_tokens={}",
            self.model_kind,
            self.style,
            self.tie_lm_head,
            self.input_rmsnorm,
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
pub enum ScalarError {
    Io(std::io::Error),
    Tokenizer(TokenizerError),
    EvalGuard(EvalGuardError),
    EmptyDataset,
    InvalidTemperature(f64),
}

impl fmt::Display for ScalarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Tokenizer(err) => write!(f, "{err}"),
            Self::EvalGuard(err) => write!(f, "{err}"),
            Self::EmptyDataset => write!(f, "dataset must contain at least one token pair"),
            Self::InvalidTemperature(value) => {
                write!(f, "temperature must be finite and > 0, got {value}")
            }
        }
    }
}

impl std::error::Error for ScalarError {}

impl From<std::io::Error> for ScalarError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<TokenizerError> for ScalarError {
    fn from(err: TokenizerError) -> Self {
        Self::Tokenizer(err)
    }
}

impl From<EvalGuardError> for ScalarError {
    fn from(err: EvalGuardError) -> Self {
        Self::EvalGuard(err)
    }
}

pub fn train(config: &TrainConfig) -> Result<TrainMetrics, ScalarError> {
    let base_text = data::load_text(&config.data_path)?;
    match config.model_kind {
        ModelKind::Bigram => train_from_text(
            &base_text,
            config.steps,
            config.seed,
            config.style,
            config.tie_lm_head,
            config.input_rmsnorm,
        ),
        ModelKind::MiniGpt => minigpt::train_from_text(
            &base_text,
            config.steps,
            config.seed,
            config.style,
            config.tie_lm_head,
            config.input_rmsnorm,
        ),
    }
}

pub fn sample(config: &SampleConfig) -> Result<String, ScalarError> {
    let base_text = data::load_text(&config.data_path)?;
    match config.model_kind {
        ModelKind::Bigram => sample_from_text(
            &base_text,
            &config.prompt,
            config.max_new_tokens,
            config.temperature,
            config.seed,
            config.style,
        ),
        ModelKind::MiniGpt => minigpt::sample_from_text(
            &base_text,
            &config.prompt,
            config.max_new_tokens,
            config.temperature,
            config.seed,
            config.style,
        ),
    }
}

pub(super) fn should_eval(step: usize, total_steps: usize, eval_every: usize) -> bool {
    if step == total_steps {
        return true;
    }
    eval_every > 0 && step % eval_every == 0
}

fn train_from_text(
    text: &str,
    steps: usize,
    seed: u64,
    style: Style,
    tie_lm_head: bool,
    input_rmsnorm: bool,
) -> Result<TrainMetrics, ScalarError> {
    let corpus = styled_corpus(text, style);
    let tokenizer = data::Tokenizer::from_text(&corpus);
    let token_ids = tokenizer.encode_with_bos(&corpus)?;
    let (train_pairs, val_pairs) = split_train_val_pairs(&token_ids)?;

    let mut model = ScalarBigram::new(tokenizer.vocab_size(), seed, tie_lm_head, input_rmsnorm);
    let parameters = model.parameters();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);

    let mut losses = Vec::with_capacity(steps.max(1));
    let mut val_loss = Some(mean_nll_loss(&model, eval_pairs_slice(&val_pairs)));
    let started = Instant::now();

    if steps == 0 {
        losses.push(mean_nll_loss(&model, eval_pairs_slice(&train_pairs)));
    } else {
        for step in 0..steps {
            let pair_idx = rng.gen_range(0..train_pairs.len());
            let (context_id, target_id) = train_pairs[pair_idx];
            let loss = model.train_step(context_id, target_id, DEFAULT_LEARNING_RATE, &parameters);
            losses.push(loss);
            let step_idx = step + 1;
            if should_eval(step_idx, steps, EVAL_EVERY) {
                val_loss = Some(mean_nll_loss(&model, eval_pairs_slice(&val_pairs)));
            }
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
        model_kind: ModelKind::Bigram,
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

fn sample_from_text(
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
    let pairs = build_pairs(&token_ids)?;
    let mut transition_counts = build_transition_counts(tokenizer.vocab_size(), &pairs);
    if style == Style::Futuristic {
        apply_futuristic_boost(&mut transition_counts, &tokenizer, tokenizer.bos_id());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut generated = prompt.to_string();
    let bos_id = tokenizer.bos_id();

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

fn eval_pairs_slice(pairs: &[(usize, usize)]) -> &[(usize, usize)] {
    let eval_len = pairs.len().min(EVAL_PAIRS);
    &pairs[..eval_len]
}

fn mean_nll_loss(model: &ScalarBigram, pairs: &[(usize, usize)]) -> f64 {
    let total = pairs
        .iter()
        .map(|(context_id, target_id)| model.nll_loss(*context_id, *target_id).data())
        .sum::<f64>();
    total / (pairs.len() as f64)
}

fn split_train_val_pairs(
    token_ids: &[usize],
) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>), ScalarError> {
    let (train_tokens, val_tokens) = data::split_train_val(token_ids, VAL_FRACTION);
    let train_pairs = match build_pairs(&train_tokens) {
        Ok(pairs) => pairs,
        Err(_) => build_pairs(token_ids)?,
    };
    let val_pairs = match build_pairs(&val_tokens) {
        Ok(pairs) => pairs,
        Err(_) => train_pairs.clone(),
    };
    Ok((train_pairs, val_pairs))
}

fn build_pairs(token_ids: &[usize]) -> Result<Vec<(usize, usize)>, ScalarError> {
    if token_ids.len() < 2 {
        return Err(ScalarError::EmptyDataset);
    }
    Ok(token_ids
        .windows(2)
        .map(|window| (window[0], window[1]))
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::config::Style;

    use super::{ScalarError, sample_from_text, train_from_text};

    #[test]
    fn train_loss_drops_on_repetitive_text() {
        let baseline = train_from_text("abababababababab", 0, 42, Style::Classic, true, false)
            .expect("baseline metrics");
        let trained = train_from_text("abababababababab", 400, 42, Style::Classic, true, false)
            .expect("trained metrics");

        assert!(trained.mean_loss_last_n < baseline.final_loss);
        assert!(trained.final_loss.is_finite());
    }

    #[test]
    fn sample_respects_prompt_and_length() {
        let output = sample_from_text("abababababab", "ab", 12, 1.0, 7, Style::Classic)
            .expect("sample text");
        assert!(output.starts_with("ab"));
        assert_eq!(output.len(), 14);
    }

    #[test]
    fn sample_rejects_invalid_temperature() {
        let err = sample_from_text("abab", "", 5, 0.0, 1, Style::Classic)
            .expect_err("invalid temperature");
        match err {
            ScalarError::InvalidTemperature(value) => assert_eq!(value, 0.0),
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn empty_input_errors() {
        let err =
            train_from_text("", 1, 0, Style::Classic, true, false).expect_err("empty dataset");
        match err {
            ScalarError::EmptyDataset => {}
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
        let tied = train_from_text("abababababababab", 0, 7, Style::Classic, true, false)
            .expect("tied metrics");
        let untied = train_from_text("abababababababab", 0, 7, Style::Classic, false, false)
            .expect("untied metrics");
        assert!(untied.parameter_count > tied.parameter_count);
    }

    #[test]
    fn train_uses_split_and_reports_validation_loss() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let metrics = train_from_text(text, 5, 31, Style::Classic, true, false).expect("metrics");

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
