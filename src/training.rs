use std::fmt;

use crate::config::Style;
use crate::data;
use rand::rngs::StdRng;
use rand::Rng;

pub const DEFAULT_CHECKPOINT_DIR: &str = "results/checkpoints";
pub const VAL_FRACTION: f32 = 0.1;
pub const EVAL_EVERY: usize = 20;
pub const EVAL_PAIRS: usize = 1024;

const WARMUP_RATIO: f64 = 0.0;
const WARMDOWN_RATIO: f64 = 0.5;
const FINAL_LR_FRAC: f64 = 0.0;

pub const FUTURISTIC_CORPUS: &str = "\
neon skylines hum over orbital transit lanes.\n\
quantum couriers sync with neural uplinks at dawn.\n\
autonomous swarms calibrate reactor lattices in silence.\n\
synthetic pilots chart wormhole routes beyond saturn.\n";

pub type TokenPair = (usize, usize);
pub type PairSplits = (Vec<TokenPair>, Vec<TokenPair>);

#[derive(Debug)]
pub enum TrainingDataError {
    EmptyDataset,
}

impl fmt::Display for TrainingDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => write!(f, "dataset must contain at least one token pair"),
        }
    }
}

impl std::error::Error for TrainingDataError {}

pub fn styled_corpus(base: &str, style: Style) -> String {
    match style {
        Style::Classic => base.to_string(),
        Style::Futuristic => format!("{base}\n{FUTURISTIC_CORPUS}"),
    }
}

pub fn apply_futuristic_boost(
    transition_counts: &mut [Vec<u64>],
    tokenizer: &crate::data::Tokenizer,
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

pub fn build_transition_counts(vocab_size: usize, pairs: &[(usize, usize)]) -> Vec<Vec<u64>> {
    let mut counts = vec![vec![1_u64; vocab_size]; vocab_size];
    for (context_id, target_id) in pairs {
        counts[*context_id][*target_id] += 1;
    }
    counts
}

pub fn sample_index(
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

pub fn weighted_choice(weights: &[f64], rng: &mut StdRng) -> usize {
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

pub fn should_eval(step: usize, total_steps: usize, eval_every: usize) -> bool {
    if step == total_steps {
        return true;
    }
    eval_every > 0 && step.is_multiple_of(eval_every)
}

pub fn lr_multiplier(step: usize, total_steps: usize) -> f64 {
    if total_steps == 0 {
        return 1.0;
    }
    let warmup_iters = (WARMUP_RATIO * (total_steps as f64)).round() as usize;
    let warmdown_iters = (WARMDOWN_RATIO * (total_steps as f64)).round() as usize;
    if warmup_iters > 0 && step < warmup_iters {
        return (step + 1) as f64 / (warmup_iters as f64);
    }
    if warmdown_iters == 0 || step <= total_steps.saturating_sub(warmdown_iters) {
        return 1.0;
    }
    let progress = (total_steps - step) as f64 / (warmdown_iters as f64);
    progress + (1.0 - progress) * FINAL_LR_FRAC
}

pub fn eval_pairs_slice(pairs: &[(usize, usize)]) -> &[(usize, usize)] {
    let eval_len = pairs.len().min(EVAL_PAIRS);
    &pairs[..eval_len]
}

pub fn build_pairs(token_ids: &[usize]) -> Result<Vec<TokenPair>, TrainingDataError> {
    if token_ids.len() < 2 {
        return Err(TrainingDataError::EmptyDataset);
    }
    Ok(token_ids
        .windows(2)
        .map(|window| (window[0], window[1]))
        .collect())
}

pub fn split_train_val_tokens(
    token_ids: &[usize],
) -> Result<(Vec<usize>, Vec<usize>), TrainingDataError> {
    if build_pairs(token_ids).is_err() {
        return Err(TrainingDataError::EmptyDataset);
    }

    let (train_tokens, val_tokens) = data::split_train_val(token_ids, VAL_FRACTION);

    let train_tokens = match build_pairs(&train_tokens) {
        Ok(_) => train_tokens,
        Err(_) => token_ids.to_vec(),
    };
    let val_tokens = match build_pairs(&val_tokens) {
        Ok(_) => val_tokens,
        Err(_) => train_tokens.clone(),
    };

    Ok((train_tokens, val_tokens))
}

pub fn split_train_val_pairs(token_ids: &[usize]) -> Result<PairSplits, TrainingDataError> {
    let (train_tokens, val_tokens) = split_train_val_tokens(token_ids)?;
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
