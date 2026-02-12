use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Scalar,
    Tensor,
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar => write!(f, "scalar"),
            Self::Tensor => write!(f, "tensor"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Bigram,
    MiniGpt,
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bigram => write!(f, "bigram"),
            Self::MiniGpt => write!(f, "mini-gpt"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Classic,
    Futuristic,
}

impl fmt::Display for Style {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Classic => write!(f, "classic"),
            Self::Futuristic => write!(f, "futuristic"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    pub mode: Mode,
    pub model_kind: ModelKind,
    pub style: Style,
    pub tie_lm_head: bool,
    pub input_rmsnorm: bool,
    pub steps: usize,
    pub data_path: PathBuf,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampleConfig {
    pub mode: Mode,
    pub model_kind: ModelKind,
    pub style: Style,
    pub temperature: f64,
    pub max_new_tokens: usize,
    pub prompt: String,
    pub data_path: PathBuf,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AblateConfig {
    pub style: Style,
    pub steps: usize,
    pub data_path: PathBuf,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppCommand {
    Train(TrainConfig),
    Sample(SampleConfig),
    Ablate(AblateConfig),
}
