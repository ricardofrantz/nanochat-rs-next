use std::ffi::OsString;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::config::{
    AblateConfig, AppCommand, Mode, ModelKind, Optimizer, RuntimeConfig, SampleConfig, Style,
    TrainConfig,
};
use crate::training;

#[derive(Debug, Parser)]
#[command(
    name = "nanochat-rs-next",
    version,
    about = "Rust nanochat benchmark CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    Train(TrainArgs),
    Sample(SampleArgs),
    Ablate(AblateArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMode {
    Scalar,
    Tensor,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliModelKind {
    Bigram,
    MiniGpt,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliStyle {
    Classic,
    Futuristic,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliOptimizer {
    Sgd,
    #[value(name = "adamw", alias = "adam-w")]
    AdamW,
}

impl From<CliMode> for Mode {
    fn from(mode: CliMode) -> Self {
        match mode {
            CliMode::Scalar => Self::Scalar,
            CliMode::Tensor => Self::Tensor,
        }
    }
}

impl From<CliModelKind> for ModelKind {
    fn from(model_kind: CliModelKind) -> Self {
        match model_kind {
            CliModelKind::Bigram => Self::Bigram,
            CliModelKind::MiniGpt => Self::MiniGpt,
        }
    }
}

impl From<CliStyle> for Style {
    fn from(style: CliStyle) -> Self {
        match style {
            CliStyle::Classic => Self::Classic,
            CliStyle::Futuristic => Self::Futuristic,
        }
    }
}

impl From<CliOptimizer> for Optimizer {
    fn from(optimizer: CliOptimizer) -> Self {
        match optimizer {
            CliOptimizer::Sgd => Self::Sgd,
            CliOptimizer::AdamW => Self::AdamW,
        }
    }
}

#[derive(Debug, Args)]
struct RuntimeArgs {
    #[arg(long, value_enum, default_value_t = CliMode::Scalar)]
    mode: CliMode,
    #[arg(long, value_enum, default_value_t = CliModelKind::Bigram)]
    model_kind: CliModelKind,
    #[arg(long, value_enum, default_value_t = CliStyle::Futuristic)]
    style: CliStyle,
    #[arg(long, default_value = "input.txt")]
    data: PathBuf,
    #[arg(long, default_value_t = 1337)]
    seed: u64,
}

#[derive(Debug, Args)]
struct TrainArgs {
    #[command(flatten)]
    runtime: RuntimeArgs,
    #[arg(long, value_enum, default_value_t = CliOptimizer::Sgd)]
    optimizer: CliOptimizer,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    tie_lm_head: bool,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    input_rmsnorm: bool,
    #[arg(long, default_value_t = 500)]
    steps: usize,
    #[arg(long, default_value_t = 0)]
    checkpoint_every: usize,
    #[arg(long, default_value = training::DEFAULT_CHECKPOINT_DIR)]
    checkpoint_dir: PathBuf,
}

#[derive(Debug, Args)]
struct SampleArgs {
    #[command(flatten)]
    runtime: RuntimeArgs,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 200)]
    max_new_tokens: usize,
    #[arg(long, default_value = "")]
    prompt: String,
}

#[derive(Debug, Args)]
struct AblateArgs {
    #[arg(long, value_enum, default_value_t = CliStyle::Futuristic)]
    style: CliStyle,
    #[arg(long, default_value_t = 500)]
    steps: usize,
    #[arg(long, default_value = "input.txt")]
    data: PathBuf,
    #[arg(long, default_value_t = 1337)]
    seed: u64,
}

pub fn parse_command() -> AppCommand {
    from_cli(Cli::parse())
}

pub fn try_command_from_iter<I, T>(iter: I) -> Result<AppCommand, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = Cli::try_parse_from(iter)?;
    Ok(from_cli(cli))
}

fn from_cli(cli: Cli) -> AppCommand {
    let to_runtime = |args: RuntimeArgs| RuntimeConfig {
        mode: args.mode.into(),
        model_kind: args.model_kind.into(),
        style: args.style.into(),
        data_path: args.data,
        seed: args.seed,
    };

    match cli.command {
        CliCommand::Train(args) => AppCommand::Train(TrainConfig {
            runtime: to_runtime(args.runtime),
            optimizer: args.optimizer.into(),
            tie_lm_head: args.tie_lm_head,
            input_rmsnorm: args.input_rmsnorm,
            steps: args.steps,
            checkpoint_every: args.checkpoint_every,
            checkpoint_dir: args.checkpoint_dir,
        }),
        CliCommand::Sample(args) => AppCommand::Sample(SampleConfig {
            runtime: to_runtime(args.runtime),
            temperature: args.temperature,
            max_new_tokens: args.max_new_tokens,
            prompt: args.prompt,
        }),
        CliCommand::Ablate(args) => AppCommand::Ablate(AblateConfig {
            style: args.style.into(),
            steps: args.steps,
            data_path: args.data,
            seed: args.seed,
        }),
    }
}

#[cfg(test)]
mod tests {
    use clap::error::ErrorKind;

    use super::*;

    #[test]
    fn parses_train_defaults() {
        let command = try_command_from_iter(["nanochat-rs-next", "train"]).expect("valid command");
        let AppCommand::Train(config) = command else {
            panic!("expected train command");
        };

        assert_eq!(config.runtime.mode, Mode::Scalar);
        assert_eq!(config.runtime.model_kind, ModelKind::Bigram);
        assert_eq!(config.runtime.style, Style::Futuristic);
        assert!(config.tie_lm_head);
        assert!(!config.input_rmsnorm);
        assert_eq!(config.steps, 500);
        assert_eq!(config.runtime.data_path, PathBuf::from("input.txt"));
        assert_eq!(config.runtime.seed, 1337);
        assert_eq!(config.checkpoint_every, 0);
        assert_eq!(
            config.checkpoint_dir,
            PathBuf::from(training::DEFAULT_CHECKPOINT_DIR)
        );
    }

    #[test]
    fn parses_train_variant_overrides() {
        let command = try_command_from_iter([
            "nanochat-rs-next",
            "train",
            "--tie-lm-head=false",
            "--input-rmsnorm=true",
            "--optimizer",
            "adamw",
        ])
        .expect("valid train command");
        let AppCommand::Train(config) = command else {
            panic!("expected train command");
        };

        assert!(!config.tie_lm_head);
        assert!(config.input_rmsnorm);
        assert_eq!(config.optimizer, Optimizer::AdamW);
    }

    #[test]
    fn parses_sample_overrides() {
        let command = try_command_from_iter([
            "nanochat-rs-next",
            "sample",
            "--mode",
            "scalar",
            "--style",
            "classic",
            "--temperature",
            "0.6",
            "--max-new-tokens",
            "64",
            "--prompt",
            "hi",
            "--seed",
            "7",
        ])
        .expect("valid sample command");
        let AppCommand::Sample(config) = command else {
            panic!("expected sample command");
        };

        assert_eq!(config.runtime.mode, Mode::Scalar);
        assert_eq!(config.runtime.model_kind, ModelKind::Bigram);
        assert_eq!(config.runtime.style, Style::Classic);
        assert!((config.temperature - 0.6).abs() < 1e-12);
        assert_eq!(config.max_new_tokens, 64);
        assert_eq!(config.prompt, "hi");
        assert_eq!(config.runtime.seed, 7);
    }

    #[test]
    fn parses_sample_defaults() {
        let command =
            try_command_from_iter(["nanochat-rs-next", "sample"]).expect("valid sample command");
        let AppCommand::Sample(config) = command else {
            panic!("expected sample command");
        };

        assert_eq!(config.runtime.mode, Mode::Scalar);
        assert_eq!(config.runtime.model_kind, ModelKind::Bigram);
        assert_eq!(config.runtime.style, Style::Futuristic);
        assert!((config.temperature - 0.8).abs() < 1e-12);
        assert_eq!(config.max_new_tokens, 200);
        assert_eq!(config.prompt, "");
        assert_eq!(config.runtime.data_path, PathBuf::from("input.txt"));
        assert_eq!(config.runtime.seed, 1337);
    }

    #[test]
    fn supports_help_flag() {
        let err = try_command_from_iter(["nanochat-rs-next", "--help"]).expect_err("help exits");
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
    }

    #[test]
    fn parses_ablate_defaults() {
        let command =
            try_command_from_iter(["nanochat-rs-next", "ablate"]).expect("valid ablate command");
        let AppCommand::Ablate(config) = command else {
            panic!("expected ablate command");
        };

        assert_eq!(config.style, Style::Futuristic);
        assert_eq!(config.steps, 500);
        assert_eq!(config.data_path, PathBuf::from("input.txt"));
        assert_eq!(config.seed, 1337);
    }

    #[test]
    fn parses_mini_gpt_model_kind() {
        let command =
            try_command_from_iter(["nanochat-rs-next", "train", "--model-kind", "mini-gpt"])
                .expect("valid train command");
        let AppCommand::Train(config) = command else {
            panic!("expected train command");
        };
        assert_eq!(config.runtime.model_kind, ModelKind::MiniGpt);
    }

    #[test]
    fn parses_tensor_mode_and_optimizer() {
        let command = try_command_from_iter([
            "nanochat-rs-next",
            "train",
            "--mode",
            "tensor",
            "--optimizer",
            "adamw",
        ])
        .expect("valid train command");
        let AppCommand::Train(config) = command else {
            panic!("expected train command");
        };

        assert_eq!(config.runtime.mode, Mode::Tensor);
        assert_eq!(config.optimizer, Optimizer::AdamW);
    }

    #[test]
    fn parses_tensor_mode_with_mini_gpt_model_kind() {
        let command = try_command_from_iter([
            "nanochat-rs-next",
            "train",
            "--mode",
            "tensor",
            "--model-kind",
            "mini-gpt",
            "--optimizer",
            "sgd",
        ])
        .expect("valid train command");
        let AppCommand::Train(config) = command else {
            panic!("expected train command");
        };

        assert_eq!(config.runtime.mode, Mode::Tensor);
        assert_eq!(config.runtime.model_kind, ModelKind::MiniGpt);
        assert_eq!(config.optimizer, Optimizer::Sgd);
    }
}
