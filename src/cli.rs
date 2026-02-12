use std::ffi::OsString;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::config::{AblateConfig, AppCommand, Mode, ModelKind, SampleConfig, Style, TrainConfig};

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

#[derive(Debug, Args)]
struct TrainArgs {
    #[arg(long, value_enum, default_value_t = CliMode::Scalar)]
    mode: CliMode,
    #[arg(long, value_enum, default_value_t = CliModelKind::Bigram)]
    model_kind: CliModelKind,
    #[arg(long, value_enum, default_value_t = CliStyle::Futuristic)]
    style: CliStyle,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    tie_lm_head: bool,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    input_rmsnorm: bool,
    #[arg(long, default_value_t = 500)]
    steps: usize,
    #[arg(long, default_value = "input.txt")]
    data: PathBuf,
    #[arg(long, default_value_t = 1337)]
    seed: u64,
}

#[derive(Debug, Args)]
struct SampleArgs {
    #[arg(long, value_enum, default_value_t = CliMode::Scalar)]
    mode: CliMode,
    #[arg(long, value_enum, default_value_t = CliModelKind::Bigram)]
    model_kind: CliModelKind,
    #[arg(long, value_enum, default_value_t = CliStyle::Futuristic)]
    style: CliStyle,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 200)]
    max_new_tokens: usize,
    #[arg(long, default_value = "")]
    prompt: String,
    #[arg(long, default_value = "input.txt")]
    data: PathBuf,
    #[arg(long, default_value_t = 1337)]
    seed: u64,
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
    match cli.command {
        CliCommand::Train(args) => AppCommand::Train(TrainConfig {
            mode: args.mode.into(),
            model_kind: args.model_kind.into(),
            style: args.style.into(),
            tie_lm_head: args.tie_lm_head,
            input_rmsnorm: args.input_rmsnorm,
            steps: args.steps,
            data_path: args.data,
            seed: args.seed,
        }),
        CliCommand::Sample(args) => AppCommand::Sample(SampleConfig {
            mode: args.mode.into(),
            model_kind: args.model_kind.into(),
            style: args.style.into(),
            temperature: args.temperature,
            max_new_tokens: args.max_new_tokens,
            prompt: args.prompt,
            data_path: args.data,
            seed: args.seed,
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

        assert_eq!(config.mode, Mode::Scalar);
        assert_eq!(config.model_kind, ModelKind::Bigram);
        assert_eq!(config.style, Style::Futuristic);
        assert!(config.tie_lm_head);
        assert!(!config.input_rmsnorm);
        assert_eq!(config.steps, 500);
        assert_eq!(config.data_path, PathBuf::from("input.txt"));
        assert_eq!(config.seed, 1337);
    }

    #[test]
    fn parses_train_variant_overrides() {
        let command = try_command_from_iter([
            "nanochat-rs-next",
            "train",
            "--tie-lm-head=false",
            "--input-rmsnorm=true",
        ])
        .expect("valid train command");
        let AppCommand::Train(config) = command else {
            panic!("expected train command");
        };

        assert!(!config.tie_lm_head);
        assert!(config.input_rmsnorm);
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
        .expect("valid command");
        let AppCommand::Sample(config) = command else {
            panic!("expected sample command");
        };

        assert_eq!(config.mode, Mode::Scalar);
        assert_eq!(config.model_kind, ModelKind::Bigram);
        assert_eq!(config.style, Style::Classic);
        assert!((config.temperature - 0.6).abs() < 1e-12);
        assert_eq!(config.max_new_tokens, 64);
        assert_eq!(config.prompt, "hi");
        assert_eq!(config.seed, 7);
    }

    #[test]
    fn parses_sample_defaults() {
        let command =
            try_command_from_iter(["nanochat-rs-next", "sample"]).expect("valid sample command");
        let AppCommand::Sample(config) = command else {
            panic!("expected sample command");
        };

        assert_eq!(config.mode, Mode::Scalar);
        assert_eq!(config.model_kind, ModelKind::Bigram);
        assert_eq!(config.style, Style::Futuristic);
        assert!((config.temperature - 0.8).abs() < 1e-12);
        assert_eq!(config.max_new_tokens, 200);
        assert_eq!(config.prompt, "");
        assert_eq!(config.data_path, PathBuf::from("input.txt"));
        assert_eq!(config.seed, 1337);
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
        assert_eq!(config.model_kind, ModelKind::MiniGpt);
    }
}
