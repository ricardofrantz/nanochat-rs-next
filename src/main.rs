use std::process;

use nanochat_rs_next::{cli, config::AppCommand, config::Mode, experiments, scalar, tensor};

fn run_or_exit(
    mode: Mode,
    scalar: impl FnOnce() -> Result<String, String>,
    tensor: impl FnOnce() -> Result<String, String>,
    context: &str,
) -> String {
    let result = match mode {
        Mode::Scalar => scalar(),
        Mode::Tensor => tensor(),
    };

    match result {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{context} failed: {err}");
            process::exit(1);
        }
    }
}

fn main() {
    let command = cli::parse_command();
    match command {
        AppCommand::Train(config) => {
            let mode = config.runtime.mode;
            let output = run_or_exit(
                mode,
                || {
                    scalar::train(&config)
                        .map(|value| value.to_string())
                        .map_err(|err| err.to_string())
                },
                || {
                    tensor::train(&config)
                        .map(|value| value.to_string())
                        .map_err(|err| err.to_string())
                },
                "train",
            );
            println!("{output}");
        }
        AppCommand::Sample(config) => {
            let mode = config.runtime.mode;
            let output = run_or_exit(
                mode,
                || {
                    scalar::sample(&config)
                        .map(|value| value.to_string())
                        .map_err(|err| err.to_string())
                },
                || {
                    tensor::sample(&config)
                        .map(|value| value.to_string())
                        .map_err(|err| err.to_string())
                },
                "sample",
            );
            println!("{output}");
        }
        AppCommand::Ablate(config) => match experiments::run_ablation(&config) {
            Ok(report) => println!("{report}"),
            Err(err) => {
                eprintln!("ablate failed: {err}");
                process::exit(1);
            }
        },
    }
}
