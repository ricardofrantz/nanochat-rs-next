use std::process;

use nanochat_rs_next::{cli, config::AppCommand, config::Mode, experiments, scalar, tensor};

fn main() {
    let command = cli::parse_command();
    match command {
        AppCommand::Train(config) => match config.mode {
            Mode::Scalar => match scalar::train(&config) {
                Ok(metrics) => println!("{metrics}"),
                Err(err) => {
                    eprintln!("scalar train failed: {err}");
                    process::exit(1);
                }
            },
            Mode::Tensor => match tensor::train(&config) {
                Ok(metrics) => println!("{metrics}"),
                Err(err) => {
                    eprintln!("tensor train failed: {err}");
                    process::exit(1);
                }
            },
        },
        AppCommand::Sample(config) => match config.mode {
            Mode::Scalar => match scalar::sample(&config) {
                Ok(output) => println!("{output}"),
                Err(err) => {
                    eprintln!("scalar sample failed: {err}");
                    process::exit(1);
                }
            },
            Mode::Tensor => match tensor::sample(&config) {
                Ok(output) => println!("{output}"),
                Err(err) => {
                    eprintln!("tensor sample failed: {err}");
                    process::exit(1);
                }
            },
        },
        AppCommand::Ablate(config) => match experiments::run_ablation(&config) {
            Ok(report) => println!("{report}"),
            Err(err) => {
                eprintln!("ablate failed: {err}");
                process::exit(1);
            }
        },
    }
}
