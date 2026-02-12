use std::process;

use microgpt_rs_lab::{cli, config::AppCommand, config::Mode, experiments, scalar, tensor};

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
            Mode::Tensor => println!("{}", tensor::train_stub(&config)),
        },
        AppCommand::Sample(config) => match config.mode {
            Mode::Scalar => match scalar::sample(&config) {
                Ok(output) => println!("{output}"),
                Err(err) => {
                    eprintln!("scalar sample failed: {err}");
                    process::exit(1);
                }
            },
            Mode::Tensor => println!("{}", tensor::sample_stub(&config)),
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
