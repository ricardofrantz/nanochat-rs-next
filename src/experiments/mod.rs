use std::fmt;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::{
    AblateConfig, LrSchedule, Mode, ModelKind, Optimizer, RuntimeConfig, TrainConfig,
};
use crate::scalar::{self, TrainMetrics};
use crate::training;

#[derive(Debug, Clone, Copy)]
struct AblationVariant {
    name: &'static str,
    tie_lm_head: bool,
    input_rmsnorm: bool,
}

const VARIANTS: [AblationVariant; 4] = [
    AblationVariant {
        name: "tied_no_input_rmsnorm",
        tie_lm_head: true,
        input_rmsnorm: false,
    },
    AblationVariant {
        name: "tied_input_rmsnorm",
        tie_lm_head: true,
        input_rmsnorm: true,
    },
    AblationVariant {
        name: "untied_no_input_rmsnorm",
        tie_lm_head: false,
        input_rmsnorm: false,
    },
    AblationVariant {
        name: "untied_input_rmsnorm",
        tie_lm_head: false,
        input_rmsnorm: true,
    },
];

#[derive(Debug, Clone)]
struct AblationRecord {
    variant: &'static str,
    tie_lm_head: bool,
    input_rmsnorm: bool,
    style: String,
    parameter_count: usize,
    steps: usize,
    seed: u64,
    final_loss: f64,
    mean_loss_last_n: f64,
    last_n: usize,
    steps_per_sec: f64,
    tokens_per_sec: f64,
    vocab_size: usize,
    train_tokens: usize,
}

impl AblationRecord {
    fn from_metrics(variant: AblationVariant, metrics: &TrainMetrics, seed: u64) -> Self {
        Self {
            variant: variant.name,
            tie_lm_head: variant.tie_lm_head,
            input_rmsnorm: variant.input_rmsnorm,
            style: metrics.style.to_string(),
            parameter_count: metrics.parameter_count,
            steps: metrics.steps,
            seed,
            final_loss: metrics.final_loss,
            mean_loss_last_n: metrics.mean_loss_last_n,
            last_n: metrics.last_n,
            steps_per_sec: metrics.steps_per_sec,
            tokens_per_sec: metrics.tokens_per_sec,
            vocab_size: metrics.vocab_size,
            train_tokens: metrics.train_tokens,
        }
    }

    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{:.6},{:.6},{},{:.6},{:.6},{},{}",
            self.variant,
            self.tie_lm_head,
            self.input_rmsnorm,
            self.style,
            self.parameter_count,
            self.steps,
            self.seed,
            self.final_loss,
            self.mean_loss_last_n,
            self.last_n,
            self.steps_per_sec,
            self.tokens_per_sec,
            self.vocab_size,
            self.train_tokens
        )
    }

    fn to_json_line(&self) -> String {
        format!(
            "{{\"variant\":\"{}\",\"tie_lm_head\":{},\"input_rmsnorm\":{},\"style\":\"{}\",\"parameter_count\":{},\"steps\":{},\"seed\":{},\"final_loss\":{:.6},\"mean_loss_last_n\":{:.6},\"last_n\":{},\"steps_per_sec\":{:.6},\"tokens_per_sec\":{:.6},\"vocab_size\":{},\"train_tokens\":{}}}",
            self.variant,
            self.tie_lm_head,
            self.input_rmsnorm,
            self.style,
            self.parameter_count,
            self.steps,
            self.seed,
            self.final_loss,
            self.mean_loss_last_n,
            self.last_n,
            self.steps_per_sec,
            self.tokens_per_sec,
            self.vocab_size,
            self.train_tokens
        )
    }
}

#[derive(Debug)]
pub struct AblationReport {
    records: Vec<AblationRecord>,
    csv_path: PathBuf,
    jsonl_path: PathBuf,
}

impl fmt::Display for AblationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "mode=scalar ablate variants={} csv={} jsonl={}",
            self.records.len(),
            self.csv_path.display(),
            self.jsonl_path.display()
        )?;
        writeln!(
            f,
            "variant,tie_lm_head,input_rmsnorm,style,params,final_loss,mean_loss_last_n,steps_per_sec,tokens_per_sec"
        )?;
        for record in &self.records {
            writeln!(
                f,
                "{},{},{},{},{},{:.6},{:.6},{:.2},{:.2}",
                record.variant,
                record.tie_lm_head,
                record.input_rmsnorm,
                record.style,
                record.parameter_count,
                record.final_loss,
                record.mean_loss_last_n,
                record.steps_per_sec,
                record.tokens_per_sec
            )?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum AblationError {
    Io(std::io::Error),
    Scalar(scalar::ScalarError),
}

impl fmt::Display for AblationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Scalar(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for AblationError {}

impl From<std::io::Error> for AblationError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<scalar::ScalarError> for AblationError {
    fn from(err: scalar::ScalarError) -> Self {
        Self::Scalar(err)
    }
}

pub fn run_ablation(config: &AblateConfig) -> Result<AblationReport, AblationError> {
    let mut records = Vec::with_capacity(VARIANTS.len());
    for variant in VARIANTS {
        let train_config = TrainConfig {
            runtime: RuntimeConfig {
                mode: Mode::Scalar,
                model_kind: ModelKind::Bigram,
                style: config.style,
                data_path: config.data_path.clone(),
                seed: config.seed,
            },
            optimizer: Optimizer::Sgd,
            lr_schedule: LrSchedule::Linear,
            tie_lm_head: variant.tie_lm_head,
            input_rmsnorm: variant.input_rmsnorm,
            steps: config.steps,
            checkpoint_every: 0,
            checkpoint_dir: PathBuf::from(training::DEFAULT_CHECKPOINT_DIR),
        };
        let metrics = scalar::train(&train_config)?;
        records.push(AblationRecord::from_metrics(variant, &metrics, config.seed));
    }

    let (csv_path, jsonl_path) = make_artifact_paths(config);
    write_csv(&csv_path, &records)?;
    write_jsonl(&jsonl_path, &records)?;

    Ok(AblationReport {
        records,
        csv_path,
        jsonl_path,
    })
}

fn make_artifact_paths(config: &AblateConfig) -> (PathBuf, PathBuf) {
    let run_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let stem = format!(
        "ablate_style-{}_seed-{}_steps-{}_{}",
        config.style, config.seed, config.steps, run_id
    );
    let base = PathBuf::from("results");
    (
        base.join(format!("{stem}.csv")),
        base.join(format!("{stem}.jsonl")),
    )
}

fn write_csv(path: &PathBuf, records: &[AblationRecord]) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    writeln!(
        file,
        "variant,tie_lm_head,input_rmsnorm,style,parameter_count,steps,seed,final_loss,mean_loss_last_n,last_n,steps_per_sec,tokens_per_sec,vocab_size,train_tokens"
    )?;
    for record in records {
        writeln!(file, "{}", record.to_csv_row())?;
    }
    Ok(())
}

fn write_jsonl(path: &PathBuf, records: &[AblationRecord]) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for record in records {
        writeln!(file, "{}", record.to_json_line())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::config::{AblateConfig, Style};

    use super::{VARIANTS, run_ablation};

    #[test]
    fn variant_matrix_has_all_four_combinations() {
        assert_eq!(VARIANTS.len(), 4);
        let combos: Vec<(bool, bool)> = VARIANTS
            .iter()
            .map(|v| (v.tie_lm_head, v.input_rmsnorm))
            .collect();
        assert!(combos.contains(&(true, false)));
        assert!(combos.contains(&(true, true)));
        assert!(combos.contains(&(false, false)));
        assert!(combos.contains(&(false, true)));
    }

    #[test]
    fn ablation_writes_style_tagged_artifacts() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let data_path = PathBuf::from(format!("ablate_test_input_{unique}.txt"));
        fs::write(&data_path, "abababababababab").expect("write test corpus");

        let config = AblateConfig {
            style: Style::Futuristic,
            steps: 8,
            data_path: data_path.clone(),
            seed: 5,
        };
        let report = run_ablation(&config).expect("ablation report");

        let csv_name = report.csv_path.to_string_lossy();
        let jsonl_name = report.jsonl_path.to_string_lossy();
        assert!(csv_name.contains("style-futuristic"));
        assert!(jsonl_name.contains("style-futuristic"));

        assert!(report.csv_path.exists());
        assert!(report.jsonl_path.exists());

        let csv = fs::read_to_string(&report.csv_path).expect("read csv");
        let jsonl = fs::read_to_string(&report.jsonl_path).expect("read jsonl");
        assert_eq!(csv.lines().count(), 5);
        assert_eq!(jsonl.lines().count(), 4);
        assert!(csv.contains("parameter_count"));
        assert!(jsonl.contains("\"parameter_count\""));

        let tied_params = report
            .records
            .iter()
            .find(|record| record.tie_lm_head)
            .map(|record| record.parameter_count)
            .expect("tied record");
        let untied_params = report
            .records
            .iter()
            .find(|record| !record.tie_lm_head)
            .map(|record| record.parameter_count)
            .expect("untied record");
        assert!(untied_params > tied_params);

        let _ = fs::remove_file(&data_path);
        let _ = fs::remove_file(&report.csv_path);
        let _ = fs::remove_file(&report.jsonl_path);
    }
}
