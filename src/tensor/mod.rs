use crate::config::{SampleConfig, TrainConfig};

pub fn train_stub(config: &TrainConfig) -> String {
    format!(
        "tensor train is not implemented yet (style={}, steps={}, seed={}, data={})",
        config.style,
        config.steps,
        config.seed,
        config.data_path.display()
    )
}

pub fn sample_stub(config: &SampleConfig) -> String {
    format!(
        "tensor sample is not implemented yet (style={}, temperature={}, max_new_tokens={}, seed={}, data={}, prompt={:?})",
        config.style,
        config.temperature,
        config.max_new_tokens,
        config.seed,
        config.data_path.display(),
        config.prompt
    )
}
