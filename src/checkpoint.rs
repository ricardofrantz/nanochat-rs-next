use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum CheckpointEvalError<E> {
    Io(std::io::Error),
    Eval(E),
}

impl<E: fmt::Display> fmt::Display for CheckpointEvalError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Eval(err) => write!(f, "{err}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for CheckpointEvalError<E> {}

pub fn persist_checkpoint(path: &Path, payload: &[u8]) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = temporary_checkpoint_path(path);
    fs::write(&tmp_path, payload)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

pub fn persist_then_eval<T, E, F>(
    checkpoint_path: &Path,
    payload: &[u8],
    eval_fn: F,
) -> Result<T, CheckpointEvalError<E>>
where
    F: FnOnce(&Path) -> Result<T, E>,
{
    persist_checkpoint(checkpoint_path, payload).map_err(CheckpointEvalError::Io)?;
    eval_fn(checkpoint_path).map_err(CheckpointEvalError::Eval)
}

fn temporary_checkpoint_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("checkpoint.bin");
    path.with_file_name(format!("{file_name}.tmp"))
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::persist_then_eval;

    #[test]
    fn checkpoint_before_eval_persists_file() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let checkpoint_path = PathBuf::from(format!(
            "results/test_artifacts/checkpoint_order_{unique}.bin"
        ));
        let payload = b"step=10;loss=1.234";

        let seen_len = persist_then_eval(
            &checkpoint_path,
            payload,
            |path| -> Result<usize, Infallible> {
                assert!(path.exists(), "checkpoint must exist before eval");
                let bytes = fs::read(path).expect("read checkpoint");
                assert_eq!(bytes, payload);
                Ok(bytes.len())
            },
        )
        .expect("checkpoint then eval should succeed");

        assert_eq!(seen_len, payload.len());
        let _ = fs::remove_file(&checkpoint_path);
    }
}
