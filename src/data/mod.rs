use std::fs;
use std::io;
use std::path::Path;

pub mod tokenizer;

pub use tokenizer::{BOS_TOKEN, Tokenizer, TokenizerError};

pub fn load_text(path: &Path) -> io::Result<String> {
    fs::read_to_string(path)
}

pub fn split_train_val(tokens: &[usize], val_fraction: f32) -> (Vec<usize>, Vec<usize>) {
    if tokens.len() < 2 {
        return (tokens.to_vec(), Vec::new());
    }

    let fraction = val_fraction.clamp(0.0, 1.0);
    if fraction <= 0.0 {
        return (tokens.to_vec(), Vec::new());
    }
    if fraction >= 1.0 {
        return (Vec::new(), tokens.to_vec());
    }

    let mut val_len = ((tokens.len() as f32) * fraction).round() as usize;
    val_len = val_len.clamp(1, tokens.len() - 1);
    let split_idx = tokens.len() - val_len;

    (tokens[..split_idx].to_vec(), tokens[split_idx..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::split_train_val;

    #[test]
    fn split_preserves_order() {
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (train, val) = split_train_val(&tokens, 0.2);

        assert_eq!(train, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(val, vec![8, 9]);
    }

    #[test]
    fn split_handles_small_input() {
        let tokens = vec![42];
        let (train, val) = split_train_val(&tokens, 0.2);

        assert_eq!(train, vec![42]);
        assert!(val.is_empty());
    }

    #[test]
    fn split_clamps_extreme_fractions() {
        let tokens = vec![1, 2, 3];

        let (all_train, no_val) = split_train_val(&tokens, 0.0);
        assert_eq!(all_train, tokens);
        assert!(no_val.is_empty());

        let (no_train, all_val) = split_train_val(&tokens, 1.0);
        assert!(no_train.is_empty());
        assert_eq!(all_val, tokens);
    }
}
