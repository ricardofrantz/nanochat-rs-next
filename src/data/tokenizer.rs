use std::collections::{BTreeSet, HashMap};
use std::fmt;

pub const BOS_TOKEN: &str = "<BOS>";
const BOS_ID: usize = 0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokenizer {
    id_to_char: Vec<char>,
    char_to_id: HashMap<char, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    UnknownToken(char),
    UnknownId(usize),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownToken(ch) => write!(f, "unknown token {ch:?}"),
            Self::UnknownId(id) => write!(f, "unknown token id {id}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl Tokenizer {
    pub fn from_text(text: &str) -> Self {
        let unique_chars: BTreeSet<char> = text.chars().collect();
        let id_to_char: Vec<char> = unique_chars.into_iter().collect();
        let char_to_id: HashMap<char, usize> = id_to_char
            .iter()
            .enumerate()
            .map(|(idx, ch)| (*ch, idx + 1))
            .collect();

        Self {
            id_to_char,
            char_to_id,
        }
    }

    pub fn bos_id(&self) -> usize {
        BOS_ID
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len() + 1
    }

    pub fn id_for_char(&self, ch: char) -> Option<usize> {
        self.char_to_id.get(&ch).copied()
    }

    pub fn char_for_id(&self, id: usize) -> Option<char> {
        if id == BOS_ID {
            return None;
        }
        self.id_to_char.get(id - 1).copied()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        text.chars()
            .map(|ch| self.id_for_char(ch).ok_or(TokenizerError::UnknownToken(ch)))
            .collect()
    }

    pub fn encode_with_bos(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        let mut ids = Vec::with_capacity(text.chars().count() + 1);
        ids.push(BOS_ID);
        ids.extend(self.encode(text)?);
        Ok(ids)
    }

    pub fn decode(&self, ids: &[usize]) -> Result<String, TokenizerError> {
        let mut text = String::new();
        for id in ids {
            if *id == BOS_ID {
                continue;
            }
            let ch = self
                .char_for_id(*id)
                .ok_or(TokenizerError::UnknownId(*id))?;
            text.push(ch);
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::{BOS_ID, Tokenizer, TokenizerError};

    #[test]
    fn deterministic_vocab_ids() {
        let tokenizer = Tokenizer::from_text("cab");
        assert_eq!(tokenizer.bos_id(), BOS_ID);
        assert_eq!(tokenizer.vocab_size(), 4);
        assert_eq!(tokenizer.id_for_char('a'), Some(1));
        assert_eq!(tokenizer.id_for_char('b'), Some(2));
        assert_eq!(tokenizer.id_for_char('c'), Some(3));
    }

    #[test]
    fn round_trip_and_bos_handling() {
        let tokenizer = Tokenizer::from_text("abba");
        let ids = tokenizer.encode("abba").expect("known chars");
        assert_eq!(tokenizer.decode(&ids).expect("known ids"), "abba");

        let ids_with_bos = tokenizer.encode_with_bos("abba").expect("known chars");
        assert_eq!(ids_with_bos[0], BOS_ID);
        assert_eq!(tokenizer.decode(&ids_with_bos).expect("known ids"), "abba");
    }

    #[test]
    fn unknown_char_errors() {
        let tokenizer = Tokenizer::from_text("abc");
        let err = tokenizer.encode("abz").expect_err("z not in vocab");
        assert_eq!(err, TokenizerError::UnknownToken('z'));
    }

    #[test]
    fn unknown_id_errors() {
        let tokenizer = Tokenizer::from_text("abc");
        let err = tokenizer.decode(&[1, 999]).expect_err("999 invalid id");
        assert_eq!(err, TokenizerError::UnknownId(999));
    }
}
