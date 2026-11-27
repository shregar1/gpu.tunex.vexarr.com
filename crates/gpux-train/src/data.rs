use std::sync::Arc;
use std::path::Path;
use gpux_tensor::{GpuContext, Tensor};

pub struct TextDataset {
    tokens: Vec<u32>,
    seq_len: usize,
    cursor: usize,
}

impl TextDataset {
    pub fn from_file(path: &Path, tokenizer: &Tokenizer) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let tokens = tokenizer.encode(&text);
        Ok(Self {
            tokens,
            seq_len: 64,
            cursor: 0,
        })
    }

    pub fn from_text(text: &str, tokenizer: &Tokenizer) -> Self {
        let tokens = tokenizer.encode(text);
        Self {
            tokens,
            seq_len: 64,
            cursor: 0,
        }
    }

    pub fn from_tokens(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            seq_len: 64,
            cursor: 0,
        }
    }

    pub fn set_seq_len(&mut self, seq_len: usize) {
        self.seq_len = seq_len;
    }

    pub fn total_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn next_batch(
        &mut self,
        ctx: &Arc<GpuContext>,
        batch_size: usize,
    ) -> gpux_core::Result<(Tensor, Tensor)> {
        let chunk = self.seq_len + 1;

        if self.cursor + batch_size * chunk > self.tokens.len() {
            self.cursor = 0;
        }

        let mut input_data = Vec::with_capacity(batch_size * self.seq_len);
        let mut target_data = Vec::with_capacity(batch_size * self.seq_len);

        for b in 0..batch_size {
            let start = self.cursor + b * chunk;
            for t in 0..self.seq_len {
                input_data.push(self.tokens[start + t]);
                target_data.push(self.tokens[start + t + 1]);
            }
        }

        self.cursor += batch_size * chunk;

        let input_ids = Tensor::from_uint_data(ctx, vec![batch_size, self.seq_len], &input_data)?;
        let targets = Tensor::from_uint_data(ctx, vec![batch_size * self.seq_len], &target_data)?;

        Ok((input_ids, targets))
    }

    pub fn num_batches(&self, batch_size: usize) -> usize {
        let chunk = self.seq_len + 1;
        self.tokens.len() / (batch_size * chunk)
    }
}

pub struct Tokenizer {
    vocab: Vec<String>,
    token_to_id: std::collections::HashMap<String, u32>,
}

impl Tokenizer {
    pub fn byte_level() -> Self {
        let mut vocab = Vec::with_capacity(256);
        let mut token_to_id = std::collections::HashMap::new();
        for i in 0..256u32 {
            let s = format!("<{:02x}>", i);
            token_to_id.insert(s.clone(), i);
            vocab.push(s);
        }
        Self { vocab, token_to_id }
    }

    pub fn char_level(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut vocab = Vec::with_capacity(chars.len());
        let mut token_to_id = std::collections::HashMap::new();
        for (i, c) in chars.iter().enumerate() {
            let s = c.to_string();
            token_to_id.insert(s.clone(), i as u32);
            vocab.push(s);
        }
        Self { vocab, token_to_id }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.vocab.len() == 256 && self.vocab[0].starts_with('<') {
            text.as_bytes().iter().map(|&b| b as u32).collect()
        } else {
            text.chars()
                .filter_map(|c| self.token_to_id.get(&c.to_string()).copied())
                .collect()
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        if self.vocab.len() == 256 && self.vocab[0].starts_with('<') {
            let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
            String::from_utf8_lossy(&bytes).to_string()
        } else {
            tokens
                .iter()
                .filter_map(|&t| self.vocab.get(t as usize))
                .cloned()
                .collect()
        }
    }
}
