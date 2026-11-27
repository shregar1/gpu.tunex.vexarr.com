use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;
use regex::Regex;

/// One slice of pre-split input: either a raw text region (still needs BPE)
/// or a recognized special token id (passes straight through).
enum Segment<'a> {
    Text(&'a str),
    Special(u32),
}

/// Which BPE flavor the loaded `tokenizer.json` uses.
///
/// * `GptByteLevel` — GPT-2 / Qwen / SmolLM2 style: byte-level BPE with
///   `Ġ` marking word boundaries, no byte fallback.
/// * `SentencePiece` — Llama / TinyLlama / Mistral / Gemma style: prepend
///   `▁` to input, replace spaces with `▁`, BPE on the resulting string,
///   byte fallback (`<0xXX>` tokens) for any character not in the vocab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerKind {
    GptByteLevel,
    SentencePiece,
}

const SP_SPACE: char = '\u{2581}'; // ▁ — SentencePiece "lower one eighth block"

static PRE_TOKENIZE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"
    ).unwrap()
});

pub struct BpeTokenizer {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    /// Merge rank table — lower index = higher priority. Built once at load
    /// time so encode() doesn't re-scan the whole list per merge.
    merge_ranks: HashMap<(String, String), usize>,
    special_tokens: HashMap<String, u32>,
    /// Reverse lookup so decode() can recognize special token ids and
    /// emit their literal string without going through the byte-level
    /// inverse mapping (which would mangle the angle brackets / pipes).
    special_ids: HashMap<u32, String>,
    /// Special-token surface forms, sorted longest-first so encode()
    /// matches them greedily (e.g. `<|im_end|>` before `<|im`).
    special_patterns: Vec<(String, u32)>,
    vocab_size: usize,
    /// Tokenizer flavor (auto-detected from tokenizer.json).
    kind: TokenizerKind,
    /// SentencePiece byte-fallback table: `byte_to_id[b]` is the id of the
    /// `<0xXX>` token for byte `b`, if present in the vocab. Empty for
    /// GPT-2 style tokenizers.
    byte_to_id: Vec<Option<u32>>,
    /// SentencePiece "Strip" decoder config: strip N leading occurrences of
    /// the strip char (typically " ", count 1) after replacing `▁`.
    strip_leading_space: bool,
}

impl BpeTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read tokenizer: {}", e))?;
        Self::from_json(&data)
    }

    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let root: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| format!("invalid tokenizer JSON: {}", e))?;

        let model = root.get("model")
            .ok_or("missing 'model' in tokenizer.json")?;

        let vocab_obj = model.get("vocab")
            .and_then(|v| v.as_object())
            .ok_or("missing 'model.vocab'")?;

        let mut encoder = HashMap::with_capacity(vocab_obj.len());
        let mut decoder = HashMap::with_capacity(vocab_obj.len());

        for (token, id) in vocab_obj {
            let id = id.as_u64().ok_or("vocab id not u64")? as u32;
            encoder.insert(token.clone(), id);
            decoder.insert(id, token.clone());
        }

        let merges_arr = model.get("merges")
            .and_then(|v| v.as_array())
            .ok_or("missing 'model.merges'")?;

        let mut merge_ranks: HashMap<(String, String), usize> =
            HashMap::with_capacity(merges_arr.len());
        for (rank, merge) in merges_arr.iter().enumerate() {
            // Tokenizers can emit merges as either `"a b"` or `["a", "b"]`.
            let pair = if let Some(s) = merge.as_str() {
                let mut it = s.splitn(2, ' ');
                let a = it.next().unwrap_or("").to_string();
                let b = it.next().unwrap_or("").to_string();
                (a, b)
            } else if let Some(arr) = merge.as_array() {
                let a = arr.first().and_then(|v| v.as_str()).unwrap_or("").to_string();
                let b = arr.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string();
                (a, b)
            } else {
                continue;
            };
            if !pair.0.is_empty() && !pair.1.is_empty() {
                merge_ranks.insert(pair, rank);
            }
        }

        let mut special_tokens = HashMap::new();
        if let Some(added) = root.get("added_tokens") {
            if let Some(arr) = added.as_array() {
                for token in arr {
                    if let (Some(content), Some(id)) = (
                        token.get("content").and_then(|c| c.as_str()),
                        token.get("id").and_then(|i| i.as_u64()),
                    ) {
                        special_tokens.insert(content.to_string(), id as u32);
                        encoder.insert(content.to_string(), id as u32);
                        decoder.insert(id as u32, content.to_string());
                    }
                }
            }
        }

        let vocab_size = encoder.len();

        let special_ids: HashMap<u32, String> =
            special_tokens.iter().map(|(k, v)| (*v, k.clone())).collect();
        let mut special_patterns: Vec<(String, u32)> =
            special_tokens.iter().map(|(k, v)| (k.clone(), *v)).collect();
        // Greedy match: longest patterns first
        special_patterns.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // Detect tokenizer flavor.
        // SentencePiece signals (any one is enough):
        //   - model.byte_fallback == true
        //   - normalizer contains a Prepend with "▁" or a Replace " " -> "▁"
        //   - decoder contains a Replace "▁" -> " "
        let mut kind = TokenizerKind::GptByteLevel;
        if model.get("byte_fallback").and_then(|v| v.as_bool()).unwrap_or(false) {
            kind = TokenizerKind::SentencePiece;
        }
        if has_metaspace(root.get("normalizer")) || has_metaspace(root.get("decoder")) {
            kind = TokenizerKind::SentencePiece;
        }

        // Build byte-fallback table (`<0xXX>` ids) if SentencePiece.
        let mut byte_to_id = vec![None; 256];
        if kind == TokenizerKind::SentencePiece {
            for b in 0u8..=255 {
                let key = format!("<0x{:02X}>", b);
                if let Some(&id) = encoder.get(&key) {
                    byte_to_id[b as usize] = Some(id);
                }
            }
        }

        // Detect "Strip" decoder step (Llama strips one leading space after `▁` swap).
        let strip_leading_space = decoder_has_leading_strip(root.get("decoder"));

        Ok(Self {
            encoder,
            decoder,
            merge_ranks,
            special_tokens,
            special_ids,
            special_patterns,
            vocab_size,
            kind,
            byte_to_id,
            strip_leading_space,
        })
    }

    /// Which tokenizer flavor was loaded (auto-detected).
    pub fn kind(&self) -> TokenizerKind { self.kind }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        // First pass: split the input on special-token surface forms so the
        // BPE pipeline never sees them. Keeps `<|im_end|>` (etc.) intact even
        // though our regex pre-tokenizer would otherwise shred it.
        let mut tokens = Vec::new();
        for segment in self.split_on_special(text) {
            match segment {
                Segment::Special(id) => tokens.push(id),
                Segment::Text(s) => self.encode_no_special(s, &mut tokens),
            }
        }
        tokens
    }

    /// Greedy left-to-right scan that yields alternating Text/Special segments.
    fn split_on_special<'a>(&'a self, text: &'a str) -> Vec<Segment<'a>> {
        if self.special_patterns.is_empty() {
            return vec![Segment::Text(text)];
        }
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut text_start = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            // Try each special pattern (longest first) at position i.
            let mut hit: Option<(usize, u32)> = None;
            for (pat, id) in &self.special_patterns {
                let pb = pat.as_bytes();
                if i + pb.len() <= bytes.len() && &bytes[i..i + pb.len()] == pb {
                    hit = Some((pb.len(), *id));
                    break;
                }
            }
            if let Some((len, id)) = hit {
                if i > text_start {
                    out.push(Segment::Text(&text[text_start..i]));
                }
                out.push(Segment::Special(id));
                i += len;
                text_start = i;
            } else {
                // Advance by one full UTF-8 char (not byte) to keep slices valid.
                let ch_len = text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                i += ch_len;
            }
        }
        if text_start < bytes.len() {
            out.push(Segment::Text(&text[text_start..]));
        }
        out
    }

    /// BPE encoder for plain-text segments. Dispatches by kind.
    fn encode_no_special(&self, text: &str, out: &mut Vec<u32>) {
        match self.kind {
            TokenizerKind::GptByteLevel => self.encode_gpt_bytelevel(text, out),
            TokenizerKind::SentencePiece => self.encode_sentencepiece(text, out),
        }
    }

    fn encode_gpt_bytelevel(&self, text: &str, out: &mut Vec<u32>) {
        let words = self.pre_tokenize(text);
        for word in &words {
            if let Some(&id) = self.special_tokens.get(word.as_str()) {
                out.push(id);
                continue;
            }
            let symbols: Vec<String> = word
                .chars()
                .map(|c| self.char_to_byte_token(c))
                .collect();
            self.apply_merges_and_emit(symbols, out);
        }
    }

    /// SentencePiece BPE: prepend `▁`, replace spaces with `▁`, then BPE on
    /// the resulting string. Each unicode char becomes one initial symbol;
    /// chars not in the vocab are decomposed via byte fallback (`<0xXX>`).
    fn encode_sentencepiece(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() { return; }
        // Normalize: prepend ▁, replace spaces with ▁ (matches Llama normalizer).
        let mut normalized = String::with_capacity(text.len() + 4);
        normalized.push(SP_SPACE);
        for c in text.chars() {
            if c == ' ' { normalized.push(SP_SPACE); } else { normalized.push(c); }
        }
        // Initial symbols: one per char (as a string); BPE merges will fuse them.
        let symbols: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();
        self.apply_merges_and_emit(symbols, out);
    }

    /// Apply BPE merges to a sequence of starting symbols using the rank
    /// table, then emit token ids. Falls back to `<0xXX>` byte tokens for
    /// any leaf symbol not in the vocab.
    fn apply_merges_and_emit(&self, mut symbols: Vec<String>, out: &mut Vec<u32>) {
        if symbols.is_empty() { return; }

        // Standard rank-based BPE: at each step, find the merge pair with
        // the lowest rank (highest priority) and apply it across the sequence.
        // O(n²) per word in the worst case but fine for typical word lengths.
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, position)
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self.merge_ranks
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if best.map_or(true, |(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            match best {
                None => break,
                Some((_, pos)) => {
                    let merged = format!("{}{}", symbols[pos], symbols[pos + 1]);
                    symbols[pos] = merged;
                    symbols.remove(pos + 1);
                }
            }
        }

        for sym in &symbols {
            if let Some(&id) = self.encoder.get(sym) {
                out.push(id);
            } else if matches!(self.kind, TokenizerKind::SentencePiece) {
                // Byte fallback: emit `<0xXX>` token for each UTF-8 byte.
                for &b in sym.as_bytes() {
                    if let Some(id) = self.byte_to_id[b as usize] {
                        out.push(id);
                    }
                    // If the byte token isn't in vocab either, drop silently
                    // (matches HF tokenizers behavior with no <unk>).
                }
            }
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        match self.kind {
            TokenizerKind::GptByteLevel => self.decode_gpt_bytelevel(tokens),
            TokenizerKind::SentencePiece => self.decode_sentencepiece(tokens),
        }
    }

    fn decode_gpt_bytelevel(&self, tokens: &[u32]) -> String {
        // Group consecutive non-special tokens together, run them through the
        // byte-level inverse mapping in one pass, then emit special tokens as
        // their literal surface form (e.g. `<|im_end|>`). Without this split,
        // the byte mapper mangles `<`, `|`, `>` inside special-token strings.
        let mut out = String::new();
        let mut bpe_buf = String::new();
        for &id in tokens {
            if let Some(special) = self.special_ids.get(&id) {
                if !bpe_buf.is_empty() {
                    out.push_str(&self.byte_tokens_to_string(&bpe_buf));
                    bpe_buf.clear();
                }
                out.push_str(special);
            } else if let Some(piece) = self.decoder.get(&id) {
                bpe_buf.push_str(piece);
            }
        }
        if !bpe_buf.is_empty() {
            out.push_str(&self.byte_tokens_to_string(&bpe_buf));
        }
        out
    }

    fn decode_sentencepiece(&self, tokens: &[u32]) -> String {
        // SentencePiece decode pipeline (matches Llama tokenizer.json):
        //   1. Concatenate piece strings, BUT collect byte-fallback `<0xXX>`
        //      tokens into raw bytes so we can rebuild multibyte UTF-8.
        //   2. Replace `▁` with space.
        //   3. Optionally strip the first leading space (Llama convention).
        // Special tokens are emitted as their literal surface form.
        let mut out = String::new();
        let mut byte_run: Vec<u8> = Vec::new();
        let mut text_buf = String::new();

        let flush_byte_run = |run: &mut Vec<u8>, target: &mut String| {
            if !run.is_empty() {
                target.push_str(&String::from_utf8_lossy(run));
                run.clear();
            }
        };

        for &id in tokens {
            if let Some(special) = self.special_ids.get(&id) {
                flush_byte_run(&mut byte_run, &mut text_buf);
                out.push_str(&self.sp_finalize(&text_buf));
                text_buf.clear();
                out.push_str(special);
                continue;
            }
            // Byte-fallback token? Accumulate raw bytes.
            if let Some(piece) = self.decoder.get(&id) {
                if let Some(byte) = parse_byte_token(piece) {
                    byte_run.push(byte);
                } else {
                    flush_byte_run(&mut byte_run, &mut text_buf);
                    text_buf.push_str(piece);
                }
            }
        }
        flush_byte_run(&mut byte_run, &mut text_buf);
        out.push_str(&self.sp_finalize(&text_buf));
        out
    }

    fn sp_finalize(&self, s: &str) -> String {
        // NOTE: we deliberately do NOT apply the SentencePiece "Strip leading
        // space" decoder step here. That step is meant to clean up the
        // artifact from prepending `▁` during *encoding* of a full prompt — but
        // we're called per-token during streaming, where every token's leading
        // `▁` is a real space the model emitted between words. Stripping would
        // produce `isParis` instead of `is Paris`.
        //
        // Callers that decode a full token sequence in one shot AND need the
        // Llama-style leading-space strip can use `decode_full`.
        s.replace(SP_SPACE, " ")
    }

    /// Decode a token sequence as a single unit, applying any tokenizer-config
    /// post-processing (e.g. SentencePiece "Strip leading space"). Use this
    /// when decoding a complete prompt or response in one call. For per-token
    /// streaming, use `decode` instead.
    pub fn decode_full(&self, tokens: &[u32]) -> String {
        let mut s = self.decode(tokens);
        if self.kind == TokenizerKind::SentencePiece && self.strip_leading_space {
            if s.starts_with(' ') { s.remove(0); }
        }
        s
    }

    /// Returns true if the token id is a special token (e.g. `<|im_end|>`).
    /// Useful for stop-condition checks during generation.
    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains_key(&id)
    }

    /// Lookup the surface form of a special token by id, if any.
    pub fn special_str(&self, id: u32) -> Option<&str> {
        self.special_ids.get(&id).map(|s| s.as_str())
    }

    pub fn encode_special(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }

    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        PRE_TOKENIZE_RE.find_iter(text).map(|m| m.as_str().to_string()).collect()
    }

    fn char_to_byte_token(&self, c: char) -> String {
        let s = c.to_string();
        if self.encoder.contains_key(&s) {
            return s;
        }

        let mut buf = [0u8; 4];
        let bytes = c.encode_utf8(&mut buf).as_bytes();
        if bytes.len() == 1 {
            let byte_repr = byte_to_unicode(bytes[0]);
            return byte_repr.to_string();
        }

        bytes.iter().map(|&b| byte_to_unicode(b).to_string()).collect::<Vec<_>>().join("")
    }

    fn byte_tokens_to_string(&self, s: &str) -> String {
        let bytes: Vec<u8> = s.chars()
            .map(|c| unicode_to_byte(c).unwrap_or(c as u8))
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Parse a SentencePiece byte-fallback token like `"<0x1F>"` into the byte
/// value `0x1F`. Returns `None` for any other piece.
fn parse_byte_token(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() != 6 || &bytes[..3] != b"<0x" || bytes[5] != b'>' {
        return None;
    }
    u8::from_str_radix(&s[3..5], 16).ok()
}

/// Walk a tokenizer.json `normalizer` or `decoder` subtree looking for any
/// SentencePiece marker (Prepend/Replace involving `▁`, or a Metaspace step).
fn has_metaspace(v: Option<&serde_json::Value>) -> bool {
    let v = match v { Some(v) => v, None => return false };
    if let Some(t) = v.get("type").and_then(|t| t.as_str()) {
        match t {
            "Metaspace" => return true,
            "Prepend" => {
                if v.get("prepend").and_then(|s| s.as_str()) == Some("\u{2581}") {
                    return true;
                }
            }
            "Replace" => {
                let pat = v.get("pattern")
                    .and_then(|p| p.get("String").and_then(|s| s.as_str())
                        .or_else(|| p.as_str()));
                let content = v.get("content").and_then(|c| c.as_str());
                if pat == Some(" ") && content == Some("\u{2581}") { return true; }
                if pat == Some("\u{2581}") && content == Some(" ") { return true; }
            }
            _ => {}
        }
    }
    // Recurse into Sequence containers
    for key in &["normalizers", "decoders", "pretokenizers"] {
        if let Some(arr) = v.get(*key).and_then(|x| x.as_array()) {
            for child in arr {
                if has_metaspace(Some(child)) { return true; }
            }
        }
    }
    false
}

/// True if the decoder config has a Strip step that removes a leading space
/// (Llama's typical `{ "type": "Strip", "content": " ", "start": 1, "stop": 0 }`).
fn decoder_has_leading_strip(v: Option<&serde_json::Value>) -> bool {
    let v = match v { Some(v) => v, None => return false };
    if v.get("type").and_then(|t| t.as_str()) == Some("Strip") {
        let content = v.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let start = v.get("start").and_then(|s| s.as_u64()).unwrap_or(0);
        if content == " " && start >= 1 { return true; }
    }
    if let Some(arr) = v.get("decoders").and_then(|x| x.as_array()) {
        for child in arr {
            if decoder_has_leading_strip(Some(child)) { return true; }
        }
    }
    false
}

fn byte_to_unicode(b: u8) -> char {
    match b {
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => b as char,
        _ => char::from_u32(0x100 + b as u32).unwrap_or('?'),
    }
}

fn unicode_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    match cp {
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => Some(cp as u8),
        0x100..=0x1FF => Some((cp - 0x100) as u8),
        _ => None,
    }
}
