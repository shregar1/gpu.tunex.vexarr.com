use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use gpux_core::KernelProfiler;
use gpux_tensor::{GpuContext, Tensor};
use gpux_nn::llama::{LlamaModel, LlamaConfig};
use gpux_nn::kv_cache::KvCache;
use gpux_train::bpe::BpeTokenizer;
use gpux_train::safetensors::load_safetensors_dir;

const DEFAULT_MODEL: &str = "Qwen/Qwen2.5-0.5B";

struct Args {
    model_id: String,
    lora_path: Option<String>,
    prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
    quantize: Option<String>,
    no_cache: bool,
    debug: bool,
    /// Speculative draft size (PLD). 0 = disabled.
    spec_k: usize,
    /// PLD n-gram match length.
    spec_ngram: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        model_id: DEFAULT_MODEL.to_string(),
        lora_path: None,
        prompt: None,
        max_tokens: 50,
        temperature: 0.7,
        quantize: None,
        no_cache: false,
        debug: false,
        spec_k: 0,
        spec_ngram: 3,
    };

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" | "-m" => { i += 1; args.model_id = argv[i].clone(); }
            "--lora" | "-l" => { i += 1; args.lora_path = Some(argv[i].clone()); }
            "--prompt" | "-p" => { i += 1; args.prompt = Some(argv[i].clone()); }
            "--max-tokens" | "-n" => { i += 1; args.max_tokens = argv[i].parse().unwrap_or(50); }
            "--temperature" | "-t" => { i += 1; args.temperature = argv[i].parse().unwrap_or(0.7); }
            "--quantize" | "-q" => { i += 1; args.quantize = Some(argv[i].clone()); }
            "--no-cache" => { args.no_cache = true; }
            "--debug" => { args.debug = true; }
            "--spec-k" => { i += 1; args.spec_k = argv[i].parse().unwrap_or(0); }
            "--spec-ngram" => { i += 1; args.spec_ngram = argv[i].parse().unwrap_or(3); }
            other if !other.starts_with('-') && args.model_id == DEFAULT_MODEL => {
                args.model_id = other.to_string();
            }
            _ => {}
        }
        i += 1;
    }
    args
}

fn model_dir(model_id: &str) -> PathBuf {
    let safe_name = model_id.replace('/', "--");
    dirs().join(safe_name)
}

fn dirs() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("gpux").join("models")
}

fn download_model(model_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir = model_dir(model_id);
    if dir.exists() {
        let has_safetensors = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"));

        if has_safetensors {
            println!("Model already cached at {}", dir.display());
            return Ok(dir);
        }
    }
    std::fs::create_dir_all(&dir)?;

    let base_url = format!("https://huggingface.co/{}/resolve/main", model_id);

    let files = [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    ];

    for file in &files {
        let url = format!("{}/{}", base_url, file);
        let out_path = dir.join(file);

        if out_path.exists() {
            println!("  {} (cached)", file);
            continue;
        }

        println!("  Downloading {}...", file);
        let status = std::process::Command::new("curl")
            .args(["-L", "-o", out_path.to_str().unwrap(), &url, "--progress-bar", "-f"])
            .status()?;

        if !status.success() {
            let index_url = format!("{}/model.safetensors.index.json", base_url);
            if *file == "model.safetensors" {
                println!("  Single shard not found, trying sharded model...");
                let index_path = dir.join("model.safetensors.index.json");
                let status = std::process::Command::new("curl")
                    .args(["-L", "-o", index_path.to_str().unwrap(), &index_url, "--progress-bar", "-f"])
                    .status()?;
                if !status.success() {
                    return Err(format!("failed to download {}", file).into());
                }

                let index_data = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_data)?;
                let weight_map = index["weight_map"].as_object()
                    .ok_or("missing weight_map in index")?;

                let mut shard_files: Vec<String> = weight_map.values()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                shard_files.sort();
                shard_files.dedup();

                for shard in &shard_files {
                    let shard_path = dir.join(shard);
                    if shard_path.exists() {
                        println!("  {} (cached)", shard);
                        continue;
                    }
                    let shard_url = format!("{}/{}", base_url, shard);
                    println!("  Downloading {}...", shard);
                    let status = std::process::Command::new("curl")
                        .args(["-L", "-o", shard_path.to_str().unwrap(), &shard_url, "--progress-bar", "-f"])
                        .status()?;
                    if !status.success() {
                        return Err(format!("failed to download shard {}", shard).into());
                    }
                }
            } else {
                return Err(format!("failed to download {}", file).into());
            }
        }
    }

    Ok(dir)
}

fn load_config(dir: &Path) -> Result<LlamaConfig, Box<dyn std::error::Error>> {
    let config_path = dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_str)?;

    let head_dim = config.get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            let hidden = config["hidden_size"].as_u64().unwrap_or(1024) as usize;
            let heads = config["num_attention_heads"].as_u64().unwrap_or(16) as usize;
            hidden / heads
        });

    Ok(LlamaConfig {
        vocab_size: config["vocab_size"].as_u64().unwrap_or(151936) as usize,
        hidden_size: config["hidden_size"].as_u64().unwrap_or(1024) as usize,
        intermediate_size: config["intermediate_size"].as_u64().unwrap_or(3072) as usize,
        num_hidden_layers: config["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
        num_attention_heads: config["num_attention_heads"].as_u64().unwrap_or(16) as usize,
        num_key_value_heads: config["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        head_dim,
        rms_norm_eps: config.get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32,
        rope_theta: config.get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(1000000.0) as f32,
        max_position_embeddings: config.get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(32768) as usize,
        tie_word_embeddings: config.get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
    })
}

fn sample_token(logits_data: &[f32], vocab: usize, temperature: f32) -> u32 {
    use rand::Rng;
    let last_logits = &logits_data[logits_data.len() - vocab..];

    if temperature < 0.01 {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in last_logits.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        max_idx as u32
    } else {
        let max_val = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = last_logits.iter()
            .map(|&v| ((v - max_val) / temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        let probs: Vec<f32> = probs.iter().map(|&p| p / sum).collect();

        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        let mut chosen = 0u32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                chosen = i as u32;
                break;
            }
        }
        chosen
    }
}

/// Greedy argmax over a flat logits row.
fn argmax_row(logits: &[f32]) -> u32 {
    let mut max_idx = 0u32;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i as u32;
        }
    }
    max_idx
}

/// Prompt Lookup Decoding draft: search backwards through `tokens` for the most
/// recent occurrence of the trailing `n_gram` tokens. If found at position `p`,
/// propose `tokens[p+n_gram .. p+n_gram+draft_k]` as draft tokens.
///
/// Returns an empty vec if no match found or sequence is too short.
fn pld_draft(tokens: &[u32], n_gram: usize, draft_k: usize) -> Vec<u32> {
    if draft_k == 0 || tokens.len() < n_gram + 1 {
        return Vec::new();
    }
    let pos = tokens.len();
    let needle = &tokens[pos - n_gram..pos];

    // Search backwards (most recent earlier match wins) — that's typically the
    // most relevant one (e.g. immediately repeated structure in code/templates).
    for start in (0..pos - n_gram).rev() {
        if &tokens[start..start + n_gram] == needle {
            let after = start + n_gram;
            let available = pos - after;
            let take = available.min(draft_k);
            if take == 0 { return Vec::new(); }
            return tokens[after..after + take].to_vec();
        }
    }
    Vec::new()
}

fn is_eos(token: u32, tokenizer: &BpeTokenizer) -> bool {
    if let Some(eos) = tokenizer.encode_special("<|endoftext|>") {
        if token == eos { return true; }
    }
    if let Some(eos) = tokenizer.encode_special("<|im_end|>") {
        if token == eos { return true; }
    }
    false
}

fn debug_compare(
    model: &LlamaModel,
    ctx: &Arc<GpuContext>,
    tokenizer: &BpeTokenizer,
    prompt: &str,
) -> gpux_core::Result<()> {
    let prompt_tokens = tokenizer.encode(prompt);
    let vocab = model.config.vocab_size;

    eprintln!("\n=== DEBUG: Comparing cached vs uncached ===");
    eprintln!("Prompt tokens: {:?}", prompt_tokens);

    // Step 1: Uncached prefill (5 tokens → get logits for position 4)
    let input_5 = Tensor::from_uint_data(ctx, vec![1, prompt_tokens.len()], &prompt_tokens)?;
    let uncached_logits_5 = model.forward_lm(&input_5)?;
    let uncached_data_5 = uncached_logits_5.to_vec();
    let uncached_last_5 = &uncached_data_5[uncached_data_5.len() - vocab..];
    let uncached_token_0 = sample_token(&uncached_data_5, vocab, 0.0);
    eprintln!("Uncached prefill → token {} (\"{}\")", uncached_token_0, tokenizer.decode(&[uncached_token_0]));
    eprintln!("  logits[0..5]: {:?}", &uncached_last_5[..5]);

    // Step 2: Cached prefill (same 5 tokens)
    let max_seq = prompt_tokens.len() + 10;
    let mut cache = KvCache::new(ctx, &model.config, max_seq, 1)?;
    let cached_logits_5 = model.forward_lm_cached(&input_5, &mut cache)?;
    let cached_data_5 = cached_logits_5.to_vec();
    let cached_last_5 = &cached_data_5[cached_data_5.len() - vocab..];
    let cached_token_0 = sample_token(&cached_data_5, vocab, 0.0);
    eprintln!("Cached prefill  → token {} (\"{}\")", cached_token_0, tokenizer.decode(&[cached_token_0]));
    eprintln!("  logits[0..5]: {:?}", &cached_last_5[..5]);

    // Compare prefill logits
    let mut max_diff_prefill = 0.0f32;
    let mut max_diff_idx_prefill = 0;
    for i in 0..vocab {
        let diff = (uncached_last_5[i] - cached_last_5[i]).abs();
        if diff > max_diff_prefill {
            max_diff_prefill = diff;
            max_diff_idx_prefill = i;
        }
    }
    eprintln!("Prefill max logit diff: {:.6} at index {}", max_diff_prefill, max_diff_idx_prefill);

    // Step 3: Test with tight cache (max_seq = exact tokens)
    let next_token = uncached_token_0;
    let mut tokens_6 = prompt_tokens.clone();
    tokens_6.push(next_token);
    let input_6 = Tensor::from_uint_data(ctx, vec![1, tokens_6.len()], &tokens_6)?;
    let input_1 = Tensor::from_uint_data(ctx, vec![1, 1], &[next_token])?;

    // Test with tight cache (max_seq = 6)
    eprintln!("\n--- Test with tight cache (max_seq=6) ---");
    let mut tight_cache = KvCache::new(ctx, &model.config, 6, 1)?;
    let _ = model.forward_lm_cached(&input_5, &mut tight_cache)?;
    let tight_logits = model.forward_lm_cached(&input_1, &mut tight_cache)?;
    let tight_data = tight_logits.to_vec();
    let tight_token = sample_token(&tight_data, vocab, 0.0);
    eprintln!("Tight cache decode → token {} (\"{}\")", tight_token, tokenizer.decode(&[tight_token]));

    // Test with loose cache (max_seq = 15)
    eprintln!("\n--- Test with loose cache (max_seq=15) ---");
    let mut loose_cache = KvCache::new(ctx, &model.config, 15, 1)?;
    let _ = model.forward_lm_cached(&input_5, &mut loose_cache)?;
    let loose_logits = model.forward_lm_cached(&input_1, &mut loose_cache)?;
    let loose_data = loose_logits.to_vec();
    let loose_token = sample_token(&loose_data, vocab, 0.0);
    eprintln!("Loose cache decode → token {} (\"{}\")", loose_token, tokenizer.decode(&[loose_token]));

    // Compare
    let mut max_diff = 0.0f32;
    for i in 0..vocab {
        let diff = (tight_data[tight_data.len() - vocab + i] - loose_data[loose_data.len() - vocab + i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    eprintln!("Tight vs loose logit diff: {:.6}", max_diff);

    // Step 4: Now run both for final logits comparison
    let uncached_logits_6 = model.forward_lm(&input_6)?;
    let uncached_data_6 = uncached_logits_6.to_vec();
    let uncached_last_6 = &uncached_data_6[uncached_data_6.len() - vocab..];
    let uncached_token_1 = sample_token(&uncached_data_6, vocab, 0.0);
    eprintln!("\nUncached decode (6 tokens) → token {} (\"{}\")", uncached_token_1, tokenizer.decode(&[uncached_token_1]));

    let cached_logits_6 = model.forward_lm_cached(&input_1, &mut cache)?;
    let cached_data_6 = cached_logits_6.to_vec();
    let cached_last_6 = &cached_data_6[cached_data_6.len() - vocab..];
    let cached_token_1 = sample_token(&cached_data_6, vocab, 0.0);
    eprintln!("Cached decode (1 token)    → token {} (\"{}\")", cached_token_1, tokenizer.decode(&[cached_token_1]));

    // Compare decode logits
    let mut max_diff_decode = 0.0f32;
    let mut max_diff_idx_decode = 0;
    for i in 0..vocab {
        let diff = (uncached_last_6[i] - cached_last_6[i]).abs();
        if diff > max_diff_decode {
            max_diff_decode = diff;
            max_diff_idx_decode = i;
        }
    }
    eprintln!("Decode max logit diff: {:.6} at index {}", max_diff_decode, max_diff_idx_decode);

    // Top-5 tokens from each
    let mut uncached_sorted: Vec<(usize, f32)> = uncached_last_6.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let mut cached_sorted: Vec<(usize, f32)> = cached_last_6.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    uncached_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    cached_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\nUncached top-5:");
    for (i, v) in &uncached_sorted[..5] {
        eprintln!("  {} ({:.4}) = \"{}\"", i, v, tokenizer.decode(&[*i as u32]));
    }
    eprintln!("Cached top-5:");
    for (i, v) in &cached_sorted[..5] {
        eprintln!("  {} ({:.4}) = \"{}\"", i, v, tokenizer.decode(&[*i as u32]));
    }

    eprintln!("=== END DEBUG ===\n");
    Ok(())
}

fn generate_cached(
    model: &LlamaModel,
    ctx: &Arc<GpuContext>,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> gpux_core::Result<String> {
    let prompt_tokens = tokenizer.encode(prompt);
    let max_seq = prompt_tokens.len() + max_tokens;
    ctx.device.reset_scratch();
    let mut cache = KvCache::new(ctx, &model.config, max_seq, 1)?;
    let vocab = model.config.vocab_size;

    print!("{}", prompt);
    std::io::stdout().flush().ok();

    let prompt_tokens_len = prompt_tokens.len();
    let prefill_start = Instant::now();
    let input = Tensor::from_uint_data(ctx, vec![1, prompt_tokens.len()], &prompt_tokens)?;
    let logits = model.forward_lm_cached(&input, &mut cache)?;
    let logits_data = logits.to_vec();
    ctx.device.reset_scratch();
    let mut next_token = sample_token(&logits_data, vocab, temperature);
    let prefill_ms = prefill_start.elapsed().as_millis();

    let mut all_tokens = prompt_tokens;

    let decode_start = Instant::now();
    for _ in 0..max_tokens {
        all_tokens.push(next_token);
        let piece = tokenizer.decode(&[next_token]);
        print!("{}", piece);
        std::io::stdout().flush().ok();

        if is_eos(next_token, tokenizer) { break; }

        ctx.device.reset_scratch();
        let input = Tensor::from_uint_data(ctx, vec![1, 1], &[next_token])?;
        let logits = model.forward_lm_cached(&input, &mut cache)?;
        let logits_data = logits.to_vec();
        if std::env::var("GPUX_DEBUG_SCRATCH").is_ok() {
            eprintln!("  [scratch] used: {} KB", ctx.device.scratch_used() / 1024);
        }
        next_token = sample_token(&logits_data, vocab, temperature);
    }
    let decode_ms = decode_start.elapsed().as_millis();
    let decode_tokens = all_tokens.len() - prompt_tokens_len;
    println!();
    eprintln!("  prefill: {}ms, decode: {} tokens in {}ms ({:.1} tok/s)",
        prefill_ms, decode_tokens, decode_ms,
        decode_tokens as f64 / (decode_ms as f64 / 1000.0));

    Ok(tokenizer.decode(&all_tokens))
}

/// Adaptive PLD scheduler. Tracks rolling acceptance rate over the last N verify
/// calls and decides whether to attempt spec at all. With our fast decode baseline
/// (~190 tok/s on Q4_0), spec only wins when expected accepted tokens per verify
/// exceeds verify_cost / decode_cost ratio (empirically ~3 for K=4 → accept ≥ 0.65).
///
/// Strategy:
///   • Always try spec for the first `WARMUP` PLD-eligible candidates
///   • After warmup: if rolling accept rate < SKIP_THRESHOLD, skip spec entirely
///     (just run base decode for the next `COOLDOWN` tokens, then re-evaluate)
struct PldScheduler {
    recent_accepts: std::collections::VecDeque<f32>,
    cooldown_remaining: usize,
    enabled_calls: u64,
    skipped_calls: u64,
}

impl PldScheduler {
    const WINDOW: usize = 8;          // rolling window of last 8 verify calls
    const WARMUP: usize = 3;          // try spec at least 3 times before evaluating
    const SKIP_THRESHOLD: f32 = 0.55; // skip spec if recent accept rate below this
    const COOLDOWN: usize = 16;       // tokens to wait before re-trying spec

    fn new() -> Self {
        Self {
            recent_accepts: std::collections::VecDeque::with_capacity(Self::WINDOW),
            cooldown_remaining: 0,
            enabled_calls: 0,
            skipped_calls: 0,
        }
    }

    fn should_attempt(&mut self) -> bool {
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            self.skipped_calls += 1;
            return false;
        }
        if self.recent_accepts.len() < Self::WARMUP {
            return true;
        }
        let avg: f32 =
            self.recent_accepts.iter().sum::<f32>() / self.recent_accepts.len() as f32;
        if avg < Self::SKIP_THRESHOLD {
            self.cooldown_remaining = Self::COOLDOWN;
            self.skipped_calls += 1;
            false
        } else {
            true
        }
    }

    fn record(&mut self, accept_fraction: f32) {
        self.enabled_calls += 1;
        if self.recent_accepts.len() == Self::WINDOW {
            self.recent_accepts.pop_front();
        }
        self.recent_accepts.push_back(accept_fraction);
    }
}

/// Speculative decoding via Prompt Lookup (greedy / temperature=0 only).
/// Each step: find n-gram from earlier in the sequence, propose K draft tokens,
/// verify with one multi-token forward pass, accept the longest matching prefix
/// + the model's correction token. KV cache is rewound to discard rejected entries.
fn generate_speculative(
    model: &LlamaModel,
    ctx: &Arc<GpuContext>,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    max_tokens: usize,
    spec_k: usize,
    spec_ngram: usize,
) -> gpux_core::Result<String> {
    let prompt_tokens = tokenizer.encode(prompt);
    let max_seq = prompt_tokens.len() + max_tokens + spec_k + 8;
    ctx.device.reset_scratch();
    let mut cache = KvCache::new(ctx, &model.config, max_seq, 1)?;
    let vocab = model.config.vocab_size;

    print!("{}", prompt);
    std::io::stdout().flush().ok();

    let prompt_tokens_len = prompt_tokens.len();
    let prefill_start = Instant::now();
    let input = Tensor::from_uint_data(ctx, vec![1, prompt_tokens.len()], &prompt_tokens)?;
    let logits = model.forward_lm_cached(&input, &mut cache)?;
    let logits_data = logits.to_vec();
    ctx.device.reset_scratch();
    // logits has shape [prompt_len, vocab]. We want the LAST row to predict the next token.
    let mut next_token = argmax_row(&logits_data[(prompt_tokens.len() - 1) * vocab..]);
    let prefill_ms = prefill_start.elapsed().as_millis();

    let mut all_tokens = prompt_tokens;
    let mut accepted_total: usize = 0;
    let mut spec_calls: usize = 0;
    let mut spec_drafts_offered: usize = 0;
    let mut spec_drafts_accepted: usize = 0;
    let mut sched = PldScheduler::new();

    let decode_start = Instant::now();
    while all_tokens.len() - prompt_tokens_len < max_tokens {
        // Push the token we're about to feed and emit it to stdout.
        all_tokens.push(next_token);
        let piece = tokenizer.decode(&[next_token]);
        print!("{}", piece);
        std::io::stdout().flush().ok();
        accepted_total += 1;

        if is_eos(next_token, tokenizer) { break; }
        if all_tokens.len() - prompt_tokens_len >= max_tokens { break; }

        // PLD only when scheduler thinks it's worth attempting.
        let drafts = if sched.should_attempt() {
            pld_draft(&all_tokens, spec_ngram, spec_k)
        } else {
            Vec::new()
        };

        if drafts.is_empty() {
            // No draft hit — fall back to single-token decode.
            ctx.device.reset_scratch();
            let input = Tensor::from_uint_data(ctx, vec![1, 1], &[next_token])?;
            let logits = model.forward_lm_cached(&input, &mut cache)?;
            let logits_data = logits.to_vec();
            next_token = argmax_row(&logits_data);
        } else {
            // Verify pass: feed [next_token, draft_0, draft_1, ..., draft_{K-1}].
            // Logits row i predicts the token at position (current_pos + i + 1).
            let pos_before = cache.current_pos;
            let mut verify = Vec::with_capacity(1 + drafts.len());
            verify.push(next_token);
            verify.extend_from_slice(&drafts);

            ctx.device.reset_scratch();
            let input = Tensor::from_uint_data(ctx, vec![1, verify.len()], &verify)?;
            let logits = model.forward_lm_cached(&input, &mut cache)?;
            let logits_data = logits.to_vec();
            // logits shape: [verify.len(), vocab]

            spec_calls += 1;
            spec_drafts_offered += drafts.len();

            // Walk the matches in lockstep: row i predicts what should follow
            // verify[i]. Accept while predictions match the proposed drafts.
            let mut accept = 0usize;
            for i in 0..drafts.len() {
                let pred = argmax_row(&logits_data[i * vocab..(i + 1) * vocab]);
                if pred == drafts[i] {
                    accept += 1;
                } else {
                    break;
                }
            }
            spec_drafts_accepted += accept;
            // Feed acceptance fraction to scheduler so it can skip future spec
            // attempts on novel/non-repetitive content.
            sched.record(accept as f32 / drafts.len() as f32);

            // Emit accepted drafts. (We've already pushed `next_token` above.)
            for i in 0..accept {
                all_tokens.push(drafts[i]);
                let piece = tokenizer.decode(&[drafts[i]]);
                print!("{}", piece);
                accepted_total += 1;
                if is_eos(drafts[i], tokenizer) {
                    // Don't continue past EOS even within a verified batch.
                    std::io::stdout().flush().ok();
                    let decode_ms = decode_start.elapsed().as_millis();
                    eprintln!(
                        "\n  prefill: {}ms, decode: {} tokens in {}ms ({:.1} tok/s) | spec: {}/{} drafts accepted ({:.0}%) over {} verify calls",
                        prefill_ms, accepted_total, decode_ms,
                        accepted_total as f64 / (decode_ms as f64 / 1000.0),
                        spec_drafts_accepted, spec_drafts_offered,
                        if spec_drafts_offered > 0 { 100.0 * spec_drafts_accepted as f64 / spec_drafts_offered as f64 } else { 0.0 },
                        spec_calls);
                    return Ok(tokenizer.decode(&all_tokens));
                }
            }
            std::io::stdout().flush().ok();

            // The model's predicted next token at the boundary (after the
            // accepted prefix) becomes the next `next_token` to feed.
            next_token = argmax_row(&logits_data[accept * vocab..(accept + 1) * vocab]);

            // Rewind the KV cache: keep entries for [next_token, drafts[0..accept]] = accept+1
            // positions past `pos_before`. The remaining entries from the verify pass become
            // stale and will be overwritten on the next dispatch.
            cache.current_pos = pos_before + accept + 1;
        }
    }

    let decode_ms = decode_start.elapsed().as_millis();
    println!();
    let acc_pct = if spec_drafts_offered > 0 {
        100.0 * spec_drafts_accepted as f64 / spec_drafts_offered as f64
    } else { 0.0 };
    eprintln!(
        "  [adaptive PLD: enabled {} verify calls, skipped {} (low recent accept rate)]",
        sched.enabled_calls, sched.skipped_calls);
    eprintln!(
        "  prefill: {}ms, decode: {} tokens in {}ms ({:.1} tok/s) | spec: {}/{} drafts accepted ({:.0}%) over {} verify calls",
        prefill_ms, accepted_total, decode_ms,
        accepted_total as f64 / (decode_ms as f64 / 1000.0),
        spec_drafts_accepted, spec_drafts_offered, acc_pct, spec_calls);

    Ok(tokenizer.decode(&all_tokens))
}

fn generate_uncached(
    model: &LlamaModel,
    ctx: &Arc<GpuContext>,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> gpux_core::Result<String> {
    let mut tokens = tokenizer.encode(prompt);
    let vocab = model.config.vocab_size;

    print!("{}", prompt);
    std::io::stdout().flush().ok();

    for _ in 0..max_tokens {
        let seq_len = tokens.len();
        let input = Tensor::from_uint_data(ctx, vec![1, seq_len], &tokens)?;
        let logits = model.forward_lm(&input)?;
        let logits_data = logits.to_vec();
        ctx.device.reset_scratch();
        let next_token = sample_token(&logits_data, vocab, temperature);

        tokens.push(next_token);
        let piece = tokenizer.decode(&[next_token]);
        print!("{}", piece);
        std::io::stdout().flush().ok();

        if is_eos(next_token, tokenizer) { break; }
    }
    println!();

    Ok(tokenizer.decode(&tokens))
}

fn load_lora_config(path: &Path) -> Result<(usize, f32), Box<dyn std::error::Error>> {
    let config_path = path.join("adapter_config.json");
    let data = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&data)?;
    let rank = config["r"].as_u64().unwrap_or(8) as usize;
    let alpha = config["lora_alpha"].as_f64().unwrap_or(16.0) as f32;
    Ok((rank, alpha))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = parse_args();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║     GPUX - Apple Silicon SLM Inference            ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    println!("Model: {}", args.model_id);
    if let Some(ref lora) = args.lora_path {
        println!("LoRA:  {}", lora);
    }
    if let Some(ref q) = args.quantize {
        println!("Quant: {}", q);
    }
    println!("Cache: {}", if args.no_cache { "disabled" } else { "enabled" });

    println!("\n[1/5] Downloading model...");
    let model_dir = download_model(&args.model_id)?;

    println!("\n[2/5] Loading config and tokenizer...");
    let config = load_config(&model_dir)?;
    println!("  Architecture: {}L/{}H({}KV)/{}D",
        config.num_hidden_layers, config.num_attention_heads,
        config.num_key_value_heads, config.hidden_size);
    println!("  Vocab: {}, Head dim: {}", config.vocab_size, config.head_dim);

    let tokenizer = BpeTokenizer::from_file(&model_dir.join("tokenizer.json"))?;
    println!("  Tokenizer loaded ({} tokens)", tokenizer.vocab_size());

    println!("\n[3/5] Loading model weights...");
    let ctx = GpuContext::new()?;
    println!("  GPU: {}", ctx.device.name());

    let load_start = Instant::now();
    let mut model = LlamaModel::new(&ctx, config)?;
    println!("  Model initialized: {} params", model.parameter_count());

    let mut weights = load_safetensors_dir(&model_dir, &ctx)?;
    println!("  Loaded {} tensors from safetensors", weights.len());

    if let Some(ref quant) = args.quantize {
        println!("  Quantizing weights to {}...", quant);
        let mut quantized = 0;
        let keys: Vec<String> = weights.keys().cloned().collect();
        for name in keys {
            let t = &weights[&name];
            // Skip embedding for Q4_K (no embedding_forward_q4_k kernel yet).
            let is_embed = name.contains("embed");
            let is_q4k_embed_skip = matches!(quant.as_str(), "q4_k" | "Q4_K" | "q4_k_m" | "Q4_K_M") && is_embed;
            if name.contains(".weight") && t.shape().ndim() == 2
                && !name.contains("norm")
                && !is_q4k_embed_skip
                && t.numel() % 32 == 0
            {
                let q = match quant.as_str() {
                    "q8_0" | "Q8_0" => t.quantize_q8_0()?,
                    "q4_0" | "Q4_0" => t.quantize_q4_0()?,
                    // Q4_K_M requires last dim divisible by 256; Qwen2.5-0.5B has
                    // hidden=896 which IS divisible by 32 but NOT by 256, so the
                    // Q-K-M kernel can't quantize many of its weights. We skip
                    // those instead of erroring — they stay at f16, which is
                    // strictly fine but means partial quantization.
                    "q4_k" | "Q4_K" | "q4_k_m" | "Q4_K_M" => {
                        if t.numel() % 256 != 0 { continue; }
                        t.quantize_q4_k()?
                    }
                    _ => continue,
                };
                weights.insert(name, q);
                quantized += 1;
            }
        }
        println!("  Quantized {} weight matrices", quantized);
    }

    model.load_weights(&weights)?;
    ctx.queue.synchronize()?;

    drop(weights);
    println!("  Weights loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    if let Some(ref lora_path) = args.lora_path {
        println!("\n[4/5] Loading LoRA adapter...");
        let lora_dir = Path::new(lora_path);
        let (rank, alpha) = load_lora_config(lora_dir)?;
        println!("  LoRA rank={}, alpha={}", rank, alpha);

        let lora_weights = load_safetensors_dir(lora_dir, &ctx)?;
        println!("  Loaded {} LoRA tensors", lora_weights.len());
        let n = model.load_lora_weights(&lora_weights, rank, alpha)?;
        println!("  Applied LoRA to {} layers", n);
    } else {
        println!("\n[4/5] No LoRA adapter (use --lora <path> to load one)");
    }

    println!("\n[5/5] Verifying tokenizer...");
    {
        let test = "The capital of France is";
        let ids = tokenizer.encode(test);
        println!("  \"{}\" -> {:?}", test, ids);
        let expected = vec![785u32, 6722, 315, 9625, 374];
        if ids == expected {
            println!("  Tokenizer OK");
        } else {
            println!("  WARNING: expected {:?}", expected);
        }
    }

    if args.debug {
        let test_prompt = args.prompt.as_deref().unwrap_or("The capital of France is");
        debug_compare(&model, &ctx, &tokenizer, test_prompt)?;
    }

    // GPUX_PROFILE=1: serialize every dispatch via per-kernel sync + timing.
    // Slows decode dramatically but reveals exact GPU time per kernel.
    let profiling = std::env::var("GPUX_PROFILE").is_ok();
    if profiling {
        // Warm up once (compile pipelines, warm caches) before flipping the flag,
        // so first-call overhead doesn't pollute the profile.
        let warm_input = Tensor::from_uint_data(&ctx, vec![1, 1], &[0u32])?;
        let mut warm_cache = gpux_nn::kv_cache::KvCache::new(&ctx, &model.config, 8, 1)?;
        let _ = model.forward_lm_cached(&warm_input, &mut warm_cache)?;
        ctx.queue.synchronize()?;
        ctx.device.reset_scratch();
        eprintln!("[profile] enabled — every dispatch will be sync'd individually");
        KernelProfiler::global().enable();
        KernelProfiler::global().reset();
    }

    // Activate scratch allocator for decode — eliminates Metal buffer allocation overhead
    ctx.device.activate_scratch();

    println!("\n{}", "─".repeat(60));
    println!("Generating text...\n");

    let prompts: Vec<&str> = if let Some(ref p) = args.prompt {
        vec![p.as_str()]
    } else {
        vec![
            "The capital of France is",
            "def fibonacci(n):\n",
            "Once upon a time",
        ]
    };

    for prompt in &prompts {
        println!("Prompt: \"{}\"", prompt);
        print!("Output: ");
        let start = Instant::now();
        if args.no_cache {
            let _ = generate_uncached(&model, &ctx, &tokenizer, prompt, args.max_tokens, args.temperature)?;
        } else if args.spec_k > 0 {
            // Speculative decoding is greedy-only. Force temperature=0 (warn if user set otherwise).
            if args.temperature >= 0.01 {
                eprintln!("  [warn] --spec-k forces greedy decoding (temperature ignored)");
            }
            let _ = generate_speculative(&model, &ctx, &tokenizer, prompt,
                args.max_tokens, args.spec_k, args.spec_ngram)?;
        } else {
            let _ = generate_cached(&model, &ctx, &tokenizer, prompt, args.max_tokens, args.temperature)?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!("  ({:.1}s, {:.1} tok/s)", elapsed, args.max_tokens as f64 / elapsed);
        println!("{}", "─".repeat(60));
    }

    if profiling {
        KernelProfiler::global().print_report("Per-kernel GPU time");
    }

    Ok(())
}
