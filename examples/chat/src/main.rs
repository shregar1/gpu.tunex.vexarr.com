//! `gpux chat` — a terminal chat REPL backed directly by the model
//! (no server hop). Maintains turn history, applies the Qwen2.5 ChatML
//! template, streams tokens to the terminal as they're generated.
//!
//! Usage:
//!   cargo run --release -p chat-example
//!   cargo run --release -p chat-example -- --model Qwen/Qwen2.5-1.5B-Instruct
//!   cargo run --release -p chat-example -- --no-quantize
//!
//! Inside the REPL:
//!   /reset     start a new conversation
//!   /system X  set/replace the system prompt
//!   /tokens N  cap response at N tokens (default 512)
//!   /temp F    set temperature (0.0 = greedy, default 0.7)
//!   /quit      exit
//!
//! Plain text → user message; assistant response streams back.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use gpux_nn::kv_cache::KvCache;
use gpux_nn::llama::{LlamaConfig, LlamaModel};
use gpux_tensor::{GpuContext, Tensor};
use gpux_train::bpe::BpeTokenizer;
use gpux_train::hub;
use gpux_train::safetensors::load_safetensors_dir;
use serde_json::Value;

const DEFAULT_MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";

// ============================================================================
// Model loading (mirrors the inference example, kept here so chat is standalone)
// ============================================================================

fn download_model(model_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Centralized in gpux_train::hub — validates Content-Length on every file
    // so a flaky network can't leave a truncated weight file on disk.
    Ok(hub::download_model(model_id, true)?)
}

fn load_config(dir: &Path) -> Result<LlamaConfig, Box<dyn std::error::Error>> {
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let head_dim = cfg.get("head_dim").and_then(|v| v.as_u64()).map(|v| v as usize)
        .unwrap_or_else(|| {
            (cfg["hidden_size"].as_u64().unwrap_or(1024)
                / cfg["num_attention_heads"].as_u64().unwrap_or(16)) as usize
        });
    Ok(LlamaConfig {
        vocab_size: cfg["vocab_size"].as_u64().unwrap_or(151936) as usize,
        hidden_size: cfg["hidden_size"].as_u64().unwrap_or(1024) as usize,
        intermediate_size: cfg["intermediate_size"].as_u64().unwrap_or(3072) as usize,
        num_hidden_layers: cfg["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
        num_attention_heads: cfg["num_attention_heads"].as_u64().unwrap_or(16) as usize,
        num_key_value_heads: cfg["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        head_dim,
        rms_norm_eps: cfg.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
        rope_theta: cfg.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(1_000_000.0) as f32,
        max_position_embeddings: cfg.get("max_position_embeddings")
            .and_then(|v| v.as_u64()).unwrap_or(32768) as usize,
        tie_word_embeddings: cfg.get("tie_word_embeddings")
            .and_then(|v| v.as_bool()).unwrap_or(true),
    })
}

// ============================================================================
// Chat template selection + simple turn history
// ============================================================================

#[derive(Clone)]
struct Message {
    role: String,
    content: String,
}

/// Which chat template to use. Auto-detected from tokenizer special tokens at
/// load time; can be overridden via `--template`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChatTemplate {
    /// `<|im_start|>{role}\n{content}<|im_end|>\n` — Qwen / SmolLM2
    ChatML,
    /// `<|user|>\n{content}</s>\n<|assistant|>\n…` — Zephyr / TinyLlama / many open chat models
    Zephyr,
    /// `[INST] {content} [/INST] {response}</s>` — Llama-2 base style
    LlamaInst,
}

/// Detect the chat template by string-matching markers in
/// `tokenizer_config.json`'s `chat_template` field — much more reliable than
/// inspecting special tokens (TinyLlama uses `<|user|>` as plain text, not
/// as a special token, so a special-tokens-only check would mis-classify it).
///
/// Falls back to inspecting special tokens, then ChatML as the universal default.
fn detect_template(tok: &BpeTokenizer, model_dir: &Path) -> ChatTemplate {
    let cfg_path = model_dir.join("tokenizer_config.json");
    if let Ok(s) = std::fs::read_to_string(&cfg_path) {
        if let Ok(v) = serde_json::from_str::<Value>(&s) {
            if let Some(template) = v.get("chat_template").and_then(|t| t.as_str()) {
                if template.contains("<|im_start|>") {
                    return ChatTemplate::ChatML;
                }
                if template.contains("<|user|>") || template.contains("<|assistant|>") {
                    return ChatTemplate::Zephyr;
                }
                if template.contains("[INST]") {
                    return ChatTemplate::LlamaInst;
                }
            }
        }
    }
    // Fallback: special-token presence
    if tok.encode_special("<|im_start|>").is_some() && tok.encode_special("<|im_end|>").is_some() {
        return ChatTemplate::ChatML;
    }
    if tok.encode_special("<|user|>").is_some() && tok.encode_special("<|assistant|>").is_some() {
        return ChatTemplate::Zephyr;
    }
    ChatTemplate::ChatML
}

fn build_prompt(
    template: ChatTemplate,
    system: Option<&str>,
    history: &[Message],
    pending_user: &str,
) -> String {
    let mut s = String::new();
    match template {
        ChatTemplate::ChatML => {
            if let Some(sys) = system {
                s.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
            }
            for msg in history {
                s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
            }
            s.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", pending_user));
            s.push_str("<|im_start|>assistant\n");
        }
        ChatTemplate::Zephyr => {
            if let Some(sys) = system {
                s.push_str(&format!("<|system|>\n{}</s>\n", sys));
            }
            for msg in history {
                let tag = if msg.role == "assistant" { "assistant" } else { "user" };
                s.push_str(&format!("<|{}|>\n{}</s>\n", tag, msg.content));
            }
            s.push_str(&format!("<|user|>\n{}</s>\n", pending_user));
            s.push_str("<|assistant|>\n");
        }
        ChatTemplate::LlamaInst => {
            // Llama-2 chat template (single system, then turns):
            //   <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_1} [/INST] {asst_1} </s>
            //   <s>[INST] {user_2} [/INST] {asst_2} </s>
            for (i, msg) in history.chunks(2).enumerate() {
                let user = msg.first().map(|m| m.content.as_str()).unwrap_or("");
                let asst = msg.get(1).map(|m| m.content.as_str()).unwrap_or("");
                if i == 0 {
                    if let Some(sys) = system {
                        s.push_str(&format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST] {} </s>", sys, user, asst));
                    } else {
                        s.push_str(&format!("[INST] {} [/INST] {} </s>", user, asst));
                    }
                } else {
                    s.push_str(&format!("<s>[INST] {} [/INST] {} </s>", user, asst));
                }
            }
            // Pending turn
            if history.is_empty() {
                if let Some(sys) = system {
                    s.push_str(&format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]", sys, pending_user));
                } else {
                    s.push_str(&format!("[INST] {} [/INST]", pending_user));
                }
            } else {
                s.push_str(&format!("<s>[INST] {} [/INST]", pending_user));
            }
        }
    }
    s
}

// ============================================================================
// Generation (streaming to stdout) + sampling
// ============================================================================

fn argmax_row(logits: &[f32]) -> u32 {
    let mut best = (0u32, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 { best = (i as u32, v); }
    }
    best.0
}

/// Tiny temperature sampler — deterministic-ish via OS time for jitter so we
/// don't pull in `rand` for a demo binary.
fn sample_temp(logits: &[f32], temperature: f32) -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|&v| ((v - max) / temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut h = DefaultHasher::new();
    now.hash(&mut h);
    let r = (h.finish() as f64 / u64::MAX as f64) as f32 * sum;
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc { return i as u32; }
    }
    (probs.len() - 1) as u32
}

fn is_end(token: u32, tok: &BpeTokenizer) -> bool {
    if !tok.is_special(token) { return false; }
    matches!(tok.special_str(token),
        Some("<|im_end|>")        // ChatML
      | Some("<|endoftext|>")     // GPT-style
      | Some("<|end_of_text|>")   // Llama-3
      | Some("<|eot_id|>")        // Llama-3 turn end
      | Some("</s>"))             // Llama-2 / Zephyr / TinyLlama
}

struct Engine {
    model: LlamaModel,
    tokenizer: BpeTokenizer,
    ctx: Arc<GpuContext>,
    model_dir: PathBuf,
}

impl Engine {
    fn load(model_id: &str, quantize: Option<&str>) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = download_model(model_id)?;
        let config = load_config(&dir)?;
        eprintln!("[chat] model: {}L/{}H({}KV)/{}D, vocab={}",
            config.num_hidden_layers, config.num_attention_heads,
            config.num_key_value_heads, config.hidden_size, config.vocab_size);

        let tokenizer = BpeTokenizer::from_file(&dir.join("tokenizer.json"))?;
        let ctx = GpuContext::new()?;
        eprintln!("[chat] gpu: {}", ctx.device.name());

        let mut model = LlamaModel::new(&ctx, config)?;
        let mut weights = load_safetensors_dir(&dir, &ctx)?;

        if let Some(q) = quantize {
            eprintln!("[chat] quantizing weights to {}…", q);
            let keys: Vec<String> = weights.keys().cloned().collect();
            for name in keys {
                let t = &weights[&name];
                if name.contains(".weight") && t.shape().ndim() == 2
                    && !name.contains("norm") && t.numel() % 32 == 0
                {
                    // Skip embedding for Q4_K (no embedding_forward_q4_k kernel yet).
                    if matches!(q, "q4_k" | "Q4_K" | "q4_k_m" | "Q4_K_M") && name.contains("embed") {
                        continue;
                    }
                    let qt = match q {
                        "q4_0" | "Q4_0" => t.quantize_q4_0()?,
                        "q8_0" | "Q8_0" => t.quantize_q8_0()?,
                        "q4_k" | "Q4_K" | "q4_k_m" | "Q4_K_M" => {
                            if t.numel() % 256 != 0 { continue; }
                            t.quantize_q4_k()?
                        }
                        _ => continue,
                    };
                    weights.insert(name, qt);
                }
            }
        }

        model.load_weights(&weights)?;
        ctx.queue.synchronize()?;
        drop(weights);
        ctx.device.activate_scratch();

        Ok(Self { model, tokenizer, ctx, model_dir: dir })
    }

    /// Generate a response, streaming tokens to stdout. Returns (text, n_tokens, elapsed).
    fn complete(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<(String, usize, f64), Box<dyn std::error::Error>> {
        let prompt_tokens = self.tokenizer.encode(prompt);
        let prompt_len = prompt_tokens.len();
        let max_seq = prompt_len + max_tokens + 8;
        self.ctx.device.reset_scratch();
        let mut cache = KvCache::new(&self.ctx, &self.model.config, max_seq, 1)?;
        let vocab = self.model.config.vocab_size;

        let t0 = Instant::now();
        let input = Tensor::from_uint_data(&self.ctx, vec![1, prompt_len], &prompt_tokens)?;
        let logits = self.model.forward_lm_cached(&input, &mut cache)?;
        let logits_data = logits.to_vec();
        self.ctx.device.reset_scratch();
        let last = &logits_data[(prompt_len - 1) * vocab..];
        let mut next = if temperature < 0.01 {
            argmax_row(last)
        } else {
            sample_temp(last, temperature)
        };

        let mut text = String::new();
        let mut produced = 0usize;
        let mut stdout = io::stdout().lock();

        for _ in 0..max_tokens {
            if is_end(next, &self.tokenizer) { break; }
            if !self.tokenizer.is_special(next) {
                let piece = self.tokenizer.decode(&[next]);
                text.push_str(&piece);
                let _ = stdout.write_all(piece.as_bytes());
                let _ = stdout.flush();
            }
            produced += 1;

            self.ctx.device.reset_scratch();
            let inp = Tensor::from_uint_data(&self.ctx, vec![1, 1], &[next])?;
            let logits = self.model.forward_lm_cached(&inp, &mut cache)?;
            let ld = logits.to_vec();
            next = if temperature < 0.01 { argmax_row(&ld) } else { sample_temp(&ld, temperature) };
        }
        let _ = stdout.write_all(b"\n");
        let elapsed = t0.elapsed().as_secs_f64();
        Ok((text, produced, elapsed))
    }
}

// ============================================================================
// REPL loop
// ============================================================================

fn print_banner() {
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│  gpux chat — Apple Silicon native LLM REPL              │");
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│  /reset    new conversation                             │");
    println!("│  /system X set/replace system prompt                    │");
    println!("│  /temp F   sampling temperature (0 = greedy)            │");
    println!("│  /tokens N max response length                          │");
    println!("│  /quit     exit                                         │");
    println!("└─────────────────────────────────────────────────────────┘");
}

fn run_repl(engine: Engine) -> Result<(), Box<dyn std::error::Error>> {
    print_banner();

    let template = detect_template(&engine.tokenizer, &engine.model_dir);
    eprintln!("[chat] tokenizer: {:?} | template: {:?}", engine.tokenizer.kind(), template);

    let mut system: Option<String> = Some(
        "You are a helpful assistant. Be concise.".to_string(),
    );
    let mut history: Vec<Message> = Vec::new();
    let mut max_tokens: usize = 512;
    let mut temperature: f32 = 0.7;

    let stdin = io::stdin();
    let mut line = String::new();
    loop {
        print!("\n> ");
        io::stdout().flush().ok();
        line.clear();
        let n = stdin.read_line(&mut line)?;
        if n == 0 { println!(); break; }   // EOF
        let cmd = line.trim();
        if cmd.is_empty() { continue; }

        // Slash commands
        if let Some(rest) = cmd.strip_prefix('/') {
            let mut parts = rest.splitn(2, char::is_whitespace);
            let head = parts.next().unwrap_or("");
            let arg = parts.next().unwrap_or("").trim();
            match head {
                "quit" | "exit" | "q" => break,
                "reset" => {
                    history.clear();
                    println!("[chat] history cleared.");
                }
                "system" => {
                    system = if arg.is_empty() { None } else { Some(arg.to_string()) };
                    println!("[chat] system prompt set.");
                }
                "temp" => {
                    if let Ok(t) = arg.parse::<f32>() {
                        temperature = t.clamp(0.0, 2.0);
                        println!("[chat] temperature = {:.2}", temperature);
                    } else { println!("[chat] usage: /temp <0..2>"); }
                }
                "tokens" => {
                    if let Ok(t) = arg.parse::<usize>() {
                        max_tokens = t.clamp(1, 8192);
                        println!("[chat] max_tokens = {}", max_tokens);
                    } else { println!("[chat] usage: /tokens <N>"); }
                }
                _ => println!("[chat] unknown command: /{}", head),
            }
            continue;
        }

        // Regular user message → build prompt, generate.
        let user_msg = cmd.to_string();
        let prompt = build_prompt(template, system.as_deref(), &history, &user_msg);
        print!("\n");
        let (reply, n_tokens, elapsed) = match engine.complete(&prompt, max_tokens, temperature) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[chat] generation error: {}", e);
                continue;
            }
        };
        let tps = if elapsed > 0.0 { n_tokens as f64 / elapsed } else { 0.0 };
        eprintln!("[chat] {} tokens in {:.2}s ({:.1} tok/s)", n_tokens, elapsed, tps);

        history.push(Message { role: "user".into(), content: user_msg });
        history.push(Message { role: "assistant".into(), content: reply });

        // Cap history to avoid unbounded prompt growth (kept simple: last 12 turns).
        while history.len() > 24 {
            history.drain(0..2);
        }
    }
    println!("[chat] bye.");
    Ok(())
}

// ============================================================================
// Entry point
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_id = DEFAULT_MODEL.to_string();
    let mut quantize: Option<String> = Some("q4_0".to_string());
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" | "-m" => { i += 1; model_id = argv[i].clone(); }
            "--quantize" | "-q" => { i += 1; quantize = Some(argv[i].clone()); }
            "--no-quantize" => { quantize = None; }
            _ => {}
        }
        i += 1;
    }

    eprintln!("[chat] loading {} (quant={:?})…", model_id, quantize);
    let engine = Engine::load(&model_id, quantize.as_deref())?;
    run_repl(engine)
}
