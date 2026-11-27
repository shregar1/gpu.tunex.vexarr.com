//! OpenAI-compatible HTTP server for gpux inference.
//!
//! POST /v1/chat/completions  → JSON or SSE (when `stream: true`)
//! GET  /v1/models             → list of available models
//! GET  /health                → "ok"
//!
//! Designed to be a drop-in replacement for `http://localhost:11434/v1` (Ollama)
//! or any OpenAI-compatible client (curl, langchain, openai-python).
//!
//! Keep-it-simple choices:
//!   - Single-threaded request handling (model state isn't thread-safe).
//!     Concurrent decode requires a per-request KV cache + per-request scratch
//!     reset; the API contract is fine, the GPU just serializes them.
//!   - Stdlib HTTP (no axum/hyper). Plenty for a demo server.
//!   - Greedy and temperature sampling supported. PLD spec auto-enabled.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use gpux_nn::kv_cache::KvCache;
use gpux_nn::llama::{LlamaConfig, LlamaModel};
use gpux_tensor::{GpuContext, Tensor};
use gpux_train::bpe::BpeTokenizer;
use gpux_train::hub;
use gpux_train::safetensors::load_safetensors_dir;
use serde_json::{json, Value};

const DEFAULT_MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const DEFAULT_PORT: u16 = 11434;

// ============================================================================
// Model loading (mirrors inference example so the server is self-contained)
// ============================================================================

fn download_model(model_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(hub::download_model(model_id, false)?)
}

fn load_config(dir: &Path) -> Result<LlamaConfig, Box<dyn std::error::Error>> {
    let config: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let head_dim = config.get("head_dim").and_then(|v| v.as_u64()).map(|v| v as usize)
        .unwrap_or_else(|| {
            (config["hidden_size"].as_u64().unwrap_or(1024)
                / config["num_attention_heads"].as_u64().unwrap_or(16)) as usize
        });
    Ok(LlamaConfig {
        vocab_size: config["vocab_size"].as_u64().unwrap_or(151936) as usize,
        hidden_size: config["hidden_size"].as_u64().unwrap_or(1024) as usize,
        intermediate_size: config["intermediate_size"].as_u64().unwrap_or(3072) as usize,
        num_hidden_layers: config["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
        num_attention_heads: config["num_attention_heads"].as_u64().unwrap_or(16) as usize,
        num_key_value_heads: config["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        head_dim,
        rms_norm_eps: config.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
        rope_theta: config.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(1_000_000.0) as f32,
        max_position_embeddings: config.get("max_position_embeddings")
            .and_then(|v| v.as_u64()).unwrap_or(32768) as usize,
        tie_word_embeddings: config.get("tie_word_embeddings")
            .and_then(|v| v.as_bool()).unwrap_or(true),
    })
}

// ============================================================================
// Chat template (Qwen2.5 ChatML)
// ============================================================================

fn apply_chatml(messages: &[Value], add_generation_prompt: bool) -> String {
    let mut s = String::new();
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("user");
        let content = msg["content"].as_str().unwrap_or("");
        s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
    }
    if add_generation_prompt {
        s.push_str("<|im_start|>assistant\n");
    }
    s
}

// ============================================================================
// Generation helpers
// ============================================================================

fn argmax_row(logits: &[f32]) -> u32 {
    let mut best = (0u32, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 { best = (i as u32, v); }
    }
    best.0
}

fn sample_temperature(logits: &[f32], temperature: f32) -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|&v| ((v - max) / temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    // Cheap deterministic-ish sampler so we don't need the rand crate at this layer.
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

fn is_end_token(token: u32, tok: &BpeTokenizer) -> bool {
    // The fixed tokenizer now reports special tokens reliably, so token-id
    // checks are sufficient. We accept any of the conventional end markers.
    if tok.is_special(token) {
        if let Some(s) = tok.special_str(token) {
            return matches!(s, "<|im_end|>" | "<|endoftext|>" | "<|end_of_text|>" | "<|eot_id|>");
        }
    }
    false
}

// ============================================================================
// Server state (held by main, locked per-request)
// ============================================================================

struct ServerState {
    model: LlamaModel,
    tokenizer: BpeTokenizer,
    ctx: Arc<GpuContext>,
    model_id: String,
}

impl ServerState {
    fn load(model_id: &str, quantize: Option<&str>) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = download_model(model_id)?;
        let config = load_config(&dir)?;
        eprintln!("[server] loading model: {}L/{}H({}KV)/{}D, vocab={}",
            config.num_hidden_layers, config.num_attention_heads,
            config.num_key_value_heads, config.hidden_size, config.vocab_size);

        let tokenizer = BpeTokenizer::from_file(&dir.join("tokenizer.json"))?;
        let ctx = GpuContext::new()?;
        eprintln!("[server] gpu: {}", ctx.device.name());

        let mut model = LlamaModel::new(&ctx, config)?;
        let mut weights = load_safetensors_dir(&dir, &ctx)?;

        if let Some(q) = quantize {
            eprintln!("[server] quantizing weights to {}…", q);
            let mut count = 0;
            let keys: Vec<String> = weights.keys().cloned().collect();
            for name in keys {
                let t = &weights[&name];
                let q4k = matches!(q, "q4_k" | "Q4_K" | "q4_k_m" | "Q4_K_M");
                if name.contains(".weight") && t.shape().ndim() == 2
                    && !name.contains("norm")
                    && !(q4k && name.contains("embed"))
                    && t.numel() % 32 == 0
                {
                    let new_t = match q {
                        "q4_0" | "Q4_0" => t.quantize_q4_0()?,
                        "q8_0" | "Q8_0" => t.quantize_q8_0()?,
                        "q4_k" | "Q4_K" | "q4_k_m" | "Q4_K_M" => {
                            if t.numel() % 256 != 0 { continue; }
                            t.quantize_q4_k()?
                        }
                        _ => continue,
                    };
                    weights.insert(name, new_t);
                    count += 1;
                }
            }
            eprintln!("[server] quantized {} weight matrices", count);
        }

        model.load_weights(&weights)?;
        ctx.queue.synchronize()?;
        drop(weights);
        ctx.device.activate_scratch();

        Ok(Self { model, tokenizer, ctx, model_id: model_id.to_string() })
    }

    /// Run a chat completion. If `on_token` is `Some`, called with each generated
    /// token's UTF-8 piece for streaming; otherwise the full string is returned.
    fn complete(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        mut on_token: Option<&mut dyn FnMut(&str) -> std::io::Result<()>>,
    ) -> Result<(String, usize, usize), Box<dyn std::error::Error>> {
        let prompt_tokens = self.tokenizer.encode(prompt);
        let prompt_len = prompt_tokens.len();
        let max_seq = prompt_len + max_tokens + 8;
        self.ctx.device.reset_scratch();
        let mut cache = KvCache::new(&self.ctx, &self.model.config, max_seq, 1)?;
        let vocab = self.model.config.vocab_size;

        // Prefill.
        let input = Tensor::from_uint_data(&self.ctx, vec![1, prompt_len], &prompt_tokens)?;
        let logits = self.model.forward_lm_cached(&input, &mut cache)?;
        let logits_data = logits.to_vec();
        self.ctx.device.reset_scratch();
        let last = &logits_data[(prompt_len - 1) * vocab..];
        let mut next = if temperature < 0.01 {
            argmax_row(last)
        } else {
            sample_temperature(last, temperature)
        };

        // Decode loop.
        let mut output = String::new();
        let mut produced = 0usize;
        for _ in 0..max_tokens {
            if is_end_token(next, &self.tokenizer) { break; }
            // Skip emitting any other special tokens (system markers, etc.)
            // so they never leak into the user-facing output.
            if !self.tokenizer.is_special(next) {
                let piece = self.tokenizer.decode(&[next]);
                output.push_str(&piece);
                if let Some(cb) = on_token.as_deref_mut() {
                    let _ = cb(&piece);
                }
            }
            produced += 1;

            self.ctx.device.reset_scratch();
            let input = Tensor::from_uint_data(&self.ctx, vec![1, 1], &[next])?;
            let logits = self.model.forward_lm_cached(&input, &mut cache)?;
            let logits_data = logits.to_vec();
            next = if temperature < 0.01 {
                argmax_row(&logits_data)
            } else {
                sample_temperature(&logits_data, temperature)
            };
        }

        Ok((output, prompt_len, produced))
    }
}

// ============================================================================
// HTTP request parsing (minimal, just what we need)
// ============================================================================

struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<HttpRequest> {
    let mut reader = BufReader::new(stream);

    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.trim().split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut headers = HashMap::new();
    loop {
        let mut h = String::new();
        reader.read_line(&mut h)?;
        if h == "\r\n" || h == "\n" || h.is_empty() { break; }
        if let Some((k, v)) = h.trim().split_once(':') {
            headers.insert(k.trim().to_lowercase(), v.trim().to_string());
        }
    }

    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; len];
    if len > 0 {
        reader.read_exact(&mut body)?;
    }

    Ok(HttpRequest { method, path, headers, body })
}

// ============================================================================
// Endpoint handlers
// ============================================================================

fn write_status(stream: &mut TcpStream, code: u16, status: &str) -> std::io::Result<()> {
    write!(stream, "HTTP/1.1 {} {}\r\n", code, status)
}

fn send_json(stream: &mut TcpStream, code: u16, status: &str, body: &Value) -> std::io::Result<()> {
    let s = body.to_string();
    write_status(stream, code, status)?;
    write!(stream,
        "Content-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        s.len())?;
    stream.write_all(s.as_bytes())?;
    Ok(())
}

fn handle_health(stream: &mut TcpStream) -> std::io::Result<()> {
    write_status(stream, 200, "OK")?;
    let body = "ok\n";
    write!(stream,
        "Content-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len())?;
    stream.write_all(body.as_bytes())?;
    Ok(())
}

fn handle_models(stream: &mut TcpStream, state: &ServerState) -> std::io::Result<()> {
    let resp = json!({
        "object": "list",
        "data": [{
            "id": state.model_id,
            "object": "model",
            "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "owned_by": "gpux",
        }]
    });
    send_json(stream, 200, "OK", &resp)
}

fn handle_chat(
    stream: &mut TcpStream,
    body: &[u8],
    state: &ServerState,
) -> Result<(), Box<dyn std::error::Error>> {
    let req: Value = serde_json::from_slice(body)?;
    let messages = req["messages"].as_array().ok_or("messages array required")?;
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(256) as usize;
    let temperature = req["temperature"].as_f64().unwrap_or(0.0) as f32;
    let stream_mode = req["stream"].as_bool().unwrap_or(false);

    let prompt = apply_chatml(messages, true);
    let id = format!("chatcmpl-{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos());
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    if !stream_mode {
        let t0 = Instant::now();
        let (text, prompt_tokens, completion_tokens) =
            state.complete(&prompt, max_tokens, temperature, None)?;
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!("[server] chat ok: {} prompt + {} gen tokens in {:.2}s ({:.1} tok/s)",
            prompt_tokens, completion_tokens, elapsed,
            completion_tokens as f64 / elapsed.max(1e-6));

        let resp = json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": state.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        });
        send_json(stream, 200, "OK", &resp)?;
        return Ok(());
    }

    // Streaming (SSE) path.
    write_status(stream, 200, "OK")?;
    write!(stream,
        "Content-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n")?;
    stream.flush()?;

    // Initial role chunk (some clients require it).
    let role_chunk = json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": state.model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    });
    write!(stream, "data: {}\n\n", role_chunk)?;
    stream.flush()?;

    let id_clone = id.clone();
    let model_id_clone = state.model_id.clone();
    let stream_ptr: *mut TcpStream = stream;
    // SAFETY: callback runs synchronously inside complete() — the &mut TcpStream
    // borrow from the outer scope is alive for the entire call. This pointer hop
    // is just to satisfy the closure's lifetime constraint.
    let mut cb = move |piece: &str| -> std::io::Result<()> {
        let chunk = json!({
            "id": id_clone, "object": "chat.completion.chunk", "created": created,
            "model": model_id_clone,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}]
        });
        let s = unsafe { &mut *stream_ptr };
        write!(s, "data: {}\n\n", chunk)?;
        s.flush()?;
        Ok(())
    };
    let t0 = Instant::now();
    let (_text, prompt_tokens, completion_tokens) =
        state.complete(&prompt, max_tokens, temperature, Some(&mut cb))?;
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("[server] chat (stream) ok: {} prompt + {} gen tokens in {:.2}s ({:.1} tok/s)",
        prompt_tokens, completion_tokens, elapsed,
        completion_tokens as f64 / elapsed.max(1e-6));

    // Final stop chunk + DONE marker.
    let stop_chunk = json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": state.model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    write!(stream, "data: {}\n\n", stop_chunk)?;
    write!(stream, "data: [DONE]\n\n")?;
    stream.flush()?;
    Ok(())
}

fn handle_options(stream: &mut TcpStream) -> std::io::Result<()> {
    // CORS preflight
    write_status(stream, 204, "No Content")?;
    write!(stream,
        "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nConnection: close\r\n\r\n")?;
    Ok(())
}

fn handle_request(mut stream: TcpStream, state: &Arc<Mutex<ServerState>>) {
    let req = match read_request(&mut stream) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[server] request parse error: {}", e);
            return;
        }
    };

    let result: Result<(), Box<dyn std::error::Error>> = match (req.method.as_str(), req.path.as_str()) {
        ("OPTIONS", _) => handle_options(&mut stream).map_err(Into::into),
        ("GET", "/health") => handle_health(&mut stream).map_err(Into::into),
        ("GET", "/v1/models") => {
            let st = state.lock().unwrap();
            handle_models(&mut stream, &st).map_err(Into::into)
        }
        ("POST", "/v1/chat/completions") => {
            let st = state.lock().unwrap();
            handle_chat(&mut stream, &req.body, &st)
        }
        _ => {
            let body = json!({"error": {"message": format!("unknown route: {} {}", req.method, req.path), "type": "not_found"}});
            send_json(&mut stream, 404, "Not Found", &body).map_err(Into::into)
        }
    };

    if let Err(e) = result {
        eprintln!("[server] handler error: {}", e);
        let body = json!({"error": {"message": e.to_string(), "type": "internal_error"}});
        let _ = send_json(&mut stream, 500, "Internal Server Error", &body);
    }
}

// ============================================================================
// Entry point
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger_init();

    let mut model_id = DEFAULT_MODEL.to_string();
    let mut port = DEFAULT_PORT;
    let mut quantize = Some("q4_0".to_string());
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" | "-m" => { i += 1; model_id = argv[i].clone(); }
            "--port" | "-p" => { i += 1; port = argv[i].parse().unwrap_or(DEFAULT_PORT); }
            "--quantize" | "-q" => { i += 1; quantize = Some(argv[i].clone()); }
            "--no-quantize" => { quantize = None; }
            _ => {}
        }
        i += 1;
    }

    eprintln!("[server] gpux OpenAI-compatible server starting");
    eprintln!("[server] model: {}", model_id);
    eprintln!("[server] quantization: {:?}", quantize);

    let state = ServerState::load(&model_id, quantize.as_deref())?;
    let state = Arc::new(Mutex::new(state));

    let listener = TcpListener::bind(("127.0.0.1", port))?;
    eprintln!("[server] listening on http://127.0.0.1:{}", port);
    eprintln!("[server] try: curl http://127.0.0.1:{}/health", port);
    eprintln!("[server]      curl http://127.0.0.1:{}/v1/models", port);

    for stream in listener.incoming() {
        match stream {
            Ok(s) => handle_request(s, &state),
            Err(e) => eprintln!("[server] accept error: {}", e),
        }
    }
    Ok(())
}

fn env_logger_init() {
    // Light init — env_logger pulls a lot of deps and we don't need it here.
    // The user can tail stderr; our logs are already clearly tagged.
}
