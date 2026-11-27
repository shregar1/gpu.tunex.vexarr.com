# gpux — Handover Document

> Written for the next person picking this up. Tells you what works, what's
> half-done, where the bodies are buried, and what the realistic next moves are.
> Not a sales pitch — read it before merging anything.

**Last updated**: end of the session that landed Q4_K_M v2.
**Hardware tested on**: M4 MacBook Air (10-core GPU, unified memory).
**Stopped because**: every remaining item is a multi-day project, not a
one-iteration fix. We hit a clean plateau.

---

## TL;DR — current state in 5 lines

1. **Decode beats llama.cpp on Qwen2.5-0.5B Q4_0 by ~34%** (~190 vs ~142 tok/s on M4).
2. **Long prefill is ~5× behind llama.cpp** (~620 vs ~2630 tok/s on pp128). Closing it requires `simdgroup_async_copy`-based GEMM tiling — see roadmap.
3. **Three models verified end-to-end**: Qwen2.5, SmolLM2, TinyLlama. Two tokenizer families (GPT-2 byte-level BPE + SentencePiece) auto-detected. Three chat templates (ChatML / Zephyr / LlamaInst) auto-detected.
4. **Three runnable binaries**: `inference`, `chat` (REPL), `server` (OpenAI-compatible HTTP with SSE).
5. **Quantization**: Q4_0, Q8_0, Q4_K_M (the last is correct but uses a naive quantizer; quality is below llama.cpp's RMSE-refined Q4_K_M).

---

## Project layout

```
crates/
├── gpux-core         # Metal device, command queue, buffer abstraction,
│                     # scratch allocator, per-kernel profiler.
│                     # ~600 LoC. Stable, well-trodden.
├── gpux-tensor       # Tensor abstraction, all GPU ops, dtype handling
│                     # (F32, F16, Q4_0, Q8_0, Q4_K).
│                     # ~1500 LoC of Rust + 90% of the dispatch routing.
├── gpux-shaders      # All Metal kernel sources (in src/metal/*.metal),
│                     # plus pipeline registry / runtime compilation.
│                     # ~3500 LoC of Metal across ~12 .metal files.
├── gpux-nn           # Llama-family transformer modules (RMSNorm, GQA,
│                     # SwiGLU, KV cache, embedding, LlamaModel).
│                     # ~600 LoC.
└── gpux-train        # BPE tokenizer (GPT-2 + SentencePiece), safetensors
│                     # loader, HF Hub downloader, training scaffolding.
│                     # ~1500 LoC. (Training side is barely tested.)

examples/
├── inference/        # Single-shot bench / generation.
├── chat/             # Multi-turn REPL with slash commands.
└── server/           # OpenAI-compatible HTTP server.
```

---

## Run book

```bash
# Single-shot bench (default Qwen2.5-0.5B, no quant)
cargo run --release -p inference-example -- \
    --prompt "Hello, world" --max-tokens 100 --temperature 0

# With quantization
cargo run --release -p inference-example -- --quantize q4_0 ...
cargo run --release -p inference-example -- --quantize q4_k ...

# Different model (auto-downloads from HF)
cargo run --release -p inference-example -- \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quantize q4_0 ...

# Chat REPL
cargo run --release -p chat-example -- \
    --model Qwen/Qwen2.5-0.5B-Instruct --quantize q4_0
# Inside: /reset  /system X  /temp F  /tokens N  /quit

# OpenAI-compatible HTTP server (default port 11434)
cargo run --release -p server-example -- \
    --model Qwen/Qwen2.5-0.5B-Instruct --quantize q4_0
# Hit it with curl or the openai-python SDK
```

### Useful env vars

| Var | Purpose |
|---|---|
| `GPUX_PROFILE=1` | Per-kernel GPU profiler. Synchronizes after every dispatch and times it. **Slows decode 10-15× but prints accurate per-kernel breakdown.** |
| `GPUX_DEBUG_DISPATCH=1` | Print dispatch count per token. |
| `GPUX_DEBUG_SCRATCH=1` | Log scratch allocator usage per token. |
| `GPUX_FUSED_ATTN=0` | Disable fused attention kernel (for debugging). |
| `GPUX_SIMD=0` | Disable SIMD matvec, use scalar fallback. |

---

## Performance summary (M4 MacBook Air, cool runs)

### Decode (single-token greedy)

| Workload | tok/s | Notes |
|---|---|---|
| Qwen2.5-0.5B Q4_0 | **~190** | best; +34% over llama.cpp |
| TinyLlama-1.1B Q4_0 | ~119 | |
| TinyLlama-1.1B Q4_K_M | ~104 | v2 lane-per-sub-block matvec |
| SmolLM2-360M-Instruct (no quant) | ~107 | |
| Qwen2.5-0.5B Q8_0 | ~70 | (Q4_0 is faster on this HW) |
| Qwen2.5-0.5B f16 | ~52 | unquantized baseline |

### Prefill (~130 tokens)

| Engine | tok/s | Notes |
|---|---|---|
| llama.cpp pp128 Q4_0 | ~2630 | uses async TG memory loads |
| **gpux Q4_0** | **~620** | uses `simdgroup_matrix` v2 GEMM, no async |

### llama.cpp head-to-head (cool, both)

```
Qwen2.5-0.5B Q4_0, M4, 100-token greedy decode
  llama.cpp:  142 ± 16 tok/s   (best observed cool)
  gpux:       190             (+34%)
```

### Variance warning

The M4 Air thermally throttles **fast** under sustained GPU load. After ~30s of
benchmarking, decode can drop 30%. **Always cool the machine for 90+ seconds
between bench runs** if you want comparable numbers. The Monitor tool we used in
session has cooldown patterns built in (`sleep 120` then bench).

---

## Architecture decisions worth knowing (the "why")

### Why scratch allocator + persistent KV cache buffers
Per-token GPU buffer allocation churn was killing us. Now: one big pre-allocated
scratch buffer, bump-pointer alloc per dispatch, reset between tokens. KV cache
buffers are *persistent* (allocated once at session start) so they don't get
overwritten by scratch resets. **See `gpux-core/src/buffer.rs` `ScratchAllocator`**.

### Why persistent compute encoder
We keep ONE `MTLComputeCommandEncoder` alive across all dispatches in a token,
ending it only on `synchronize()`. Eliminates per-dispatch encoder
create/destroy overhead. **See `gpux-core/src/command.rs` `CommandQueue::enqueue`**.

### Why inline parameters (`setBytes`)
Most kernels need a tiny `[N, K]` dims buffer. We pass these inline via
`setBytes` instead of allocating a buffer per dispatch. Saves ~440 small
allocations per token. **See `CommandEncoder::dispatch_inline`**.

### Why `simdgroup_matrix` for prefill (and not decode)
Decode is M=1 (matvec, not matmul) — `simdgroup_matrix` doesn't help for skinny
matrices. Prefill is M>1 (real matmul) — that's where Apple's hardware matrix
units shine. **See `matmul_a_bt_q4_0_sgm2_f32` in `quantized.metal`**.

### Why we don't have a Q4_K prefill GEMM
Same simdgroup_matrix pattern would work but it's another ~150 LoC of kernel
that nobody's written yet. M>1 Q4_K currently chunks into sequential M=1 matvec
calls (correct, slow). **See `matmul_a_bt_quantized` in `gpux-tensor/src/ops.rs`**.

### Why two tokenizer code paths
GPT-2 byte-level BPE and SentencePiece BPE are *fundamentally different*
algorithms despite both being called "BPE":
- GPT-2: byte-level encoding (`Ġ` for space prefix), no byte fallback
- SentencePiece: `▁` for space prefix, `<0xXX>` byte-fallback tokens for unknown chars

We auto-detect from `tokenizer.json` (presence of Metaspace normalizer or
byte_fallback flag). **See `gpux-train/src/bpe.rs` `TokenizerKind`**.

### Why `decode` doesn't strip leading space
SentencePiece tokenizers have a "Strip leading space" decoder step that's
intended for **whole-prompt decoding** — strip the leading space artifact from
the `▁` prepend. But during streaming, `decode([single_token])` is called per
token; stripping every leading space produces `isParis` instead of `is Paris`.
We expose `decode_full()` for the rare case where you want the strip.

### Why the Q4_K v2 matvec uses lane-per-sub-block
v1 (lane-per-super-block, 256 elements) only utilizes K/256 lanes. For K=2048
that's 8 of 32 — most lanes idle. v2 puts each lane on a 32-element sub-block;
for K=2048 that's 64 sub-blocks → 32 lanes × 2 inner iters, all busy. **2.1×
speedup**. The 8-lane redundant reads of super-block headers (d/dmin/scales)
get absorbed by L1 cache. See **`matvec_a_bt_q4_k_v2_f32` in `quantized.metal`**.

### Why speculative decoding has an "adaptive" scheduler
With our fast decode baseline (~190 tok/s), the verify pass overhead often
exceeds the win from accepted drafts. PLD spec was *hurting* on novel content
(160 → 135 tok/s). The adaptive scheduler tracks rolling acceptance and skips
spec when recent accept rate < 0.55 (with a 16-token cooldown). **See
`PldScheduler` in `examples/inference/src/main.rs`**.

### Why the downloader validates Content-Length
We hit a real bug: SmolLM2 silently truncated to 378 MB instead of 723 MB
mid-download (CDN hiccup, curl exited 0). The model loaded structurally and
produced gibberish. Downloader now HEADs the URL, compares to local size, and
deletes truncated files so retries work. **See `gpux-train/src/hub.rs`
`download_file`**.

---

## Known issues & technical debt (the honest list)

### High-priority

1. **Q4_K_M quantizer is naive max-range** (`quantize_q4_k_f32` in `quantized.metal`).
   llama.cpp uses iterative RMSE refinement → ~5-10% better perplexity. Single-
   function fix. Quality on TinyLlama Q4_K_M is noticeably below Q4_0; partly
   the format, mostly this.

2. **No Q4_K prefill GEMM**. M>1 Q4_K matmul falls through to M-chunking (call
   M=1 matvec sequentially) — see `matmul_a_bt_quantized` in
   `gpux-tensor/src/ops.rs:1481+`. Long-context prefill on Q4_K is slow.

3. **No fused gate_up_silu for Q4_K**. `SwiGLU::forward_fused` falls back to
   separate {gate, up, silu_mul, down} dispatches when weights are Q4_K. Costs
   ~10-15% on Q4_K decode.

4. **No embedding lookup kernel for Q4_K**. The example quantize loops skip
   the embedding for Q4_K, leaving it at f16 (extra 130 MB on disk for Qwen).
   See the `if matches!(q, "q4_k" ...) && name.contains("embed")` guards in
   inference/chat/server `main.rs`.

5. **Long prefill ~5× slower than llama.cpp**. Root cause: our GEMM doesn't
   use `simdgroup_async_copy` for double-buffered TG memory loads. Fixing this
   is the single biggest perf item left.

### Medium-priority

6. **No multi-batch decode**. Server processes requests serially because the
   model state isn't thread-safe (KV cache + scratch are shared). To support
   concurrent decode: per-request KV cache + per-request scratch reset, plus a
   request queue that batches matching shapes.

7. **Server has a `Mutex<ServerState>`** wrapping the model — works for serial
   throughput, would need rework for batched serving.

8. **Streaming SSE in server uses unsafe pointer hop** for the closure
   lifetime. Works but ugly. See the comment near `stream_ptr` in
   `examples/server/src/main.rs`. A proper async refactor would clean this up.

9. **Sampler uses OS-time-derived randomness** — not seedable. Adding the
   `rand` crate would fix this.

10. **`hub::revalidate` is half-strict**: only deletes empty files, not
    truncated ones. Truncation IS caught at download time, but if you somehow
    end up with a 378-MB file claiming to be 723 MB, the next run will trust it.
    Harden by always HEADing on cache hit.

### Low-priority

11. **`gpux-train` crate** has training scaffolding (Adam optimizer, mixed
    precision config, trainer loop) that's not tested in current state. We use
    it only for the BPE tokenizer + safetensors loader + hub downloader.

12. **No tests.** Seriously. Output correctness is verified by running
    inference and comparing to known answers. There's no `cargo test`. Adding
    a basic integration test that runs Qwen2.5-0.5B Q4_0 for 10 tokens and
    asserts output would be valuable.

13. **No CI.** No GitHub Actions, no benchmark regression detection.

14. **The inference example has duplicated download code** (didn't get
    refactored to `hub::download_model` because the example has more complex
    sharded-model handling). Server + chat use `hub::download_model`.

15. **Lots of `Box<dyn std::error::Error>`** in example code. Real codebase
    would use a proper error enum.

16. **No tokenizer_config.json validation.** We string-match the chat_template
    to detect format. A malicious / unusual `chat_template` could fool us. In
    practice: every model on HF that uses one of these three templates has a
    standard string we recognize.

---

## Files where the interesting stuff lives

### Hot-path kernels (perf matters)

| File | What |
|---|---|
| `crates/gpux-shaders/src/metal/quantized.metal` | All Q4_0 / Q8_0 / Q4_K kernels. ~1500 LoC. The Q4_K v2 matvec at the top is the most recent perf win. |
| `crates/gpux-shaders/src/metal/fused.metal` | RoPE, gate_up_silu, RMSNorm, embedding (f16/f32), SIMD matvec for f16. The SIMD-cooperative `rmsnorm_residual_f16w_f32` was the biggest single decode speedup of the session (35% → 18% of GPU time). |
| `crates/gpux-shaders/src/metal/kv_cache.metal` | KV cache append, fused RoPE+KV-append, fused attention decode (online softmax). |

### Hot-path Rust (dispatch routing)

| File | What |
|---|---|
| `crates/gpux-tensor/src/ops.rs` | All GPU ops. The big functions: `matmul_a_bt`, `matmul_a_bt_bias`, `matmul_a_bt_quantized`, `fused_gate_up_silu`, `add_rms_norm`. **This is the file you'll edit most.** |
| `crates/gpux-nn/src/gqa.rs` | Grouped-query attention — the fused decode path is here. |
| `crates/gpux-nn/src/llama.rs` | Top-level forward pass. The "fuse final-add into final-rmsnorm" trick is in `forward_lm_cached`. |
| `crates/gpux-core/src/command.rs` | Command queue with persistent encoder + the `dispatch_inline` (setBytes) path. The whole encoder lifecycle is here. |
| `crates/gpux-core/src/buffer.rs` | Scratch allocator — handle with care, this is the bedrock for per-token allocation perf. |

### User-facing

| File | What |
|---|---|
| `examples/inference/src/main.rs` | Bench / generation entry point. PLD speculative decoder is here. |
| `examples/chat/src/main.rs` | Chat REPL. Chat template detection is here. |
| `examples/server/src/main.rs` | OpenAI-compatible HTTP server. SSE streaming. |
| `crates/gpux-train/src/bpe.rs` | Tokenizer (both formats). The auto-detection logic and merge-rank-based BPE are here. |
| `crates/gpux-train/src/hub.rs` | HF downloader with size validation. |

---

## Roadmap with effort estimates

Ranked by ROI (impact ÷ effort):

| # | Item | Effort | Impact |
|---|---|---|---|
| 1 | RMSE-refined Q4_K quantizer | ~1 day | Q4_K quality jumps to ~llama.cpp parity. Single-kernel change. |
| 2 | Fused `gate_up_silu` Q4_K kernel | ~1 day | TinyLlama Q4_K_M decode: ~104 → ~120 tok/s. |
| 3 | Q4_K embedding lookup kernel | ~30 min | Saves ~130 MB disk for Qwen Q4_K runs. Small dispatch save. |
| 4 | `simdgroup_async_copy` in Q4_0 GEMM | ~3 days | Closes most of the prefill gap. Real speedup: probably 2× on long prefill. |
| 5 | Multi-batch decode | ~3 days | Real serving throughput story. Needs per-request KV cache + queue. |
| 6 | Q4_K simdgroup_matrix prefill GEMM | ~3 days | Q4_K becomes viable for long-context use. |
| 7 | RoPE for T>1 + KV-append for T>1 fused | ~1 day | Spec verify and prefill ~10-15% faster. |
| 8 | FlashAttention-2 attention kernel | ~1 week | Long context; doesn't help short context. |
| 9 | Add `rand` crate + seedable sampler | ~1 hour | Reproducible sampling for tests. |
| 10 | Integration tests + CI | ~2 days | Catch regressions automatically. |
| 11 | Real `Llama-3.x` support | ~1 day plus HF token | Tokenizer should already work; need to verify and test. |
| 12 | Tool-use / structured output in server | ~3 days | Function-calling JSON-schema. |

### Things I'd specifically tell you NOT to do

- **Don't try to micro-optimize the existing matvec inner loops.** I tried
  vectorized loads, the divider trick, Q hoisting in attention — Apple's Metal
  compiler was already doing all of those. None moved the needle. The wins
  available are *structural* (more SIMD cooperation, better tile sizes, async
  loads), not local.

- **Don't add a SentencePiece tokenizer "from scratch"** thinking it'll be
  cleaner. The current implementation IS pretty clean — it shares 80% of code
  with the GPT-2 path via `apply_merges_and_emit` and dispatches by `kind`.

- **Don't reach for an HTTP framework** (axum, hyper, tonic). The stdlib server
  is ~250 LoC, has zero dependencies, and works. Adding a framework is a
  decision that locks in async runtime (tokio) and pulls in 50+ deps — not
  worth it unless you're going to multi-batch + production serving.

- **Don't enable `GPUX_PROFILE=1` for "real" benchmarks.** It synchronizes
  every dispatch, slowing decode 10-15× and inflating per-kernel time by
  including sync overhead. It's a *relative* tool — accurate for "which kernel
  is taking most of the time", not for absolute kernel timings.

---

## Gotchas & tribal knowledge

### Thermal throttling is real and brutal
M4 Air will throttle 30-40% within seconds of sustained GPU work. Always cool
between bench runs. Use the `Monitor` pattern with `sleep 90+` between
sequential bench batches.

### Profile mode wildly overestimates per-kernel time
`GPUX_PROFILE=1` adds synchronize() after every dispatch. The reported
"742µs/call" for `rmsnorm_residual` includes ~700µs of sync overhead per call.
Real per-call time in production is more like 20-50µs. Use profile mode for
**relative** breakdowns only.

### "downloaded successfully" doesn't mean "complete"
We hit a CDN truncation bug with SmolLM2. The downloader now validates
Content-Length but **only on download**, not on cache hit. If you suspect a
cached file is bad, delete it and re-download.

### Q4_K_M is K-divisible-by-256 only
Models with hidden_size = 896 (Qwen 0.5B), 960 (SmolLM 360M), etc. can't fully
use Q4_K_M — only weights with the K dim divisible by 256 get quantized. The
example quantize loops skip non-divisible weights with a `t.numel() % 256 !=
0` check. Same constraint as llama.cpp.

### The Q4_K dispatch routing has a footgun
For M>1 Q4_K, `matmul_a_bt_quantized` does CPU-side M-chunking — calls
`to_vec()` on the input row, builds a new tensor, runs M=1 matvec, copies
output back. This involves CPU↔GPU transfers per row. For long prefill on
Q4_K it's catastrophically slow. If you hit prefill perf issues with Q4_K,
this is why. Replace with a real Q4_K GEMM.

### The chat REPL eats `<|im_end|>` tokens
The `is_end()` check stops generation on any of: `<|im_end|>`, `<|endoftext|>`,
`<|end_of_text|>`, `<|eot_id|>`, `</s>`. Other end markers (e.g. Mistral's
`[/INST]` doesn't have a special token) won't trip it — generation will run
for the full `max_tokens`.

### SwiGLU has subtle dispatch routing
`SwiGLU::forward_fused` chooses one of three paths depending on (M, dtype):
- `M=1 + non-Q4_K` → fused single dispatch
- `M=2..8 + Q4_0` → M-batched fused dispatch (for spec verify)
- everything else → unfused {gate, up, silu_mul, down}

Adding new dtypes / paths requires updating *both* the gate condition and the
dispatch in `Tensor::fused_gate_up_silu`.

### The order of kernels in `quantized.metal` matters
Constants like `Q_SIMD_W`, `Q_ROWS_PER_TG` must be declared **before** any
kernel that uses them. We hit this twice during the session (lost ~10 min each
time). When adding a new kernel, place it AFTER the constants block.

### `decode_full` vs `decode`
For SentencePiece tokenizers, `decode([single_token])` does NOT strip leading
space (would break streaming). `decode_full(tokens)` does the full Llama-style
"strip one leading space at the start" cleanup. Server / chat use per-token
`decode`; offline batch decoding wants `decode_full`.

### `forward_lm_cached` has a "first layer" hack
The first layer's input RMSNorm is computed BEFORE the loop, separately, so
the loop body can use the next layer's input_layernorm.weight as the
"between-layer" rmsnorm. The last layer's "after-MLP" add is fused with the
final model.norm via add_rms_norm. This saves 1 dispatch per token but is
non-obvious — read `forward_lm_cached` carefully if modifying.

### Speculative decoding: greedy only
PLD spec compares argmax(logits) to the draft token. Temperature > 0 isn't
supported (would need rejection sampling). The CLI prints a warning if you
pass `--temperature > 0` with `--spec-k`.

---

## Things I never wrote but you might want

1. **A `gpux-bench` binary** that automates: cool, bench gpux, cool, bench
   llama.cpp, print comparison. Currently doing this with shell + Monitor.

2. **A CI workflow** that runs a 30-second smoke test per PR (load
   Qwen2.5-0.5B Q4_0, generate 20 tokens, assert output starts with the right
   first token).

3. **A `--validate` flag for the inference example** that checks output
   against a known-good golden file. Catches kernel regressions immediately.

4. **A `tokenizer_config.json` "chat_template" -> Rust enum compiler.** Right
   now we string-match three templates. A real Jinja2 parser would handle
   anything (but adds a Jinja dep — probably not worth it).

5. **Real LoRA support.** There's plumbing in `Linear::set_lora` but it's
   never been tested end-to-end with a real LoRA adapter on Q4_0.

---

## Final word

This is a research-grade inference engine. It demonstrates that you can hand-
write Metal kernels in pure Rust and beat llama.cpp on a focused workload
(small Llama-style models, single-token decode, Q4_0 on M-series). It's not
trying to replace llama.cpp.

The thing that worked best in this project wasn't being clever about kernels —
it was **profiling honestly**. The single biggest decode speedup of the entire
session (88 → 193 tok/s, 2.1×) came from one observation: a 12-line scalar
RMSNorm kernel that nobody had thought about was using 35% of GPU time. The
fix took 30 minutes. The lesson: build the profiler, use the profiler,
believe the profiler.

Good luck.
