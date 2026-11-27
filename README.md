# gpux

**Apple Silicon native LLM inference. Pure Rust. Hand-written Metal kernels.**

Decode throughput on Qwen2.5-0.5B Q4_0 (M4 MacBook Air): **~190 tok/s** — measurably faster than llama.cpp on the same hardware/model/quantization.

```
                                    M4 MacBook Air, Qwen2.5-0.5B Q4_0
                                    100-token greedy decode

         llama.cpp Q4_0  ████████████████████████              142 tok/s
         gpux     Q4_0   ███████████████████████████████████   190 tok/s   (+34%)
```

This is a research-grade inference engine that demonstrates what's possible with
hand-tuned Metal kernels. It's **not** trying to replace llama.cpp's broad model
support, but it goes hard on a focused workload: small-to-mid Llama-style models
on Apple Silicon, with Q4_0 quantization, decode-optimized.

### Tested models

Architecture-agnostic at the Metal level — same kernels run any Llama-style
transformer (RMSNorm + GQA + SwiGLU + RoPE) regardless of dimensions. Just point
at a HuggingFace repo:

| Model                                  | Params | Tokenizer       | Q4_0 decode  | Q4_K_M decode |
| -------------------------------------- | ------ | --------------- | ------------ | ------------- |
| Qwen/Qwen2.5-0.5B-Instruct             | 494 M  | GPT-2 BPE       | **~190 t/s** | n/a (K=896 not ÷256) |
| HuggingFaceTB/SmolLM2-360M-Instruct    | 362 M  | GPT-2 BPE       | ~107 t/s     | partial only |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0     | 1.1 B  | SentencePiece   | ~119 t/s     | **~104 t/s**  |

Q4_K_M requires K dimensions to be multiples of 256 (super-block size). For
models like Qwen2.5-0.5B with K=896, only the wide MLP weights get quantized
to Q4_K and attention stays at f16 — same constraint as llama.cpp.

**Both major HuggingFace tokenizer formats are auto-detected:**
- **GPT-2 byte-level BPE** (Qwen, SmolLM2, GPT-2/3, OPT): `Ġ` for word boundaries
- **SentencePiece BPE** (Llama family, TinyLlama, Mistral, Gemma): `▁` for word
  boundaries, byte fallback (`<0xXX>` tokens) for unknown bytes, optional
  Llama-style leading-space strip

Detection is automatic — read `model.byte_fallback`, `Prepend`/`Replace`
normalizers, and `Strip` decoders from the standard `tokenizer.json` schema.

---

## What's in the box

### Three runnable binaries

```bash
# Single-shot inference benchmark — bench against your own prompts
cargo run --release -p inference-example -- --quantize q4_0 \
    --prompt "The capital of France is" --temperature 0 --max-tokens 100

# Interactive chat REPL — multi-turn conversations, streaming output
cargo run --release -p chat-example -- --model Qwen/Qwen2.5-0.5B-Instruct
#   /reset    /system X    /temp F    /tokens N    /quit

# OpenAI-compatible HTTP server — drop-in for openai-python, langchain, etc
cargo run --release -p server-example -- --model Qwen/Qwen2.5-0.5B-Instruct
#   POST http://127.0.0.1:11434/v1/chat/completions   (with stream: true for SSE)
#   GET  http://127.0.0.1:11434/v1/models
#   GET  http://127.0.0.1:11434/health
```

The server works end-to-end with the official `openai` Python package:

```python
import openai
client = openai.OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='ignored')
resp = client.chat.completions.create(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello'}],
)
print(resp.choices[0].message.content)
```

### Speculative decoding (PLD) with adaptive scheduling

```bash
cargo run --release -p inference-example -- --quantize q4_0 --spec-k 4 \
    --prompt "The capital of France is" --max-tokens 100
```

Auto-detects when verify-pass overhead exceeds the win and skips spec on
non-repetitive content. Output is byte-identical to greedy.

### Custom GPU profiler

```bash
GPUX_PROFILE=1 cargo run --release -p inference-example -- \
    --quantize q4_0 --max-tokens 30 --prompt "..."
```

Sample output:

```
=== Per-kernel GPU time ===
  kernel                                       calls  total (ms)   per (µs)  % time
  ----------------------------------------------------------------------------------
  matvec_a_bt_q4_0_f32                          1470     499.4      339.7   27.0%
  rmsnorm_residual_f16w_f32                     1488     378.9      254.6   20.5%
  fused_gate_up_silu_q4_0_f32                    720     297.0      412.5   16.1%
  matvec_a_bt_bias_q4_0_f32                      720     199.9      277.7   10.8%
  fused_attention_decode_f32                     720     164.6      228.6    8.9%
  rope_kv_append_fused_f32                       720     154.2      214.1    8.3%
  ...
```

This is the tool that surfaced our biggest perf win mid-development: the original
`rmsnorm_residual` kernel was a 1-thread-per-row scalar loop using ~35% of GPU
time. Swapping to a SIMD-cooperative version (4 SIMD groups × 32 lanes per row)
**doubled overall decode throughput** in one change.

---

## Performance

### Decode (M4 MacBook Air, Qwen2.5-0.5B Q4_0, 100-token greedy)

| Engine              | tok/s        | Notes                              |
| ------------------- | ------------ | ---------------------------------- |
| llama.cpp Q4_0      | ~142         | best observed cool-state           |
| llama.cpp Q4_K_M    | ~135         | better quality format              |
| **gpux Q4_0**       | **~190**     | **+34% vs llama.cpp Q4_0**         |
| gpux Q4_K_M         | ~104 (TinyLlama 1.1B) | naive quantizer; v2 lane-per-sub-block matvec |
| gpux Q8_0           | ~70          | (Q4_0 is faster on this hardware)  |
| gpux f16            | ~52          | unquantized baseline               |

### Prefill (~130 tokens)

| Engine              | tok/s        | Notes                              |
| ------------------- | ------------ | ---------------------------------- |
| llama.cpp pp128     | ~2630        | uses `simdgroup_matrix` + async    |
| **gpux Q4_0**       | **~620**     | uses `simdgroup_matrix`, no async  |

Long prefill is our remaining frontier — we're competitive on decode but ~4× behind
on prefill. See "What we'd need to be acquisition-grade" below.

### Session-long progression on the same M4

| Stage                                              | Decode tok/s |
| -------------------------------------------------- | ------------ |
| Initial naive baseline                             | 50           |
| Q4_0 quantization + scratch fixes                  | 84           |
| setBytes inline params + tile tuning               | 95           |
| **SIMD-cooperative `rmsnorm_residual` rewrite**    | **193**      |
| Fused RoPE + KV-append (saves 24 dispatches)       | 190          |
| simdgroup_matrix tiled GEMM (long prefill 5×)      | 190          |

The headline takeaway: **profile-driven optimization beat blind micro-tuning by 4×.**

---

## How it's built

### Crates

```
crates/
├── gpux-core         # Metal device, command queue, buffer, profiler
├── gpux-tensor       # Tensor abstraction, all GPU ops, scratch allocator
├── gpux-shaders      # All Metal kernel sources + pipeline registry
├── gpux-nn           # Llama transformer modules (RMSNorm, GQA, SwiGLU)
└── gpux-train        # Tokenizer (BPE), safetensors loader, training utils
examples/
├── inference         # Single-shot bench / generation
├── chat              # Multi-turn REPL
└── server            # OpenAI-compatible HTTP server
```

### Key Metal kernels (~3000 lines, all hand-written)

**Q4_0 (4-bit) quantized matmul family**, with N=896 to 4864 specializations:
- `matvec_a_bt_q4_0_f32` — SIMD matvec (decode, M=1)
- `matvec_a_bt_bias_q4_0_f32` — same with fused bias add
- `matmul_a_bt_q4_0_m_f32` — M-batched (M=2..8, used for spec-decode verify and short prefill)
- `matmul_a_bt_q4_0_sgm2_f32` — tiled GEMM with `simdgroup_matrix` (long prefill, 32×32 output per TG)
- `fused_gate_up_silu_q4_0_f32` and `_m_f32` — single-dispatch SwiGLU (decode + spec)
- `embedding_forward_q4_0_f32` — quantized embedding lookup

**Fusion kernels**:
- `rope_kv_append_fused_f32` — RoPE on Q/K + write K/V to cache, single dispatch
- `fused_attention_decode_f32` — online softmax attention, no intermediate buffers
- `rmsnorm_residual_f16w_f32` — SIMD-cooperative add + RMSNorm

### Architectural choices that pay off

1. **Scratch allocator with bump-pointer + persistent KV cache buffers**.
   Per-token GPU buffer churn was killing us; a single pre-allocated buffer + bump
   offsets per dispatch eliminates it.

2. **Persistent compute encoder**. We keep one `MTLComputeCommandEncoder` alive
   across all dispatches in a token, only ending it on `synchronize()`. Eliminates
   per-dispatch encoder create/destroy overhead.

3. **Inline parameters via `setBytes`**. Most kernels need a tiny `[N, K]` dims
   buffer. We pass these as inline bytes instead of allocating a buffer, saving
   ~440 small allocations per token.

4. **`simdgroup_matrix` for prefill GEMM**. Apple Silicon's hardware matrix units
   process 8×8 tiles in a few cycles. For the M>1 prefill path we use the v2 layout:
   4 SIMD groups cooperate per TG to produce a 32×32 output tile.

5. **Per-kernel GPU profiler**. Synchronizes after every dispatch and times the
   wait on CPU; gives accurate relative kernel cost. Enabled via `GPUX_PROFILE=1`.

---

## What we deliberately don't do

- **Multiple model architectures**. Llama-style only (Qwen2.5 family is verified).
  Adding Llama-3, Phi, Gemma is straightforward but not done.
- **Multi-batch decode**. Server processes requests serially. Adding concurrent
  decode requires per-request KV caches + a request queue.
- **Q4_K_M / Q5_K_M / IQ-* quantization**. Q4_0 only. Q4_K would close some quality
  gap with llama.cpp at the same bandwidth.
- **Long-context attention**. We don't have FlashAttention-2; our online-softmax
  attention is fine up to ~2K context but starts to lose efficiency past that.
- **Vision / multimodal**. Text-only.
- **Training**. There's a `gpux-train` crate with autograd-style ops for pedagogical
  reasons but inference is the focus.

---

## What we'd need to be production-grade

In rough order of impact:

1. **Multi-batch decode** — the biggest serving win. ~3 days.
2. **Llama-3 / Phi-3 / Gemma-2 architecture support** — ~1 week.
3. **Q4_K_M quantization + matvec/gate_up_silu kernels** — ~1 week.
4. **Async TG memory loads in GEMM (`simdgroup_async_copy`)** — closes prefill gap. ~3 days.
5. **FlashAttention-2** — long context performance. ~1 week.
6. **HF token streaming + tool use in server** — ~2 days.
7. **Continuous batching** — production serving. ~2 weeks.

---

## Building & running

```bash
# Requires: macOS on Apple Silicon, Rust 1.75+, ~2GB free disk for model cache
git clone <this-repo> && cd gpux
cargo build --release

# Quick test
cargo run --release -p inference-example -- --quantize q4_0 \
    --prompt "Hello, world" --max-tokens 30 --temperature 0
```

Models auto-download on first use into `~/.cache/gpux/models/`. Default is
Qwen2.5-0.5B; the chat and server examples default to Qwen2.5-0.5B-Instruct.

### Useful environment variables

```bash
GPUX_PROFILE=1            # per-kernel GPU profiler (slower, but shows breakdown)
GPUX_DEBUG_DISPATCH=1     # log dispatch count per token
GPUX_DEBUG_SCRATCH=1      # log scratch allocator usage
GPUX_FUSED_ATTN=0         # disable fused attention kernel (debug)
GPUX_SIMD=0               # disable SIMD matvec (use scalar fallback)
```

---

## Why this exists

Started as "can pure Rust + hand-written Metal beat llama.cpp on small models?"

The answer turned out to be yes for the specific workload of single-token greedy
decode on Qwen2.5-0.5B Q4_0 on M-series hardware. Whether it's *useful* depends on
what you're doing — llama.cpp is a much broader project — but it's a real
demonstration that Apple Silicon Metal compute kernels can be tuned competitive
with the established C++ implementations when you're focused on a narrow target.

License: MIT OR Apache-2.0.
