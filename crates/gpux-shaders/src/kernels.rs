pub const ELEMENTWISE_SHADER: &str = include_str!("metal/elementwise.metal");
pub const MATMUL_SHADER: &str = include_str!("metal/matmul.metal");
pub const REDUCE_SHADER: &str = include_str!("metal/reduce.metal");
pub const SOFTMAX_SHADER: &str = include_str!("metal/softmax.metal");
pub const LAYERNORM_SHADER: &str = include_str!("metal/layernorm.metal");
pub const ATTENTION_SHADER: &str = include_str!("metal/attention.metal");
pub const EMBEDDING_SHADER: &str = include_str!("metal/embedding.metal");
pub const OPTIMIZER_SHADER: &str = include_str!("metal/optimizer.metal");
pub const TRANSPOSE_SHADER: &str = include_str!("metal/transpose.metal");
pub const CAST_SHADER: &str = include_str!("metal/cast.metal");
pub const KV_CACHE_SHADER: &str = include_str!("metal/kv_cache.metal");
pub const QUANTIZED_SHADER: &str = include_str!("metal/quantized.metal");
pub const FUSED_SHADER: &str = include_str!("metal/fused.metal");

pub struct ShaderSource {
    pub source: &'static str,
    pub functions: &'static [&'static str],
}

pub fn all_shaders() -> Vec<ShaderSource> {
    vec![
        ShaderSource {
            source: ELEMENTWISE_SHADER,
            functions: &[
                "add_f32", "add_scalar_f32", "sub_f32", "mul_f32", "mul_scalar_f32",
                "div_f32", "neg_f32", "exp_f32", "log_f32", "sqrt_f32", "rsqrt_f32",
                "tanh_f32", "relu_f32", "relu_backward_f32", "gelu_f32",
                "gelu_backward_f32", "silu_f32", "fill_f32", "pow_f32", "clamp_f32",
            ],
        },
        ShaderSource {
            source: MATMUL_SHADER,
            functions: &[
                "matmul_f32", "matmul_at_b_f32", "matmul_a_bt_f32",
                "batched_matmul_f32", "batched_matmul_a_bt_f32",
                "batched_matmul_at_b_f32", "outer_product_acc_f32",
            ],
        },
        ShaderSource {
            source: REDUCE_SHADER,
            functions: &[
                "sum_rows_f32", "sum_all_f32", "max_rows_f32", "mean_rows_f32",
                "var_rows_f32", "sum_cols_f32", "argmax_rows_f32",
            ],
        },
        ShaderSource {
            source: SOFTMAX_SHADER,
            functions: &[
                "softmax_f32", "softmax_backward_f32",
                "cross_entropy_forward_f32", "cross_entropy_backward_f32",
            ],
        },
        ShaderSource {
            source: LAYERNORM_SHADER,
            functions: &[
                "layernorm_forward_f32", "layernorm_backward_f32",
                "layernorm_grad_gamma_beta_f32", "rmsnorm_forward_f32",
            ],
        },
        ShaderSource {
            source: ATTENTION_SHADER,
            functions: &[
                "causal_mask_f32", "attention_scale_f32", "attention_scores_f32",
                "attention_weighted_values_f32", "reshape_for_attention_f32",
                "reshape_from_attention_f32", "rope_f32",
                "gqa_attention_scores_f32", "gqa_attention_weighted_values_f32",
            ],
        },
        ShaderSource {
            source: EMBEDDING_SHADER,
            functions: &[
                "embedding_forward_f32", "embedding_backward_f32",
                "add_position_embedding_f32",
            ],
        },
        ShaderSource {
            source: OPTIMIZER_SHADER,
            functions: &[
                "adamw_step_f32", "sgd_momentum_step_f32",
                "clip_grad_norm_f32", "squared_norm_f32",
            ],
        },
        ShaderSource {
            source: TRANSPOSE_SHADER,
            functions: &[
                "transpose_2d_f32", "copy_f32", "accumulate_f32",
                "broadcast_add_bias_f32", "concat_cols_f32", "slice_cols_f32",
            ],
        },
        ShaderSource {
            source: CAST_SHADER,
            functions: &[
                "cast_f32_to_f16", "cast_f16_to_f32",
                "add_f16", "mul_f16", "mul_scalar_f16", "gelu_f16",
                "matmul_f16", "matmul_a_bt_f16", "matmul_at_b_f16",
                "layernorm_forward_f16",
            ],
        },
        ShaderSource {
            source: KV_CACHE_SHADER,
            functions: &[
                "kv_cache_append_f32", "kv_cache_append_fused_f32",
                "rope_kv_append_fused_f32",
                "rope_with_offset_f32",
                "fused_attention_decode_f32",
                "gqa_attention_scores_cached_f32",
                "gqa_attention_weighted_values_cached_f32",
            ],
        },
        ShaderSource {
            source: QUANTIZED_SHADER,
            functions: &[
                "matmul_q8_0_f32", "matmul_q4_0_f32",
                "matmul_a_bt_q4_0_m_f32",
                "matmul_a_bt_q4_0_sgm_f32",
                "matmul_a_bt_q4_0_sgm2_f32",
                "matmul_a_bt_q4_0_sgm3_f32",
                "matmul_a_bt_q4_0_sgm4_f32",
                "matvec_a_bt_q4_0_f32", "matvec_a_bt_bias_q4_0_f32",
                "matvec_a_bt_q8_0_f32", "matvec_a_bt_bias_q8_0_f32",
                "matvec_a_bt_q4_k_f32", "matvec_a_bt_bias_q4_k_f32",
                "matvec_a_bt_q4_k_v2_f32", "matvec_a_bt_bias_q4_k_v2_f32",
                "quantize_q4_k_f32",
                "fused_gate_up_silu_q4_0_f32", "fused_gate_up_silu_q8_0_f32",
                "fused_gate_up_silu_q4_0_m_f32",
                "embedding_forward_q4_0_f32", "embedding_forward_q8_0_f32",
                "quantize_q8_0_f32", "quantize_q4_0_f32",
                "dequantize_q8_0_f32", "dequantize_q4_0_f32",
            ],
        },
        ShaderSource {
            source: FUSED_SHADER,
            functions: &[
                "rope_fused_f32", "rope_with_offset_fused_f32",
                "silu_mul_f32", "rmsnorm_residual_f32",
                "fused_gate_up_silu_f16w_f32", "fused_gate_up_silu_f32",
                "matvec_a_bt_f32", "matvec_a_bt_bias_f32",
                "matvec_scalar_f16w_f32",
                "matvec_a_bt_f16w_f32", "matvec_a_bt_bias_f16w_f32",
                "matmul_a_bt_f16w_f32",
                "embedding_forward_f16w_f32",
                "rmsnorm_forward_f16w_f32", "rmsnorm_residual_f16w_f32",
            ],
        },
    ]
}
