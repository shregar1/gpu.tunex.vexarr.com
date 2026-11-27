use std::sync::Arc;
use gpux_tensor::{GpuContext, Tensor};
use crate::llama::LlamaConfig;

pub struct KvCache {
    pub k_cache: Vec<Tensor>,
    pub v_cache: Vec<Tensor>,
    pub current_pos: usize,
    pub max_seq_len: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub batch_size: usize,
}

impl KvCache {
    pub fn new(
        ctx: &Arc<GpuContext>,
        config: &LlamaConfig,
        max_seq_len: usize,
        batch_size: usize,
    ) -> gpux_core::Result<Self> {
        let bh_kv = batch_size * config.num_key_value_heads;

        let mut k_cache = Vec::with_capacity(config.num_hidden_layers);
        let mut v_cache = Vec::with_capacity(config.num_hidden_layers);

        for _ in 0..config.num_hidden_layers {
            k_cache.push(Tensor::zeros_persistent(ctx, vec![bh_kv, max_seq_len, config.head_dim])?);
            v_cache.push(Tensor::zeros_persistent(ctx, vec![bh_kv, max_seq_len, config.head_dim])?);
        }

        Ok(Self {
            k_cache,
            v_cache,
            current_pos: 0,
            max_seq_len,
            n_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            batch_size,
        })
    }

    pub fn append_kv(
        &self,
        layer_idx: usize,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> gpux_core::Result<()> {
        Tensor::kv_cache_append_fused(
            new_k, new_v,
            &self.k_cache[layer_idx], &self.v_cache[layer_idx],
            self.current_pos, self.max_seq_len,
        )
    }

    pub fn advance(&mut self, n_tokens: usize) {
        self.current_pos += n_tokens;
    }

    pub fn seq_len(&self) -> usize {
        self.current_pos
    }

    pub fn reset(&mut self) {
        self.current_pos = 0;
    }

    pub fn k_at(&self, layer: usize) -> &Tensor {
        &self.k_cache[layer]
    }

    pub fn v_at(&self, layer: usize) -> &Tensor {
        &self.v_cache[layer]
    }
}
