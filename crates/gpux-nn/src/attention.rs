use std::sync::Arc;
use gpux_tensor::{GpuContext, Tensor};
use crate::linear::Linear;
use crate::module::Module;

pub struct CausalSelfAttention {
    pub qkv_proj: Linear,
    pub out_proj: Linear,
    pub n_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
}

pub struct AttentionCache {
    pub input: Tensor,
    pub q: Tensor,        // [B*H, T, D]
    pub k: Tensor,        // [B*H, T, D]
    pub v: Tensor,        // [B*H, T, D]
    pub attn_weights: Tensor, // [B*H, T, T]
    pub attn_out_pre_reshape: Tensor, // [B*H, T, D]
    pub attn_out: Tensor, // [B, T, C] (after reshape, before out_proj)
}

impl CausalSelfAttention {
    pub fn new(ctx: &Arc<GpuContext>, embed_dim: usize, n_heads: usize) -> gpux_core::Result<Self> {
        assert_eq!(embed_dim % n_heads, 0);
        let head_dim = embed_dim / n_heads;

        let qkv_proj = Linear::new(ctx, embed_dim, 3 * embed_dim, true)?;
        let out_proj = Linear::new(ctx, embed_dim, embed_dim, true)?;

        Ok(Self {
            qkv_proj,
            out_proj,
            n_heads,
            head_dim,
            embed_dim,
        })
    }

    pub fn forward_with_cache(&self, x: &Tensor) -> gpux_core::Result<(Tensor, AttentionCache)> {
        let batch = x.shape().dims()[0];
        let seq_len = x.shape().dims()[1];

        let qkv = self.qkv_proj.forward(x)?;
        let (q_2d, k_2d, v_2d) = qkv.split_qkv(self.embed_dim)?;
        let q_3d = q_2d.reshape(vec![batch, seq_len, self.embed_dim]);
        let k_3d = k_2d.reshape(vec![batch, seq_len, self.embed_dim]);
        let v_3d = v_2d.reshape(vec![batch, seq_len, self.embed_dim]);

        let q = q_3d.reshape_for_attention(batch, seq_len, self.n_heads, self.head_dim)?;
        let k = k_3d.reshape_for_attention(batch, seq_len, self.n_heads, self.head_dim)?;
        let v = v_3d.reshape_for_attention(batch, seq_len, self.n_heads, self.head_dim)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = q.attention_scores(&k, scale)?;
        let attn_weights = scores.softmax()?;
        let attn_out_pre = attn_weights.attention_weighted_values(&v)?;
        let attn_out = attn_out_pre.reshape_from_attention(batch, seq_len, self.n_heads, self.head_dim)?;

        let output = self.out_proj.forward(&attn_out)?;

        let cache = AttentionCache {
            input: x.clone(),
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            attn_weights,
            attn_out_pre_reshape: attn_out_pre,
            attn_out,
        };

        Ok((output, cache))
    }

    pub fn forward_attn(&self, x: &Tensor) -> gpux_core::Result<Tensor> {
        let (output, _) = self.forward_with_cache(x)?;
        Ok(output)
    }
}

impl Module for CausalSelfAttention {
    fn forward(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        self.forward_attn(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.qkv_proj.parameters();
        params.extend(self.out_proj.parameters());
        params
    }
}
