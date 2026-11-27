use std::sync::Arc;
use gpux_tensor::{GpuContext, Tensor};
use crate::attention::AttentionCache;
use crate::module::Module;
use crate::transformer::{GPT, TransformerBlock};

pub struct GPTGrads {
    pub token_emb_grad: Tensor,
    pub pos_emb_grad: Tensor,
    pub block_grads: Vec<TransformerBlockGrads>,
    pub ln_final_gamma_grad: Tensor,
    pub ln_final_beta_grad: Tensor,
    pub lm_head_weight_grad: Tensor,
}

pub struct TransformerBlockGrads {
    pub ln1_gamma_grad: Tensor,
    pub ln1_beta_grad: Tensor,
    pub qkv_weight_grad: Tensor,
    pub qkv_bias_grad: Tensor,
    pub out_proj_weight_grad: Tensor,
    pub out_proj_bias_grad: Tensor,
    pub ln2_gamma_grad: Tensor,
    pub ln2_beta_grad: Tensor,
    pub fc1_weight_grad: Tensor,
    pub fc1_bias_grad: Tensor,
    pub fc2_weight_grad: Tensor,
    pub fc2_bias_grad: Tensor,
}

pub struct ForwardCache {
    pub tok_emb: Tensor,
    pub hidden_after_pos: Tensor,
    pub block_caches: Vec<BlockForwardCache>,
    pub final_normed: Tensor,
    pub final_ln_mean: Tensor,
    pub final_ln_rstd: Tensor,
    pub logits: Tensor,
}

pub struct BlockForwardCache {
    pub input: Tensor,
    pub ln1_out: Tensor,
    pub ln1_mean: Tensor,
    pub ln1_rstd: Tensor,
    pub attn_cache: AttentionCache,
    pub attn_proj_out: Tensor,
    pub residual1: Tensor,
    pub ln2_out: Tensor,
    pub ln2_mean: Tensor,
    pub ln2_rstd: Tensor,
    pub fc1_out: Tensor,
    pub gelu_input: Tensor,
    pub mlp_out: Tensor,
    pub output: Tensor,
}

impl GPT {
    pub fn forward_with_cache(&self, input_ids: &Tensor) -> gpux_core::Result<ForwardCache> {
        let tok_emb = self.token_embedding.lookup(input_ids)?;
        let hidden_after_pos = self.position_embedding.forward(&tok_emb)?;

        let mut x = hidden_after_pos.clone();
        let mut block_caches = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            let input = x.clone();

            let (ln1_out, ln1_mean, ln1_rstd) = block.ln1.forward_with_stats(&input)?;
            let (attn_proj_out, attn_cache) = block.attn.forward_with_cache(&ln1_out)?;
            let residual1 = input.add(&attn_proj_out)?;

            let (ln2_out, ln2_mean, ln2_rstd) = block.ln2.forward_with_stats(&residual1)?;
            let fc1_out = block.mlp.fc1.forward(&ln2_out)?;
            let gelu_input = fc1_out.clone();
            let activated = fc1_out.gelu()?;
            let mlp_out = block.mlp.fc2.forward(&activated)?;

            x = residual1.add(&mlp_out)?;

            block_caches.push(BlockForwardCache {
                input,
                ln1_out,
                ln1_mean,
                ln1_rstd,
                attn_cache,
                attn_proj_out,
                residual1,
                ln2_out,
                ln2_mean,
                ln2_rstd,
                fc1_out: gelu_input.clone(),
                gelu_input,
                mlp_out,
                output: x.clone(),
            });
        }

        let (final_normed, final_ln_mean, final_ln_rstd) = self.ln_final.forward_with_stats(&x)?;
        let logits = self.lm_head.forward(&final_normed)?;

        Ok(ForwardCache {
            tok_emb,
            hidden_after_pos,
            block_caches,
            final_normed,
            final_ln_mean,
            final_ln_rstd,
            logits,
        })
    }

    pub fn backward(
        &self,
        cache: &ForwardCache,
        grad_logits: &Tensor,
        input_ids: &Tensor,
        _ctx: &Arc<GpuContext>,
    ) -> gpux_core::Result<GPTGrads> {
        let batch = cache.logits.shape().dims()[0];
        let seq_len = cache.logits.shape().dims()[1];
        let vocab = cache.logits.shape().dims()[2];
        let embed = self.config.embed_dim;

        // === LM head backward (zero-copy reshapes) ===
        let grad_logits_2d = grad_logits.reshape(vec![batch * seq_len, vocab]);
        let final_normed_2d = cache.final_normed.reshape(vec![batch * seq_len, embed]);

        let lm_head_weight_grad = grad_logits_2d.matmul_at_b(&final_normed_2d)?;
        let grad_final_normed_2d = grad_logits_2d.matmul(&self.lm_head.weight)?;
        let grad_final_normed = grad_final_normed_2d.reshape(vec![batch, seq_len, embed]);

        // === Final LayerNorm backward ===
        let x_before_final_ln = if self.blocks.is_empty() {
            &cache.hidden_after_pos
        } else {
            &cache.block_caches.last().unwrap().output
        };

        let (grad_before_final_ln, ln_final_gamma_grad, ln_final_beta_grad) =
            grad_final_normed.layernorm_backward(
                x_before_final_ln,
                &self.ln_final.gamma,
                &cache.final_ln_mean,
                &cache.final_ln_rstd,
            )?;

        // === Transformer blocks backward (reverse order) ===
        let mut grad_x = grad_before_final_ln;
        let mut block_grads = Vec::with_capacity(self.blocks.len());

        for (_i, (block, block_cache)) in self.blocks.iter().zip(cache.block_caches.iter()).enumerate().rev() {
            let bg = self.backward_block(block, block_cache, &grad_x)?;
            grad_x = bg.grad_input;
            block_grads.push(bg.grads);
        }
        block_grads.reverse();

        // === Position + token embedding gradients ===
        let pos_emb_grad = grad_x.sum_cols()?;
        let token_emb_grad = grad_x.embedding_backward(input_ids, self.config.vocab_size)?;

        Ok(GPTGrads {
            token_emb_grad,
            pos_emb_grad,
            block_grads,
            ln_final_gamma_grad,
            ln_final_beta_grad,
            lm_head_weight_grad,
        })
    }

    fn backward_block(
        &self,
        block: &TransformerBlock,
        cache: &BlockForwardCache,
        grad_output: &Tensor,
    ) -> gpux_core::Result<BlockBackwardResult> {
        let batch = grad_output.shape().dims()[0];
        let seq_len = grad_output.shape().dims()[1];
        let embed_dim = grad_output.shape().dims()[2];
        let bt = batch * seq_len;

        // === MLP backward ===
        let activated = cache.fc1_out.gelu()?;
        let grad_mlp_2d = grad_output.reshape(vec![bt, embed_dim]);
        let activated_2d = activated.reshape(vec![bt, 4 * embed_dim]);

        let fc2_weight_grad = grad_mlp_2d.matmul_at_b(&activated_2d)?;
        let fc2_bias_grad = grad_mlp_2d.sum_cols()?;
        let grad_activated = grad_mlp_2d.matmul(&block.mlp.fc2.weight)?;

        let fc1_out_2d = cache.fc1_out.reshape(vec![bt, 4 * embed_dim]);
        let grad_fc1_out = grad_activated.gelu_backward(&fc1_out_2d)?;

        let ln2_out_2d = cache.ln2_out.reshape(vec![bt, embed_dim]);
        let fc1_weight_grad = grad_fc1_out.matmul_at_b(&ln2_out_2d)?;
        let fc1_bias_grad = grad_fc1_out.sum_cols()?;

        let grad_ln2_out = grad_fc1_out.matmul(&block.mlp.fc1.weight)?;
        let grad_ln2_3d = grad_ln2_out.reshape(vec![batch, seq_len, embed_dim]);

        let (grad_residual1_from_mlp, ln2_gamma_grad, ln2_beta_grad) =
            grad_ln2_3d.layernorm_backward(
                &cache.residual1,
                &block.ln2.gamma,
                &cache.ln2_mean,
                &cache.ln2_rstd,
            )?;

        let grad_residual1 = grad_residual1_from_mlp.add(grad_output)?;

        // === Attention backward (full) ===
        let n_heads = block.attn.n_heads;
        let head_dim = block.attn.head_dim;
        let bh = batch * n_heads;

        let grad_attn_2d = grad_residual1.reshape(vec![bt, embed_dim]);
        let attn_out_2d = cache.attn_cache.attn_out.reshape(vec![bt, embed_dim]);

        let out_proj_weight_grad = grad_attn_2d.matmul_at_b(&attn_out_2d)?;
        let out_proj_bias_grad = grad_attn_2d.sum_cols()?;
        let grad_attn_out_2d = grad_attn_2d.matmul(&block.attn.out_proj.weight)?;

        let grad_attn_out_3d = grad_attn_out_2d.reshape(vec![batch, seq_len, embed_dim]);
        let grad_attn_out_heads = grad_attn_out_3d.reshape_for_attention(batch, seq_len, n_heads, head_dim)?;

        let grad_weights = grad_attn_out_heads.batched_matmul_a_bt(&cache.attn_cache.v)?;
        let grad_v = cache.attn_cache.attn_weights.batched_matmul_at_b(&grad_attn_out_heads)?;

        let grad_weights_2d = grad_weights.reshape(vec![bh * seq_len, seq_len]);
        let weights_2d = cache.attn_cache.attn_weights.reshape(vec![bh * seq_len, seq_len]);
        let grad_scores_2d = grad_weights_2d.softmax_backward(&weights_2d)?;
        let grad_scores = grad_scores_2d.reshape(vec![bh, seq_len, seq_len]);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let grad_q = grad_scores.batched_matmul(&cache.attn_cache.k)?.mul_scalar(scale)?;
        let grad_k = grad_scores.batched_matmul_at_b(&cache.attn_cache.q)?.mul_scalar(scale)?;

        let grad_q_3d = grad_q.reshape_from_attention(batch, seq_len, n_heads, head_dim)?;
        let grad_k_3d = grad_k.reshape_from_attention(batch, seq_len, n_heads, head_dim)?;
        let grad_v_3d = grad_v.reshape_from_attention(batch, seq_len, n_heads, head_dim)?;

        let grad_q_2d = grad_q_3d.reshape(vec![bt, embed_dim]);
        let grad_k_2d = grad_k_3d.reshape(vec![bt, embed_dim]);
        let grad_v_2d = grad_v_3d.reshape(vec![bt, embed_dim]);
        let grad_qkv_2d = Tensor::concat_qkv(&grad_q_2d, &grad_k_2d, &grad_v_2d)?;

        let ln1_out_2d = cache.ln1_out.reshape(vec![bt, embed_dim]);
        let qkv_weight_grad = grad_qkv_2d.matmul_at_b(&ln1_out_2d)?;
        let qkv_bias_grad = grad_qkv_2d.sum_cols()?;
        let grad_ln1_2d = grad_qkv_2d.matmul(&block.attn.qkv_proj.weight)?;
        let grad_ln1_3d = grad_ln1_2d.reshape(vec![batch, seq_len, embed_dim]);

        let (grad_input_from_attn, ln1_gamma_grad, ln1_beta_grad) =
            grad_ln1_3d.layernorm_backward(
                &cache.input,
                &block.ln1.gamma,
                &cache.ln1_mean,
                &cache.ln1_rstd,
            )?;

        let grad_input = grad_input_from_attn.add(&grad_residual1)?;

        Ok(BlockBackwardResult {
            grad_input,
            grads: TransformerBlockGrads {
                ln1_gamma_grad,
                ln1_beta_grad,
                qkv_weight_grad,
                qkv_bias_grad,
                out_proj_weight_grad,
                out_proj_bias_grad,
                ln2_gamma_grad,
                ln2_beta_grad,
                fc1_weight_grad,
                fc1_bias_grad,
                fc2_weight_grad,
                fc2_bias_grad,
            },
        })
    }
}

struct BlockBackwardResult {
    grad_input: Tensor,
    grads: TransformerBlockGrads,
}

impl GPTGrads {
    pub fn all_grads(&self) -> Vec<&Tensor> {
        let mut grads = vec![
            &self.token_emb_grad,
            &self.pos_emb_grad,
        ];
        for bg in &self.block_grads {
            grads.push(&bg.ln1_gamma_grad);
            grads.push(&bg.ln1_beta_grad);
            grads.push(&bg.qkv_weight_grad);
            grads.push(&bg.qkv_bias_grad);
            grads.push(&bg.out_proj_weight_grad);
            grads.push(&bg.out_proj_bias_grad);
            grads.push(&bg.ln2_gamma_grad);
            grads.push(&bg.ln2_beta_grad);
            grads.push(&bg.fc1_weight_grad);
            grads.push(&bg.fc1_bias_grad);
            grads.push(&bg.fc2_weight_grad);
            grads.push(&bg.fc2_bias_grad);
        }
        grads.push(&self.ln_final_gamma_grad);
        grads.push(&self.ln_final_beta_grad);
        grads.push(&self.lm_head_weight_grad);
        grads
    }
}
