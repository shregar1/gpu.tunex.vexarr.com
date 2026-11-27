use std::sync::Arc;
use std::path::Path;
use std::io::{Read, Write, BufReader, BufWriter};
use gpux_tensor::{GpuContext, Tensor};
use crate::attention::CausalSelfAttention;
use crate::embedding::{Embedding, PositionEmbedding};
use crate::layernorm::LayerNorm;
use crate::linear::Linear;
use crate::mlp::MLP;
use crate::module::Module;

pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: CausalSelfAttention,
    pub ln2: LayerNorm,
    pub mlp: MLP,
}

impl TransformerBlock {
    pub fn new(ctx: &Arc<GpuContext>, embed_dim: usize, n_heads: usize) -> gpux_core::Result<Self> {
        let mlp_hidden = 4 * embed_dim;
        Ok(Self {
            ln1: LayerNorm::new(ctx, embed_dim)?,
            attn: CausalSelfAttention::new(ctx, embed_dim, n_heads)?,
            ln2: LayerNorm::new(ctx, embed_dim)?,
            mlp: MLP::new(ctx, embed_dim, mlp_hidden)?,
        })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        // Pre-norm architecture (GPT-2 style)
        let normed = self.ln1.forward(input)?;
        let attn_out = self.attn.forward(&normed)?;
        let residual1 = input.add(&attn_out)?;

        let normed2 = self.ln2.forward(&residual1)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        let residual2 = residual1.add(&mlp_out)?;

        Ok(residual2)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.ln1.parameters();
        params.extend(self.attn.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.mlp.parameters());
        params
    }
}

pub struct GPTConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub embed_dim: usize,
}

impl GPTConfig {
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            max_seq_len: 1024,
            n_layers: 12,
            n_heads: 12,
            embed_dim: 768,
        }
    }

    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50257,
            max_seq_len: 1024,
            n_layers: 24,
            n_heads: 16,
            embed_dim: 1024,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            max_seq_len: 128,
            n_layers: 4,
            n_heads: 4,
            embed_dim: 128,
        }
    }

    pub fn small_lm() -> Self {
        Self {
            vocab_size: 32000,
            max_seq_len: 512,
            n_layers: 6,
            n_heads: 8,
            embed_dim: 512,
        }
    }
}

pub struct GPT {
    pub token_embedding: Embedding,
    pub position_embedding: PositionEmbedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_final: LayerNorm,
    pub lm_head: Linear,
    pub config: GPTConfig,
}

impl GPT {
    pub fn new(ctx: &Arc<GpuContext>, config: GPTConfig) -> gpux_core::Result<Self> {
        let token_embedding = Embedding::new(ctx, config.vocab_size, config.embed_dim)?;
        let position_embedding = PositionEmbedding::new(ctx, config.max_seq_len, config.embed_dim)?;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(TransformerBlock::new(ctx, config.embed_dim, config.n_heads)?);
        }

        let ln_final = LayerNorm::new(ctx, config.embed_dim)?;
        let lm_head = Linear::new(ctx, config.embed_dim, config.vocab_size, false)?;

        Ok(Self {
            token_embedding,
            position_embedding,
            blocks,
            ln_final,
            lm_head,
            config,
        })
    }

    pub fn forward_lm(&self, input_ids: &Tensor) -> gpux_core::Result<Tensor> {
        // Token embeddings
        let tok_emb = self.token_embedding.lookup(input_ids)?;

        // Add position embeddings
        let hidden = self.position_embedding.forward(&tok_emb)?;

        // Transformer blocks
        let mut x = hidden;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Final layer norm
        let x = self.ln_final.forward(&x)?;

        // LM head -> logits
        let logits = self.lm_head.forward(&x)?;

        Ok(logits)
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        count += self.token_embedding.weight.numel();
        count += self.position_embedding.weight.numel();
        for block in &self.blocks {
            count += block.parameters().iter().map(|p| p.numel()).sum::<usize>();
        }
        count += self.ln_final.parameters().iter().map(|p| p.numel()).sum::<usize>();
        count += self.lm_head.parameters().iter().map(|p| p.numel()).sum::<usize>();
        count
    }

    pub fn all_parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![
            &self.token_embedding.weight,
            &self.position_embedding.weight,
        ];
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_final.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    pub fn save_checkpoint(&self, path: &Path) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(b"GPUX")?;
        let version: u32 = 1;
        w.write_all(&version.to_le_bytes())?;

        let config_vals: [u32; 5] = [
            self.config.vocab_size as u32,
            self.config.max_seq_len as u32,
            self.config.n_layers as u32,
            self.config.n_heads as u32,
            self.config.embed_dim as u32,
        ];
        for v in &config_vals {
            w.write_all(&v.to_le_bytes())?;
        }

        let params = self.all_parameters();
        let num_params: u32 = params.len() as u32;
        w.write_all(&num_params.to_le_bytes())?;

        for param in &params {
            let dims = param.shape().dims();
            let ndim: u32 = dims.len() as u32;
            w.write_all(&ndim.to_le_bytes())?;
            for &d in dims {
                w.write_all(&(d as u32).to_le_bytes())?;
            }
            let data = param.to_vec();
            let bytes: &[u8] = bytemuck::cast_slice(&data);
            w.write_all(bytes)?;
        }

        w.flush()?;
        Ok(())
    }

    pub fn load_checkpoint(ctx: &Arc<GpuContext>, path: &Path) -> gpux_core::Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| gpux_core::GpuError::Other(format!("Failed to open checkpoint: {}", e)))?;
        let mut r = BufReader::new(file);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)
            .map_err(|e| gpux_core::GpuError::Other(format!("Read error: {}", e)))?;
        if &magic != b"GPUX" {
            return Err(gpux_core::GpuError::Other("Invalid checkpoint magic".into()));
        }

        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)
            .map_err(|e| gpux_core::GpuError::Other(format!("Read error: {}", e)))?;
        let _version = u32::from_le_bytes(buf4);

        let mut config_vals = [0u32; 5];
        for v in &mut config_vals {
            r.read_exact(&mut buf4)
                .map_err(|e| gpux_core::GpuError::Other(format!("Read error: {}", e)))?;
            *v = u32::from_le_bytes(buf4);
        }

        let config = GPTConfig {
            vocab_size: config_vals[0] as usize,
            max_seq_len: config_vals[1] as usize,
            n_layers: config_vals[2] as usize,
            n_heads: config_vals[3] as usize,
            embed_dim: config_vals[4] as usize,
        };

        let model = GPT::new(ctx, config)?;

        r.read_exact(&mut buf4)
            .map_err(|e| gpux_core::GpuError::Other(format!("Read error: {}", e)))?;
        let num_params = u32::from_le_bytes(buf4) as usize;

        let params = model.all_parameters();
        if num_params != params.len() {
            return Err(gpux_core::GpuError::Other(
                format!("Checkpoint has {} params, model has {}", num_params, params.len()),
            ));
        }

        for param in &params {
            r.read_exact(&mut buf4)
                .map_err(|e| gpux_core::GpuError::Other(format!("Read error: {}", e)))?;
            let ndim = u32::from_le_bytes(buf4) as usize;

            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                r.read_exact(&mut buf4)
                    .map_err(|e| gpux_core::GpuError::Other(format!("Read error: {}", e)))?;
                shape.push(u32::from_le_bytes(buf4) as usize);
            }

            let numel: usize = shape.iter().product();
            let mut float_bytes = vec![0u8; numel * 4];
            r.read_exact(&mut float_bytes)
                .map_err(|e| gpux_core::GpuError::Other(format!("Read error: {}", e)))?;
            let data: &[f32] = bytemuck::cast_slice(&float_bytes);
            param.buffer.write_data(data);
        }

        Ok(model)
    }
}
