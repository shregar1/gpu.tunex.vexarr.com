use std::sync::Arc;
use gpux_tensor::{GpuContext, Tensor};
use crate::module::Module;

pub struct Embedding {
    pub weight: Tensor, // [vocab_size, embed_dim]
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl Embedding {
    pub fn new(ctx: &Arc<GpuContext>, vocab_size: usize, embed_dim: usize) -> gpux_core::Result<Self> {
        let weight = Tensor::randn(ctx, [vocab_size, embed_dim], 0.02)?;
        Ok(Self {
            weight,
            vocab_size,
            embed_dim,
        })
    }

    pub fn lookup(&self, indices: &Tensor) -> gpux_core::Result<Tensor> {
        self.weight.embedding_lookup(indices)
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        self.lookup(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }
}

pub struct PositionEmbedding {
    pub weight: Tensor, // [max_seq_len, embed_dim]
}

impl PositionEmbedding {
    pub fn new(ctx: &Arc<GpuContext>, max_seq_len: usize, embed_dim: usize) -> gpux_core::Result<Self> {
        let weight = Tensor::randn(ctx, [max_seq_len, embed_dim], 0.02)?;
        Ok(Self { weight })
    }

    pub fn forward(&self, x: &Tensor) -> gpux_core::Result<Tensor> {
        x.add_position_embedding(&self.weight)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }
}
