use std::sync::Arc;
use gpux_tensor::{GpuContext, Tensor};
use crate::module::Module;

pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub eps: f32,
    pub dim: usize,
}

impl LayerNorm {
    pub fn new(ctx: &Arc<GpuContext>, dim: usize) -> gpux_core::Result<Self> {
        let gamma = Tensor::ones(ctx, [dim])?;
        let beta = Tensor::zeros(ctx, [dim])?;
        Ok(Self {
            gamma,
            beta,
            eps: 1e-5,
            dim,
        })
    }

    pub fn forward_with_stats(&self, input: &Tensor) -> gpux_core::Result<(Tensor, Tensor, Tensor)> {
        input.layer_norm(&self.gamma, &self.beta, self.eps)
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        let (out, _, _) = self.forward_with_stats(input)?;
        Ok(out)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }
}
