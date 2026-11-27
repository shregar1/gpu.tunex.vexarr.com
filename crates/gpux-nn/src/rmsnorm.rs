use std::sync::Arc;
use gpux_tensor::{GpuContext, Tensor};
use crate::module::Module;

pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(ctx: &Arc<GpuContext>, dim: usize) -> gpux_core::Result<Self> {
        let weight = Tensor::ones(ctx, [dim])?;
        Ok(Self { weight, eps: 1e-6 })
    }

    pub fn forward_norm(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        let (out, _) = input.rms_norm(&self.weight, self.eps)?;
        Ok(out)
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        self.forward_norm(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }
}
