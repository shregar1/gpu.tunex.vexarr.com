use std::sync::Arc;
use gpux_tensor::{GpuContext, Tensor};
use crate::linear::Linear;
use crate::module::Module;

pub struct MLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl MLP {
    pub fn new(ctx: &Arc<GpuContext>, embed_dim: usize, hidden_dim: usize) -> gpux_core::Result<Self> {
        let fc1 = Linear::new(ctx, embed_dim, hidden_dim, true)?;
        let fc2 = Linear::new(ctx, hidden_dim, embed_dim, true)?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward_with_cache(&self, x: &Tensor) -> gpux_core::Result<(Tensor, Tensor)> {
        let hidden = self.fc1.forward(x)?;
        let activated = hidden.gelu()?;
        let out = self.fc2.forward(&activated)?;
        Ok((out, hidden))
    }
}

impl Module for MLP {
    fn forward(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        let (out, _) = self.forward_with_cache(input)?;
        Ok(out)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}
