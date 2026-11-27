use std::sync::Arc;
use gpux_tensor::{DType, GpuContext, Tensor};
use crate::linear::Linear;
use crate::module::Module;

pub struct SwiGLU {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl SwiGLU {
    pub fn new(ctx: &Arc<GpuContext>, hidden_size: usize, intermediate_size: usize) -> gpux_core::Result<Self> {
        let gate_proj = Linear::new(ctx, hidden_size, intermediate_size, false)?;
        let up_proj = Linear::new(ctx, hidden_size, intermediate_size, false)?;
        let down_proj = Linear::new(ctx, intermediate_size, hidden_size, false)?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }
}

impl SwiGLU {
    pub fn forward_fused(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        let batch_size = input.shape().numel() / input.shape().last_dim();

        // Fused fast path:
        //   • M=1: any non-Q4_K dtype (Q4_K has no fused gate_up_silu kernel yet)
        //   • M=2..8 with Q4_0 weights: spec-decode verify path uses the M-batched
        //     gate_up_silu kernel — single dispatch instead of {gate, up, silu_mul}.
        let dtype = self.gate_proj.weight.dtype;
        let m1_fused_supported = batch_size == 1 && dtype != DType::Q4_K;
        let q4_m_fast = batch_size > 1 && batch_size <= 8 && dtype == DType::Q4_0;
        if m1_fused_supported || q4_m_fast {
            let orig_shape = input.shape().clone();
            let in_features = orig_shape.last_dim();
            let _out_features = self.gate_proj.weight.shape().dims()[0];
            let input_2d = input.reshape(vec![batch_size, in_features]);
            let hidden = input_2d.fused_gate_up_silu(&self.gate_proj.weight, &self.up_proj.weight)?;
            let down_out = self.down_proj.forward(&hidden)?;
            if orig_shape.ndim() > 2 {
                let mut out_dims = orig_shape.dims()[..orig_shape.ndim() - 1].to_vec();
                out_dims.push(self.down_proj.weight.shape().dims()[0]);
                return Ok(down_out.reshape(out_dims));
            }
            return Ok(down_out);
        }

        // Fallback: separate gate, up, silu_mul, down (used for prefill M>8 or non-Q4_0 M>1).
        let gate = self.gate_proj.forward(input)?;
        let up = self.up_proj.forward(input)?;
        let hidden = gate.silu_mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

impl Module for SwiGLU {
    fn forward(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        self.forward_fused(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.gate_proj.parameters();
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }
}
