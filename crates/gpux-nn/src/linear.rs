use std::sync::Arc;
use gpux_tensor::{GpuContext, Tensor};
use crate::module::Module;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub lora_a: Option<Tensor>,
    pub lora_b: Option<Tensor>,
    pub lora_scaling: f32,
}

impl Linear {
    pub fn new(ctx: &Arc<GpuContext>, in_features: usize, out_features: usize, use_bias: bool) -> gpux_core::Result<Self> {
        let weight = Tensor::kaiming_uniform(ctx, [out_features, in_features])?;
        let bias = if use_bias {
            Some(Tensor::zeros(ctx, [out_features])?)
        } else {
            None
        };
        Ok(Self { weight, bias, lora_a: None, lora_b: None, lora_scaling: 1.0 })
    }

    pub fn set_lora(&mut self, lora_a: Tensor, lora_b: Tensor, alpha: f32, rank: f32) {
        self.lora_scaling = alpha / rank;
        self.lora_a = Some(lora_a);
        self.lora_b = Some(lora_b);
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> gpux_core::Result<Tensor> {
        let orig_shape = input.shape().clone();
        let in_features = orig_shape.last_dim();
        let batch_size = orig_shape.numel() / in_features;
        let out_features = self.weight.shape().dims()[0];

        let input_2d = input.reshape(vec![batch_size, in_features]);

        let mut out = if self.weight.dtype.is_quantized() {
            // Fused quantized matvec+bias for decode if bias present and no LoRA
            if let Some(ref bias) = self.bias {
                if self.lora_a.is_none() {
                    input_2d.matmul_a_bt_bias_quantized(&self.weight, bias)?
                } else {
                    input_2d.matmul_a_bt_quantized(&self.weight)?
                }
            } else {
                input_2d.matmul_a_bt_quantized(&self.weight)?
            }
        } else if let Some(ref bias) = self.bias {
            // Fused matmul+bias for decode (M=1) — saves a dispatch
            if self.lora_a.is_none() {
                input_2d.matmul_a_bt_bias(&self.weight, bias)?
            } else {
                input_2d.matmul_a_bt(&self.weight)?
            }
        } else {
            input_2d.matmul_a_bt(&self.weight)?
        };

        if let (Some(ref a), Some(ref b)) = (&self.lora_a, &self.lora_b) {
            let lora_out = input_2d.matmul_a_bt(a)?;
            let lora_out = lora_out.matmul_a_bt(b)?;
            let lora_out = lora_out.mul_scalar(self.lora_scaling)?;
            out = out.add(&lora_out)?;
        }

        // Bias already handled in fused matvec+bias paths (both f16 and quantized).
        // Only add separately when LoRA is present (the LoRA path uses non-bias matmul).
        let out = if self.bias.is_some() && self.lora_a.is_some() {
            out.broadcast_add_bias(self.bias.as_ref().unwrap())?
        } else {
            out
        };

        if orig_shape.ndim() > 2 {
            let mut out_dims = orig_shape.dims()[..orig_shape.ndim() - 1].to_vec();
            out_dims.push(out_features);
            Ok(out.reshape(out_dims))
        } else {
            Ok(out)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
}
