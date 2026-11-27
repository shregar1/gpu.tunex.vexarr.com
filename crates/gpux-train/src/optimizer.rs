use std::sync::Arc;
use gpux_tensor::{GpuContext, Tensor, Shape};

#[derive(Clone)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: Option<f32>,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: Some(1.0),
        }
    }
}

pub struct AdamW {
    pub config: AdamWConfig,
    m_states: Vec<Tensor>,
    v_states: Vec<Tensor>,
    step: u32,
    ctx: Arc<GpuContext>,
}

impl AdamW {
    pub fn new(ctx: &Arc<GpuContext>, params: &[&Tensor], config: AdamWConfig) -> gpux_core::Result<Self> {
        let mut m_states = Vec::with_capacity(params.len());
        let mut v_states = Vec::with_capacity(params.len());

        for param in params {
            m_states.push(Tensor::zeros(ctx, param.shape().clone())?);
            v_states.push(Tensor::zeros(ctx, param.shape().clone())?);
        }

        Ok(Self {
            config,
            m_states,
            v_states,
            step: 0,
            ctx: ctx.clone(),
        })
    }

    pub fn step(&mut self, params: &[&Tensor], grads: &[Tensor]) -> gpux_core::Result<()> {
        self.step += 1;

        // Global gradient clipping (computed on CPU, applied on GPU)
        let clip_coeff = if let Some(max_norm) = self.config.max_grad_norm {
            let mut total_norm_sq = 0.0f32;
            for grad in grads {
                let data = grad.to_vec();
                total_norm_sq += data.iter().map(|x| x * x).sum::<f32>();
            }
            let total_norm = total_norm_sq.sqrt();
            let coeff = (max_norm / (total_norm + 1e-6)).min(1.0);
            coeff
        } else {
            1.0
        };

        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            if i >= self.m_states.len() {
                break;
            }
            if param.shape() != grad.shape() {
                continue;
            }

            // Apply gradient clipping + AdamW on GPU
            let actual_grad = if clip_coeff < 1.0 {
                grad.mul_scalar(clip_coeff)?
            } else {
                grad.clone()
            };

            param.adamw_step(
                &actual_grad,
                &self.m_states[i],
                &self.v_states[i],
                self.config.lr,
                self.config.beta1,
                self.config.beta2,
                self.config.eps,
                self.config.weight_decay,
                self.step,
            )?;
        }

        Ok(())
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }

    pub fn current_step(&self) -> u32 {
        self.step
    }
}
