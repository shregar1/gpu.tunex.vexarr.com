use std::sync::Arc;
use std::time::Instant;
use gpux_tensor::{GpuContext, Tensor};
use gpux_nn::transformer::GPT;
use crate::optimizer::{AdamW, AdamWConfig};
use crate::trainer::TrainStats;

pub struct MixedPrecisionTrainer {
    pub optimizer: Option<AdamW>,
    pub config: MixedPrecisionConfig,
    loss_scale: f32,
    steps_since_scale_update: u32,
    consecutive_ok: u32,
}

#[derive(Clone)]
pub struct MixedPrecisionConfig {
    pub optimizer: AdamWConfig,
    pub max_steps: usize,
    pub warmup_steps: usize,
    pub max_lr: f32,
    pub min_lr: f32,
    pub initial_loss_scale: f32,
    pub scale_growth_interval: u32,
    pub scale_growth_factor: f32,
    pub scale_backoff_factor: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            optimizer: AdamWConfig::default(),
            max_steps: 1000,
            warmup_steps: 100,
            max_lr: 6e-4,
            min_lr: 6e-5,
            initial_loss_scale: 1024.0,
            scale_growth_interval: 200,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
        }
    }
}

impl MixedPrecisionTrainer {
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let loss_scale = config.initial_loss_scale;
        Self {
            optimizer: None,
            config,
            loss_scale,
            steps_since_scale_update: 0,
            consecutive_ok: 0,
        }
    }

    pub fn cosine_lr(&self, step: usize) -> f32 {
        if step < self.config.warmup_steps {
            self.config.max_lr * (step as f32 / self.config.warmup_steps as f32)
        } else {
            let decay_ratio = (step - self.config.warmup_steps) as f32
                / (self.config.max_steps - self.config.warmup_steps).max(1) as f32;
            let coeff = 0.5 * (1.0 + (std::f32::consts::PI * decay_ratio).cos());
            self.config.min_lr + coeff * (self.config.max_lr - self.config.min_lr)
        }
    }

    fn has_nan_or_inf(grad: &Tensor) -> bool {
        let data = grad.to_vec();
        data.iter().any(|&v| v.is_nan() || v.is_infinite())
    }

    pub fn train_step(
        &mut self,
        model: &GPT,
        input_ids: &Tensor,
        targets: &Tensor,
        step: usize,
        ctx: &Arc<GpuContext>,
    ) -> gpux_core::Result<TrainStats> {
        let start = Instant::now();

        let cache = model.forward_with_cache(input_ids)?;

        let batch = cache.logits.shape().dims()[0];
        let seq_len = cache.logits.shape().dims()[1];
        let vocab = cache.logits.shape().dims()[2];

        let logits_2d = cache.logits.reshape(vec![batch * seq_len, vocab]);
        let (loss, softmax_out) = logits_2d.cross_entropy_loss(targets)?;
        let loss_val = loss.to_vec()[0];

        // Scaled backward for mixed precision stability
        let scaled_grad_scale = self.loss_scale / (batch * seq_len) as f32;
        let grad_logits_2d = softmax_out.cross_entropy_backward(targets, scaled_grad_scale)?;
        let grad_logits = grad_logits_2d.reshape(vec![batch, seq_len, vocab]);

        let grads = model.backward(&cache, &grad_logits, input_ids, ctx)?;

        let params = model.all_parameters();
        let grad_tensors = grads.all_grads();

        // Check for overflow
        let has_overflow = grad_tensors.iter().any(|g| Self::has_nan_or_inf(g));

        if has_overflow {
            self.loss_scale *= self.config.scale_backoff_factor;
            self.consecutive_ok = 0;
            if step % 50 == 0 {
                log::warn!("Gradient overflow at step {}, reducing loss scale to {}", step, self.loss_scale);
            }
        } else {
            // Unscale gradients and step
            let inv_scale = 1.0 / self.loss_scale;
            let mut matched_params = Vec::new();
            let mut matched_grads = Vec::new();
            for (p, g) in params.iter().zip(grad_tensors.iter()) {
                if p.shape() == g.shape() {
                    matched_params.push(*p);
                    matched_grads.push((*g).mul_scalar(inv_scale)?);
                }
            }

            if self.optimizer.is_none() {
                self.optimizer = Some(AdamW::new(ctx, &matched_params, self.config.optimizer.clone())?);
            }

            let lr = self.cosine_lr(step);
            if let Some(ref mut opt) = self.optimizer {
                opt.set_lr(lr);
                opt.step(&matched_params, &matched_grads)?;
            }

            self.consecutive_ok += 1;
            if self.consecutive_ok >= self.config.scale_growth_interval {
                self.loss_scale *= self.config.scale_growth_factor;
                self.consecutive_ok = 0;
            }
        }

        let elapsed = start.elapsed();
        let tokens = (batch * seq_len) as f32;

        Ok(TrainStats {
            step,
            loss: loss_val,
            tokens_per_sec: tokens / elapsed.as_secs_f32(),
            lr: self.cosine_lr(step),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }

    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }
}
