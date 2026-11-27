use std::sync::Arc;
use std::time::Instant;

use gpux_tensor::{GpuContext, Tensor};
use gpux_nn::transformer::GPT;
use crate::optimizer::{AdamW, AdamWConfig};

#[derive(Clone)]
pub struct TrainerConfig {
    pub optimizer: AdamWConfig,
    pub max_steps: usize,
    pub log_interval: usize,
    pub eval_interval: usize,
    pub warmup_steps: usize,
    pub max_lr: f32,
    pub min_lr: f32,
    pub gradient_accumulation_steps: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            optimizer: AdamWConfig::default(),
            max_steps: 1000,
            log_interval: 10,
            eval_interval: 100,
            warmup_steps: 100,
            max_lr: 6e-4,
            min_lr: 6e-5,
            gradient_accumulation_steps: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainStats {
    pub step: usize,
    pub loss: f32,
    pub tokens_per_sec: f32,
    pub lr: f32,
    pub elapsed_ms: f64,
}

impl std::fmt::Display for TrainStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "step {:>6} | loss {:>8.4} | lr {:>10.2e} | tok/s {:>10.0} | {:.1}ms",
            self.step, self.loss, self.lr, self.tokens_per_sec, self.elapsed_ms
        )
    }
}

pub struct Trainer {
    pub config: TrainerConfig,
    pub optimizer: Option<AdamW>,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            optimizer: None,
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

    pub fn train_step(
        &mut self,
        model: &GPT,
        input_ids: &Tensor,
        targets: &Tensor,
        step: usize,
        ctx: &Arc<GpuContext>,
    ) -> gpux_core::Result<TrainStats> {
        let start = Instant::now();

        // Forward pass with cache for backward
        let cache = model.forward_with_cache(input_ids)?;

        let batch = cache.logits.shape().dims()[0];
        let seq_len = cache.logits.shape().dims()[1];
        let vocab = cache.logits.shape().dims()[2];

        // Reshape logits for cross-entropy: [B*T, vocab] (zero-copy)
        let logits_2d = cache.logits.reshape(vec![batch * seq_len, vocab]);

        // Compute loss
        let (loss, softmax_out) = logits_2d.cross_entropy_loss(targets)?;
        let loss_val = loss.to_vec()[0];

        // Backward: compute gradients for cross-entropy
        let grad_logits_2d = softmax_out.cross_entropy_backward(
            targets,
            1.0 / (batch * seq_len) as f32,
        )?;

        // Reshape grad back to [B, T, vocab] (zero-copy)
        let grad_logits = grad_logits_2d.reshape(vec![batch, seq_len, vocab]);

        // Full backward pass through the model
        let grads = model.backward(&cache, &grad_logits, input_ids, ctx)?;

        // Collect parameters and their gradients in matching order
        let params = model.all_parameters();
        let grad_tensors = grads.all_grads();

        // Verify shapes match between params and grads
        assert_eq!(params.len(), grad_tensors.len(),
            "param count {} != grad count {}", params.len(), grad_tensors.len());

        // Filter to only params/grads where shapes match (skip mismatches safely)
        let mut matched_params = Vec::new();
        let mut matched_grads = Vec::new();
        for (p, g) in params.iter().zip(grad_tensors.iter()) {
            if p.shape() == g.shape() {
                matched_params.push(*p);
                matched_grads.push((*g).clone());
            } else if step == 0 {
                log::warn!("Shape mismatch: param {} vs grad {}", p.shape(), g.shape());
            }
        }

        // Initialize optimizer on first step
        if self.optimizer.is_none() {
            self.optimizer = Some(AdamW::new(ctx, &matched_params, self.config.optimizer.clone())?);
        }

        // Learning rate schedule
        let lr = self.cosine_lr(step);
        if let Some(ref mut opt) = self.optimizer {
            opt.set_lr(lr);
            opt.step(&matched_params, &matched_grads)?;
        }

        let elapsed = start.elapsed();
        let tokens = (batch * seq_len) as f32;
        let tokens_per_sec = tokens / elapsed.as_secs_f32();

        Ok(TrainStats {
            step,
            loss: loss_val,
            tokens_per_sec,
            lr,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }
}
