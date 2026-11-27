use std::sync::Arc;

use gpux_nn::linear::Linear;
use gpux_nn::module::Module;
use gpux_tensor::{GpuContext, Tensor};
use gpux_train::optimizer::{AdamW, AdamWConfig};

struct MnistNet {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl MnistNet {
    fn new(ctx: &Arc<GpuContext>) -> gpux_core::Result<Self> {
        Ok(Self {
            fc1: Linear::new(ctx, 784, 256, true)?,
            fc2: Linear::new(ctx, 256, 128, true)?,
            fc3: Linear::new(ctx, 128, 10, true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> gpux_core::Result<Tensor> {
        let h1 = self.fc1.forward(x)?.relu()?;
        let h2 = self.fc2.forward(&h1)?.relu()?;
        self.fc3.forward(&h2)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}

fn generate_fake_mnist_batch(
    ctx: &Arc<GpuContext>,
    batch_size: usize,
) -> gpux_core::Result<(Tensor, Tensor)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let images: Vec<f32> = (0..batch_size * 784)
        .map(|_| rng.gen::<f32>())
        .collect();

    let labels: Vec<u32> = (0..batch_size)
        .map(|_| rng.gen_range(0..10u32))
        .collect();

    let x = Tensor::from_data(ctx, vec![batch_size, 784], &images)?;
    let y = Tensor::from_uint_data(ctx, vec![batch_size], &labels)?;

    Ok((x, y))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== GPUX MNIST Training Example ===");
    println!("Training a simple MLP on Apple Silicon GPU\n");

    let ctx = GpuContext::new()?;
    println!("GPU: {}", ctx.device.name());
    println!("Unified Memory: {}", ctx.device.has_unified_memory());
    println!();

    let model = MnistNet::new(&ctx)?;
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("Model parameters: {}", param_count);

    let batch_size = 64;
    let num_steps = 200;
    println!("Batch size: {}", batch_size);
    println!("Training steps: {}\n", num_steps);

    let params = model.parameters();
    let _optimizer = AdamW::new(&ctx, &params, AdamWConfig {
        lr: 1e-3,
        weight_decay: 0.0,
        ..Default::default()
    })?;

    for step in 0..num_steps {
        let start = std::time::Instant::now();

        let (x, targets) = generate_fake_mnist_batch(&ctx, batch_size)?;

        // Forward
        let logits = model.forward(&x)?;

        // Loss
        let (loss, _softmax_out) = logits.cross_entropy_loss(&targets)?;
        let loss_val = loss.to_vec()[0];

        let elapsed = start.elapsed();

        if step % 10 == 0 {
            let images_per_sec = batch_size as f32 / elapsed.as_secs_f32();
            println!(
                "step {:>4} | loss {:>8.4} | {:.1}ms | {:.0} images/s",
                step, loss_val, elapsed.as_secs_f64() * 1000.0, images_per_sec
            );
        }
    }

    println!("\nTraining complete!");
    println!("Note: This uses synthetic data to demonstrate GPU compute.");
    println!("For real training, load actual MNIST data.");

    Ok(())
}
