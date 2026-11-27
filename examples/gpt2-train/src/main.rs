use std::sync::Arc;
use std::time::Instant;

use gpux_tensor::{GpuContext, Tensor};
use gpux_nn::transformer::{GPT, GPTConfig};
use gpux_train::trainer::{Trainer, TrainerConfig};
use gpux_train::optimizer::AdamWConfig;

fn generate_random_batch(
    ctx: &Arc<GpuContext>,
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
) -> gpux_core::Result<(Tensor, Tensor)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let input_ids: Vec<u32> = (0..batch_size * seq_len)
        .map(|_| rng.gen_range(0..vocab_size as u32))
        .collect();

    let targets: Vec<u32> = (0..batch_size * seq_len)
        .map(|_| rng.gen_range(0..vocab_size as u32))
        .collect();

    let x = Tensor::from_uint_data(ctx, vec![batch_size, seq_len], &input_ids)?;
    let y = Tensor::from_uint_data(ctx, vec![batch_size * seq_len], &targets)?;

    Ok((x, y))
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║          GPUX - GPT-2 Training Example          ║");
    println!("║      Apple Silicon GPU Compute via Metal         ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let ctx = GpuContext::new()?;
    println!("GPU Device: {}", ctx.device.name());
    println!("Unified Memory: {}", ctx.device.has_unified_memory());
    println!("Max Buffer Size: {} MB", ctx.device.max_buffer_length() / 1024 / 1024);
    println!();

    // Use tiny config for demo (fits in any Apple Silicon)
    let config = GPTConfig::tiny();
    println!("Model Config:");
    println!("  vocab_size:   {}", config.vocab_size);
    println!("  max_seq_len:  {}", config.max_seq_len);
    println!("  n_layers:     {}", config.n_layers);
    println!("  n_heads:      {}", config.n_heads);
    println!("  embed_dim:    {}", config.embed_dim);

    println!("\nInitializing model...");
    let init_start = Instant::now();
    let model = GPT::new(&ctx, config)?;
    let init_time = init_start.elapsed();
    println!("Model initialized in {:.1}ms", init_time.as_secs_f64() * 1000.0);
    println!("Total parameters: {}", format_params(model.parameter_count()));
    println!();

    let batch_size = 4;
    let seq_len = 64;
    let vocab_size = 256;
    let num_steps = 200;

    println!("Training Config:");
    println!("  batch_size:   {}", batch_size);
    println!("  seq_len:      {}", seq_len);
    println!("  num_steps:    {}", num_steps);
    println!();

    let mut trainer = Trainer::new(TrainerConfig {
        optimizer: AdamWConfig {
            lr: 1e-4,
            weight_decay: 0.01,
            max_grad_norm: Some(0.5),
            ..Default::default()
        },
        max_steps: num_steps,
        warmup_steps: 10,
        max_lr: 3e-4,
        min_lr: 1e-5,
        ..Default::default()
    });

    println!("Training...");
    println!("{:-<70}", "");

    for step in 0..num_steps {
        let (input_ids, targets) = generate_random_batch(&ctx, batch_size, seq_len, vocab_size)?;

        let stats = trainer.train_step(&model, &input_ids, &targets, step, &ctx)?;

        if step % 20 == 0 || step == num_steps - 1 {
            println!("{}", stats);
        }
    }

    println!("{:-<70}", "");
    println!("\nTraining complete!");
    println!("Note: This trains on random data to demonstrate GPU compute pipeline.");
    println!("For real LLM training, provide actual tokenized text data.");
    println!("\nTo train a real model:");
    println!("  1. Tokenize your dataset");
    println!("  2. Use GPTConfig::small_lm() or GPTConfig::gpt2_small()");
    println!("  3. Feed real token sequences as input_ids");

    Ok(())
}
