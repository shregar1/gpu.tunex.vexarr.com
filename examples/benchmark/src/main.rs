use std::sync::Arc;
use std::time::Instant;

use gpux_tensor::{GpuContext, Tensor};
use gpux_nn::transformer::{GPT, GPTConfig};


fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p]; // b is transposed
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn cpu_matmul_threaded(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            row[j] = sum;
        }
    });
    c
}

fn bench_matmul(ctx: &Arc<GpuContext>) -> gpux_core::Result<()> {
    println!("=== Matrix Multiply Benchmark ===");
    println!("{:<30} {:>10} {:>10} {:>10}", "Operation", "Size", "Time", "GFLOPS");
    println!("{:-<65}", "");

    let sizes: &[(usize, usize, usize)] = &[
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ];

    for &(m, n, k) in sizes {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001) % 1.0).collect();
        let b_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.002) % 1.0).collect();
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let label = format!("{}x{}x{}", m, n, k);

        // CPU single-threaded
        let start = Instant::now();
        let iters = if m <= 512 { 3 } else { 1 };
        for _ in 0..iters {
            let _ = cpu_matmul(&a_data, &b_data, m, n, k);
        }
        let cpu_time = start.elapsed().as_secs_f64() / iters as f64;
        let cpu_gflops = flops / cpu_time / 1e9;
        println!("{:<30} {:>10} {:>9.1}ms {:>9.1}", "CPU (1 thread)", label, cpu_time * 1000.0, cpu_gflops);

        // CPU multi-threaded
        let start = Instant::now();
        for _ in 0..iters {
            let _ = cpu_matmul_threaded(&a_data, &b_data, m, n, k);
        }
        let cpu_mt_time = start.elapsed().as_secs_f64() / iters as f64;
        let cpu_mt_gflops = flops / cpu_mt_time / 1e9;
        println!("{:<30} {:>10} {:>9.1}ms {:>9.1}", "CPU (multi-thread)", label, cpu_mt_time * 1000.0, cpu_mt_gflops);

        // GPU
        let a_gpu = Tensor::from_data(ctx, vec![m, k], &a_data)?;
        let b_gpu = Tensor::from_data(ctx, vec![n, k], &b_data)?;

        // warmup
        let _ = a_gpu.matmul_a_bt(&b_gpu)?;

        let gpu_iters = iters.max(5);
        let start = Instant::now();
        for _ in 0..gpu_iters {
            let _ = a_gpu.matmul_a_bt(&b_gpu)?;
        }
        let gpu_time = start.elapsed().as_secs_f64() / gpu_iters as f64;
        let gpu_gflops = flops / gpu_time / 1e9;
        let speedup_st = cpu_time / gpu_time;
        let speedup_mt = cpu_mt_time / gpu_time;
        println!("{:<30} {:>10} {:>9.1}ms {:>9.1}  ({:.1}x vs 1T, {:.1}x vs MT)",
            "Metal GPU", label, gpu_time * 1000.0, gpu_gflops, speedup_st, speedup_mt);
        println!();
    }

    Ok(())
}

fn cpu_forward_pass(weights: &[Vec<f32>], biases: &[Vec<f32>], input: &[f32], dims: &[(usize, usize)]) -> Vec<f32> {
    let mut x = input.to_vec();
    for (layer, &(in_dim, out_dim)) in dims.iter().enumerate() {
        let w = &weights[layer];
        let b = &biases[layer];
        let batch = x.len() / in_dim;
        let mut out = vec![0.0f32; batch * out_dim];
        for i in 0..batch {
            for j in 0..out_dim {
                let mut sum = b[j];
                for p in 0..in_dim {
                    sum += x[i * in_dim + p] * w[j * in_dim + p];
                }
                out[i * out_dim + j] = sum.max(0.0); // ReLU
            }
        }
        x = out;
    }
    x
}

fn bench_forward_pass(ctx: &Arc<GpuContext>) -> gpux_core::Result<()> {
    println!("=== Transformer Forward Pass Benchmark ===");

    let config = GPTConfig {
        vocab_size: 65,
        max_seq_len: 256,
        n_layers: 6,
        n_heads: 6,
        embed_dim: 192,
    };

    let model = GPT::new(ctx, config)?;
    let params = model.parameter_count();
    println!("Model: 6L/6H/192D, {} params", params);
    println!("{:<40} {:>10} {:>10}", "Operation", "Time", "tok/s");
    println!("{:-<65}", "");

    let batch_sizes = [1, 4, 8, 16];
    let seq_len = 128;

    for &batch in &batch_sizes {
        let total_tokens = batch * seq_len;
        let input_data: Vec<u32> = (0..total_tokens).map(|i| (i % 65) as u32).collect();

        // GPU forward
        let input_gpu = Tensor::from_uint_data(ctx, vec![batch, seq_len], &input_data)?;

        // warmup
        let _ = model.forward_lm(&input_gpu)?;

        let iters = 5;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = model.forward_lm(&input_gpu)?;
        }
        let gpu_time = start.elapsed().as_secs_f64() / iters as f64;
        let gpu_toks = total_tokens as f64 / gpu_time;
        let label = format!("GPU  (batch={}, seq={})", batch, seq_len);
        println!("{:<40} {:>9.1}ms {:>9.0}", label, gpu_time * 1000.0, gpu_toks);
    }

    // CPU comparison: simple MLP with equivalent FLOPs
    println!();
    println!("CPU comparison (equivalent-FLOP MLP):");
    let dims = vec![
        (192, 768), (768, 768), (768, 768), (768, 192),
    ];
    let weights: Vec<Vec<f32>> = dims.iter()
        .map(|&(i, o)| (0..o * i).map(|x| (x as f32 * 0.001) % 0.1).collect())
        .collect();
    let biases: Vec<Vec<f32>> = dims.iter()
        .map(|&(_, o)| vec![0.0f32; o])
        .collect();

    for &batch in &batch_sizes {
        let total_tokens = batch * 128;
        let input: Vec<f32> = (0..total_tokens * 192).map(|i| (i as f32 * 0.001) % 1.0).collect();
        let start = Instant::now();
        let iters = 3;
        for _ in 0..iters {
            let _ = cpu_forward_pass(&weights, &biases, &input, &dims);
        }
        let cpu_time = start.elapsed().as_secs_f64() / iters as f64;
        let cpu_toks = total_tokens as f64 / cpu_time;
        let label = format!("CPU  (batch={}, seq={})", batch, 128);
        println!("{:<40} {:>9.1}ms {:>9.0}", label, cpu_time * 1000.0, cpu_toks);
    }

    Ok(())
}

fn bench_training_step(ctx: &Arc<GpuContext>) -> gpux_core::Result<()> {
    println!();
    println!("=== Training Step Benchmark ===");

    let config = GPTConfig {
        vocab_size: 65,
        max_seq_len: 256,
        n_layers: 6,
        n_heads: 6,
        embed_dim: 192,
    };

    let model = GPT::new(ctx, config)?;

    let trainer_config = gpux_train::trainer::TrainerConfig {
        optimizer: gpux_train::optimizer::AdamWConfig {
            lr: 1e-3,
            weight_decay: 0.1,
            max_grad_norm: Some(1.0),
            ..Default::default()
        },
        max_steps: 100,
        warmup_steps: 10,
        max_lr: 1e-3,
        min_lr: 1e-4,
        ..Default::default()
    };

    let mut trainer = gpux_train::trainer::Trainer::new(trainer_config);

    let batch_size = 8;
    let seq_len = 128;
    let total_tokens = batch_size * seq_len;

    let input_data: Vec<u32> = (0..total_tokens).map(|i| (i % 65) as u32).collect();
    let target_data: Vec<u32> = (0..total_tokens).map(|i| ((i + 1) % 65) as u32).collect();
    let input = Tensor::from_uint_data(ctx, vec![batch_size, seq_len], &input_data)?;
    let target = Tensor::from_uint_data(ctx, vec![batch_size, seq_len], &target_data)?;

    // warmup
    let _ = trainer.train_step(&model, &input, &target, 0, ctx)?;

    println!("{:<40} {:>10} {:>10} {:>10}", "Operation", "Time", "tok/s", "Loss");
    println!("{:-<65}", "");

    let iters = 10;
    let start = Instant::now();
    let mut last_loss = 0.0;
    for i in 0..iters {
        let stats = trainer.train_step(&model, &input, &target, i + 1, ctx)?;
        last_loss = stats.loss;
    }
    let avg_time = start.elapsed().as_secs_f64() / iters as f64;
    let toks = total_tokens as f64 / avg_time;
    println!("{:<40} {:>9.1}ms {:>9.0} {:>9.4}",
        format!("GPU train (batch={}, seq={})", batch_size, seq_len),
        avg_time * 1000.0, toks, last_loss);

    println!();
    println!("Note: CPU training comparison omitted — a full transformer");
    println!("backward pass on CPU would take ~50-100x longer than forward.");
    println!("CPU forward-only numbers above give the lower bound estimate.");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║       GPUX - CPU vs GPU Benchmark                 ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let ctx = GpuContext::new()?;
    println!("GPU: {}", ctx.device.name());
    println!("Cores: {} (Apple M-series GPU)  ", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    println!();

    bench_matmul(&ctx)?;
    println!();
    bench_forward_pass(&ctx)?;
    bench_training_step(&ctx)?;

    println!();
    println!("═══════════════════════════════════════════════════");

    Ok(())
}
