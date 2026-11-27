use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use gpux_tensor::{GpuContext, Tensor};
use gpux_nn::transformer::{GPT, GPTConfig};
use gpux_train::data::{TextDataset, Tokenizer};
use gpux_train::trainer::{Trainer, TrainerConfig};
use gpux_train::optimizer::AdamWConfig;

fn generate_text(model: &GPT, ctx: &Arc<GpuContext>, tokenizer: &Tokenizer, prompt: &str, max_tokens: usize, temperature: f32) -> gpux_core::Result<String> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut tokens = tokenizer.encode(prompt);
    let max_seq = model.config.max_seq_len;

    for _ in 0..max_tokens {
        let seq_len = tokens.len().min(max_seq);
        let input_slice = &tokens[tokens.len() - seq_len..];
        let input = Tensor::from_uint_data(ctx, vec![1, seq_len], input_slice)?;

        let logits = model.forward_lm(&input)?;
        let logits_data = logits.to_vec();
        let vocab = model.config.vocab_size;
        let last_logits = &logits_data[logits_data.len() - vocab..];

        if temperature < 0.01 {
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &v) in last_logits.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }
            tokens.push(max_idx as u32);
        } else {
            let max_val = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let probs: Vec<f32> = last_logits.iter()
                .map(|&v| ((v - max_val) / temperature).exp())
                .collect();
            let sum: f32 = probs.iter().sum();
            let probs: Vec<f32> = probs.iter().map(|&p| p / sum).collect();

            let r: f32 = rng.gen();
            let mut cumsum = 0.0;
            let mut chosen = 0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    chosen = i;
                    break;
                }
            }
            tokens.push(chosen as u32);
        }
    }

    Ok(tokenizer.decode(&tokens))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║       GPUX - Apple Silicon LLM Training          ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let ctx = GpuContext::new()?;
    println!("GPU: {}", ctx.device.name());
    println!();

    // Load text
    let text_path = std::env::args().nth(1)
        .unwrap_or_else(|| "data/shakespeare.txt".to_string());
    let path = Path::new(&text_path);

    if !path.exists() {
        eprintln!("Error: text file not found at '{}'", text_path);
        eprintln!("Usage: text-train [path/to/text.txt]");
        eprintln!("Download Shakespeare: curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/shakespeare.txt");
        std::process::exit(1);
    }
    println!("Loading: {}", text_path);
    let text = std::fs::read_to_string(path)?;

    let tokenizer = Tokenizer::char_level(&text);
    let vocab_size = tokenizer.vocab_size();
    println!("Vocab: {} chars, {} tokens", vocab_size, text.len());

    let seq_len = 128;
    let batch_size = 8;
    let num_steps = 2000;

    let mut dataset = TextDataset::from_text(&text, &tokenizer);
    dataset.set_seq_len(seq_len);
    println!("Batches: {}", dataset.num_batches(batch_size));

    let config = GPTConfig {
        vocab_size,
        max_seq_len: 256,
        n_layers: 6,
        n_heads: 6,
        embed_dim: 192,
    };

    let model = GPT::new(&ctx, config)?;
    println!("Model: {}L/{}H/{}D, {} params",
        model.config.n_layers, model.config.n_heads, model.config.embed_dim,
        model.parameter_count());
    println!();

    let mut trainer = Trainer::new(TrainerConfig {
        optimizer: AdamWConfig {
            lr: 1e-3,
            weight_decay: 0.1,
            max_grad_norm: Some(1.0),
            ..Default::default()
        },
        max_steps: num_steps,
        warmup_steps: 100,
        max_lr: 1e-3,
        min_lr: 1e-4,
        ..Default::default()
    });

    println!("Training {} steps (batch={}, seq={})...", num_steps, batch_size, seq_len);
    println!("{:-<70}", "");

    let train_start = Instant::now();
    let mut best_loss = f32::MAX;

    for step in 0..num_steps {
        let (input_ids, targets) = dataset.next_batch(&ctx, batch_size)?;
        let stats = trainer.train_step(&model, &input_ids, &targets, step, &ctx)?;

        if stats.loss < best_loss {
            best_loss = stats.loss;
        }

        if step % 100 == 0 || step == num_steps - 1 {
            println!("{}", stats);
        }

        if step > 0 && step % 500 == 0 {
            let sample = generate_text(&model, &ctx, &tokenizer, "\nHAMLET:\n", 200, 0.8)?;
            let display: String = sample.chars().take(200).collect();
            println!("  >>> {}", display.replace('\n', "\n  >>> "));
            println!();
        }
    }

    let total_time = train_start.elapsed();
    println!("{:-<70}", "");
    println!("Done in {:.1}s | best loss: {:.4}", total_time.as_secs_f64(), best_loss);
    println!();

    println!("=== Generated Samples ===");
    let prompts = [
        "\nHAMLET:\nTo be, or not to be",
        "\nFirst Citizen:\n",
        "\nKING HENRY:\n",
        "\nROMEO:\nO, ",
    ];
    for prompt in &prompts {
        let generated = generate_text(&model, &ctx, &tokenizer, prompt, 300, 0.8)?;
        let display: String = generated.chars().take(250).collect();
        println!("{}", display);
        println!("---");
    }

    let ckpt_path = Path::new("shakespeare_model.gpux");
    model.save_checkpoint(ckpt_path)?;
    let size = std::fs::metadata(ckpt_path)?.len();
    println!("\nCheckpoint: {:.1} MB", size as f64 / 1024.0 / 1024.0);
    std::fs::remove_file(ckpt_path)?;

    Ok(())
}
