#!/usr/bin/env python3
"""
LoRA fine-tuning for Qwen/Llama models on Apple Silicon (MPS backend).

Usage:
    python tools/finetune.py --model Qwen/Qwen2.5-0.5B --data data.txt --output ./my-adapter
    python tools/finetune.py --model Qwen/Qwen2.5-0.5B --data dataset.jsonl --output ./my-adapter
    python tools/finetune.py --model Qwen/Qwen2.5-0.5B --data imdb --output ./my-adapter --split train

Output is a PEFT LoRA adapter (adapter_model.safetensors + adapter_config.json)
that can be loaded by GPUX inference: cargo run --release -p inference-example -- --lora ./my-adapter
"""

import argparse
import os
import json
import torch
from pathlib import Path


def detect_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_data(data_path, split, tokenizer, max_length):
    from datasets import load_dataset, Dataset

    path = Path(data_path)

    if path.exists() and path.suffix == ".txt":
        texts = path.read_text().strip().split("\n\n")
        ds = Dataset.from_dict({"text": texts})
    elif path.exists() and path.suffix == ".jsonl":
        ds = load_dataset("json", data_files=str(path), split="train")
    elif path.exists() and path.suffix == ".json":
        ds = load_dataset("json", data_files=str(path), split="train")
    else:
        ds = load_dataset(data_path, split=split)

    if "text" not in ds.column_names:
        for col in ["content", "sentence", "document", "input"]:
            if col in ds.column_names:
                ds = ds.rename_column(col, "text")
                break

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds = ds.map(lambda x: {"labels": x["input_ids"]})
    ds.set_format("torch")
    return ds


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for GPUX")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model ID")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to .txt/.jsonl file, or HuggingFace dataset name")
    parser.add_argument("--output", type=str, default="./lora-adapter",
                        help="Output directory for LoRA adapter")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (for HF datasets)")

    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--target-modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Modules to apply LoRA to")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 mixed precision (MPS does not support BF16, will use FP16 instead)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (auto-detects mps/cuda/cpu)")

    args = parser.parse_args()
    device = args.device or detect_device()

    print(f"{'='*60}")
    print(f"  GPUX LoRA Fine-tuning")
    print(f"{'='*60}")
    print(f"  Model:    {args.model}")
    print(f"  Data:     {args.data}")
    print(f"  Device:   {device}")
    print(f"  LoRA:     rank={args.rank}, alpha={args.alpha}")
    print(f"  Targets:  {args.target_modules}")
    print(f"  Epochs:   {args.epochs}, LR: {args.lr}")
    print(f"  Output:   {args.output}")
    print(f"{'='*60}\n")

    print("[1/4] Loading model and tokenizer...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Base model: {total_params:,} params")

    print("\n[2/4] Applying LoRA...")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    model.print_trainable_parameters()

    print("\n[3/4] Loading dataset...")
    dataset = load_data(args.data, args.split, tokenizer, args.max_length)
    print(f"  {len(dataset)} samples, max_length={args.max_length}")

    print("\n[4/4] Training...")
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    use_fp16 = device == "mps" and args.bf16
    use_bf16 = device == "cuda" and args.bf16

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch",
        report_to="none",
        use_mps_device=(device == "mps"),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    print(f"\nSaving adapter to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    adapter_path = Path(args.output) / "adapter_model.safetensors"
    if adapter_path.exists():
        size_mb = adapter_path.stat().st_size / (1024 * 1024)
        print(f"  Adapter saved: {size_mb:.1f} MB")

    print(f"\nDone! Load in GPUX with:")
    print(f"  cargo run --release -p inference-example -- --lora {args.output}")


if __name__ == "__main__":
    main()
