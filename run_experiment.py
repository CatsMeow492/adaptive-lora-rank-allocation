#!/usr/bin/env python3
"""Main experiment runner for adaptive LoRA rank allocation study."""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import set_seed

from src.data.loader import load_sst2_dataset, load_wikitext2_dataset, get_data_collator
from src.models.factory import (
    load_model_and_tokenizer,
    create_lora_config,
    get_default_target_modules,
)
from src.train import train_model, get_training_args


def get_experiment_config(config_id: str) -> Dict[str, Any]:
    """Get experiment configuration by ID."""
    configs = {
        # Baselines
        "B-FP": {
            "name": "Baseline Fixed-rank FP16",
            "quantization_bits": None,
            "lora_rank": 8,
            "lora_adaptive": False,
        },
        "B-Q4": {
            "name": "Baseline 4-bit QLoRA",
            "quantization_bits": 4,
            "lora_rank": 8,
            "lora_adaptive": False,
        },
        "B-Ada": {
            "name": "Baseline AdaLoRA FP16",
            "quantization_bits": None,
            "lora_rank": 8,
            "lora_adaptive": True,
            "init_r": 12,
            "target_r": 4,
        },
        
        # Joint methods
        "Joint-1": {
            "name": "Joint 4-bit + AdaLoRA",
            "quantization_bits": 4,
            "lora_rank": 8,
            "lora_adaptive": True,
            "init_r": 12,
            "target_r": 4,
        },
        "Joint-2": {
            "name": "Joint Mixed-precision + AdaLoRA",
            "quantization_bits": 4,  # Will be modified per layer
            "lora_rank": 8,
            "lora_adaptive": True,
            "init_r": 12,
            "target_r": 4,
            "mixed_precision": True,
        },
        "Joint-3": {
            "name": "Joint Attention 8-bit FFN 4-bit + Manual Ranks",
            "quantization_bits": 4,
            "lora_rank": 8,
            "lora_adaptive": False,
            "mixed_precision": True,
            "manual_ranks": {"attention": 6, "ffn": 10},
        },
    }
    
    if config_id not in configs:
        raise ValueError(f"Unknown config ID: {config_id}")
    
    return configs[config_id]


def run_single_experiment(
    config_id: str,
    task: str,
    model_name: str,
    output_dir: str,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Get experiment configuration
    config = get_experiment_config(config_id)
    run_name = f"{config_id}_{task}_{model_name.replace('/', '_')}"
    
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"Task: {task}")
    print(f"Model: {model_name}")
    print(f"Run name: {run_name}")
    print(f"{'='*60}")
    
    # Determine task type
    task_type = "classification" if task == "sst2" else "language_modeling"
    
    # Get target modules
    target_modules = get_default_target_modules(model_name)
    
    # Create LoRA config
    lora_config = create_lora_config(
        task_type=task_type,
        rank=config.get("lora_rank", 8),
        alpha=config.get("lora_alpha", 16),
        dropout=config.get("lora_dropout", 0.1),
        target_modules=target_modules,
        adaptive=config.get("lora_adaptive", False),
        init_r=config.get("init_r", 12),
        target_r=config.get("target_r", 4),
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        task_type=task_type,
        quantization_bits=config.get("quantization_bits"),
        lora_config=lora_config,
        device_map="auto",
    )
    
    # Load dataset
    if task == "sst2":
        dataset = load_sst2_dataset(tokenizer, max_length=128)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    elif task == "wikitext2":
        dataset = load_wikitext2_dataset(tokenizer, max_length=512)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Get data collator
    data_collator = get_data_collator(tokenizer, task_type)
    
    # Create training arguments
    training_args = get_training_args(
        output_dir=os.path.join(output_dir, run_name),
        task_type=task_type,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_steps=500,
        save_steps=1000,
        use_fp16=None,
        seed=seed,
    )
    
    # Train the model
    results = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        training_args=training_args,
        task_type=task_type,
        run_name=run_name,
    )
    
    # Add config info to results
    results["config"] = config
    results["config_id"] = config_id
    results["task"] = task
    results["model_name"] = model_name
    results["run_name"] = run_name
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run adaptive LoRA rank allocation experiments")
    parser.add_argument("--config", type=str, required=True, 
                       choices=["B-FP", "B-Q4", "B-Ada", "Joint-1", "Joint-2", "Joint-3"],
                       help="Experiment configuration ID")
    parser.add_argument("--task", type=str, required=True,
                       choices=["sst2", "wikitext2"],
                       help="Task to run")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., bert-base-uncased, gpt2)")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    results = run_single_experiment(
        config_id=args.config,
        task=args.task,
        model_name=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Save consolidated results
    results_file = os.path.join(args.output_dir, f"results_{args.config}_{args.task}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Trainable parameters: {results['trainable_params']:,}")
    print(f"Peak memory: {results['peak_memory_mb']:.2f} MB")
    
    if args.task == "sst2":
        print(f"Accuracy: {results['eval_metrics']['eval_accuracy']:.4f}")
    else:
        print(f"Perplexity: {np.exp(results['eval_metrics']['eval_loss']):.4f}")


if __name__ == "__main__":
    main() 