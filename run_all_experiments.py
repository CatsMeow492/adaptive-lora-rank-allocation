#!/usr/bin/env python3
"""Run all experiments in the experimental matrix."""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import torch


def run_experiment_matrix(
    output_dir: str = "./results",
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
) -> List[Dict[str, Any]]:
    """Run the full experiment matrix."""
    
    # Auto-detect MPS and adjust batch size
    import platform
    is_mac_mps = (
        platform.system() == "Darwin" and 
        platform.processor() == "arm" and 
        torch.backends.mps.is_available()
    )
    
    # Apply MPS-safe settings
    if is_mac_mps and batch_size > 4:
        print(f"MPS detected: reducing batch size from {batch_size} to 4 for stability")
        batch_size = 4
    
    # Experiment configurations
    configs = ["B-FP", "B-Q4", "B-Ada", "Joint-1", "Joint-2", "Joint-3"]
    tasks = ["sst2", "wikitext2"]
    models = {
        "sst2": "bert-base-uncased",
        "wikitext2": "gpt2"
    }
    
    results = []
    total_experiments = len(configs) * len(tasks)
    current_experiment = 0
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for config in configs:
        for task in tasks:
            current_experiment += 1
            model = models[task]
            
            print(f"\n{'='*80}")
            print(f"Experiment {current_experiment}/{total_experiments}: {config} on {task}")
            print(f"{'='*80}")
            
            # Build command
            cmd = [
                "python", "run_experiment.py",
                "--config", config,
                "--task", task,
                "--model", model,
                "--output-dir", output_dir,
                "--seed", str(seed),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--learning-rate", str(learning_rate),
            ]
            
            # Add MPS-safe flag if detected
            if is_mac_mps:
                cmd.append("--mps-safe")
                # Add FP32 flag for language modeling tasks on MPS
                if task == "wikitext2":
                    cmd.append("--fp32")
            
            # Run experiment
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
                
                if result.returncode == 0:
                    print(f"✅ Success: {config} on {task}")
                    
                    # Load results
                    results_file = os.path.join(output_dir, f"results_{config}_{task}.json")
                    if os.path.exists(results_file):
                        with open(results_file, "r") as f:
                            exp_results = json.load(f)
                        results.append(exp_results)
                    else:
                        print(f"⚠️  Results file not found: {results_file}")
                else:
                    print(f"❌ Failed: {config} on {task}")
                    print(f"Error: {result.stderr}")
                    
                    # Check for MPS-specific errors
                    if "MPS" in result.stderr or "total bytes of NDArray > 2**32" in result.stderr:
                        print("💡 MPS tensor size limit error - try using Docker with GPU backend")
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout: {config} on {task}")
            except Exception as e:
                print(f"💥 Exception: {config} on {task} - {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Completed {len(results)}/{total_experiments} experiments in {total_time:.2f} seconds")
    print(f"{'='*80}")
    
    return results


def create_summary_table(results: List[Dict[str, Any]], output_dir: str):
    """Create a summary table of all results."""
    
    # Convert to DataFrame
    summary_data = []
    for result in results:
        row = {
            "config_id": result["config_id"],
            "task": result["task"],
            "model": result["model_name"],
            "trainable_params": result["trainable_params"],
            "trainable_percent": result["trainable_percent"],
            "peak_memory_mb": result["peak_memory_mb"],
            "training_time_seconds": result["training_time_seconds"],
            "train_loss": result["train_loss"],
        }
        
        # Add task-specific metrics
        if result["task"] == "sst2":
            row["accuracy"] = result["eval_metrics"]["eval_accuracy"]
            row["f1"] = result["eval_metrics"]["eval_f1"]
        else:
            row["eval_loss"] = result["eval_metrics"]["eval_loss"]
            row["perplexity"] = result["eval_metrics"]["eval_perplexity"] if "eval_perplexity" in result["eval_metrics"] else None
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_file = os.path.join(output_dir, "experiment_summary.csv")
    df.to_csv(summary_file, index=False)
    
    print(f"Summary table saved to: {summary_file}")
    
    # Print formatted table
    print("\nExperiment Summary:")
    print("=" * 120)
    print(df.to_string(index=False))
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all experiments in the matrix")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run all experiments
    results = run_experiment_matrix(
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Create summary table
    if results:
        summary_df = create_summary_table(results, args.output_dir)
        
        # Save consolidated results
        consolidated_file = os.path.join(args.output_dir, "all_results.json")
        with open(consolidated_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nConsolidated results saved to: {consolidated_file}")
    else:
        print("No successful experiments to summarize.")


if __name__ == "__main__":
    main() 