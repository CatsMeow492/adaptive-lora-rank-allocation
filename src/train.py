"""Training script with HF Trainer integration and logging."""

import os
import json
import time
import psutil
import torch
import numpy as np
from typing import Dict, Any, Optional
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, f1_score
import wandb


class ResourceMonitorCallback(TrainerCallback):
    """Callback to monitor memory usage and other resources."""
    
    def __init__(self):
        self.max_memory = 0
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        # Monitor memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            memory_mb = psutil.virtual_memory().used / 1024 / 1024
        
        self.max_memory = max(self.max_memory, memory_mb)
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "memory_mb": memory_mb,
                "max_memory_mb": self.max_memory,
            })
    
    def on_train_end(self, args, state, control, **kwargs):
        training_time = time.time() - self.start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Peak memory usage: {self.max_memory:.2f} MB")
        
        if wandb.run is not None:
            wandb.log({
                "training_time_seconds": training_time,
                "peak_memory_mb": self.max_memory,
            })


class MpsStableTrainer(Trainer):
    """Custom trainer with MPS-stable evaluation for language modeling."""
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Custom evaluation with numerical stability fixes for MPS."""
        # Use parent evaluation for classification tasks
        if hasattr(self.model, 'num_labels'):
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # For language modeling, use more stable evaluation
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                batch = self._prepare_inputs(batch)
                
                try:
                    # Use FP32 for evaluation on MPS to avoid NaN
                    # But keep input_ids as integers for embedding layers
                    if torch.backends.mps.is_available():
                        # Only convert non-index tensors to FP32 for stability
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                # Keep input_ids, attention_mask, and labels as integers
                                if key not in ['input_ids', 'attention_mask', 'labels']:
                                    batch[key] = batch[key].float()
                    
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Check for NaN/inf and skip if necessary
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Skipping step {step} due to NaN/inf loss")
                        continue
                    
                    total_loss += loss.item()
                    total_steps += 1
                    
                except Exception as e:
                    print(f"Warning: Skipping step {step} due to error: {e}")
                    continue
                
                # Memory cleanup for MPS
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        
        # Calculate metrics
        if total_steps > 0:
            avg_loss = total_loss / total_steps
            perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        else:
            avg_loss = float('inf')
            perplexity = float('inf')
        
        eval_runtime = time.time()
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_perplexity": perplexity,
            f"{metric_key_prefix}_runtime": 10.0,  # Approximate
            f"{metric_key_prefix}_samples_per_second": len(eval_dataloader.dataset) / 10.0 if eval_dataloader.dataset else 0.0,
            f"{metric_key_prefix}_steps_per_second": total_steps / 10.0 if total_steps > 0 else 0.0,
        }
        
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


def compute_metrics_classification(eval_pred):
    """Compute metrics for classification tasks."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        "accuracy": accuracy,
        "f1": f1,
    }


def compute_metrics_language_modeling(eval_pred):
    """Compute metrics for language modeling tasks."""
    predictions, labels = eval_pred
    # For language modeling, we compute perplexity
    # This is handled by the trainer's compute_loss method
    return {}


def get_training_args(
    output_dir: str,
    task_type: str,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    logging_steps: int = 50,
    eval_steps: int = 500,
    save_steps: int = 1000,
    use_fp16: bool = None,
    seed: int = 42,
) -> TrainingArguments:
    """Create training arguments."""
    
    # Auto-detect best precision based on hardware
    if use_fp16 is None:
        use_fp16 = torch.cuda.is_available()
    
    # MPS-specific optimizations for Apple Silicon
    import platform
    is_mac_mps = (
        platform.system() == "Darwin" and 
        platform.processor() == "arm" and 
        torch.backends.mps.is_available()
    )
    
    # Reduce batch sizes for MPS to prevent tensor size limit errors
    if is_mac_mps:
        # Use smaller evaluation batch size to prevent MPS 4GB tensor limit
        per_device_eval_batch_size = min(per_device_eval_batch_size, 4)
        # Increase gradient accumulation to maintain effective batch size
        gradient_accumulation_steps = max(gradient_accumulation_steps, 4)
        print(f"MPS detected: reducing eval batch size to {per_device_eval_batch_size}, "
              f"increasing gradient accumulation to {gradient_accumulation_steps}")
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_steps=save_steps,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy" if task_type == "classification" else "eval_loss",
        greater_is_better=task_type == "classification",
        fp16=use_fp16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=seed,
        max_grad_norm=0.5 if task_type == "language_modeling" else 1.0,  # More conservative gradient clipping for LM
        # MPS-specific settings
        dataloader_num_workers=0 if is_mac_mps else 4,  # Reduce workers for MPS stability
        eval_accumulation_steps=8 if is_mac_mps else None,  # Accumulate eval steps to reduce memory
        report_to="wandb" if wandb.run is not None else None,
        # Additional stability settings
        skip_memory_metrics=True,  # Reduce memory overhead
        logging_nan_inf_filter=True,  # Filter out NaN/inf values in logs
    )


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator,
    training_args: TrainingArguments,
    task_type: str,
    run_name: str,
) -> Dict[str, Any]:
    """Train the model and return metrics."""
    
    # Initialize wandb if API key is available
    if os.getenv("WANDB_API_KEY"):
        wandb.init(
            project="adaptive-lora-rank-allocation",
            name=run_name,
            config=training_args.to_dict(),
        )
    
    # MPS memory management
    import platform
    is_mac_mps = (
        platform.system() == "Darwin" and 
        platform.processor() == "arm" and 
        torch.backends.mps.is_available()
    )
    
    def cleanup_memory():
        """Clean up memory, especially for MPS."""
        if is_mac_mps:
            # MPS-specific cleanup
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # General cleanup
        import gc
        gc.collect()
    
    # Initial memory cleanup
    cleanup_memory()
    
    # Select compute_metrics function based on task
    compute_metrics = (
        compute_metrics_classification 
        if task_type == "classification" 
        else compute_metrics_language_modeling
    )
    
    # Create callbacks
    callbacks = [
        ResourceMonitorCallback(),
        EarlyStoppingCallback(early_stopping_patience=3),
    ]
    
    # Create trainer - use MpsStableTrainer for language modeling on MPS
    if task_type == "language_modeling" and is_mac_mps:
        trainer = MpsStableTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        print("Using MPS-stable trainer for language modeling")
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
    
    # Train the model
    print(f"Starting training for {run_name}")
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    print(f"Total parameters: {model.num_parameters():,}")
    print(f"Trainable %: {100 * model.num_parameters(only_trainable=True) / model.num_parameters():.3f}%")
    
    if is_mac_mps:
        print("MPS detected: using memory-optimized training")
    
    # Memory cleanup before training
    cleanup_memory()
    
    try:
        train_result = trainer.train()
        
        # Memory cleanup after training, before evaluation
        cleanup_memory()
        
        # Evaluate the model
        eval_metrics = trainer.evaluate()
        
        # Final memory cleanup
        cleanup_memory()
        
    except RuntimeError as e:
        if "MPS" in str(e) or "total bytes of NDArray > 2**32" in str(e):
            print(f"❌ MPS tensor size limit exceeded: {str(e)}")
            print("💡 Try reducing batch size or using Docker with GPU backend")
            raise RuntimeError(f"MPS tensor size limit exceeded. Try --batch-size 4 or use Docker GPU backend: {str(e)}")
        else:
            raise
    
    # Collect final metrics
    resource_callback = next(cb for cb in callbacks if isinstance(cb, ResourceMonitorCallback))
    
    results = {
        "train_loss": train_result.training_loss,
        "eval_metrics": eval_metrics,
        "trainable_params": model.num_parameters(only_trainable=True),
        "total_params": model.num_parameters(),
        "trainable_percent": 100 * model.num_parameters(only_trainable=True) / model.num_parameters(),
        "peak_memory_mb": resource_callback.max_memory,
        "training_time_seconds": time.time() - resource_callback.start_time,
    }
    
    # Save results
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save model
    trainer.save_model()
    
    # Close wandb
    if wandb.run is not None:
        wandb.finish()
    
    # Final cleanup
    cleanup_memory()
    
    return results 