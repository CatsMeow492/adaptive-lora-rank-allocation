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
    learning_rate: float = 2e-4,
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
        report_to="wandb" if wandb.run is not None else None,
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
    
    # Create trainer
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
    
    train_result = trainer.train()
    
    # Evaluate the model
    eval_metrics = trainer.evaluate()
    
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
    
    return results 