#!/usr/bin/env python3
"""Test script to train BERT without LoRA to isolate training issues."""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from src.data.loader import load_sst2_dataset, get_data_collator
from src.train import compute_metrics_classification

def test_no_lora():
    """Test training without LoRA."""
    set_seed(42)
    
    # Load model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    
    # Load dataset
    dataset = load_sst2_dataset(tokenizer, max_length=128)
    train_dataset = dataset["train"].select(range(1000))  # Small subset for testing
    eval_dataset = dataset["validation"].select(range(100))
    
    # Get data collator
    data_collator = get_data_collator(tokenizer, "classification")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./test_results",
        num_train_epochs=1,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=25,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="no",
        fp16=False,  # Disable FP16 for debugging
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_classification,
    )
    
    # Train the model
    print("Starting training without LoRA...")
    print(f"Trainable parameters: {model.num_parameters():,}")
    
    train_result = trainer.train()
    
    # Evaluate the model
    eval_metrics = trainer.evaluate()
    
    print("\nTraining Results:")
    print(f"Train loss: {train_result.training_loss:.4f}")
    print(f"Eval accuracy: {eval_metrics['eval_accuracy']:.4f}")
    print(f"Eval F1: {eval_metrics['eval_f1']:.4f}")
    
    return train_result, eval_metrics

if __name__ == "__main__":
    test_no_lora() 