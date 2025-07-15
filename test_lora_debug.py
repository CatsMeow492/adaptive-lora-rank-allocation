#!/usr/bin/env python3
"""Debug script to test LoRA configuration step by step."""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from src.data.loader import load_sst2_dataset, get_data_collator
from src.train import compute_metrics_classification

def test_lora_debug():
    """Test LoRA with different configurations."""
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
    
    # Test different LoRA configurations
    configs_to_test = [
        {
            "name": "Standard LoRA",
            "config": LoraConfig(
                r=8,
                lora_alpha=8,  # Try alpha = r instead of 16
                lora_dropout=0.1,
                target_modules=["query", "value", "key", "dense"],
                task_type=TaskType.SEQ_CLS,
            )
        },
        {
            "name": "Low rank LoRA", 
            "config": LoraConfig(
                r=4,
                lora_alpha=4,
                lora_dropout=0.1,
                target_modules=["query", "value"],  # Fewer modules
                task_type=TaskType.SEQ_CLS,
            )
        },
        {
            "name": "High alpha LoRA",
            "config": LoraConfig(
                r=8,
                lora_alpha=32,  # Higher alpha
                lora_dropout=0.1,
                target_modules=["query", "value"],
                task_type=TaskType.SEQ_CLS,
            )
        }
    ]
    
    # Load dataset (small subset for quick testing)
    dataset = load_sst2_dataset(tokenizer, max_length=128)
    train_dataset = dataset["train"].select(range(500))
    eval_dataset = dataset["validation"].select(range(50))
    data_collator = get_data_collator(tokenizer, "classification")
    
    for config_info in configs_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {config_info['name']}")
        print(f"{'='*60}")
        
        # Create fresh model
        test_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
        )
        
        # Apply LoRA
        lora_config = config_info['config']
        test_model = get_peft_model(test_model, lora_config)
        
        print(f"Trainable parameters: {test_model.num_parameters(only_trainable=True):,}")
        print(f"Total parameters: {test_model.num_parameters():,}")
        print(f"Trainable %: {100 * test_model.num_parameters(only_trainable=True) / test_model.num_parameters():.3f}%")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./test_results_{config_info['name'].replace(' ', '_').lower()}",
            num_train_epochs=0.2,  # Very short training
            learning_rate=1e-4,  # Conservative learning rate
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_steps=10,
            eval_steps=25,
            eval_strategy="steps",
            save_strategy="no",
            fp16=False,
            max_grad_norm=1.0,
            remove_unused_columns=False,
            report_to="none",
        )
        
        # Create trainer
        trainer = Trainer(
            model=test_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_classification,
        )
        
        try:
            # Train for a few steps
            print("Training for a few steps...")
            train_result = trainer.train()
            
            # Evaluate
            eval_metrics = trainer.evaluate()
            
            print(f"✅ SUCCESS")
            print(f"  Train loss: {train_result.training_loss:.4f}")
            print(f"  Eval accuracy: {eval_metrics['eval_accuracy']:.4f}")
            print(f"  Eval F1: {eval_metrics['eval_f1']:.4f}")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            
        # Clean up
        del test_model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    test_lora_debug() 