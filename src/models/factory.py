"""Factory functions for model loading with quantization and LoRA."""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from typing import Dict, Any, Tuple, Optional


def create_quantization_config(bits: int = 4, device_map: str = "auto") -> BitsAndBytesConfig:
    """Create quantization configuration."""
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            device_map=device_map,
        )
    else:
        raise ValueError(f"Unsupported quantization bits: {bits}")


def create_lora_config(
    task_type: str,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Optional[list] = None,
    adaptive: bool = False,
    init_r: int = 12,
    target_r: int = 4,
) -> LoraConfig:
    """Create LoRA configuration."""
    task_type_map = {
        "classification": TaskType.SEQ_CLS,
        "language_modeling": TaskType.CAUSAL_LM,
    }
    
    if task_type not in task_type_map:
        raise ValueError(f"Unknown task type: {task_type}")
    
    if adaptive:
        return AdaLoraConfig(
            init_r=init_r,
            target_r=target_r,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            deltaT=10,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            task_type=task_type_map[task_type],
        )
    else:
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            task_type=task_type_map[task_type],
        )


def load_model_and_tokenizer(
    model_name: str,
    task_type: str,
    quantization_bits: Optional[int] = None,
    lora_config: Optional[LoraConfig] = None,
    device_map: str = "auto",
) -> Tuple[Any, AutoTokenizer]:
    """Load model and tokenizer with optional quantization and LoRA."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create quantization config if specified
    quantization_config = None
    if quantization_bits is not None:
        quantization_config = create_quantization_config(quantization_bits, device_map)
    
    # Load model based on task type
    if task_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # SST-2 has 2 labels
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16 if quantization_config is None else None,
        )
    elif task_type == "language_modeling":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16 if quantization_config is None else None,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Prepare model for k-bit training if quantized
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA if specified
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def get_default_target_modules(model_name: str) -> list:
    """Get default target modules for LoRA based on model architecture."""
    if "bert" in model_name.lower():
        return ["query", "value", "key", "dense"]
    elif "gpt" in model_name.lower():
        return ["c_attn", "c_proj", "c_fc"]
    elif "t5" in model_name.lower():
        return ["q", "v", "k", "o", "wi", "wo"]
    else:
        # Generic fallback
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 