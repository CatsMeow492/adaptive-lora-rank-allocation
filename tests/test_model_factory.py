"""Unit tests for model factory module."""

import pytest
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, AdaLoraConfig
from src.models.factory import (
    create_quantization_config,
    create_lora_config,
    get_default_target_modules,
)


class TestModelFactory:
    """Test cases for model factory functions."""
    
    def test_create_quantization_config_4bit(self):
        """Test 4-bit quantization config creation."""
        config = create_quantization_config(bits=4)
        assert isinstance(config, BitsAndBytesConfig)
        assert config.load_in_4bit == True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant == True
    
    def test_create_quantization_config_8bit(self):
        """Test 8-bit quantization config creation."""
        config = create_quantization_config(bits=8)
        assert isinstance(config, BitsAndBytesConfig)
        assert config.load_in_8bit == True
    
    def test_create_quantization_config_invalid_bits(self):
        """Test quantization config with invalid bits."""
        with pytest.raises(ValueError):
            create_quantization_config(bits=16)
    
    def test_create_lora_config_standard(self):
        """Test standard LoRA config creation."""
        config = create_lora_config(
            task_type="classification",
            rank=8,
            alpha=16,
            dropout=0.1,
            adaptive=False,
        )
        assert isinstance(config, LoraConfig)
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
    
    def test_create_lora_config_adaptive(self):
        """Test adaptive LoRA config creation."""
        config = create_lora_config(
            task_type="classification",
            adaptive=True,
            init_r=12,
            target_r=4,
        )
        assert isinstance(config, AdaLoraConfig)
        assert config.init_r == 12
        assert config.target_r == 4
    
    def test_create_lora_config_invalid_task(self):
        """Test LoRA config with invalid task type."""
        with pytest.raises(ValueError):
            create_lora_config(task_type="invalid_task")
    
    def test_get_default_target_modules_bert(self):
        """Test default target modules for BERT."""
        modules = get_default_target_modules("bert-base-uncased")
        expected = ["query", "value", "key", "dense"]
        assert modules == expected
    
    def test_get_default_target_modules_gpt(self):
        """Test default target modules for GPT."""
        modules = get_default_target_modules("gpt2")
        expected = ["c_attn", "c_proj", "c_fc"]
        assert modules == expected
    
    def test_get_default_target_modules_t5(self):
        """Test default target modules for T5."""
        modules = get_default_target_modules("t5-small")
        expected = ["q", "v", "k", "o", "wi", "wo"]
        assert modules == expected
    
    def test_get_default_target_modules_generic(self):
        """Test default target modules for generic model."""
        modules = get_default_target_modules("some-random-model")
        expected = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert modules == expected 