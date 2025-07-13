"""Unit tests for data loader module."""

import pytest
from transformers import AutoTokenizer
from src.data.loader import load_sst2_dataset, load_wikitext2_dataset, get_data_collator


class TestDataLoader:
    """Test cases for data loading utilities."""
    
    def test_load_sst2_dataset(self):
        """Test SST-2 dataset loading."""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        dataset = load_sst2_dataset(tokenizer, max_length=128)
        
        assert "train" in dataset
        assert "validation" in dataset
        assert "test" in dataset
        
        # Check that tokenization worked
        train_sample = dataset["train"][0]
        assert "input_ids" in train_sample
        assert "attention_mask" in train_sample
        assert "labels" in train_sample
    
    def test_load_wikitext2_dataset(self):
        """Test WikiText-2 dataset loading."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        dataset = load_wikitext2_dataset(tokenizer, max_length=512)
        
        assert "train" in dataset
        assert "validation" in dataset
        assert "test" in dataset
        
        # Check that tokenization worked
        train_sample = dataset["train"][0]
        assert "input_ids" in train_sample
        assert "attention_mask" in train_sample
    
    def test_get_data_collator_classification(self):
        """Test data collator for classification."""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        collator = get_data_collator(tokenizer, "classification")
        assert collator is not None
    
    def test_get_data_collator_language_modeling(self):
        """Test data collator for language modeling."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        collator = get_data_collator(tokenizer, "language_modeling")
        assert collator is not None
    
    def test_get_data_collator_invalid_task(self):
        """Test data collator with invalid task type."""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        with pytest.raises(ValueError):
            get_data_collator(tokenizer, "invalid_task") 