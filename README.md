# Adaptive LoRA Rank Allocation with Mixed-Precision Quantization

This repository contains the experimental code for investigating whether jointly optimizing LoRA rank allocation and mixed-precision quantization yields better efficiency-performance trade-offs than existing baselines on laptop-class hardware.

## 🎯 Project Overview

We explore the combination of:
- **Adaptive LoRA rank allocation** (AdaLoRA) - dynamically assigning ranks to layers based on importance
- **Mixed-precision quantization** (8-bit/4-bit) - using different precisions for different layers
- **Resource constraints** - all experiments run on Apple Silicon MacBooks with ≤16GB RAM

## 🏗️ Architecture

```
src/
├── data/           # Dataset loading utilities
├── models/         # Model factory functions
└── train.py        # Training pipeline with monitoring

run_experiment.py   # Single experiment runner
run_all_experiments.py  # Full experiment matrix
tests/              # Unit tests
```

## 📋 Experimental Matrix

| Config ID | Method | Quantization | LoRA Strategy |
|-----------|--------|-------------|---------------|
| B-FP | Baseline | FP16 | Fixed rank=8 |
| B-Q4 | QLoRA | 4-bit | Fixed rank=8 |
| B-Ada | AdaLoRA | FP16 | Adaptive (12→4) |
| Joint-1 | Joint | 4-bit | Adaptive (12→4) |
| Joint-2 | Joint | Mixed 8/4-bit | Adaptive (12→4) |
| Joint-3 | Joint | Mixed 8/4-bit | Manual ranks |

Tasks: **SST-2** (classification) and **WikiText-2** (language modeling)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/adaptive-lora-rank-allocation.git
cd adaptive-lora-rank-allocation

# Install dependencies
make install

# Development setup (optional)
make setup
```

### Running Experiments

```bash
# Single experiment
python run_experiment.py --config B-FP --task sst2 --model bert-base-uncased

# Full experiment matrix (12 runs)
python run_all_experiments.py

# Quick test run
make test-run
```

### Environment Variables

Create `.env` file (optional):
```bash
WANDB_API_KEY=your_wandb_key  # For experiment tracking
HF_TOKEN=your_hf_token        # For private models
```

## 🧪 Experiment Configurations

### Baselines
- **B-FP**: Standard LoRA with FP16 weights
- **B-Q4**: QLoRA with 4-bit quantization
- **B-Ada**: AdaLoRA with FP16 weights

### Joint Methods
- **Joint-1**: AdaLoRA + 4-bit quantization
- **Joint-2**: AdaLoRA + mixed-precision (critical layers 8-bit, others 4-bit)
- **Joint-3**: Manual rank allocation + mixed-precision

## 📊 Results Analysis

Results are saved in `results/` directory:
- `experiment_summary.csv` - Consolidated metrics table
- `all_results.json` - Full experiment data
- Individual run directories with checkpoints

Key metrics tracked:
- Task performance (accuracy/perplexity)
- Trainable parameters (% of total)
- Peak memory usage (MB)
- Training time (seconds)

## 🔧 Development

### Testing
```bash
make test        # Run unit tests
make lint        # Check code quality
make format      # Format code
```

### Adding New Experiments
1. Add config to `get_experiment_config()` in `run_experiment.py`
2. Update experiment matrix in `run_all_experiments.py`
3. Add tests in `tests/`

## 📝 Hardware Requirements

- **Minimum**: MacBook with M1/M2 chip, 8GB RAM
- **Recommended**: MacBook with M3 chip, 16GB+ RAM
- **GPU**: Uses MPS when available, falls back to CPU for quantization

## 📖 Implementation Details

### Key Libraries
- 🤗 Transformers ≥4.40 (model loading)
- PEFT ≥0.10 (LoRA/AdaLoRA)
- bitsandbytes ≥0.43 (quantization)
- PyTorch ≥2.2 (training)

### Quantization Strategy
- **4-bit**: NF4 quantization with double quantization
- **8-bit**: Standard int8 quantization
- **Mixed**: Per-layer precision allocation

### LoRA Configuration
- **Fixed**: Uniform rank across all layers
- **Adaptive**: SVD-based pruning (AdaLoRA)
- **Manual**: Task-specific rank patterns

## 🎨 Memory Bank Integration

This project uses an enhanced memory management system:
- `.memory/` directory contains project knowledge base
- Automatic updates after successful experiments
- Semantic search capabilities for project context

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `make lint` and `make test`
5. Submit a pull request

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)

## 📄 License

MIT License - see LICENSE file for details.

## 🙋 Support

For questions or issues:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Include system information and error logs 