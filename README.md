# Adaptive LoRA Rank Allocation with Mixed-Precision Quantization

This project investigates joint optimization of LoRA rank allocation and mixed-precision quantization for efficient fine-tuning on laptop-class hardware.

## Quick Start

### Local Development (Mac/Linux)

```bash
# Install dependencies
pip install -r requirements.txt

# Run single experiment
python run_experiment.py --config B-FP --task sst2 --model bert-base-uncased

# Run full experiment matrix
python run_all_experiments.py --epochs 3 --batch-size 8
```

### Apple Silicon (MPS) Compatibility

Apple Silicon Macs have MPS (Metal Performance Shaders) limitations that can cause tensor size errors. The system automatically detects and applies MPS-safe settings:

```bash
# Automatic MPS detection (recommended)
python run_experiment.py --config B-FP --task sst2 --model bert-base-uncased

# Force MPS-safe settings (smaller batch sizes)
python run_experiment.py --config B-FP --task sst2 --model bert-base-uncased --mps-safe

# Manual batch size reduction
python run_experiment.py --config B-FP --task sst2 --model bert-base-uncased --batch-size 4
```

**If you encounter MPS tensor size errors**, try:
1. Reduce batch size: `--batch-size 4` or `--batch-size 2`
2. Use Docker with GPU backend (recommended for quantization)
3. Disable quantization: `--quant-backend none`

### Docker Deployment (GPU Recommended)

```bash
# Build and run on GPU
./scripts/docker-run.sh --build --gpu --wandb-key $WANDB_API_KEY run-all

# Run single experiment on CPU
./scripts/docker-run.sh --cpu run B-FP sst2

# Interactive development
./scripts/docker-run.sh --gpu shell
```

## Experiment Configurations

- **B-FP**: Baseline fixed-rank LoRA (FP16)
- **B-Q4**: Baseline 4-bit QLoRA  
- **B-Ada**: Baseline AdaLoRA (adaptive rank)
- **Joint-1/2/3**: Combined adaptive rank + quantization

## Quantization Backends

Use `--quant-backend` to control quantization strategy:

- `auto`: Auto-detect best backend (default)
- `cuda-4bit`: 4-bit quantization on GPU
- `cuda-8bit`: 8-bit quantization on GPU  
- `cpu-int8`: 8-bit CPU quantization
- `none`: Disable quantization

Examples:
```bash
# Force CPU quantization
python run_experiment.py --config B-Q4 --task sst2 --quant-backend cpu-int8

# GPU-only 4-bit (fails on non-CUDA)
python run_experiment.py --config B-Q4 --task sst2 --quant-backend cuda-4bit

# Disable quantization
python run_experiment.py --config B-Q4 --task sst2 --quant-backend none
```

## Troubleshooting

### MPS Tensor Size Limit Error

If you see:
```
MPSNDArray.mm:788: failed assertion `[MPSNDArray initWithDevice:descriptor:] Error: total bytes of NDArray > 2**32'
```

**Solutions:**
1. **Use smaller batch sizes**: `--batch-size 4` or `--batch-size 2`
2. **Use Docker with GPU**: Recommended for quantization experiments
3. **Disable quantization**: `--quant-backend none` for local testing

### High Memory Usage

For systems with limited RAM:
- Reduce batch size: `--batch-size 4`
- Use gradient accumulation (automatically adjusted)
- Close other applications

### Quantization Issues

On Apple Silicon:
- Use `--quant-backend none` for local development
- Use Docker with GPU for full quantization experiments
- CPU quantization: `--quant-backend cpu-int8` (slower but stable)

## Results Structure

```
results/
├── results_B-FP_sst2.json      # Individual experiment results
├── results_B-Q4_wikitext2.json
├── all_results.json            # Aggregated results
└── experiment_summary.csv      # Summary table
```

## Hardware Requirements

- **Local**: 16GB+ RAM, Apple M-series or Intel/AMD CPU
- **Docker**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **Quantization**: CUDA-compatible GPU or CPU fallback 