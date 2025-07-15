#!/bin/bash
# Docker deployment helper for adaptive-lora-rank-allocation experiments

set -e

# Configuration
IMAGE_NAME="adaptive-lora-gpu"
CONTAINER_NAME="adaptive-lora-experiment"
RESULTS_DIR="./results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Options:"
    echo "  --build         Build the Docker image"
    echo "  --gpu           Enable GPU support (requires NVIDIA Docker)"
    echo "  --cpu           Force CPU-only execution"
    echo "  --results DIR   Results output directory (default: ./results)"
    echo "  --wandb-key KEY Set WANDB API key"
    echo "  --hf-token TOK  Set Hugging Face token"
    echo ""
    echo "Commands:"
    echo "  run-all         Run complete experiment matrix"
    echo "  run CONFIG TASK Run single experiment (e.g. B-FP sst2)"
    echo "  shell           Interactive shell in container"
    echo ""
    echo "Examples:"
    echo "  $0 --build --gpu run-all"
    echo "  $0 --cpu run B-FP sst2"
    echo "  $0 --gpu --wandb-key \$WANDB_API_KEY run-all"
}

# Parse arguments
BUILD=false
GPU_SUPPORT=""
RESULTS_DIR="./results"
WANDB_KEY=""
HF_TOKEN=""
COMMAND=""
CONFIG=""
TASK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --gpu)
            GPU_SUPPORT="--gpus all"
            shift
            ;;
        --cpu)
            GPU_SUPPORT=""
            shift
            ;;
        --results)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --wandb-key)
            WANDB_KEY="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        run-all)
            COMMAND="run-all"
            shift
            ;;
        run)
            COMMAND="run"
            CONFIG="$2"
            TASK="$3"
            shift 3
            ;;
        shell)
            COMMAND="shell"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Create results directory
mkdir -p "$RESULTS_DIR"

# Build Docker image if requested
if [ "$BUILD" = true ]; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t "$IMAGE_NAME" .
    echo -e "${GREEN}Docker image built successfully${NC}"
fi

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo -e "${RED}Docker image '$IMAGE_NAME' not found. Run with --build first.${NC}"
    exit 1
fi

# Auto-detect GPU support if not specified
if [ -z "$GPU_SUPPORT" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}NVIDIA GPU detected, enabling GPU support${NC}"
        GPU_SUPPORT="--gpus all"
    else
        echo -e "${YELLOW}No GPU detected, using CPU-only mode${NC}"
        GPU_SUPPORT=""
    fi
fi

# Prepare environment variables
ENV_ARGS=""
if [ -n "$WANDB_KEY" ]; then
    ENV_ARGS="$ENV_ARGS -e WANDB_API_KEY=$WANDB_KEY"
fi
if [ -n "$HF_TOKEN" ]; then
    ENV_ARGS="$ENV_ARGS -e HF_TOKEN=$HF_TOKEN"
fi

# Execute command
case $COMMAND in
    run-all)
        echo -e "${GREEN}Running complete experiment matrix...${NC}"
        docker run --rm $GPU_SUPPORT $ENV_ARGS \
            -v "$(pwd)/$RESULTS_DIR:/workspace/results" \
            --name "$CONTAINER_NAME" \
            "$IMAGE_NAME"
        ;;
    run)
        if [ -z "$CONFIG" ] || [ -z "$TASK" ]; then
            echo -e "${RED}Error: CONFIG and TASK required for single run${NC}"
            echo "Usage: $0 run CONFIG TASK"
            exit 1
        fi
        echo -e "${GREEN}Running single experiment: $CONFIG on $TASK${NC}"
        docker run --rm $GPU_SUPPORT $ENV_ARGS \
            -v "$(pwd)/$RESULTS_DIR:/workspace/results" \
            --name "$CONTAINER_NAME" \
            "$IMAGE_NAME" \
            python run_experiment.py --config "$CONFIG" --task "$TASK" \
            --model bert-base-uncased --output-dir /workspace/results
        ;;
    shell)
        echo -e "${GREEN}Starting interactive shell...${NC}"
        docker run --rm -it $GPU_SUPPORT $ENV_ARGS \
            -v "$(pwd)/$RESULTS_DIR:/workspace/results" \
            --name "$CONTAINER_NAME" \
            "$IMAGE_NAME" /bin/bash
        ;;
    "")
        echo -e "${RED}No command specified${NC}"
        usage
        exit 1
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        usage
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}" 