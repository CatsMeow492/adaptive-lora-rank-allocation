FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python 3.11 and basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        git \
        curl \
        ca-certificates \
        build-essential && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /workspace

# Copy only requirements first for efficient layer caching
COPY requirements.txt ./

# Install PyTorch + CUDA matching base image
RUN pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python deps
RUN pip install -r requirements.txt

# bitsandbytes with CUDA support
RUN BITSANDBYTES_VERSION=0.43.0 \
    pip install bitsandbytes==${BITSANDBYTES_VERSION} --no-binary bitsandbytes

# Copy the rest of the project
COPY . .

# Expose default wandb port (optional)
EXPOSE 8080

ENTRYPOINT ["python", "run_all_experiments.py"] 