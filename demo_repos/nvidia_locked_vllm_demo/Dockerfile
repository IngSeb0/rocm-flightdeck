FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="demo-team"
LABEL description="NVIDIA-locked vLLM inference demo"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CUDA_HOME=/usr/local/cuda

CMD ["python", "app.py"]
