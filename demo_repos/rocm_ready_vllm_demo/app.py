"""ROCm-ready vLLM/PyTorch demo fixture."""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_device() -> torch.device:
    return device


def describe_runtime() -> str:
    return "ROCm PyTorch exposes AMD GPUs through the CUDA-compatible PyTorch API."
