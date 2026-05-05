"""ROCm vLLM serving fixture."""

import os
import sys


def build_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-hf"),
        "--device",
        "cuda",
    ]
