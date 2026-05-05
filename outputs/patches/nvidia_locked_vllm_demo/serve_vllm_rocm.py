# =============================================================================
# AGENT-GENERATED FILE — ROCm FlightDeck starter template
# Review and adapt before using in production.
# Original files are NOT modified by this tool.
# =============================================================================
"""
vLLM ROCm serving script — starter template for AMD MI300X.
Adapt model name, tensor-parallel degree, and quantization to your setup.
"""

import os
import subprocess
import sys


def main() -> None:
    model = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
    tp = os.environ.get("TENSOR_PARALLEL", "1")
    port = os.environ.get("PORT", "8000")

    # HIP device selection (set to match your MI300X slot)
    hip_devices = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    os.environ["HIP_VISIBLE_DEVICES"] = hip_devices

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", tp,
        "--port", port,
        "--device", "cuda",   # ROCm PyTorch exposes HIP GPUs as CUDA device
        "--dtype", "float16",  # or bfloat16 for MI300X
    ]

    print(f"Starting vLLM ROCm server: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
